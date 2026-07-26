from __future__ import annotations

import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek
from pydantic import BaseModel, Field, field_validator
from bs4 import BeautifulSoup


REDUCE_PROMPT_VERSION = "filing-reduce-v1"
SECTION_ORDER = ("business", "risk_factors", "mda")
Materiality = Literal["high", "medium", "low"]
RiskStatus = Literal["potential", "realized", "mixed"]


def _as_list(value) -> list:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _normalize_source_ids(value) -> list[str]:
    return [str(item).strip().upper() for item in _as_list(value) if str(item).strip()]


class BusinessPoint(BaseModel):
    title_cn: str
    analysis_cn: str
    source_ids: list[str]

    @field_validator("source_ids", mode="before")
    @classmethod
    def normalize_ids(cls, value):
        return _normalize_source_ids(value)


class BusinessReduce(BaseModel):
    section_summary_cn: str
    key_points: list[BusinessPoint]
    open_questions_cn: list[str] = Field(default_factory=list)

    @field_validator("open_questions_cn", mode="before")
    @classmethod
    def normalize_questions(cls, value):
        return [str(item).strip() for item in _as_list(value) if str(item).strip()]


class RiskPoint(BaseModel):
    title_cn: str
    assessment_cn: str
    status: RiskStatus
    source_ids: list[str]

    @field_validator("source_ids", mode="before")
    @classmethod
    def normalize_ids(cls, value):
        return _normalize_source_ids(value)


class RiskReduce(BaseModel):
    section_summary_cn: str
    key_risks: list[RiskPoint]
    risk_connections_cn: list[str] = Field(default_factory=list)
    open_questions_cn: list[str] = Field(default_factory=list)

    @field_validator("risk_connections_cn", "open_questions_cn", mode="before")
    @classmethod
    def normalize_lists(cls, value):
        return [str(item).strip() for item in _as_list(value) if str(item).strip()]


class MDAPoint(BaseModel):
    title_cn: str
    analysis_cn: str
    source_ids: list[str]

    @field_validator("source_ids", mode="before")
    @classmethod
    def normalize_ids(cls, value):
        return _normalize_source_ids(value)


class MDAReduce(BaseModel):
    section_summary_cn: str
    key_metrics: list[MDAPoint]
    management_themes: list[MDAPoint]
    open_questions_cn: list[str] = Field(default_factory=list)

    @field_validator("open_questions_cn", mode="before")
    @classmethod
    def normalize_questions(cls, value):
        return [str(item).strip() for item in _as_list(value) if str(item).strip()]


COMMON_SYSTEM = """你是一名审慎的 SEC 财报研究助理，负责把已验证的 Map 结果汇总成研究底稿。
只能使用输入 JSON 中的信息，不使用外部知识，不提供投资建议，也不能把潜在风险改写成已经发生的事实。
source_ids 引用的是输入候选记录的全局编号，只能逐字使用真实存在的编号，不得创造编号。
每一个关键结论必须包含至少一个 source_id。中文表述应简洁、具体，并区分事实、管理层解释和风险。
只返回一个 JSON 对象，不要返回 Markdown。"""

BUSINESS_SYSTEM = COMMON_SYSTEM + """
输出 section_summary_cn、key_points、open_questions_cn。
key_points 提取 4 到 8 个最重要的业务主题，每项包含 title_cn、analysis_cn、source_ids。
合并重复信息，但不要增加输入中没有的因果关系。"""

RISK_SYSTEM = COMMON_SYSTEM + """
输出 section_summary_cn、key_risks、risk_connections_cn、open_questions_cn。
key_risks 提取 6 到 12 个最重要的风险主题，每项包含 title_cn、assessment_cn、status、source_ids。
status 只能是 potential、realized、mixed，并且必须与所引用记录的语气一致。
risk_connections_cn 只描述输入可直接支持的风险联动，不需要 source_ids；证据不足时返回空数组。"""

MDA_SYSTEM = COMMON_SYSTEM + """
输出 section_summary_cn、key_metrics、management_themes、open_questions_cn。
key_metrics 提取 5 到 12 个最重要的财务指标或趋势，每项包含 title_cn、analysis_cn、source_ids，优先覆盖收入、地区或产品表现、毛利、费用、现金或资本回报等输入实际提供的主题。
management_themes 提取 2 到 8 个管理层讨论主题，每项也包含 title_cn、analysis_cn、source_ids。
数值、期间、单位和变化方向必须忠于输入，不自行计算。"""


def _make_prompt(system_prompt: str) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "human",
                """ticker: {ticker}
form: {form}
section: {section}

<validated_map_candidates>
{candidates_json}
</validated_map_candidates>

上一次校验反馈（首次调用为空）：
{validation_feedback}""",
            ),
        ]
    )


REDUCE_CONFIG = {
    "business": (BusinessReduce, _make_prompt(BUSINESS_SYSTEM)),
    "risk_factors": (RiskReduce, _make_prompt(RISK_SYSTEM)),
    "mda": (MDAReduce, _make_prompt(MDA_SYSTEM)),
}


class FilingReducer:
    def __init__(
        self,
        company_dir: Path,
        map_cache_dir: Path,
        *,
        model_name: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0,
        max_attempts: int = 2,
        request_delay_seconds: float = 1.0,
    ) -> None:
        self.company_dir = company_dir
        self.map_cache_dir = map_cache_dir
        self.output_path = company_dir / "reduce_v1.json"
        self.memo_path = company_dir / "research_memo_draft.md"
        self.model_name = model_name or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY", "")
        self.base_url = base_url or os.getenv(
            "DEEPSEEK_BASE_URL", "https://api.deepseek.com"
        )
        self.temperature = temperature
        self.max_attempts = max_attempts
        self.request_delay_seconds = request_delay_seconds
        self._llm: ChatDeepSeek | None = None

    def run(
        self,
        chunks: list[dict],
        *,
        max_new_calls: int | None = 3,
    ) -> dict:
        ticker = chunks[0]["ticker"] if chunks else ""
        form = chunks[0]["form"] if chunks else ""
        candidates, collection = self.collect_candidates(chunks)
        fingerprint = self._fingerprint(candidates)
        cache = self._load_or_initialize_cache(
            ticker=ticker,
            form=form,
            fingerprint=fingerprint,
            candidates=candidates,
            collection=collection,
        )

        cached_sections = [
            section for section in SECTION_ORDER if self._section_is_current(cache, section)
        ]
        pending = [section for section in SECTION_ORDER if section not in cached_sections]
        selected = pending if max_new_calls is None else pending[:max_new_calls]
        stats = {
            "input_fingerprint": fingerprint,
            "cached_before": len(cached_sections),
            "pending_before": len(pending),
            "selected_new_calls": len(selected),
            "new_successes": 0,
            "validation_failed": 0,
            "exceptions": [],
        }

        for index, section in enumerate(selected, start=1):
            print(f"[Reduce {index}/{len(selected)}] 汇总 {section}")
            try:
                section_result = self._reduce_section(
                    ticker=ticker,
                    form=form,
                    section=section,
                    candidates=candidates,
                )
                cache["sections"][section] = section_result
                if section_result["validation"]["structure_passed"]:
                    stats["new_successes"] += 1
                else:
                    stats["validation_failed"] += 1
            except Exception as exc:
                stats["exceptions"].append({"section": section, "error": str(exc)})
            self._finalize_cache(cache)
            if index < len(selected):
                time.sleep(self.request_delay_seconds)

        self._finalize_cache(cache)
        complete = cache["validation"]["all_sections_passed"]
        if complete:
            self._write_memo(cache)

        cached_after = sum(
            self._section_is_current(cache, section) for section in SECTION_ORDER
        )
        stats.update(
            {
                "cached_after": cached_after,
                "pending_after": len(SECTION_ORDER) - cached_after,
                "complete": complete,
                "output_file": str(self.output_path),
                "memo_file": str(self.memo_path) if complete else None,
            }
        )
        return stats

    def collect_candidates(self, chunks: list[dict]) -> tuple[dict, dict]:
        raw = {"business": [], "risks": [], "mda_metrics": [], "mda_facts": []}
        for chunk in chunks:
            cache_path = self.map_cache_dir / f"{chunk['chunk_id']}.json"
            if not cache_path.exists():
                raise FileNotFoundError(f"缺少 Map 缓存: {cache_path}")
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if not cached.get("validation", {}).get("structure_passed"):
                raise ValueError(f"Map 缓存未通过结构校验: {chunk['chunk_id']}")
            self._collect_chunk(raw, cached)

        before = {key: len(value) for key, value in raw.items()}
        deduped = {
            "business": self._dedupe_text_records(raw["business"], "fact_cn", "business"),
            "risks": self._dedupe_text_records(raw["risks"], "risk_cn", "risk"),
            "mda_metrics": self._dedupe_metrics(raw["mda_metrics"]),
            "mda_facts": self._dedupe_text_records(raw["mda_facts"], "fact_cn", "mda"),
        }
        prefixes = {
            "business": "B",
            "risks": "RISK",
            "mda_metrics": "M",
            "mda_facts": "D",
        }
        for key, records in deduped.items():
            records.sort(key=self._record_sort_key)
            for index, record in enumerate(records, start=1):
                record["global_id"] = f"{prefixes[key]}{index:03d}"
                record["canonical_quotes"] = self._canonical_quotes(record["evidence"])

        after = {key: len(value) for key, value in deduped.items()}
        return deduped, {"before_dedup": before, "after_dedup": after}

    def _collect_chunk(self, raw: dict, cached: dict) -> None:
        metadata = cached["chunk_metadata"]
        output = cached["model_output"]
        resolved = cached["resolved_evidence"]
        review_status = cached["validation"].get("semantic_review_status", "pending")
        section = metadata["section"]

        if section == "business":
            for item in output.get("facts", []):
                raw["business"].append(
                    self._make_record(item, metadata, resolved.get(item["fact_id"], []), review_status)
                )
        elif section == "risk_factors":
            for item in output.get("risks", []):
                raw["risks"].append(
                    self._make_record(item, metadata, resolved.get(item["risk_id"], []), review_status)
                )
        elif section == "mda":
            for item in output.get("metrics", []):
                raw["mda_metrics"].append(
                    self._make_record(
                        item,
                        metadata,
                        resolved.get("metrics", {}).get(item["metric_id"], []),
                        review_status,
                    )
                )
            for item in output.get("facts", []):
                raw["mda_facts"].append(
                    self._make_record(
                        item,
                        metadata,
                        resolved.get("facts", {}).get(item["fact_id"], []),
                        review_status,
                    )
                )

    @staticmethod
    def _make_record(item: dict, metadata: dict, evidence: list[dict], review_status: str) -> dict:
        local_id = item.get("fact_id") or item.get("risk_id") or item.get("metric_id")
        record = {
            key: value
            for key, value in item.items()
            if key not in {"fact_id", "risk_id", "metric_id", "evidence_ids"}
        }
        record["source_refs"] = [
            {
                "chunk_id": metadata["chunk_id"],
                "local_item_id": local_id,
                "semantic_review_status": review_status,
            }
        ]
        record["evidence"] = FilingReducer._dedupe_evidence(evidence)
        record["semantic_review_status"] = review_status
        return record

    def _dedupe_text_records(self, records: list[dict], text_key: str, kind: str) -> list[dict]:
        kept: list[dict] = []
        for record in sorted(records, key=self._record_sort_key):
            match = next(
                (candidate for candidate in kept if self._is_duplicate(record, candidate, text_key, kind)),
                None,
            )
            if match is None:
                kept.append(record)
            else:
                self._merge_records(match, record, kind)
        return kept

    def _is_duplicate(self, left: dict, right: dict, text_key: str, kind: str) -> bool:
        left_text = self._normalize_text(left.get(text_key, ""))
        right_text = self._normalize_text(right.get(text_key, ""))
        if not left_text or not right_text:
            return False
        ratio = SequenceMatcher(None, left_text, right_text).ratio()
        if kind != "risk":
            return ratio >= 0.86
        if ratio >= 0.9:
            return True
        left_ranges = {
            (item.get("source_start"), item.get("source_end"))
            for item in left.get("evidence", [])
        }
        right_ranges = {
            (item.get("source_start"), item.get("source_end"))
            for item in right.get("evidence", [])
        }
        if left_ranges & right_ranges and ratio >= 0.5:
            return True
        left_detail = self._normalize_text(
            " ".join(left.get("causes_cn", []) + left.get("potential_impacts_cn", []))
        )
        right_detail = self._normalize_text(
            " ".join(right.get("causes_cn", []) + right.get("potential_impacts_cn", []))
        )
        detail_ratio = SequenceMatcher(None, left_detail, right_detail).ratio()
        return ratio >= 0.75 and detail_ratio >= 0.82

    def _dedupe_metrics(self, records: list[dict]) -> list[dict]:
        kept: list[dict] = []
        by_key: dict[tuple[str, str], dict] = {}
        for record in sorted(records, key=self._record_sort_key):
            key = (
                self._normalize_text(record.get("metric_name", "")),
                self._normalize_text(record.get("current_period", "")),
            )
            if key not in by_key:
                by_key[key] = record
                kept.append(record)
            else:
                self._merge_records(by_key[key], record, "metric")
        return kept

    def _merge_records(self, target: dict, incoming: dict, kind: str) -> None:
        target["source_refs"] = self._unique_dicts(
            target["source_refs"] + incoming["source_refs"],
            ("chunk_id", "local_item_id"),
        )
        target["evidence"] = self._dedupe_evidence(target["evidence"] + incoming["evidence"])
        target["semantic_review_status"] = (
            "reviewed"
            if all(
                ref.get("semantic_review_status") == "reviewed"
                for ref in target["source_refs"]
            )
            else "pending"
        )
        materiality_rank = {"low": 0, "medium": 1, "high": 2}
        if materiality_rank.get(incoming.get("materiality"), -1) > materiality_rank.get(
            target.get("materiality"), -1
        ):
            target["materiality"] = incoming["materiality"]
        if kind == "risk":
            target["causes_cn"] = self._unique_strings(
                target.get("causes_cn", []) + incoming.get("causes_cn", [])
            )
            target["potential_impacts_cn"] = self._unique_strings(
                target.get("potential_impacts_cn", [])
                + incoming.get("potential_impacts_cn", [])
            )
            statuses = {target.get("status"), incoming.get("status")}
            if len(statuses - {None}) > 1:
                target["status"] = "mixed"
        elif kind == "metric":
            current_explanation = target.get("management_explanation_cn") or ""
            incoming_explanation = incoming.get("management_explanation_cn") or ""
            if len(incoming_explanation) > len(current_explanation):
                target["management_explanation_cn"] = incoming_explanation

    def _reduce_section(
        self,
        *,
        ticker: str,
        form: str,
        section: str,
        candidates: dict,
    ) -> dict:
        schema, prompt = REDUCE_CONFIG[section]
        section_candidates = self._section_candidates(section, candidates)
        allowed_ids = self._allowed_ids(section, candidates)
        prompt_candidates = self._prompt_candidates(section_candidates)
        candidates_json = json.dumps(prompt_candidates, ensure_ascii=False, indent=2)
        feedback = ""
        last_error: Exception | None = None

        for attempt in range(1, self.max_attempts + 1):
            try:
                chain = prompt | self._get_llm().bind(
                    response_format={"type": "json_object"}
                )
                raw_message = chain.invoke(
                    {
                        "ticker": ticker,
                        "form": form,
                        "section": section,
                        "candidates_json": candidates_json,
                        "validation_feedback": feedback,
                    }
                )
                output = self._parse_output(raw_message.content, section, schema)
                errors = self._validate_output(
                    section, output, allowed_ids, section_candidates
                )
                if errors:
                    feedback = "；".join(errors)
                    last_error = ValueError(feedback)
                    if attempt < self.max_attempts:
                        time.sleep(2)
                        continue
                return {
                    "run_metadata": {
                        "prompt_version": REDUCE_PROMPT_VERSION,
                        "analyzed_at_utc": datetime.now(timezone.utc).isoformat(),
                        "model": self.model_name,
                        "temperature": self.temperature,
                        "attempt": attempt,
                        "token_usage": raw_message.usage_metadata or {},
                    },
                    "model_output": output.model_dump(),
                    "validation": {
                        "structure_passed": not errors,
                        "structure_errors": errors,
                        "semantic_review_status": "pending",
                    },
                }
            except Exception as exc:
                last_error = exc
                feedback = str(exc)
                if attempt < self.max_attempts:
                    time.sleep(2)

        raise RuntimeError(f"{section} Reduce 失败: {last_error}")

    @staticmethod
    def _parse_output(raw_content, section: str, schema):
        if not isinstance(raw_content, str):
            raw_content = str(raw_content)
        text = raw_content.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError("Reduce JSON 顶层必须是对象")
        aliases = {
            "summary_cn": "section_summary_cn",
            "summary": "section_summary_cn",
            "business_points": "key_points",
            "risks": "key_risks",
            "metrics": "key_metrics",
            "themes": "management_themes",
            "questions_cn": "open_questions_cn",
        }
        for alias, standard in aliases.items():
            if alias in payload and standard not in payload:
                payload[standard] = payload.pop(alias)
        if section == "business":
            payload.setdefault("open_questions_cn", [])
        elif section == "risk_factors":
            payload.setdefault("risk_connections_cn", [])
            payload.setdefault("open_questions_cn", [])
        else:
            payload.setdefault("key_metrics", [])
            payload.setdefault("management_themes", [])
            payload.setdefault("open_questions_cn", [])
        return schema.model_validate(payload)

    @staticmethod
    def _validate_output(
        section: str,
        output,
        allowed_ids: set[str],
        section_candidates: dict,
    ) -> list[str]:
        errors: list[str] = []
        if section == "business":
            points = output.key_points
            if not 4 <= len(points) <= 8:
                errors.append("Business key_points 数量必须在 4 到 8 之间")
        elif section == "risk_factors":
            points = output.key_risks
            if not 6 <= len(points) <= 12:
                errors.append("Risk key_risks 数量必须在 6 到 12 之间")
        else:
            points = output.key_metrics + output.management_themes
            if not 5 <= len(output.key_metrics) <= 12:
                errors.append("MD&A key_metrics 数量必须在 5 到 12 之间")
            if not 2 <= len(output.management_themes) <= 8:
                errors.append("MD&A management_themes 数量必须在 2 到 8 之间")

        for index, point in enumerate(points, start=1):
            if not point.source_ids:
                errors.append(f"第 {index} 个结论缺少 source_ids")
                continue
            invalid = sorted(set(point.source_ids) - allowed_ids)
            if invalid:
                errors.append(f"第 {index} 个结论引用了无效编号: {', '.join(invalid)}")
            if section == "risk_factors" and not invalid:
                candidate_index = {
                    record["global_id"]: record
                    for record in section_candidates["risks"]
                }
                source_statuses = {
                    candidate_index[source_id].get("status")
                    for source_id in point.source_ids
                }
                if point.status == "potential" and source_statuses - {"potential"}:
                    errors.append(f"第 {index} 个风险的 potential 状态与引用记录不一致")
                if point.status == "realized" and source_statuses - {"realized"}:
                    errors.append(f"第 {index} 个风险的 realized 状态与引用记录不一致")
                if point.status == "mixed" and source_statuses == {"potential"}:
                    errors.append(f"第 {index} 个风险缺少支持 mixed 状态的引用记录")
        embedded_ids = FilingReducer._find_global_ids(output.model_dump())
        invalid_embedded = sorted(embedded_ids - allowed_ids)
        if invalid_embedded:
            errors.append(
                "Reduce 文本中出现无效全局编号: " + ", ".join(invalid_embedded)
            )
        return errors

    def _get_llm(self) -> ChatDeepSeek:
        if self._llm is None:
            if not self.api_key:
                raise ValueError("请先在 .env 中配置 DEEPSEEK_API_KEY")
            self._llm = ChatDeepSeek(
                model=self.model_name,
                api_key=self.api_key,
                base_url=self.base_url,
                temperature=self.temperature,
                request_timeout=120,
                max_retries=1,
            )
        return self._llm

    def _load_or_initialize_cache(
        self,
        *,
        ticker: str,
        form: str,
        fingerprint: str,
        candidates: dict,
        collection: dict,
    ) -> dict:
        if self.output_path.exists():
            cached = json.loads(self.output_path.read_text(encoding="utf-8"))
            if (
                cached.get("run_metadata", {}).get("input_fingerprint") == fingerprint
                and cached.get("run_metadata", {}).get("prompt_version")
                == REDUCE_PROMPT_VERSION
            ):
                return cached
        return {
            "run_metadata": {
                "prompt_version": REDUCE_PROMPT_VERSION,
                "input_fingerprint": fingerprint,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            },
            "filing": {"ticker": ticker, "form": form},
            "collection": collection,
            "candidates": candidates,
            "sections": {},
            "validation": {
                "all_sections_passed": False,
                "completed_sections": [],
                "pending_sections": list(SECTION_ORDER),
            },
        }

    def _finalize_cache(self, cache: dict) -> None:
        completed = [
            section for section in SECTION_ORDER if self._section_is_current(cache, section)
        ]
        cache["run_metadata"]["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        cache["validation"] = {
            "all_sections_passed": len(completed) == len(SECTION_ORDER),
            "completed_sections": completed,
            "pending_sections": [section for section in SECTION_ORDER if section not in completed],
        }
        self.output_path.write_text(
            json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @staticmethod
    def _section_is_current(cache: dict, section: str) -> bool:
        result = cache.get("sections", {}).get(section, {})
        return bool(
            result.get("validation", {}).get("structure_passed")
            and result.get("run_metadata", {}).get("prompt_version")
            == REDUCE_PROMPT_VERSION
        )

    @staticmethod
    def _section_candidates(section: str, candidates: dict) -> dict:
        if section == "business":
            return {"business": candidates["business"]}
        if section == "risk_factors":
            return {"risks": candidates["risks"]}
        return {
            "mda_metrics": candidates["mda_metrics"],
            "mda_facts": candidates["mda_facts"],
        }

    @staticmethod
    def _prompt_candidates(section_candidates: dict) -> dict:
        compact = {}
        for key, records in section_candidates.items():
            compact[key] = []
            for record in records:
                item = {
                    field: value
                    for field, value in record.items()
                    if field not in {"evidence", "source_refs", "semantic_review_status"}
                }
                item["canonical_quotes"] = record.get("canonical_quotes", [])[:2]
                compact[key].append(item)
        return compact

    @staticmethod
    def _allowed_ids(section: str, candidates: dict) -> set[str]:
        selected = FilingReducer._section_candidates(section, candidates)
        return {
            record["global_id"]
            for records in selected.values()
            for record in records
        }

    @staticmethod
    def _fingerprint(candidates: dict) -> str:
        payload = json.dumps(
            {"prompt_version": REDUCE_PROMPT_VERSION, "candidates": candidates},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _normalize_text(value: str) -> str:
        return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", str(value).lower())

    @staticmethod
    def _record_sort_key(record: dict) -> tuple:
        evidence = record.get("evidence", [])
        start = min((item.get("source_start", 10**12) for item in evidence), default=10**12)
        source_ref = record.get("source_refs", [{}])[0]
        return start, source_ref.get("chunk_id", ""), source_ref.get("local_item_id", "")

    @staticmethod
    def _dedupe_evidence(evidence: list[dict]) -> list[dict]:
        unique: dict[tuple, dict] = {}
        for item in evidence:
            normalized = {
                key: item.get(key)
                for key in (
                    "text",
                    "display_text",
                    "local_start",
                    "local_end",
                    "source_start",
                    "source_end",
                )
            }
            key = (
                normalized.get("source_start"),
                normalized.get("source_end"),
                normalized.get("display_text"),
            )
            unique[key] = normalized
        return sorted(
            unique.values(),
            key=lambda item: (item.get("source_start", 10**12), item.get("source_end", 10**12)),
        )

    @staticmethod
    def _canonical_quotes(evidence: list[dict], limit: int = 3) -> list[dict]:
        return [
            {
                "source_start": item.get("source_start"),
                "source_end": item.get("source_end"),
                "quote": item.get("display_text") or item.get("text") or "",
            }
            for item in evidence[:limit]
        ]

    @staticmethod
    def _unique_strings(values: list[str]) -> list[str]:
        seen = set()
        result = []
        for value in values:
            key = FilingReducer._normalize_text(value)
            if key and key not in seen:
                seen.add(key)
                result.append(value)
        return result

    @staticmethod
    def _unique_dicts(values: list[dict], keys: tuple[str, ...]) -> list[dict]:
        seen = set()
        result = []
        for value in values:
            key = tuple(value.get(name) for name in keys)
            if key not in seen:
                seen.add(key)
                result.append(value)
        return result

    def _write_memo(self, cache: dict) -> None:
        outputs = {
            section: cache["sections"][section]["model_output"]
            for section in SECTION_ORDER
        }
        cited_ids = self._cited_ids(outputs)
        candidate_index = {
            record["global_id"]: record
            for records in cache["candidates"].values()
            for record in records
        }
        filing_metadata = self._load_filing_metadata()
        ticker = cache["filing"]["ticker"]
        form = cache["filing"]["form"]
        lines = [
            f"# {ticker} {form} 研究备忘录（草稿）",
            "",
            "> 本文由自动化 Pipeline 基于 SEC 原文生成，尚需人工语义复核，不构成投资建议。",
            "",
            "## 基本信息",
            "",
            f"- 公司：{filing_metadata.get('name', ticker)}",
            f"- 报告类型：{form}",
            f"- 报告期：{filing_metadata.get('report_date', '未记录')}",
            f"- 申报日期：{filing_metadata.get('filing_date', '未记录')}",
            f"- SEC 原文：{filing_metadata.get('url', '未记录')}",
            f"- Reduce 模型：{cache['sections']['business']['run_metadata'].get('model', '')}",
            "",
            "## 执行摘要",
            "",
            f"- 业务：{outputs['business']['section_summary_cn']}",
            f"- 风险：{outputs['risk_factors']['section_summary_cn']}",
            f"- MD&A：{outputs['mda']['section_summary_cn']}",
            "",
            "## 业务概览",
            "",
        ]
        for point in outputs["business"]["key_points"]:
            lines.extend(self._memo_point(point["title_cn"], point["analysis_cn"], point["source_ids"]))

        lines.extend(["## 关键风险", ""])
        for point in outputs["risk_factors"]["key_risks"]:
            title = f"{point['title_cn']}（{point['status']}）"
            lines.extend(self._memo_point(title, point["assessment_cn"], point["source_ids"]))
        connections = outputs["risk_factors"].get("risk_connections_cn", [])
        if connections:
            lines.extend(["### 风险联动", ""])
            lines.extend(f"- {item}" for item in connections)
            lines.append("")

        lines.extend(["## MD&A 核心指标", ""])
        for point in outputs["mda"]["key_metrics"]:
            lines.extend(self._memo_point(point["title_cn"], point["analysis_cn"], point["source_ids"]))
        lines.extend(["## 管理层讨论主题", ""])
        for point in outputs["mda"]["management_themes"]:
            lines.extend(self._memo_point(point["title_cn"], point["analysis_cn"], point["source_ids"]))

        questions = (
            outputs["business"].get("open_questions_cn", [])
            + outputs["risk_factors"].get("open_questions_cn", [])
            + outputs["mda"].get("open_questions_cn", [])
        )
        lines.extend(["## 待进一步核查", ""])
        if questions:
            lines.extend(f"- {item}" for item in self._unique_strings(questions))
        else:
            lines.append("- 当前 Reduce 输出未提出额外核查问题；仍需人工完成语义与重要性复核。")
        lines.extend(["", "## 证据索引", ""])

        for source_id in cited_ids:
            record = candidate_index[source_id]
            lines.extend(self._evidence_entry(source_id, record))

        self.memo_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    @staticmethod
    def _memo_point(title: str, text: str, source_ids: list[str]) -> list[str]:
        links = "、".join(f"[{source_id}](#evidence-{source_id.lower()})" for source_id in source_ids)
        return [f"### {title}", "", f"{text}（{links}）", ""]

    @staticmethod
    def _cited_ids(outputs: dict) -> list[str]:
        ids = []
        for point in outputs["business"]["key_points"]:
            ids.extend(point["source_ids"])
        for point in outputs["risk_factors"]["key_risks"]:
            ids.extend(point["source_ids"])
        for point in outputs["mda"]["key_metrics"] + outputs["mda"]["management_themes"]:
            ids.extend(point["source_ids"])
        return list(dict.fromkeys(ids))

    @staticmethod
    def _evidence_entry(source_id: str, record: dict) -> list[str]:
        label = (
            record.get("fact_cn")
            or record.get("risk_cn")
            or f"{record.get('metric_name', '')}: {record.get('current_value', '')}"
        )
        label = label.replace("州援助决定", "欧盟国家援助裁定")
        chunks = ", ".join(ref["chunk_id"] for ref in record.get("source_refs", []))
        lines = [
            f'<a id="evidence-{source_id.lower()}"></a>',
            f"### {source_id}",
            "",
            f"- Map 结论：{label}",
            f"- 来源块：{chunks}",
            f"- 重要性：{record.get('materiality', '未标记')}",
            f"- 语义复核：{record.get('semantic_review_status', 'pending')}",
        ]
        if record.get("status"):
            lines.append(f"- 风险状态：{record['status']}")
        lines.append("")
        for quote in record.get("canonical_quotes", []):
            lines.append(
                f"> 原文位置 {quote.get('source_start')}-{quote.get('source_end')}：{quote.get('quote', '')}"
            )
            lines.append(">")
        lines.append("")
        return lines

    def _load_filing_metadata(self) -> dict:
        files = sorted(self.company_dir.glob("*.metadata.json"), reverse=True)
        if files:
            return json.loads(files[0].read_text(encoding="utf-8"))

        html_files = sorted(
            list(self.company_dir.glob("*.htm"))
            + list(self.company_dir.glob("*.html")),
            reverse=True,
        )
        if not html_files:
            return {}
        html_path = html_files[0]
        soup = BeautifulSoup(html_path.read_bytes(), "html.parser")

        def ix_value(name: str) -> str | None:
            tag = soup.find(
                lambda item: item.name
                and str(item.get("name", "")).lower() == name.lower()
            )
            if not tag:
                return None
            value = re.sub(r"\s+", " ", tag.get_text(" ", strip=True))
            return value.replace(" ,", ",")

        filename_match = re.match(
            r"(?P<filing_date>\d{4}-\d{2}-\d{2})_.*?_"
            r"(?P<accession>\d{10}-\d{2}-\d{6})_(?P<document>.+\.html?)$",
            html_path.name,
            re.IGNORECASE,
        )
        metadata = {
            "name": ix_value("dei:EntityRegistrantName"),
            "report_date": ix_value("dei:DocumentPeriodEndDate"),
        }
        if filename_match:
            accession = filename_match.group("accession")
            cik = str(int(accession[:10]))
            metadata.update(
                {
                    "filing_date": filename_match.group("filing_date"),
                    "accession_number": accession,
                    "primary_document": filename_match.group("document"),
                    "url": (
                        "https://www.sec.gov/Archives/edgar/data/"
                        f"{cik}/{accession.replace('-', '')}/"
                        f"{filename_match.group('document')}"
                    ),
                }
            )
        return {key: value for key, value in metadata.items() if value}

    @staticmethod
    def _find_global_ids(value) -> set[str]:
        if isinstance(value, dict):
            return set().union(
                *(FilingReducer._find_global_ids(item) for item in value.values())
            ) if value else set()
        if isinstance(value, list):
            return set().union(
                *(FilingReducer._find_global_ids(item) for item in value)
            ) if value else set()
        if not isinstance(value, str):
            return set()
        return set(
            re.findall(r"(?<![A-Z0-9])(?:RISK\d{3}|B\d{3}|M\d{3}|D\d{3})(?!\d)", value.upper())
        )
