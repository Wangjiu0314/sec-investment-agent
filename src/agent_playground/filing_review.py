from __future__ import annotations

import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek
from pydantic import BaseModel, Field, field_validator

from .filing_reduce import FilingReducer, SECTION_ORDER


REVIEW_PROMPT_VERSION = "filing-semantic-review-v1"
Verdict = Literal["supported", "partially_supported", "unsupported"]
RiskStatus = Literal["potential", "realized", "mixed"]


def _as_list(value) -> list:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


class ClaimReview(BaseModel):
    claim_id: str
    verdict: Verdict
    corrected_text_cn: str = ""
    source_ids: list[str] = Field(default_factory=list)
    issues_cn: list[str] = Field(default_factory=list)
    corrected_status: RiskStatus | None = None

    @field_validator("corrected_text_cn", mode="before")
    @classmethod
    def normalize_corrected_text(cls, value):
        return "" if value is None else str(value).strip()

    @field_validator("claim_id", mode="before")
    @classmethod
    def normalize_claim_id(cls, value):
        return str(value).strip().upper()

    @field_validator("source_ids", mode="before")
    @classmethod
    def normalize_source_ids(cls, value):
        return [
            str(item).strip().upper()
            for item in _as_list(value)
            if str(item).strip()
        ]

    @field_validator("issues_cn", mode="before")
    @classmethod
    def normalize_issues(cls, value):
        return [str(item).strip() for item in _as_list(value) if str(item).strip()]


class SectionReview(BaseModel):
    reviews: list[ClaimReview]
    section_notes_cn: list[str] = Field(default_factory=list)

    @field_validator("section_notes_cn", mode="before")
    @classmethod
    def normalize_notes(cls, value):
        return [str(item).strip() for item in _as_list(value) if str(item).strip()]


REVIEW_SYSTEM = """你是一名独立、保守的 SEC 财报语义审查员。
你的任务不是重新写一份报告，而是逐条检查已有中文 claim 是否被给定 SEC 证据完整支持。

必须遵守：
1. supported 表示 claim 的每一个事实、数字、因果关系和风险语气都由证据直接支持。
2. 只有部分内容成立，或只需最小修改即可成立时，使用 partially_supported，并在 corrected_text_cn 中给出修正版。
3. 没有足够证据时使用 unsupported，不得凭常识补全。
4. may、could、can、might、potential 等条件性原文不能写成“已经、已导致、曾造成”等现实结果。
5. “措施已经宣布”不等于“措施已经造成某项经济影响”。分别判断事件是否发生和影响是否发生。
6. 数字、期间和单位应尽量保留 SEC 原文格式。不要把 dollars in millions 自行换算成“亿美元”。
7. State Aid Decision 在本财报语境中译为“欧盟国家援助裁定”，不能译为“州援助决定”。
8. corrected_text_cn 中不要写 B001、RISK001、M001、D001 等编号；编号只放在 source_ids。
9. source_ids 只能使用 evidence_catalog 中真实存在且能支持修正版的全局编号。
10. summary、connection、open_question 也必须分配支持它们的 source_ids。
11. key_risk 必须返回 corrected_status，且只能是 potential、realized、mixed。其他 claim 的 corrected_status 返回 null。
12. 必须返回全部 claim_id，不能遗漏、增加或重复。

只返回一个 JSON 对象，包含 reviews 和 section_notes_cn。"""

REVIEW_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", REVIEW_SYSTEM),
        (
            "human",
            """ticker: {ticker}
form: {form}
section: {section}

<claims_to_review>
{claims_json}
</claims_to_review>

<evidence_catalog>
{evidence_json}
</evidence_catalog>

上一次 Python 校验反馈（首次调用为空）：
{validation_feedback}""",
        ),
    ]
)


class SemanticReviewer:
    def __init__(
        self,
        company_dir: Path,
        *,
        model_name: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0,
        max_attempts: int = 2,
        request_delay_seconds: float = 1.0,
    ) -> None:
        self.company_dir = company_dir
        self.reduce_path = company_dir / "reduce_v1.json"
        self.output_path = company_dir / "semantic_review_v1.json"
        self.memo_path = company_dir / "research_memo.md"
        self.model_name = model_name or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY", "")
        self.base_url = base_url or os.getenv(
            "DEEPSEEK_BASE_URL", "https://api.deepseek.com"
        )
        self.temperature = temperature
        self.max_attempts = max_attempts
        self.request_delay_seconds = request_delay_seconds
        self._llm: ChatDeepSeek | None = None

    def run(self, *, max_new_calls: int | None = 3) -> dict:
        reduce_cache = self._load_reduce_cache()
        claims = self.build_claims(reduce_cache)
        fingerprint = self._fingerprint(reduce_cache, claims)
        cache = self._load_or_initialize_cache(reduce_cache, claims, fingerprint)
        self._revalidate_cached_sections(cache, claims, reduce_cache["candidates"])

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
            print(f"[Review {index}/{len(selected)}] 审查 {section}")
            try:
                result = self._review_section(
                    section=section,
                    filing=reduce_cache["filing"],
                    claims=claims[section],
                    candidates=self._section_candidates(
                        section, reduce_cache["candidates"]
                    ),
                )
                cache["sections"][section] = result
                if result["validation"]["structure_passed"]:
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
            self._write_reviewed_memo(reduce_cache, cache)

        cached_after = sum(
            self._section_is_current(cache, section) for section in SECTION_ORDER
        )
        decision_counts = self._decision_counts(cache) if complete else {}
        stats.update(
            {
                "cached_after": cached_after,
                "pending_after": len(SECTION_ORDER) - cached_after,
                "complete": complete,
                "decision_counts": decision_counts,
                "output_file": str(self.output_path),
                "memo_file": str(self.memo_path) if complete else None,
            }
        )
        return stats

    def build_claims(self, reduce_cache: dict) -> dict[str, list[dict]]:
        outputs = {
            section: reduce_cache["sections"][section]["model_output"]
            for section in SECTION_ORDER
        }
        claims = {section: [] for section in SECTION_ORDER}

        business = outputs["business"]
        business_point_ids = self._collect_source_ids(business["key_points"])
        claims["business"].append(
            self._claim(
                "BUS_SUMMARY",
                "summary",
                "业务摘要",
                business["section_summary_cn"],
                business_point_ids,
            )
        )
        for index, point in enumerate(business["key_points"], start=1):
            claims["business"].append(
                self._claim(
                    f"BUS{index:03d}",
                    "key_point",
                    point["title_cn"],
                    point["analysis_cn"],
                    point["source_ids"],
                )
            )
        for index, question in enumerate(business.get("open_questions_cn", []), start=1):
            claims["business"].append(
                self._claim(
                    f"BUSQ{index:03d}",
                    "open_question",
                    "待核查",
                    question,
                    self._extract_global_ids(question),
                )
            )

        risks = outputs["risk_factors"]
        risk_point_ids = self._collect_source_ids(risks["key_risks"])
        claims["risk_factors"].append(
            self._claim(
                "RISK_SUMMARY",
                "summary",
                "风险摘要",
                risks["section_summary_cn"],
                risk_point_ids,
            )
        )
        for index, point in enumerate(risks["key_risks"], start=1):
            claims["risk_factors"].append(
                self._claim(
                    f"RC{index:03d}",
                    "key_risk",
                    point["title_cn"],
                    point["assessment_cn"],
                    point["source_ids"],
                    point["status"],
                )
            )
        for index, connection in enumerate(
            risks.get("risk_connections_cn", []), start=1
        ):
            claims["risk_factors"].append(
                self._claim(
                    f"RCON{index:03d}",
                    "connection",
                    "风险联动",
                    connection,
                    [],
                )
            )
        for index, question in enumerate(risks.get("open_questions_cn", []), start=1):
            claims["risk_factors"].append(
                self._claim(
                    f"RISKQ{index:03d}",
                    "open_question",
                    "待核查",
                    question,
                    self._extract_global_ids(question),
                )
            )

        mda = outputs["mda"]
        mda_ids = self._collect_source_ids(
            mda["key_metrics"] + mda["management_themes"]
        )
        claims["mda"].append(
            self._claim(
                "MDA_SUMMARY",
                "summary",
                "MD&A 摘要",
                mda["section_summary_cn"],
                mda_ids,
            )
        )
        for index, point in enumerate(mda["key_metrics"], start=1):
            claims["mda"].append(
                self._claim(
                    f"MC{index:03d}",
                    "key_metric",
                    point["title_cn"],
                    point["analysis_cn"],
                    point["source_ids"],
                )
            )
        for index, point in enumerate(mda["management_themes"], start=1):
            claims["mda"].append(
                self._claim(
                    f"MT{index:03d}",
                    "management_theme",
                    point["title_cn"],
                    point["analysis_cn"],
                    point["source_ids"],
                )
            )
        for index, question in enumerate(mda.get("open_questions_cn", []), start=1):
            claims["mda"].append(
                self._claim(
                    f"MDAQ{index:03d}",
                    "open_question",
                    "待核查",
                    question,
                    self._extract_global_ids(question),
                )
            )
        return claims

    @staticmethod
    def _claim(
        claim_id: str,
        kind: str,
        title_cn: str,
        text_cn: str,
        source_ids: list[str],
        declared_status: str | None = None,
    ) -> dict:
        return {
            "claim_id": claim_id,
            "kind": kind,
            "title_cn": title_cn,
            "text_cn": text_cn,
            "initial_source_ids": list(dict.fromkeys(source_ids)),
            "declared_status": declared_status,
        }

    def _review_section(
        self,
        *,
        section: str,
        filing: dict,
        claims: list[dict],
        candidates: dict,
    ) -> dict:
        allowed_ids = {
            record["global_id"]
            for records in candidates.values()
            for record in records
        }
        prompt_candidates = self._prompt_candidates(candidates)
        feedback = ""
        last_error: Exception | None = None
        chain = REVIEW_PROMPT | self._get_llm().bind(
            response_format={"type": "json_object"}
        )

        for attempt in range(1, self.max_attempts + 1):
            try:
                raw_message = chain.invoke(
                    {
                        "ticker": filing["ticker"],
                        "form": filing["form"],
                        "section": section,
                        "claims_json": json.dumps(
                            claims, ensure_ascii=False, indent=2
                        ),
                        "evidence_json": json.dumps(
                            prompt_candidates, ensure_ascii=False, indent=2
                        ),
                        "validation_feedback": feedback,
                    }
                )
                output = self._parse_output(raw_message.content)
                self._normalize_reviews(output, claims)
                errors = self._validate_output(
                    section=section,
                    output=output,
                    claims=claims,
                    allowed_ids=allowed_ids,
                )
                if errors:
                    feedback = "；".join(errors)
                    last_error = ValueError(feedback)
                    if attempt < self.max_attempts:
                        time.sleep(2)
                        continue
                return {
                    "run_metadata": {
                        "prompt_version": REVIEW_PROMPT_VERSION,
                        "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
                        "model": self.model_name,
                        "temperature": self.temperature,
                        "attempt": attempt,
                        "token_usage": raw_message.usage_metadata or {},
                    },
                    "model_output": output.model_dump(),
                    "validation": {
                        "structure_passed": not errors,
                        "structure_errors": errors,
                    },
                }
            except Exception as exc:
                last_error = exc
                feedback = str(exc)
                if attempt < self.max_attempts:
                    time.sleep(2)
        raise RuntimeError(f"{section} Semantic Review 失败: {last_error}")

    @staticmethod
    def _parse_output(raw_content) -> SectionReview:
        if not isinstance(raw_content, str):
            raw_content = str(raw_content)
        text = raw_content.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
        payload = json.loads(text)
        if isinstance(payload, list):
            payload = {"reviews": payload}
        if not isinstance(payload, dict):
            raise ValueError("Semantic Review JSON 顶层必须是对象")
        aliases = {
            "claims": "reviews",
            "claim_reviews": "reviews",
            "notes_cn": "section_notes_cn",
        }
        for alias, standard in aliases.items():
            if alias in payload and standard not in payload:
                payload[standard] = payload.pop(alias)
        payload.setdefault("section_notes_cn", [])
        return SectionReview.model_validate(payload)

    @staticmethod
    def _fill_supported_text(output: SectionReview, claims: list[dict]) -> None:
        claim_index = {claim["claim_id"]: claim for claim in claims}
        for review in output.reviews:
            if (
                review.verdict == "supported"
                and not review.corrected_text_cn
                and review.claim_id in claim_index
            ):
                review.corrected_text_cn = claim_index[review.claim_id]["text_cn"]

    @staticmethod
    def _normalize_reviews(output: SectionReview, claims: list[dict]) -> None:
        SemanticReviewer._fill_supported_text(output, claims)
        for review in output.reviews:
            review.corrected_text_cn = SemanticReviewer._strip_embedded_ids(
                review.corrected_text_cn
            )
            if "州援助决定" in review.corrected_text_cn:
                review.corrected_text_cn = review.corrected_text_cn.replace(
                    "州援助决定", "欧盟国家援助裁定"
                )
                if review.verdict == "supported":
                    review.verdict = "partially_supported"
                review.issues_cn.append(
                    "专业术语修正：State Aid Decision 译为欧盟国家援助裁定。"
                )
            if review.verdict == "partially_supported" and not review.issues_cn:
                review.issues_cn = ["Reviewer 已修正文案，但未单独返回问题说明。"]

    @staticmethod
    def _strip_embedded_ids(text: str) -> str:
        global_id = r"(?:RISK\d{3}|B\d{3}|M\d{3}|D\d{3})"
        value = re.sub(
            rf"来源[:：]\s*{global_id}(?:\s*[,，、]\s*{global_id})*[。.]?",
            "",
            text,
            flags=re.IGNORECASE,
        )
        value = re.sub(
            rf"已在\s*{global_id}\s*中提及",
            "已在相关披露中提及",
            value,
            flags=re.IGNORECASE,
        )
        value = re.sub(
            rf"[（(]\s*{global_id}(?:\s*[,，、]\s*{global_id})*\s*[）)]",
            "",
            value,
            flags=re.IGNORECASE,
        )
        value = re.sub(global_id, "", value, flags=re.IGNORECASE)
        value = re.sub(r"\s+([，。；：？！])", r"\1", value)
        value = re.sub(r"\s{2,}", " ", value)
        return value.strip()

    @staticmethod
    def _validate_output(
        *,
        section: str,
        output: SectionReview,
        claims: list[dict],
        allowed_ids: set[str],
    ) -> list[str]:
        errors: list[str] = []
        expected = {claim["claim_id"]: claim for claim in claims}
        returned_ids = [review.claim_id for review in output.reviews]
        duplicates = sorted(
            claim_id for claim_id in set(returned_ids) if returned_ids.count(claim_id) > 1
        )
        if duplicates:
            errors.append("重复 claim_id: " + ", ".join(duplicates))
        missing = sorted(set(expected) - set(returned_ids))
        extra = sorted(set(returned_ids) - set(expected))
        if missing:
            errors.append("缺少 claim_id: " + ", ".join(missing))
        if extra:
            errors.append("出现未知 claim_id: " + ", ".join(extra))

        for review in output.reviews:
            claim = expected.get(review.claim_id)
            if claim is None:
                continue
            invalid = sorted(set(review.source_ids) - allowed_ids)
            if invalid:
                errors.append(
                    f"{review.claim_id} 使用无效 source_ids: {', '.join(invalid)}"
                )
            if review.verdict != "unsupported" and not review.source_ids:
                errors.append(f"{review.claim_id} 通过或修正后缺少 source_ids")
            if review.verdict != "unsupported" and not review.corrected_text_cn.strip():
                errors.append(f"{review.claim_id} 通过或修正后缺少 corrected_text_cn")
            embedded = FilingReducer._find_global_ids(review.corrected_text_cn)
            if embedded:
                errors.append(
                    f"{review.claim_id} corrected_text_cn 中仍包含编号: "
                    + ", ".join(sorted(embedded))
                )
            if claim["kind"] == "key_risk" and review.corrected_status is None:
                errors.append(f"{review.claim_id} key_risk 缺少 corrected_status")
            if claim["kind"] != "key_risk" and review.corrected_status is not None:
                errors.append(f"{review.claim_id} 非 key_risk 不应返回 corrected_status")
        expects_connections = any(
            claim["kind"] == "connection" for claim in claims
        )
        if section == "risk_factors" and expects_connections and not any(
            review.claim_id.startswith("RCON") for review in output.reviews
        ):
            errors.append("Risk Review 缺少风险联动审查")
        return errors

    def _revalidate_cached_sections(
        self,
        cache: dict,
        claims: dict[str, list[dict]],
        candidates: dict,
    ) -> None:
        for section, result in cache.get("sections", {}).items():
            if (
                section not in SECTION_ORDER
                or result.get("run_metadata", {}).get("prompt_version")
                != REVIEW_PROMPT_VERSION
            ):
                continue
            output = SectionReview.model_validate(result["model_output"])
            self._normalize_reviews(output, claims[section])
            allowed_ids = {
                record["global_id"]
                for records in self._section_candidates(section, candidates).values()
                for record in records
            }
            errors = self._validate_output(
                section=section,
                output=output,
                claims=claims[section],
                allowed_ids=allowed_ids,
            )
            result["model_output"] = output.model_dump()
            result["validation"] = {
                "structure_passed": not errors,
                "structure_errors": errors,
            }

    def _load_reduce_cache(self) -> dict:
        if not self.reduce_path.exists():
            raise FileNotFoundError(f"缺少 Reduce 缓存: {self.reduce_path}")
        cached = json.loads(self.reduce_path.read_text(encoding="utf-8"))
        if not cached.get("validation", {}).get("all_sections_passed"):
            raise ValueError("Reduce 尚未全部通过，不能开始语义审查")
        return cached

    def _load_or_initialize_cache(
        self,
        reduce_cache: dict,
        claims: dict,
        fingerprint: str,
    ) -> dict:
        if self.output_path.exists():
            cached = json.loads(self.output_path.read_text(encoding="utf-8"))
            if (
                cached.get("run_metadata", {}).get("input_fingerprint") == fingerprint
                and cached.get("run_metadata", {}).get("prompt_version")
                == REVIEW_PROMPT_VERSION
            ):
                return cached
        now = datetime.now(timezone.utc).isoformat()
        return {
            "run_metadata": {
                "prompt_version": REVIEW_PROMPT_VERSION,
                "input_fingerprint": fingerprint,
                "created_at_utc": now,
                "updated_at_utc": now,
            },
            "filing": reduce_cache["filing"],
            "claims": claims,
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
            == REVIEW_PROMPT_VERSION
        )

    @staticmethod
    def _section_candidates(section: str, candidates: dict) -> dict:
        return FilingReducer._section_candidates(section, candidates)

    @staticmethod
    def _prompt_candidates(candidates: dict) -> dict:
        compact = {}
        for key, records in candidates.items():
            compact[key] = []
            for record in records:
                item = {
                    field: value
                    for field, value in record.items()
                    if field not in {"evidence", "source_refs"}
                }
                item["canonical_quotes"] = record.get("canonical_quotes", [])[:3]
                compact[key].append(item)
        return compact

    @staticmethod
    def _fingerprint(reduce_cache: dict, claims: dict) -> str:
        payload = json.dumps(
            {
                "prompt_version": REVIEW_PROMPT_VERSION,
                "reduce_input_fingerprint": reduce_cache["run_metadata"][
                    "input_fingerprint"
                ],
                "reduce_outputs": {
                    section: reduce_cache["sections"][section]["model_output"]
                    for section in SECTION_ORDER
                },
                "claims": claims,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _collect_source_ids(points: list[dict]) -> list[str]:
        return list(
            dict.fromkeys(
                source_id
                for point in points
                for source_id in point.get("source_ids", [])
            )
        )

    @staticmethod
    def _extract_global_ids(text: str) -> list[str]:
        return list(
            dict.fromkeys(
                re.findall(
                    r"(?<![A-Z0-9])(?:RISK\d{3}|B\d{3}|M\d{3}|D\d{3})(?!\d)",
                    text.upper(),
                )
            )
        )

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

    @staticmethod
    def _decision_counts(cache: dict) -> dict:
        counts = {"supported": 0, "partially_supported": 0, "unsupported": 0}
        for section in SECTION_ORDER:
            for review in cache["sections"][section]["model_output"]["reviews"]:
                counts[review["verdict"]] += 1
        return counts

    def _write_reviewed_memo(self, reduce_cache: dict, review_cache: dict) -> None:
        reviews = {
            section: {
                item["claim_id"]: item
                for item in review_cache["sections"][section]["model_output"]["reviews"]
            }
            for section in SECTION_ORDER
        }
        claims = review_cache["claims"]
        candidate_index = {
            record["global_id"]: record
            for records in reduce_cache["candidates"].values()
            for record in records
        }
        filing_reducer = FilingReducer(
            self.company_dir, self.company_dir / "map_analysis_v3"
        )
        metadata = filing_reducer._load_filing_metadata()
        ticker = reduce_cache["filing"]["ticker"]
        form = reduce_cache["filing"]["form"]

        lines = [
            f"# {ticker} {form} 研究备忘录",
            "",
            "> 本文基于 SEC 原文整理，仍建议由专业人员完成最终复核，不构成投资建议。",
            "",
            "## 基本信息",
            "",
            f"- 公司：{metadata.get('name', ticker)}",
            f"- 报告类型：{form}",
            f"- 报告期：{metadata.get('report_date', '未记录')}",
            f"- 申报日期：{metadata.get('filing_date', '未记录')}",
            f"- SEC 原文：{metadata.get('url', '未记录')}",
            "",
            "## 执行摘要",
            "",
        ]
        for section, label, claim_id in (
            ("business", "业务", "BUS_SUMMARY"),
            ("risk_factors", "风险", "RISK_SUMMARY"),
            ("mda", "MD&A", "MDA_SUMMARY"),
        ):
            review = reviews[section][claim_id]
            if review["verdict"] != "unsupported":
                lines.append(
                    f"- {label}：{review['corrected_text_cn']}"
                    f"{self._inline_citations(review['source_ids'])}"
                )
        lines.append("")

        lines.extend(["## 业务概览", ""])
        self._append_claim_group(
            lines, claims["business"], reviews["business"], {"key_point"}
        )

        lines.extend(["## 关键风险", ""])
        self._append_claim_group(
            lines,
            claims["risk_factors"],
            reviews["risk_factors"],
            {"key_risk"},
            include_status=True,
        )
        connections = self._accepted_claims(
            claims["risk_factors"], reviews["risk_factors"], {"connection"}
        )
        if connections:
            lines.extend(["### 风险联动", ""])
            for claim, review in connections:
                lines.append(
                    f"- {review['corrected_text_cn']}"
                    f"{self._inline_citations(review['source_ids'])}"
                )
            lines.append("")

        lines.extend(["## MD&A 核心指标", ""])
        self._append_claim_group(
            lines, claims["mda"], reviews["mda"], {"key_metric"}
        )
        lines.extend(["## 管理层讨论主题", ""])
        self._append_claim_group(
            lines, claims["mda"], reviews["mda"], {"management_theme"}
        )

        lines.extend(["## 待进一步核查", ""])
        questions = []
        for section in SECTION_ORDER:
            questions.extend(
                self._accepted_claims(
                    claims[section], reviews[section], {"open_question"}
                )
            )
        if questions:
            for _, review in questions:
                lines.append(
                    f"- {review['corrected_text_cn']}"
                    f"{self._inline_citations(review['source_ids'])}"
                )
        else:
            lines.append("- Reviewer 未保留待核查项。")
        lines.extend(["", "## 证据索引", ""])

        cited_ids = []
        for section in SECTION_ORDER:
            for review in reviews[section].values():
                if review["verdict"] != "unsupported":
                    cited_ids.extend(review["source_ids"])
        for source_id in dict.fromkeys(cited_ids):
            lines.extend(FilingReducer._evidence_entry(source_id, candidate_index[source_id]))

        self.memo_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    @staticmethod
    def _accepted_claims(
        claims: list[dict], reviews: dict[str, dict], kinds: set[str]
    ) -> list[tuple[dict, dict]]:
        return [
            (claim, reviews[claim["claim_id"]])
            for claim in claims
            if claim["kind"] in kinds
            and reviews[claim["claim_id"]]["verdict"] != "unsupported"
        ]

    def _append_claim_group(
        self,
        lines: list[str],
        claims: list[dict],
        reviews: dict[str, dict],
        kinds: set[str],
        *,
        include_status: bool = False,
    ) -> None:
        for claim, review in self._accepted_claims(claims, reviews, kinds):
            title = claim["title_cn"]
            if include_status:
                title += f"（{review['corrected_status']}）"
            lines.extend(
                [
                    f"### {title}",
                    "",
                    review["corrected_text_cn"]
                    + self._inline_citations(review["source_ids"]),
                    "",
                ]
            )

    @staticmethod
    def _inline_citations(source_ids: list[str]) -> str:
        if not source_ids:
            return ""
        links = "、".join(
            f"[{source_id}](#evidence-{source_id.lower()})"
            for source_id in source_ids
        )
        return f"（{links}）"
