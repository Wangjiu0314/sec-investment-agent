from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, get_args

from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field, field_validator


PROMPT_VERSIONS = {
    "business": "filing-map-business-v3.1",
    "risk_factors": "filing-map-risk-v3",
    "mda": "filing-map-mda-v3.2",
}

Materiality = Literal["high", "medium", "low"]
BusinessCategory = Literal[
    "business_model",
    "product",
    "service",
    "geography",
    "market",
    "distribution",
    "competition",
    "supply_chain",
    "research_development",
    "intellectual_property",
    "customer",
    "other",
]
RiskStatus = Literal["potential", "realized", "mixed"]
Direction = Literal["increase", "decrease", "flat", "not_stated"]
MetricUnit = Literal["usd_millions", "usd_billions", "percent", "other"]
MDACategory = Literal[
    "accounting_standard",
    "liquidity",
    "commitment",
    "critical_estimate",
    "legal_contingency",
    "tax",
    "other",
]


def _as_list(value):
    if value is None:
        return []
    values = value if isinstance(value, list) else [value]
    return [item for item in values if str(item).strip()]


def _normalize_id(value: str | int, prefix: str) -> str:
    text = str(value).strip().upper()
    numbers = re.findall(r"\d+", text)
    if numbers:
        return f"{prefix}{int(numbers[-1]):03d}"
    if text.startswith(prefix):
        text = text[len(prefix) :]
    return f"{prefix}{text}"


def _normalize_level(value: str) -> str:
    return {"高": "high", "中": "medium", "低": "low"}.get(
        str(value).strip(), str(value).strip()
    )


class BusinessFact(BaseModel):
    fact_id: str
    category: BusinessCategory
    fact_cn: str
    evidence_ids: list[str]
    materiality: Materiality

    @field_validator("fact_id", mode="before")
    @classmethod
    def normalize_fact_id(cls, value):
        return _normalize_id(value, "F")

    @field_validator("category", mode="before")
    @classmethod
    def normalize_business_category(cls, value):
        key = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
        aliases = {
            "products": "product",
            "services": "service",
            "markets": "market",
            "research_and_development": "research_development",
        }
        normalized = aliases.get(key, key)
        return normalized if normalized in get_args(BusinessCategory) else "other"

    @field_validator("evidence_ids", mode="before")
    @classmethod
    def normalize_evidence_ids(cls, value):
        return [_normalize_id(item, "E") for item in _as_list(value)]

    @field_validator("materiality", mode="before")
    @classmethod
    def normalize_materiality(cls, value):
        return _normalize_level(value)


class BusinessAnalysis(BaseModel):
    chunk_id: str = ""
    summary_cn: str = ""
    facts: list[BusinessFact]
    chunk_gaps_cn: list[str] = Field(default_factory=list)

    @field_validator("chunk_gaps_cn", mode="before")
    @classmethod
    def normalize_gaps(cls, value):
        return _as_list(value)


class RiskItem(BaseModel):
    risk_id: str
    category: str
    risk_cn: str
    status: RiskStatus
    causes_cn: list[str]
    potential_impacts_cn: list[str]
    evidence_ids: list[str]
    materiality: Materiality

    @field_validator("risk_id", mode="before")
    @classmethod
    def normalize_risk_id(cls, value):
        return _normalize_id(value, "R")

    @field_validator("status", mode="before")
    @classmethod
    def normalize_status(cls, value):
        aliases = {
            "潜在": "potential",
            "可能": "potential",
            "已发生": "realized",
            "现实": "realized",
            "混合": "mixed",
        }
        text = str(value).strip()
        return aliases.get(text, text)

    @field_validator("causes_cn", "potential_impacts_cn", mode="before")
    @classmethod
    def normalize_text_lists(cls, value):
        return _as_list(value)

    @field_validator("evidence_ids", mode="before")
    @classmethod
    def normalize_risk_evidence(cls, value):
        return [_normalize_id(item, "E") for item in _as_list(value)]

    @field_validator("materiality", mode="before")
    @classmethod
    def normalize_risk_materiality(cls, value):
        return _normalize_level(value)


class RiskAnalysis(BaseModel):
    chunk_id: str = ""
    summary_cn: str = ""
    risks: list[RiskItem]
    chunk_gaps_cn: list[str] = Field(default_factory=list)

    @field_validator("chunk_gaps_cn", mode="before")
    @classmethod
    def normalize_risk_gaps(cls, value):
        return _as_list(value)


class FinancialMetric(BaseModel):
    metric_id: str
    metric_name: str
    current_period: str
    current_value: str
    unit: MetricUnit
    comparison_period: str | None = None
    comparison_value: str | None = None
    change_value: str | None = None
    change_direction: Direction
    management_explanation_cn: str | None = None
    evidence_ids: list[str]
    materiality: Materiality

    @field_validator(
        "current_period",
        "current_value",
        "comparison_period",
        "comparison_value",
        "change_value",
        mode="before",
    )
    @classmethod
    def normalize_numeric_fields(cls, value):
        if value is None:
            return None
        text = str(value).strip()
        if text.lower() in {
            "not_stated",
            "not stated",
            "none",
            "null",
            "未提供",
        }:
            return None
        return text

    @field_validator("management_explanation_cn", mode="before")
    @classmethod
    def normalize_explanation(cls, value):
        if value is None or str(value).strip() in {"", "未提供", "not_stated"}:
            return None
        return str(value).strip()

    @field_validator("metric_id", mode="before")
    @classmethod
    def normalize_metric_id(cls, value):
        return _normalize_id(value, "M")

    @field_validator("unit", mode="before")
    @classmethod
    def normalize_unit(cls, value):
        aliases = {
            "百万美元": "usd_millions",
            "美元百万": "usd_millions",
            "十亿美元": "usd_billions",
            "百分比": "percent",
            "%": "percent",
            "其他": "other",
        }
        text = str(value).strip()
        return aliases.get(text, text)

    @field_validator("change_direction", mode="before")
    @classmethod
    def normalize_direction(cls, value):
        aliases = {
            "增加": "increase",
            "上升": "increase",
            "减少": "decrease",
            "下降": "decrease",
            "持平": "flat",
            "未说明": "not_stated",
        }
        text = str(value).strip()
        return aliases.get(text, text)

    @field_validator("evidence_ids", mode="before")
    @classmethod
    def normalize_metric_evidence(cls, value):
        return [_normalize_id(item, "E") for item in _as_list(value)]

    @field_validator("materiality", mode="before")
    @classmethod
    def normalize_metric_materiality(cls, value):
        return _normalize_level(value)


class MDAFact(BaseModel):
    fact_id: str
    category: MDACategory
    fact_cn: str
    evidence_ids: list[str]
    materiality: Materiality

    @field_validator("fact_id", mode="before")
    @classmethod
    def normalize_fact_id(cls, value):
        return _normalize_id(value, "F")

    @field_validator("category", mode="before")
    @classmethod
    def normalize_category(cls, value):
        key = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
        aliases = {
            "accounting_standards": "accounting_standard",
            "legal": "legal_contingency",
            "contingency": "legal_contingency",
            "commitments": "commitment",
        }
        normalized = aliases.get(key, key)
        return normalized if normalized in get_args(MDACategory) else "other"

    @field_validator("evidence_ids", mode="before")
    @classmethod
    def normalize_evidence_ids(cls, value):
        return [_normalize_id(item, "E") for item in _as_list(value)]

    @field_validator("materiality", mode="before")
    @classmethod
    def normalize_materiality(cls, value):
        return _normalize_level(value)


class MDAAnalysis(BaseModel):
    chunk_id: str = ""
    summary_cn: str = ""
    metrics: list[FinancialMetric] = Field(default_factory=list)
    facts: list[MDAFact] = Field(default_factory=list)
    chunk_gaps_cn: list[str] = Field(default_factory=list)

    @field_validator("chunk_gaps_cn", mode="before")
    @classmethod
    def normalize_mda_gaps(cls, value):
        return _as_list(value)


COMMON_SYSTEM = """你是一名审慎的 SEC 财报信息提取助手。
只能依据输入证据分析，不使用外部知识，不提供投资建议。
只能引用真实存在的 E 编号，例如 E003；不得自己编写引文。
所有编号必须保留字母前缀并作为 JSON 字符串返回。
high、medium、low 等枚举值必须使用指定英文，不得翻译。
chunk_gaps_cn 只表示当前文本块缺少的信息，不代表整份财报没有披露。
只返回一个 JSON 对象。"""

BUSINESS_SYSTEM = COMMON_SYSTEM + """
提取 4 到 8 条直接披露的业务事实，不生成分析推断。
每条事实包含 fact_id、category、fact_cn、evidence_ids、materiality。
category 只能使用 business_model、product、service、geography、market、distribution、competition、supply_chain、research_development、intellectual_property、customer、other。"""

RISK_SYSTEM = COMMON_SYSTEM + """
提取 2 到 6 项风险。每项包含 risk_id、category、risk_cn、status、causes_cn、potential_impacts_cn、evidence_ids、materiality。
status 只能是 potential、realized、mixed。
仅有 may、could、can、might、risk、uncertainty 等条件性表述时，必须标记 potential，不得写成已经发生。
只有原文明示已经发生或正在发生时才能使用 realized；同时包含现实情况和未来可能影响时使用 mixed。"""

MDA_SYSTEM = COMMON_SYSTEM + """
根据原文内容输出 metrics 和 facts，至少一类非空。
metrics 用于原文明示金额、比例、期间或比较值的财务指标，最多 12 项。每项包含 metric_id、metric_name、current_period、current_value、unit、comparison_period、comparison_value、change_value、change_direction、management_explanation_cn、evidence_ids、materiality。
facts 用于会计准则、承诺、流动性安排、关键估计、法律或有事项等叙述性披露，最多 8 项。每项包含 fact_id、category、fact_cn、evidence_ids、materiality。category 只能使用 accounting_standard、liquidity、commitment、critical_estimate、legal_contingency、tax、other。
不要机械列出全部地区或全部产品。若原文同时包含 Segment Operating Performance、Products and Services Performance、Gross Margin，结果必须覆盖这三个子章节，并至少包含一个地区指标、一个产品或服务指标、一个毛利额指标和一个毛利率指标。
unit 只能是 usd_millions、usd_billions、percent、other。change_direction 只能是 increase、decrease、flat、not_stated。
数值和期间保持原文格式；不得创造 Q1、FY、N/A 等原文没有的期间，也不得自行计算原文未披露的数值。若记录美元指标，证据必须同时覆盖 in millions、dollars in millions、billion 等单位说明和对应数字。
management_explanation_cn 只能翻译管理层明确披露的变化原因，不能猜测。"""


def _make_prompt(system_prompt: str) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "human",
                """chunk_id: {chunk_id}
ticker: {ticker}
form: {form}
section: {section_title}

<numbered_evidence>
{numbered_evidence}
</numbered_evidence>

上一次校验反馈（首次调用为空）：
{validation_feedback}""",
            ),
        ]
    )


ANALYSIS_CONFIG = {
    "business": (BusinessAnalysis, _make_prompt(BUSINESS_SYSTEM)),
    "risk_factors": (RiskAnalysis, _make_prompt(RISK_SYSTEM)),
    "mda": (MDAAnalysis, _make_prompt(MDA_SYSTEM)),
}


class MapAnalyzer:
    def __init__(
        self,
        cache_dir: Path,
        *,
        model_name: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0,
        max_attempts: int = 2,
        request_delay_seconds: float = 1.0,
    ) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY", "")
        self.base_url = base_url or os.getenv(
            "DEEPSEEK_BASE_URL", "https://api.deepseek.com"
        )
        self.temperature = temperature
        self.max_attempts = max_attempts
        self.request_delay_seconds = request_delay_seconds
        self._llm: ChatDeepSeek | None = None
        self._evidence_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=0,
            separators=["\n\n", "\n", ". ", " ", ""],
            keep_separator="end",
            add_start_index=True,
        )

    def load_current_cache(self, chunk: dict) -> dict | None:
        cache_path = self.cache_dir / f"{chunk['chunk_id']}.json"
        if not cache_path.exists():
            return None
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        expected_version = PROMPT_VERSIONS[chunk["section"]]
        if (
            cached.get("validation", {}).get("structure_passed")
            and cached.get("run_metadata", {}).get("prompt_version")
            == expected_version
        ):
            return cached
        return None

    def run_batch(
        self,
        chunks: list[dict],
        *,
        sections: set[str] | None = None,
        max_new_calls: int | None = 3,
    ) -> dict:
        self._attach_inherited_context(chunks)
        candidates = [
            chunk
            for chunk in chunks
            if sections is None or chunk["section"] in sections
        ]
        revalidated_successes = 0
        revalidation_exceptions = []
        for chunk in candidates:
            if self.load_current_cache(chunk) or not self._cache_is_repairable(chunk):
                continue
            try:
                repaired = self.revalidate_cache(chunk)
                if repaired["validation"]["structure_passed"]:
                    revalidated_successes += 1
            except Exception as exc:
                revalidation_exceptions.append(
                    {"chunk_id": chunk["chunk_id"], "error": str(exc)}
                )
        cached = [chunk for chunk in candidates if self.load_current_cache(chunk)]
        pending = [chunk for chunk in candidates if not self.load_current_cache(chunk)]
        selected = pending if max_new_calls is None else pending[:max_new_calls]

        stats = {
            "total_candidates": len(candidates),
            "cached_before": len(cached),
            "pending_before": len(pending),
            "selected_new_calls": len(selected),
            "new_successes": 0,
            "validation_failed": 0,
            "exceptions": [],
            "revalidated_successes": revalidated_successes,
            "revalidation_exceptions": revalidation_exceptions,
        }

        for index, chunk in enumerate(selected, start=1):
            print(f"[{index}/{len(selected)}] 分析 {chunk['chunk_id']}")
            try:
                result = self.analyze_chunk(chunk)
                if result["validation"]["structure_passed"]:
                    stats["new_successes"] += 1
                else:
                    stats["validation_failed"] += 1
            except Exception as exc:
                stats["exceptions"].append(
                    {"chunk_id": chunk["chunk_id"], "error": str(exc)}
                )
            if index < len(selected):
                time.sleep(self.request_delay_seconds)

        stats["cached_after"] = len(
            [chunk for chunk in candidates if self.load_current_cache(chunk)]
        )
        stats["pending_after"] = len(candidates) - stats["cached_after"]
        return stats

    def analyze_chunk(self, chunk: dict, *, force: bool = False) -> dict:
        cached = None if force else self.load_current_cache(chunk)
        if cached is not None:
            return cached

        schema, prompt = ANALYSIS_CONFIG[chunk["section"]]
        units = self._build_evidence_units(chunk)
        unit_map = {unit["evidence_id"]: unit for unit in units}
        numbered_evidence = "\n".join(
            f"[{unit['evidence_id']}] {unit['display_text']}" for unit in units
        )
        chain = prompt | self._get_llm().bind(
            response_format={"type": "json_object"}
        )
        last_error: Exception | None = None
        validation_feedback = ""

        for attempt in range(1, self.max_attempts + 1):
            try:
                raw_message = chain.invoke(
                    {
                        "chunk_id": chunk["chunk_id"],
                        "ticker": chunk["ticker"],
                        "form": chunk["form"],
                        "section_title": chunk["section_title"],
                        "numbered_evidence": numbered_evidence,
                        "validation_feedback": validation_feedback,
                    }
                )
                analysis = self._parse_model_output(
                    raw_message.content,
                    chunk["section"],
                    chunk["chunk_id"],
                    schema,
                )
                self._fill_common_fields(analysis, chunk)
                self._renumber_items(analysis)
                errors, resolved, dropped_items = self._validate_with_pruning(
                    analysis, unit_map
                )
                if errors and attempt < self.max_attempts:
                    validation_feedback = json.dumps(
                        {
                            "previous_output": analysis.model_dump(),
                            "validation_errors": errors,
                            "instruction": (
                                "只修正错误字段；期间使用证据原文；"
                                "未直接披露的 change_value 返回 null。"
                            ),
                        },
                        ensure_ascii=False,
                    )
                    time.sleep(2)
                    continue
                output = {
                    "run_metadata": {
                        "prompt_version": PROMPT_VERSIONS[chunk["section"]],
                        "analyzed_at_utc": datetime.now(timezone.utc).isoformat(),
                        "model": self.model_name,
                        "temperature": self.temperature,
                        "attempt": attempt,
                        "token_usage": raw_message.usage_metadata or {},
                    },
                    "chunk_metadata": {
                        key: chunk[key]
                        for key in (
                            "chunk_id",
                            "ticker",
                            "form",
                            "section",
                            "source_start",
                            "source_end",
                        )
                    },
                    "model_output": analysis.model_dump(),
                    "resolved_evidence": resolved,
                    "validation": {
                        "structure_passed": not errors,
                        "structure_errors": errors,
                        "dropped_items": dropped_items,
                        "semantic_review_status": "pending",
                    },
                }
                cache_path = self.cache_dir / f"{chunk['chunk_id']}.json"
                cache_path.write_text(
                    json.dumps(output, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                return output
            except Exception as exc:
                last_error = exc
                if attempt < self.max_attempts:
                    time.sleep(2)

        raise RuntimeError(f"{chunk['chunk_id']} 分析失败: {last_error}")

    def revalidate_cache(self, chunk: dict) -> dict:
        cache_path = self.cache_dir / f"{chunk['chunk_id']}.json"
        if not cache_path.exists():
            raise FileNotFoundError(f"没有找到缓存: {cache_path}")
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        previous_dropped_items = cached.get("validation", {}).get(
            "dropped_items", []
        )
        schema, _ = ANALYSIS_CONFIG[chunk["section"]]
        analysis = schema.model_validate(cached["model_output"])
        self._fill_common_fields(analysis, chunk)
        self._renumber_items(analysis)
        units = self._build_evidence_units(chunk)
        unit_map = {unit["evidence_id"]: unit for unit in units}
        errors, resolved, dropped_items = self._validate_with_pruning(
            analysis, unit_map
        )
        cached["run_metadata"]["prompt_version"] = PROMPT_VERSIONS[
            chunk["section"]
        ]
        cached["model_output"] = analysis.model_dump()
        cached["resolved_evidence"] = resolved
        cached["validation"]["structure_passed"] = not errors
        cached["validation"]["structure_errors"] = errors
        cached["validation"]["dropped_items"] = _merge_dropped_items(
            previous_dropped_items, dropped_items
        )
        if errors:
            cached["validation"]["semantic_review_status"] = "pending"
        cache_path.write_text(
            json.dumps(cached, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return cached

    def _get_llm(self) -> ChatDeepSeek:
        if self._llm is None:
            if not self.api_key:
                raise ValueError("请先在 .env 中配置 DEEPSEEK_API_KEY")
            self._llm = ChatDeepSeek(
                model=self.model_name,
                api_key=self.api_key,
                base_url=self.base_url,
                temperature=self.temperature,
                request_timeout=60,
                max_retries=1,
            )
        return self._llm

    def _build_evidence_units(self, chunk: dict) -> list[dict]:
        documents = self._evidence_splitter.create_documents([chunk["text"]])
        units = []
        for index, document in enumerate(documents, start=1):
            local_start = document.metadata["start_index"]
            local_end = local_start + len(document.page_content)
            if chunk["text"][local_start:local_end] != document.page_content:
                raise ValueError(f"{chunk['chunk_id']} 证据位置无法对应原文")
            units.append(
                {
                    "evidence_id": f"E{index:03d}",
                    "text": document.page_content,
                    "display_text": re.sub(r"\s+", " ", document.page_content),
                    "local_start": local_start,
                    "local_end": local_end,
                    "source_start": chunk["source_start"] + local_start,
                    "source_end": chunk["source_start"] + local_end,
                    "context_only": False,
                }
            )
        seen_ranges = {
            (unit["source_start"], unit["source_end"]) for unit in units
        }
        for context_unit in chunk.get("inherited_context_evidence", []):
            key = (context_unit["source_start"], context_unit["source_end"])
            if key in seen_ranges:
                continue
            copied = dict(context_unit)
            copied["evidence_id"] = f"E{len(units) + 1:03d}"
            copied["context_only"] = True
            units.append(copied)
            seen_ranges.add(key)
        return units

    def _attach_inherited_context(self, chunks: list[dict]) -> None:
        by_section: dict[str, list[dict]] = {}
        for chunk in chunks:
            by_section.setdefault(chunk["section"], []).append(chunk)
        for section_chunks in by_section.values():
            recent_units: list[dict] = []
            for chunk in sorted(section_chunks, key=lambda item: item["chunk_index"]):
                chunk["inherited_context_evidence"] = [
                    dict(unit) for unit in recent_units[-2:]
                ]
                current_units = [
                    unit
                    for unit in self._build_local_evidence_units(chunk)
                    if _contains_unit_context(unit["display_text"])
                ]
                if current_units:
                    recent_units.extend(current_units)
                    recent_units = recent_units[-2:]

    def _build_local_evidence_units(self, chunk: dict) -> list[dict]:
        inherited = chunk.pop("inherited_context_evidence", None)
        try:
            return self._build_evidence_units(chunk)
        finally:
            if inherited is not None:
                chunk["inherited_context_evidence"] = inherited

    def _cache_is_repairable(self, chunk: dict) -> bool:
        cache_path = self.cache_dir / f"{chunk['chunk_id']}.json"
        if not cache_path.exists():
            return False
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        return (
            cached.get("run_metadata", {}).get("prompt_version")
            == PROMPT_VERSIONS[chunk["section"]]
        )

    @staticmethod
    def _parse_model_output(raw_content, section: str, chunk_id: str, schema):
        if not isinstance(raw_content, str):
            raw_content = str(raw_content)
        raw_content = raw_content.strip()
        if raw_content.startswith("```"):
            raw_content = re.sub(
                r"^```(?:json)?\s*|\s*```$", "", raw_content
            )
        payload = json.loads(raw_content)
        output_key = {
            "business": "facts",
            "risk_factors": "risks",
            "mda": "metrics",
        }[section]
        if isinstance(payload, list):
            payload = {output_key: payload}
        if not isinstance(payload, dict):
            raise ValueError("模型 JSON 顶层必须是对象或数组")
        aliases = {
            "business_facts": "facts",
            "risk_items": "risks",
            "financial_metrics": "metrics",
            "narrative_facts": "facts",
            "mda_facts": "facts",
        }
        for alias, standard_name in aliases.items():
            if alias in payload and standard_name not in payload:
                payload[standard_name] = payload.pop(alias)
        payload.setdefault("chunk_id", chunk_id)
        payload.setdefault("summary_cn", "")
        payload.setdefault("chunk_gaps_cn", [])
        return schema.model_validate(payload)

    @staticmethod
    def _fill_common_fields(analysis, chunk: dict) -> None:
        analysis.chunk_id = analysis.chunk_id or chunk["chunk_id"]
        if analysis.summary_cn:
            return
        if isinstance(analysis, RiskAnalysis):
            analysis.summary_cn = "；".join(item.risk_cn for item in analysis.risks[:3])
        elif isinstance(analysis, MDAAnalysis):
            if analysis.metrics:
                analysis.summary_cn = "；".join(
                    f"{item.metric_name}: {item.current_value}"
                    for item in analysis.metrics[:4]
                )
            else:
                analysis.summary_cn = "；".join(
                    item.fact_cn for item in analysis.facts[:3]
                )
        else:
            analysis.summary_cn = "；".join(
                item.fact_cn for item in analysis.facts[:3]
            )

    @staticmethod
    def _renumber_items(analysis) -> None:
        if isinstance(analysis, BusinessAnalysis):
            for index, item in enumerate(analysis.facts, start=1):
                item.fact_id = f"F{index:03d}"
        elif isinstance(analysis, RiskAnalysis):
            for index, item in enumerate(analysis.risks, start=1):
                item.risk_id = f"R{index:03d}"
        else:
            for index, item in enumerate(analysis.metrics, start=1):
                item.metric_id = f"M{index:03d}"
            for index, item in enumerate(analysis.facts, start=1):
                item.fact_id = f"F{index:03d}"

    def _validate(self, analysis, unit_map: dict[str, dict]):
        if isinstance(analysis, BusinessAnalysis):
            return self._validate_business(analysis, unit_map)
        if isinstance(analysis, RiskAnalysis):
            return self._validate_risk(analysis, unit_map)
        return self._validate_mda(analysis, unit_map)

    def _validate_with_pruning(self, analysis, unit_map: dict[str, dict]):
        errors, resolved = self._validate(analysis, unit_map)
        dropped_items = []
        if not errors or not isinstance(analysis, MDAAnalysis):
            return errors, resolved, dropped_items

        hard_failure_ids = {
            match.group(1)
            for error in errors
            if (
                match := re.match(
                    r"^(M\d{3}) 的 (?:current_value|comparison_value)=", error
                )
            )
        }
        if not hard_failure_ids:
            return errors, resolved, dropped_items
        remaining_metrics = [
            metric
            for metric in analysis.metrics
            if metric.metric_id not in hard_failure_ids
        ]
        if not remaining_metrics and not analysis.facts:
            return errors, resolved, dropped_items

        dropped_items = [
            {
                "item_type": "metric",
                "reason": "关键数值无法由证据直接支持",
                "item": metric.model_dump(),
            }
            for metric in analysis.metrics
            if metric.metric_id in hard_failure_ids
        ]
        analysis.metrics = remaining_metrics
        self._renumber_items(analysis)
        errors, resolved = self._validate(analysis, unit_map)
        return errors, resolved, dropped_items

    @staticmethod
    def _resolve_evidence(ids, unit_map, errors, owner_id):
        resolved = []
        for evidence_id in ids:
            unit = unit_map.get(evidence_id)
            if unit is None:
                errors.append(f"{owner_id} 使用了无效证据编号 {evidence_id}")
            else:
                resolved.append(unit)
        return resolved

    def _validate_business(self, analysis, unit_map):
        errors = []
        resolved = {}
        if not 4 <= len(analysis.facts) <= 8:
            errors.append("Business facts 数量不在 4 到 8 之间")
        for fact in analysis.facts:
            resolved[fact.fact_id] = self._resolve_evidence(
                fact.evidence_ids, unit_map, errors, fact.fact_id
            )
        return errors, resolved

    def _validate_risk(self, analysis, unit_map):
        errors = []
        resolved = {}
        if not 2 <= len(analysis.risks) <= 6:
            errors.append("Risk items 数量不在 2 到 6 之间")
        for risk in analysis.risks:
            evidence = self._resolve_evidence(
                risk.evidence_ids, unit_map, errors, risk.risk_id
            )
            resolved[risk.risk_id] = evidence
        return errors, resolved

    def _validate_mda(self, analysis, unit_map):
        errors = []
        resolved = {"metrics": {}, "facts": {}}
        if not analysis.metrics and not analysis.facts:
            errors.append("MD&A metrics 和 facts 不能同时为空")
        if len(analysis.metrics) > 12:
            errors.append("MD&A metrics 数量超过 12")
        if len(analysis.facts) > 8:
            errors.append("MD&A facts 数量超过 8")
        for metric in analysis.metrics:
            evidence = self._resolve_evidence(
                metric.evidence_ids, unit_map, errors, metric.metric_id
            )
            resolved["metrics"][metric.metric_id] = evidence
            evidence_text = " ".join(unit["display_text"] for unit in evidence)
            numeric_fields = {
                "current_value": metric.current_value,
                "comparison_value": metric.comparison_value,
                "change_value": metric.change_value,
            }
            for field_name, value in numeric_fields.items():
                if (
                    value
                    and not _numeric_value_supported(value, evidence_text)
                    and field_name in {"current_value", "comparison_value"}
                ):
                    matching_unit = next(
                        (
                            unit
                            for unit in unit_map.values()
                            if _numeric_value_supported(value, unit["display_text"])
                            or _money_value_supported(
                                value, metric.unit, unit["display_text"]
                            )
                        ),
                        None,
                    )
                    if matching_unit and matching_unit not in evidence:
                        evidence.append(matching_unit)
                        metric.evidence_ids.append(matching_unit["evidence_id"])
                        evidence_text = " ".join(
                            unit["display_text"] for unit in evidence
                        )
                if value and not _numeric_value_supported(value, evidence_text):
                    if _money_value_supported(value, metric.unit, evidence_text):
                        continue
                    if field_name == "change_value":
                        metric.change_value = None
                        continue
                    errors.append(
                        f"{metric.metric_id} 的 {field_name}={value} 无法在证据中找到"
                    )
            for period_name, period in (
                ("current_period", metric.current_period),
                ("comparison_period", metric.comparison_period),
            ):
                if period and not _period_supported(period, evidence_text):
                    matching_period = next(
                        (
                            unit
                            for unit in unit_map.values()
                            if _period_supported(period, unit["display_text"])
                        ),
                        None,
                    )
                    if matching_period and matching_period not in evidence:
                        evidence.append(matching_period)
                        metric.evidence_ids.append(matching_period["evidence_id"])
                        evidence_text += " " + matching_period["display_text"]
                    else:
                        errors.append(
                            f"{metric.metric_id} 的 {period_name}={period} 无法在证据中找到"
                        )
            if metric.unit in {"usd_millions", "usd_billions"}:
                expected_scale = (
                    "millions" if metric.unit == "usd_millions" else "billions"
                )
                if not _unit_supported(expected_scale, evidence_text):
                    matching_unit = next(
                        (
                            unit
                            for unit in unit_map.values()
                            if _unit_supported(expected_scale, unit["display_text"])
                        ),
                        None,
                    )
                    if matching_unit and matching_unit not in evidence:
                        evidence.append(matching_unit)
                        metric.evidence_ids.append(matching_unit["evidence_id"])
                        evidence_text += " " + matching_unit["display_text"]
                    elif not _money_value_supported(
                        metric.current_value, metric.unit, evidence_text
                    ):
                        errors.append(
                            f"{metric.metric_id} 缺少 {expected_scale} 单位证据"
                        )
            current_number = _parse_number(metric.current_value)
            comparison_number = _parse_number(metric.comparison_value)
            if current_number is not None and comparison_number is not None:
                expected_direction = (
                    "increase"
                    if current_number > comparison_number
                    else "decrease"
                    if current_number < comparison_number
                    else "flat"
                )
                metric.change_direction = expected_direction

        for fact in analysis.facts:
            resolved["facts"][fact.fact_id] = self._resolve_evidence(
                fact.evidence_ids, unit_map, errors, fact.fact_id
            )

        all_text = " ".join(
            unit["display_text"]
            for unit in unit_map.values()
            if not unit.get("context_only")
        ).lower()
        metric_names = [metric.metric_name.lower() for metric in analysis.metrics]
        if "net sales by reportable segment" in all_text:
            regional_terms = (
                "americas",
                "europe",
                "greater china",
                "japan",
                "asia pacific",
            )
            if not any(
                any(term in name for term in regional_terms)
                for name in metric_names
            ):
                errors.append("MD&A 缺少地区分部指标")
        if "net sales by category" in all_text:
            product_terms = (
                "iphone",
                "mac",
                "ipad",
                "wearables",
                "services net sales",
            )
            if not any(
                any(term in name for term in product_terms)
                for name in metric_names
            ):
                errors.append("MD&A 缺少产品或服务指标")
        if (
            "products and services gross margin and gross margin percentage"
            in all_text
        ):
            if not any(
                "gross margin" in name and "percentage" not in name
                for name in metric_names
            ):
                errors.append("MD&A 缺少毛利额指标")
            if not any("gross margin percentage" in name for name in metric_names):
                errors.append("MD&A 缺少毛利率指标")
        return errors, resolved


def _normalize_numeric_text(value: str) -> str:
    text = (
        str(value)
        .replace(",", "")
        .replace("$", "")
        .replace("%", "")
    )
    text = re.sub(r"\(([-+]?\d+(?:\.\d+)?)\)", r"-\1", text)
    return re.sub(r"\s+", "", text)


def _parse_number(value: str | None) -> float | None:
    if value is None:
        return None
    match = re.search(r"[-+]?\d+(?:\.\d+)?", _normalize_numeric_text(value))
    return float(match.group()) if match else None


def _numeric_value_supported(value: str | None, evidence_text: str) -> bool:
    if value is None:
        return False
    target = _normalize_numeric_text(value)
    token_pattern = re.compile(
        r"\(\s*[-+]?\$?\s*\d[\d,]*(?:\.\d+)?\s*%?\s*\)"
        r"|[-+]?\$?\s*\d[\d,]*(?:\.\d+)?\s*%?"
    )
    return any(
        _normalize_numeric_text(token) == target
        for token in token_pattern.findall(evidence_text)
    )


MONTH_ALIASES = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}


def _period_supported(period: str, evidence_text: str) -> bool:
    normalized_period = re.sub(r"\s+", " ", period).strip().lower()
    normalized_evidence = re.sub(r"\s+", " ", evidence_text).lower()
    if normalized_period in normalized_evidence:
        return True
    period_years = set(re.findall(r"(?:19|20)\d{2}", normalized_period))
    evidence_years = set(re.findall(r"(?:19|20)\d{2}", normalized_evidence))
    if not period_years or not period_years.issubset(evidence_years):
        return False
    period_months = _extract_months(normalized_period)
    evidence_months = _extract_months(normalized_evidence)
    return not period_months or period_months.issubset(evidence_months)


def _extract_months(text: str) -> set[int]:
    months = {
        number for name, number in MONTH_ALIASES.items() if name in text.lower()
    }
    months.update(int(value) for value in re.findall(r"(\d{1,2})\s*月", text))
    return {month for month in months if 1 <= month <= 12}


def _contains_unit_context(text: str) -> bool:
    lowered = text.lower()
    return bool(
        re.search(r"\bin\s+(?:thousands|millions|billions)\b", lowered)
        or re.search(r"\b(?:million|billion)\b", lowered)
    )


def _unit_supported(scale: str, evidence_text: str) -> bool:
    singular = scale.removesuffix("s")
    lowered = evidence_text.lower()
    return bool(
        re.search(rf"\bin\s+{re.escape(scale)}\b", lowered)
        or re.search(rf"\b{re.escape(singular)}\b", lowered)
    )


def _money_value_supported(
    value: str | None,
    unit: str,
    evidence_text: str,
) -> bool:
    target = _parse_number(value)
    if target is None or unit not in {"usd_millions", "usd_billions"}:
        return False
    target_millions = target if unit == "usd_millions" else target * 1000
    pattern = re.compile(
        r"\$\s*\(?\s*([\d,]+(?:\.\d+)?)\s*\)?\s*"
        r"(million|billion)s?\b",
        re.IGNORECASE,
    )
    for raw_amount, scale in pattern.findall(evidence_text):
        amount = float(raw_amount.replace(",", ""))
        evidence_millions = amount * (1000 if scale.lower() == "billion" else 1)
        if abs(evidence_millions - target_millions) < 0.0001:
            return True
    return False


def _merge_dropped_items(existing: list[dict], new_items: list[dict]) -> list[dict]:
    merged = []
    seen = set()
    for item in existing + new_items:
        key = json.dumps(item, ensure_ascii=False, sort_keys=True)
        if key not in seen:
            seen.add(key)
            merged.append(item)
    return merged
