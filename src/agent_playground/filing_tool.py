from __future__ import annotations

import json
import re
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.tools import tool

from .filing_pipeline import FilingPipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAX_PIPELINE_PASSES = 3
TICKER_PATTERN = re.compile(r"^[A-Z][A-Z0-9.-]{0,9}$")


def run_filing_analysis(
    ticker: str,
    *,
    status_only: bool = False,
) -> dict:
    normalized_ticker = ticker.strip().upper()
    if not TICKER_PATTERN.fullmatch(normalized_ticker):
        return {
            "ok": False,
            "error": "股票代码格式无效，只允许 1 到 10 位字母、数字、点或连字符。",
        }
    load_dotenv(PROJECT_ROOT / ".env", override=True)
    try:
        pipeline = FilingPipeline(
            project_root=PROJECT_ROOT,
            ticker=normalized_ticker,
            form_type="10-K",
        )
        result, pass_count, new_tasks = _run_pipeline(
            pipeline,
            status_only=status_only,
        )
    except Exception as exc:
        return {
            "ok": False,
            "ticker": normalized_ticker,
            "error": str(exc),
        }

    progress = result["progress"]
    routing_run = result.get(
        "routing_run",
        {
            "status": "legacy",
            "complete": True,
            "cached_after": 0,
            "pending_after": 0,
            "target_counts": {},
        },
    )
    reduce_run = result["reduce_run"]
    review_run = result["review_run"]
    response = {
        "ok": True,
        "ticker": result["ticker"],
        "form": result["form"],
        "complete": result["complete"],
        "routing": {
            "status": routing_run.get("status", "legacy"),
            "complete": routing_run.get("complete", True),
            "completed_chunks": routing_run.get("cached_after", 0),
            "pending_chunks": routing_run.get("pending_after", 0),
            "reviewed_candidates": routing_run.get("cached_after", 0),
            "pending_candidates": routing_run.get("pending_after", 0),
            "candidate_headings": routing_run.get("candidate_headings", 0),
            "accepted_headings": routing_run.get("accepted_headings", 0),
            "target_counts": routing_run.get("target_counts", {}),
        },
        "map": {
            section: {
                "completed": values["cached"],
                "total": values["total"],
                "pending": values["pending"],
            }
            for section, values in progress.items()
        },
        "reduce": {
            "status": reduce_run["status"],
            "completed_sections": reduce_run.get("cached_after", 0),
            "pending_sections": reduce_run.get("pending_after", 3),
        },
        "semantic_review": {
            "status": review_run["status"],
            "completed_sections": review_run.get("cached_after", 0),
            "pending_sections": review_run.get("pending_after", 3),
        },
        "running": False,
        "pipeline_passes": pass_count,
        "new_tasks_this_run": new_tasks,
        "artifacts": {
            "memo": result["artifacts"]["memo"] if result["complete"] else None,
            "status": result["status_file"],
        },
    }
    if not result["complete"]:
        diagnostics = _collect_failure_diagnostics(result)
        response["diagnostics"] = diagnostics
        if diagnostics:
            response["next_action"] = (
                "本次同步分析已经停止，没有后台任务仍在运行。"
                "请根据 diagnostics 修复或重新运行。"
            )
        elif status_only:
            response["next_action"] = (
                "本次仅查询本地状态，没有新增任务，也没有后台任务。"
                "使用 status_only=false 可继续完成剩余分析。"
            )
        else:
            response["next_action"] = (
                "本次同步分析已经停止，没有后台任务仍在运行。请重新运行。"
            )
    return response


def _run_pipeline(
    pipeline: FilingPipeline,
    *,
    status_only: bool,
) -> tuple[dict, int, int]:
    if status_only:
        result = pipeline.run(
            offline=True,
            max_new_calls=0,
            write_status=False,
        )
        return result, 1, 0

    total_new_tasks = 0
    previous_progress: tuple | None = None
    result: dict | None = None
    for pass_count in range(1, MAX_PIPELINE_PASSES + 1):
        result = pipeline.run(offline=False, max_new_calls=None)
        total_new_tasks += _selected_tasks(result)
        if result["complete"]:
            return result, pass_count, total_new_tasks

        current_progress = _progress_signature(result)
        if current_progress == previous_progress:
            return result, pass_count, total_new_tasks
        previous_progress = current_progress

    if result is None:
        raise RuntimeError("Pipeline 没有返回运行结果")
    return result, MAX_PIPELINE_PASSES, total_new_tasks


def _selected_tasks(result: dict) -> int:
    return (
        result.get("routing_run", {}).get("selected_new_calls", 0)
        + result["map_run"].get("selected_new_calls", 0)
        + result["reduce_run"].get("selected_new_calls", 0)
        + result["review_run"].get("selected_new_calls", 0)
    )


def _progress_signature(result: dict) -> tuple:
    map_pending = sum(item["pending"] for item in result["progress"].values())
    return (
        result.get("routing_run", {}).get("pending_after", 0),
        map_pending,
        result["reduce_run"].get("pending_after", 3),
        result["review_run"].get("pending_after", 3),
    )


def _collect_failure_diagnostics(result: dict) -> list[dict]:
    diagnostics = []
    routing_run = result.get("routing_run", {})
    for error in routing_run.get("coverage_errors", []):
        diagnostics.append(
            {"stage": "routing", "task": "coverage", "errors": [error]}
        )
    for exception in routing_run.get("exceptions", []):
        diagnostics.append(
            {
                "stage": "routing",
                "task": ", ".join(
                    exception.get("block_ids", [])
                    or exception.get("chunk_ids", [])
                )
                or "batch",
                "errors": [exception.get("error", "未知异常")],
            }
        )
    map_cache = result.get("artifacts", {}).get("map_cache")
    if map_cache:
        for cache_path in sorted(Path(map_cache).glob("*.json")):
            if cache_path.name == "latest_batch_status.json":
                continue
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            validation = cached.get("validation", {})
            if validation.get("structure_passed", False):
                continue
            diagnostics.append(
                {
                    "stage": "map",
                    "task": cached.get("chunk_metadata", {}).get(
                        "chunk_id", cache_path.stem
                    ),
                    "errors": validation.get("structure_errors", [])[:5],
                }
            )

    for stage, artifact_key in (
        ("reduce", "reduce"),
        ("semantic_review", "semantic_review"),
    ):
        artifact_path = result.get("artifacts", {}).get(artifact_key)
        if not artifact_path or not Path(artifact_path).exists():
            continue
        try:
            cached = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for section, section_result in cached.get("sections", {}).items():
            validation = section_result.get("validation", {})
            if validation.get("structure_passed", False):
                continue
            diagnostics.append(
                {
                    "stage": stage,
                    "task": section,
                    "errors": validation.get("structure_errors", [])[:5],
                }
            )

    for stage, run_key in (
        ("routing", "routing_run"),
        ("map", "map_run"),
        ("reduce", "reduce_run"),
        ("semantic_review", "review_run"),
    ):
        run_result = result.get(run_key, {})
        if stage == "routing":
            continue
        for exception in run_result.get("exceptions", []):
            diagnostics.append(
                {
                    "stage": stage,
                    "task": exception.get("chunk_id")
                    or exception.get("section")
                    or "unknown",
                    "errors": [exception.get("error", "未知异常")],
                }
            )
    return diagnostics[:10]


@tool
def analyze_sec_filing(
    ticker: str,
    status_only: bool = False,
) -> str:
    """分析美股公司的最新 10-K，并返回进度、审查状态和报告路径。

    ticker 是美股代码，例如 AAPL。正常分析时使用 status_only=false，工具会自动
    处理全部 Heading Router、Map、Reduce 和 Semantic Review 任务；只查看已有
    本地状态时使用 status_only=true，不新增分析任务。
    """
    result = run_filing_analysis(
        ticker,
        status_only=status_only,
    )
    return json.dumps(result, ensure_ascii=False)
