from __future__ import annotations

import json
import re
from pathlib import Path

import streamlit as st

from src.agent_playground.filing_tool import run_filing_analysis


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data" / "sec"
TICKER_PATTERN = re.compile(r"^[A-Z][A-Z0-9.-]{0,9}$")


st.set_page_config(
    page_title="SEC Filing Research",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .stApp { background: #f4f7f7; }
      [data-testid="stSidebar"] { background: #172126; }
      [data-testid="stSidebar"] * { color: #edf2f3; }
      [data-testid="stSidebar"] input,
      [data-testid="stSidebar"] [data-baseweb="select"] * { color: #172126; }
      [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #d7e0e2;
        padding: 14px 16px;
        border-radius: 6px;
      }
      [data-testid="stMetricLabel"] { color: #68777d; }
      [data-testid="stMetricValue"] { color: #172126; font-size: 1.55rem; }
      .block-container { padding-top: 2rem; padding-bottom: 3rem; }
      h1, h2, h3 { letter-spacing: 0; color: #172126; }
      .status-complete {
        display: inline-block;
        padding: 5px 9px;
        border-radius: 4px;
        background: #e8f2ef;
        color: #13715b;
        font-size: 0.82rem;
        font-weight: 700;
      }
      .status-progress {
        display: inline-block;
        padding: 5px 9px;
        border-radius: 4px;
        background: #fff3df;
        color: #9a5b17;
        font-size: 0.82rem;
        font-weight: 700;
      }
      .source-meta {
        color: #68777d;
        font-size: 0.82rem;
        margin-bottom: 8px;
      }
      .report-shell {
        background: #ffffff;
        border: 1px solid #d7e0e2;
        border-radius: 6px;
        padding: 24px 28px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def existing_tickers() -> list[str]:
    if not DATA_ROOT.exists():
        return []
    return sorted(
        directory.name.upper()
        for directory in DATA_ROOT.iterdir()
        if directory.is_dir()
    )


def latest_metadata(company_dir: Path) -> dict:
    files = sorted(company_dir.glob("*.metadata.json"), reverse=True)
    return load_json(files[0]) if files else {}


def stage_ratio(completed: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return min(1.0, max(0.0, completed / total))


def status_totals(status: dict) -> dict:
    routing = status.get("routing_run", {})
    if routing.get("status") == "legacy":
        route_completed = 0
        route_total = 0
        route_label = "旧缓存"
    else:
        route_completed = routing.get("cached_after", 0)
        route_total = routing.get("candidate_headings", routing.get("total_chunks", 0))
        route_label = f"{route_completed}/{route_total}"

    progress = status.get("progress", {})
    map_completed = sum(item.get("cached", 0) for item in progress.values())
    map_total = sum(item.get("total", 0) for item in progress.values())
    reduce_run = status.get("reduce_run", {})
    review_run = status.get("review_run", {})
    reduce_completed = reduce_run.get("cached_after", 0)
    review_completed = review_run.get("cached_after", 0)
    return {
        "route_completed": route_completed,
        "route_total": route_total,
        "route_label": route_label,
        "map_completed": map_completed,
        "map_total": map_total,
        "reduce_completed": reduce_completed,
        "review_completed": review_completed,
    }


def flatten_candidates(reduce_cache: dict | None) -> list[dict]:
    if not reduce_cache:
        return []
    labels = {
        "business": "Business",
        "risks": "Risk",
        "mda_metrics": "MD&A Metric",
        "mda_facts": "MD&A Fact",
    }
    records = []
    for group, items in reduce_cache.get("candidates", {}).items():
        for item in items:
            copied = dict(item)
            copied["_group"] = group
            copied["_group_label"] = labels.get(group, group)
            records.append(copied)
    return records


def record_title(record: dict) -> str:
    for key in ("fact_cn", "risk_cn", "metric_name", "title_cn"):
        value = record.get(key)
        if value:
            return str(value)
    return record.get("global_id", "未命名结论")


def record_summary(record: dict) -> str:
    if record.get("metric_name"):
        current = record.get("current_value") or "未披露"
        period = record.get("current_period") or ""
        explanation = record.get("management_explanation_cn") or ""
        return f"{period} {current}\n\n{explanation}".strip()
    if record.get("risk_cn"):
        causes = "；".join(record.get("causes_cn", []))
        impacts = "；".join(record.get("potential_impacts_cn", []))
        details = []
        if causes:
            details.append(f"原因：{causes}")
        if impacts:
            details.append(f"影响：{impacts}")
        return "\n\n".join(details)
    return str(record.get("fact_cn") or "")


def clean_memo(markdown: str) -> str:
    return re.sub(r'<a id="[^"]+"></a>\s*', "", markdown)


tickers = existing_tickers()
default_ticker = "ORCL" if "ORCL" in tickers else tickers[0] if tickers else "AAPL"

with st.sidebar:
    st.title("Filing Research")
    options = tickers + ["其他公司"]
    default_index = options.index(default_ticker) if default_ticker in options else 0
    selected = st.selectbox("本地公司", options, index=default_index)
    if selected == "其他公司":
        ticker = st.text_input("股票代码", value="MSFT").strip().upper()
    else:
        ticker = selected

    valid_ticker = bool(TICKER_PATTERN.fullmatch(ticker))
    analyze_clicked = st.button(
        "开始或继续分析",
        type="primary",
        use_container_width=True,
        disabled=not valid_ticker,
    )
    st.button("刷新本地状态", use_container_width=True)

    st.divider()
    st.caption("本地报告")
    for item in tickers:
        item_status = load_json(DATA_ROOT / item / "pipeline_status.json") or {}
        marker = "完成" if item_status.get("complete") else "进行中"
        st.text(f"{item:<8} {marker}")

if not valid_ticker:
    st.error("股票代码格式无效。")
    st.stop()

company_dir = DATA_ROOT / ticker
status_path = company_dir / "pipeline_status.json"

if analyze_clicked:
    with st.status(f"正在同步分析 {ticker} 10-K", expanded=True) as task_status:
        st.write("执行标题识别、Map、Reduce 和语义审查。")
        result = run_filing_analysis(ticker, status_only=False)
        st.session_state["last_analysis_result"] = result
        if not result.get("ok"):
            task_status.update(label="分析失败", state="error", expanded=True)
            st.error(result.get("error", "未知错误"))
        elif result.get("complete"):
            task_status.update(label="分析完成", state="complete", expanded=False)
        else:
            task_status.update(label="本次同步分析已停止", state="error", expanded=True)
            for diagnostic in result.get("diagnostics", []):
                st.write(
                    f"{diagnostic.get('stage')}: {diagnostic.get('task')} - "
                    + "；".join(diagnostic.get("errors", []))
                )

status = load_json(status_path)
metadata = latest_metadata(company_dir) if company_dir.exists() else {}

company_name = metadata.get("name") or ticker
header_left, header_right = st.columns([0.78, 0.22], vertical_alignment="center")
with header_left:
    st.title(company_name)
    st.caption(
        " · ".join(
            value
            for value in (
                ticker,
                metadata.get("form", "10-K"),
                f"报告期 {metadata.get('report_date')}" if metadata.get("report_date") else None,
                f"申报日 {metadata.get('filing_date')}" if metadata.get("filing_date") else None,
            )
            if value
        )
    )
with header_right:
    if status and status.get("complete"):
        st.markdown('<span class="status-complete">COMPLETE</span>', unsafe_allow_html=True)
    elif status:
        st.markdown('<span class="status-progress">IN PROGRESS</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-progress">NO LOCAL RESULT</span>', unsafe_allow_html=True)

if not status:
    st.info("当前公司还没有本地分析状态。")
    st.stop()

totals = status_totals(status)
metric_columns = st.columns(4)
metric_columns[0].metric("标题识别", totals["route_label"])
metric_columns[1].metric("Map", f"{totals['map_completed']}/{totals['map_total']}")
metric_columns[2].metric("Reduce", f"{totals['reduce_completed']}/3")
metric_columns[3].metric("Review", f"{totals['review_completed']}/3")

progress_columns = st.columns(4)
with progress_columns[0]:
    st.progress(
        stage_ratio(totals["route_completed"], totals["route_total"]),
        text="Heading Router",
    )
with progress_columns[1]:
    st.progress(
        stage_ratio(totals["map_completed"], totals["map_total"]),
        text="Map Analysis",
    )
with progress_columns[2]:
    st.progress(stage_ratio(totals["reduce_completed"], 3), text="Reduce")
with progress_columns[3]:
    st.progress(stage_ratio(totals["review_completed"], 3), text="Semantic Review")

report_tab, evidence_tab, details_tab = st.tabs(
    ["研究报告", "SEC 证据", "运行详情"]
)

memo_path = company_dir / "research_memo.md"
reduce_path = company_dir / "reduce_v1.json"
reduce_cache = load_json(reduce_path)

with report_tab:
    if status.get("complete") and memo_path.exists():
        memo = memo_path.read_text(encoding="utf-8")
        download_col, source_col = st.columns([0.24, 0.76], vertical_alignment="center")
        with download_col:
            st.download_button(
                "下载 Markdown",
                data=memo,
                file_name=f"{ticker}_research_memo.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with source_col:
            if metadata.get("url"):
                st.link_button("打开 SEC 原文", metadata["url"])
        with st.container(border=True):
            st.markdown(clean_memo(memo))
    else:
        st.info("最终研究报告将在全部阶段完成后显示。")

with evidence_tab:
    records = flatten_candidates(reduce_cache)
    if not records:
        st.info("当前没有可展示的已验证证据。")
    else:
        filter_col, select_col = st.columns([0.34, 0.66])
        with filter_col:
            group_options = ["全部", "Business", "Risk", "MD&A Metric", "MD&A Fact"]
            selected_group = st.selectbox("结论类型", group_options)
            query = st.text_input("筛选结论", value="")
        filtered = [
            record
            for record in records
            if (selected_group == "全部" or record["_group_label"] == selected_group)
            and (
                not query
                or query.lower() in record_title(record).lower()
                or query.lower() in record.get("global_id", "").lower()
            )
        ]
        with select_col:
            if not filtered:
                st.warning("没有匹配的结论。")
                selected_record = None
            else:
                labels = {
                    f"{record.get('global_id')} · {record_title(record)[:70]}": record
                    for record in filtered
                }
                selected_label = st.selectbox("已验证结论", list(labels))
                selected_record = labels[selected_label]

        if selected_record:
            conclusion_col, source_evidence_col = st.columns([0.4, 0.6])
            with conclusion_col:
                st.subheader(record_title(selected_record))
                st.caption(
                    f"{selected_record.get('global_id')} · "
                    f"{selected_record.get('_group_label')} · "
                    f"重要性 {selected_record.get('materiality', '未标注')}"
                )
                summary = record_summary(selected_record)
                if summary:
                    st.markdown(summary)
                refs = selected_record.get("source_refs", [])
                if refs:
                    st.caption(
                        "来源块：" + ", ".join(ref.get("chunk_id", "") for ref in refs)
                    )
            with source_evidence_col:
                st.subheader("SEC 原文片段")
                evidence_items = selected_record.get("evidence", [])
                if not evidence_items:
                    st.info("该结论没有展开的证据片段。")
                for index, evidence in enumerate(evidence_items, start=1):
                    st.markdown(
                        f'<div class="source-meta">证据 {index} · '
                        f'{evidence.get("source_start", "-")} → '
                        f'{evidence.get("source_end", "-")}</div>',
                        unsafe_allow_html=True,
                    )
                    st.code(
                        evidence.get("display_text") or evidence.get("text", ""),
                        language=None,
                        wrap_lines=True,
                    )
                if metadata.get("url"):
                    st.link_button("打开 SEC 原文", metadata["url"], key="evidence_sec")

with details_tab:
    last_result = st.session_state.get("last_analysis_result")
    if last_result and last_result.get("ticker") == ticker:
        diagnostics = last_result.get("diagnostics", [])
        if diagnostics:
            st.subheader("Diagnostics")
            for diagnostic in diagnostics:
                st.error(
                    f"{diagnostic.get('stage')} · {diagnostic.get('task')}: "
                    + "；".join(diagnostic.get("errors", []))
                )
    status_json = json.dumps(status, ensure_ascii=False, indent=2)
    st.download_button(
        "下载状态 JSON",
        data=status_json,
        file_name=f"{ticker}_pipeline_status.json",
        mime="application/json",
    )
    with st.expander("Pipeline 状态", expanded=False):
        st.json(status)
    with st.expander("本地文件", expanded=False):
        st.code(
            "\n".join(
                f"{key}: {value}"
                for key, value in status.get("artifacts", {}).items()
                if value
            ),
            language=None,
        )
