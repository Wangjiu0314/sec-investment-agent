from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.agent_playground.filing_pipeline import FilingPipeline


VALID_SECTIONS = {"business", "risk_factors", "mda"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="下载、解析并使用 DeepSeek 分析 SEC 财报。"
    )
    parser.add_argument("--ticker", required=True, help="美股代码，例如 AAPL")
    parser.add_argument("--form", default="10-K", help="SEC 表单类型，默认 10-K")
    parser.add_argument(
        "--sections",
        help="逗号分隔的章节：business,risk_factors,mda；默认全部",
    )
    parser.add_argument(
        "--max-new-calls",
        type=int,
        default=3,
        help="本轮最多新增的 DeepSeek 调用数，默认 3",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="处理所有待分析 chunks，忽略 --max-new-calls",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="只使用本地 SEC 文件，不访问 SEC 网络",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="重新查询 SEC 最新财报；不能与 --offline 同时使用",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="以完整 JSON 显示运行结果",
    )
    args = parser.parse_args()
    if args.offline and args.refresh:
        parser.error("--offline 和 --refresh 不能同时使用")
    if args.max_new_calls < 0:
        parser.error("--max-new-calls 不能小于 0")
    return args


def parse_sections(value: str | None) -> set[str] | None:
    if not value:
        return None
    sections = {item.strip() for item in value.split(",") if item.strip()}
    unknown = sections - VALID_SECTIONS
    if unknown:
        raise ValueError(f"不支持的章节: {', '.join(sorted(unknown))}")
    return sections


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    pipeline = FilingPipeline(
        project_root=project_root,
        ticker=args.ticker,
        form_type=args.form,
    )
    result = pipeline.run(
        max_new_calls=None if args.all else args.max_new_calls,
        sections=parse_sections(args.sections),
        refresh=args.refresh,
        offline=args.offline,
    )

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    print(f"\n{result['ticker']} {result['form']} Pipeline 状态")
    print("-" * 48)
    routing_run = result.get("routing_run", {"status": "legacy", "complete": True})
    print(f"语义路由状态：{routing_run['status']}")
    if routing_run["status"] != "legacy":
        print(
            f"标题候选：{routing_run.get('cached_after', 0)}/"
            f"{routing_run.get('candidate_headings', 0)} 完成，"
            f"本轮新增调用 {routing_run.get('selected_new_calls', 0)}"
        )
        print(f"确认正文标题：{routing_run.get('accepted_headings', 0)}")
        if routing_run.get("target_counts"):
            counts = routing_run["target_counts"]
            print(
                "目标标题："
                f"Business {counts.get('business', 0)}，"
                f"Risk {counts.get('risk_factors', 0)}，"
                f"MD&A {counts.get('mda', 0)}"
            )
    print("-" * 48)
    for section, progress in result["progress"].items():
        print(
            f"{section:>12}: {progress['cached']:>2}/{progress['total']:<2} 完成，"
            f"待处理 {progress['pending']}"
        )
    map_run = result["map_run"]
    print("-" * 48)
    print(f"本轮新增成功：{map_run['new_successes']}")
    print(f"校验失败：{map_run['validation_failed']}")
    print(f"异常：{len(map_run['exceptions'])}")
    reduce_run = result["reduce_run"]
    print("-" * 48)
    print(f"Map 全部完成：{result['map_complete']}")
    print(f"Reduce 状态：{reduce_run['status']}")
    if result["map_complete"]:
        print(
            f"Reduce 章节：{reduce_run['cached_after']}/3 完成，"
            f"本轮新增调用 {reduce_run['selected_new_calls']}"
        )
    review_run = result["review_run"]
    print("-" * 48)
    print(f"语义审查状态：{review_run['status']}")
    if reduce_run["complete"]:
        print(
            f"审查章节：{review_run['cached_after']}/3 完成，"
            f"本轮新增调用 {review_run['selected_new_calls']}"
        )
        if review_run.get("decision_counts"):
            counts = review_run["decision_counts"]
            print(
                "审查结论："
                f"支持 {counts['supported']}，"
                f"修正 {counts['partially_supported']}，"
                f"排除 {counts['unsupported']}"
            )
        if review_run.get("memo_file"):
            print(f"研究备忘录：{review_run['memo_file']}")
    print(f"全部完成：{result['complete']}")
    print(f"状态文件：{result['status_file']}")


if __name__ == "__main__":
    main()
