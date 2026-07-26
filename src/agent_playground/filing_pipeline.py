from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

from .filing_analysis import MapAnalyzer
from .filing_reduce import FilingReducer
from .filing_review import SemanticReviewer
from .filing_heading_router import (
    ANALYSIS_CHUNK_OVERLAP,
    ANALYSIS_CHUNK_SIZE,
    HeadingRouter,
    build_heading_analysis_chunks,
    extract_structural_document,
)


class FilingPipeline:
    def __init__(
        self,
        project_root: Path,
        ticker: str,
        form_type: str = "10-K",
    ) -> None:
        self.project_root = project_root.resolve()
        self.ticker = ticker.strip().upper()
        self.form_type = form_type.strip().upper()
        self.data_root = self.project_root / "data" / "sec"
        self.company_dir = self.data_root / self.ticker
        self.company_dir.mkdir(parents=True, exist_ok=True)
        load_dotenv(self.project_root / ".env", override=True)

    def run(
        self,
        *,
        max_new_calls: int | None = 3,
        sections: set[str] | None = None,
        refresh: bool = False,
        offline: bool = False,
        write_status: bool = True,
    ) -> dict:
        prepared = self._prepare_input(refresh=refresh, offline=offline)
        routing_stats = {
            "status": "legacy",
            "complete": True,
            "selected_new_calls": 0,
            "pending_after": 0,
            "exceptions": [],
            "coverage_errors": [],
        }
        if prepared["mode"] == "legacy":
            chunks_path = prepared["chunks_path"]
            chunks_data = json.loads(chunks_path.read_text(encoding="utf-8"))
            chunks = chunks_data["chunks"]
            cache_dir = self.company_dir / "map_analysis_v3"
        else:
            text_path = prepared["text_path"]
            structural_blocks_path = prepared["structural_blocks_path"]
            structural_data = json.loads(
                structural_blocks_path.read_text(encoding="utf-8")
            )
            structural_blocks = structural_data["blocks"]
            router = HeadingRouter(self.company_dir / "heading_routes_v1.json")
            routing_stats = router.run(
                structural_blocks,
                ticker=self.ticker,
                form=self.form_type,
                max_new_calls=max_new_calls,
            )
            chunks_path = text_path.with_name(
                f"{text_path.stem}_heading_chunks_v1.json"
            )
            chunks = []
            if routing_stats["complete"]:
                decisions = router.load_current_decisions(structural_blocks)
                text = text_path.read_text(encoding="utf-8")
                chunks = build_heading_analysis_chunks(
                    text=text,
                    blocks=structural_blocks,
                    decisions=decisions,
                    ticker=self.ticker,
                    form=self.form_type,
                )
                routed_data = {
                    "pipeline_version": "heading_router_v1",
                    "ticker": self.ticker,
                    "form": self.form_type,
                    "source_file": str(text_path),
                    "source_structural_blocks": str(structural_blocks_path),
                    "source_heading_routes": str(router.cache_path),
                    "chunk_config": {
                        "chunk_size": ANALYSIS_CHUNK_SIZE,
                        "chunk_overlap": ANALYSIS_CHUNK_OVERLAP,
                        "length_unit": "characters",
                    },
                    "chunks": chunks,
                }
                if write_status or not chunks_path.exists():
                    chunks_path.write_text(
                        json.dumps(routed_data, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
            cache_dir = self.company_dir / "map_analysis_v5"

        analyzer = MapAnalyzer(cache_dir)
        remaining_map_calls = (
            None
            if max_new_calls is None
            else max(0, max_new_calls - routing_stats["selected_new_calls"])
        )
        if routing_stats["complete"]:
            map_stats = analyzer.run_batch(
                chunks,
                sections=sections,
                max_new_calls=remaining_map_calls,
            )
            progress = self._build_progress(chunks, analyzer)
        else:
            map_stats = {
                "status": "waiting_for_routing",
                "selected_new_calls": 0,
                "new_successes": 0,
                "validation_failed": 0,
                "exceptions": [],
            }
            progress = {
                section: {"total": 0, "cached": 0, "pending": 0}
                for section in ("business", "risk_factors", "mda")
            }
        map_complete = routing_stats["complete"] and all(
            item["pending"] == 0 for item in progress.values()
        )
        reduce_stats = {
            "status": "waiting_for_map",
            "complete": False,
            "selected_new_calls": 0,
        }
        review_stats = {
            "status": "waiting_for_reduce",
            "complete": False,
            "selected_new_calls": 0,
        }
        if map_complete:
            reducer = FilingReducer(self.company_dir, cache_dir)
            remaining_calls = (
                None
                if max_new_calls is None
                else max(
                    0,
                    max_new_calls
                    - routing_stats["selected_new_calls"]
                    - map_stats["selected_new_calls"],
                )
            )
            reduce_stats = reducer.run(chunks, max_new_calls=remaining_calls)
            reduce_stats["status"] = (
                "complete" if reduce_stats["complete"] else "in_progress"
            )
            if reduce_stats["complete"]:
                remaining_review_calls = (
                    None
                    if max_new_calls is None
                    else max(
                        0,
                        max_new_calls
                        - routing_stats["selected_new_calls"]
                        - map_stats["selected_new_calls"]
                        - reduce_stats["selected_new_calls"],
                    )
                )
                reviewer = SemanticReviewer(self.company_dir)
                review_stats = reviewer.run(
                    max_new_calls=remaining_review_calls
                )
                review_stats["status"] = (
                    "complete" if review_stats["complete"] else "in_progress"
                )
        result = {
            "ticker": self.ticker,
            "form": self.form_type,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "artifacts": {
                "chunks": str(chunks_path),
                "structural_blocks": str(
                    prepared.get("structural_blocks_path", chunks_path)
                ),
                "routes": str(self.company_dir / "heading_routes_v1.json")
                if prepared["mode"] != "legacy"
                else None,
                "map_cache": str(cache_dir),
                "reduce": str(self.company_dir / "reduce_v1.json"),
                "semantic_review": str(
                    self.company_dir / "semantic_review_v1.json"
                ),
                "draft_memo": str(
                    self.company_dir / "research_memo_draft.md"
                ),
                "memo": str(self.company_dir / "research_memo.md"),
            },
            "routing_run": routing_stats,
            "map_run": map_stats,
            "progress": progress,
            "map_complete": map_complete,
            "reduce_run": reduce_stats,
            "review_run": review_stats,
            "complete": (
                routing_stats["complete"]
                and map_complete
                and reduce_stats["complete"]
                and review_stats["complete"]
            ),
        }
        status_path = self.company_dir / "pipeline_status.json"
        if write_status:
            status_path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        result["status_file"] = str(status_path)
        return result

    def _prepare_input(self, *, refresh: bool, offline: bool) -> dict:
        if not refresh:
            legacy_chunks = self._completed_legacy_chunks()
            if legacy_chunks is not None:
                return {"mode": "legacy", "chunks_path": legacy_chunks}

        html_path = None if refresh else self._latest_html()
        if html_path is None:
            if offline:
                raise FileNotFoundError(
                    f"offline mode has no local {self.ticker} {self.form_type} filing"
                )
            html_path = self._download_latest_filing()
        text_path, structural_blocks_path = self._prepare_structural_document(
            html_path
        )
        return {
            "mode": "heading_router",
            "text_path": text_path,
            "structural_blocks_path": structural_blocks_path,
        }

    def _completed_legacy_chunks(self) -> Path | None:
        status_path = self.company_dir / "pipeline_status.json"
        if not status_path.exists():
            return None
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
            chunks_value = status.get("artifacts", {}).get("chunks")
            if not status.get("complete") or not chunks_value:
                return None
            chunks_path = Path(chunks_value)
            if not chunks_path.is_absolute():
                chunks_path = self.project_root / chunks_path
            if (
                chunks_path.exists()
                and "_routed_chunks_v1.json" not in chunks_path.name
            ):
                return chunks_path
        except (OSError, ValueError, TypeError):
            return None
        return None

    def _prepare_structural_document(self, html_path: Path) -> tuple[Path, Path]:
        text_path = html_path.with_name(f"{html_path.stem}_structured_v1.txt")
        blocks_path = html_path.with_name(
            f"{html_path.stem}_structural_blocks_v1.json"
        )
        if text_path.exists() and blocks_path.exists():
            return text_path, blocks_path
        document = extract_structural_document(html_path)
        text_path.write_text(document["text"], encoding="utf-8")
        blocks_path.write_text(
            json.dumps(
                {
                    "pipeline_version": "heading_router_v1",
                    "ticker": self.ticker,
                    "form": self.form_type,
                    "source_html": str(html_path),
                    "source_text": str(text_path),
                    "blocks": document["blocks"],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return text_path, blocks_path

    def _latest(self, pattern: str) -> Path | None:
        files = sorted(self.company_dir.glob(pattern), reverse=True)
        return files[0] if files else None

    def _latest_html(self) -> Path | None:
        files = sorted(
            (
                path
                for path in self.company_dir.iterdir()
                if path.is_file() and path.suffix.lower() in {".htm", ".html"}
            ),
            reverse=True,
        )
        return files[0] if files else None

    def _download_latest_filing(self) -> Path:
        user_agent = os.getenv("SEC_USER_AGENT", "").strip()
        if not user_agent:
            raise ValueError("请先在 .env 中配置 SEC_USER_AGENT=姓名 邮箱")
        headers = {
            "User-Agent": user_agent,
            "Accept-Encoding": "gzip, deflate",
        }

        companies = _sec_get(
            "https://www.sec.gov/files/company_tickers.json", headers
        ).json()
        company = next(
            (
                item
                for item in companies.values()
                if item["ticker"].upper() == self.ticker
            ),
            None,
        )
        if company is None:
            raise ValueError(f"SEC 公司列表中没有找到股票代码: {self.ticker}")

        cik = str(company["cik_str"]).zfill(10)
        submissions = _sec_get(
            f"https://data.sec.gov/submissions/CIK{cik}.json", headers
        ).json()
        recent = submissions["filings"]["recent"]
        filing = None
        for index, form in enumerate(recent["form"]):
            if form == self.form_type:
                filing = {
                    "name": company["title"],
                    "cik": cik,
                    "filing_date": recent["filingDate"][index],
                    "report_date": recent["reportDate"][index],
                    "accession_number": recent["accessionNumber"][index],
                    "primary_document": recent["primaryDocument"][index],
                }
                break
        if filing is None:
            raise ValueError(
                f"没有在近期申报中找到 {self.ticker} 的 {self.form_type}"
            )

        accession_no_dashes = filing["accession_number"].replace("-", "")
        cik_no_zeros = str(int(cik))
        url = (
            "https://www.sec.gov/Archives/edgar/data/"
            f"{cik_no_zeros}/{accession_no_dashes}/{filing['primary_document']}"
        )
        filename = (
            f"{filing['filing_date']}_{self.form_type}_"
            f"{filing['accession_number']}_{filing['primary_document']}"
        )
        output_path = self.company_dir / filename
        if not output_path.exists():
            output_path.write_bytes(_sec_get(url, headers).content)

        metadata_path = output_path.with_name(f"{output_path.name}.metadata.json")
        metadata_path.write_text(
            json.dumps(
                {
                    **filing,
                    "ticker": self.ticker,
                    "form": self.form_type,
                    "url": url,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return output_path

    @staticmethod
    def _build_progress(chunks: list[dict], analyzer: MapAnalyzer) -> dict:
        progress = {}
        for section in ("business", "risk_factors", "mda"):
            section_chunks = [chunk for chunk in chunks if chunk["section"] == section]
            cached = sum(
                analyzer.load_current_cache(chunk) is not None
                for chunk in section_chunks
            )
            progress[section] = {
                "total": len(section_chunks),
                "cached": cached,
                "pending": len(section_chunks) - cached,
            }
        return progress


def _sec_get(url: str, headers: dict[str, str]) -> requests.Response:
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    return response
