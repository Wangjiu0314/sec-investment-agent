import json
import tempfile
import unittest
from pathlib import Path

from src.agent_playground.filing_reduce import (
    BusinessPoint,
    BusinessReduce,
    FilingReducer,
    RiskPoint,
    RiskReduce,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AAPL_DIR = PROJECT_ROOT / "data" / "sec" / "AAPL"
CHUNKS_PATH = AAPL_DIR / (
    "2025-10-31_10-K_0000320193-25-000079_"
    "aapl-20250927_chunks.json"
)


class FilingReducerTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.reducer = FilingReducer(
            Path(self.temp_dir.name), AAPL_DIR / "map_analysis_v3"
        )
        self.chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))["chunks"]

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_collects_stable_global_ids_and_deduplicates_overlap(self):
        candidates, collection = self.reducer.collect_candidates(self.chunks)

        self.assertEqual(collection["before_dedup"]["risks"], 58)
        self.assertEqual(collection["after_dedup"]["risks"], 56)
        self.assertEqual(candidates["business"][0]["global_id"], "B001")
        self.assertEqual(candidates["risks"][0]["global_id"], "RISK001")
        self.assertEqual(candidates["mda_metrics"][0]["global_id"], "M001")
        self.assertEqual(candidates["mda_facts"][0]["global_id"], "D001")

    def test_rejects_unknown_business_source_id(self):
        output = BusinessReduce(
            section_summary_cn="summary B998",
            key_points=[
                BusinessPoint(
                    title_cn=f"point {index}",
                    analysis_cn="analysis",
                    source_ids=["B999" if index == 1 else "B001"],
                )
                for index in range(1, 5)
            ],
        )
        section_candidates = {"business": [{"global_id": "B001"}]}

        errors = self.reducer._validate_output(
            "business", output, {"B001"}, section_candidates
        )

        self.assertTrue(any("B999" in error for error in errors))
        self.assertTrue(any("B998" in error for error in errors))

    def test_rejects_risk_status_not_supported_by_sources(self):
        output = RiskReduce(
            section_summary_cn="summary",
            key_risks=[
                RiskPoint(
                    title_cn=f"risk {index}",
                    assessment_cn="assessment",
                    status="realized" if index == 1 else "potential",
                    source_ids=["RISK001"],
                )
                for index in range(1, 7)
            ],
        )
        section_candidates = {
            "risks": [{"global_id": "RISK001", "status": "potential"}]
        }

        errors = self.reducer._validate_output(
            "risk_factors", output, {"RISK001"}, section_candidates
        )

        self.assertTrue(any("realized" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
