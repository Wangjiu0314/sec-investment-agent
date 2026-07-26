import tempfile
import unittest
from pathlib import Path

from src.agent_playground.filing_analysis import (
    FinancialMetric,
    MDAAnalysis,
    MapAnalyzer,
    _money_value_supported,
    _period_supported,
    _unit_supported,
)


class FilingAnalysisNormalizationTests(unittest.TestCase):
    def test_periods_match_across_full_and_compact_formats(self):
        evidence = "Year Ended December 31, 2024 2025"

        self.assertTrue(
            _period_supported("Year Ended December 31, 2025", evidence)
        )
        self.assertTrue(_period_supported("2024", evidence))
        self.assertTrue(
            _period_supported("截至2025年12月31日", "As of December 31, 2025")
        )

    def test_money_values_match_equivalent_million_and_billion_units(self):
        evidence = "Costs increased by $5.0 billion and lease payments were $491 million."

        self.assertTrue(_money_value_supported("5000", "usd_millions", evidence))
        self.assertTrue(_money_value_supported("0.491", "usd_billions", evidence))
        self.assertFalse(_money_value_supported("17.5", "usd_billions", evidence))

    def test_accepts_general_sec_unit_phrases(self):
        self.assertTrue(_unit_supported("millions", "Results (in millions)"))
        self.assertTrue(_unit_supported("millions", "dollars in millions"))
        self.assertTrue(_unit_supported("billions", "cash was $91.4 billion"))

    def test_clears_unsupported_change_and_computes_numeric_direction(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            analyzer = MapAnalyzer(Path(temp_dir))
            analysis = MDAAnalysis(
                metrics=[
                    FinancialMetric(
                        metric_id="M001",
                        metric_name="Revenue",
                        current_period="Year Ended December 31, 2025",
                        current_value="100",
                        comparison_period="2024",
                        comparison_value="80",
                        change_value="20",
                        unit="usd_millions",
                        change_direction="decrease",
                        evidence_ids=["E001"],
                        materiality="high",
                    )
                ]
            )
            unit_map = {
                "E001": {
                    "evidence_id": "E001",
                    "display_text": (
                        "Results (in millions), Year Ended December 31, "
                        "2024 80 2025 100"
                    ),
                }
            }

            errors, _ = analyzer._validate_mda(analysis, unit_map)

            self.assertEqual(errors, [])
            self.assertIsNone(analysis.metrics[0].change_value)
            self.assertEqual(analysis.metrics[0].change_direction, "increase")


if __name__ == "__main__":
    unittest.main()
