import json
import copy
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from src.agent_playground.filing_tool import (
    _collect_failure_diagnostics,
    analyze_sec_filing,
    run_filing_analysis,
)


def _pipeline_result() -> dict:
    return {
        "ticker": "AAPL",
        "form": "10-K",
        "complete": True,
        "progress": {
            "business": {"cached": 5, "total": 5, "pending": 0},
            "risk_factors": {"cached": 20, "total": 20, "pending": 0},
            "mda": {"cached": 6, "total": 6, "pending": 0},
        },
        "map_run": {"selected_new_calls": 0},
        "reduce_run": {
            "status": "complete",
            "cached_after": 3,
            "pending_after": 0,
            "selected_new_calls": 0,
        },
        "review_run": {
            "status": "complete",
            "cached_after": 3,
            "pending_after": 0,
            "selected_new_calls": 0,
            "decision_counts": {
                "supported": 47,
                "partially_supported": 5,
                "unsupported": 4,
            },
        },
        "artifacts": {"memo": "data/sec/AAPL/research_memo.md"},
        "status_file": "data/sec/AAPL/pipeline_status.json",
    }


class FilingToolTests(unittest.TestCase):
    def test_rejects_invalid_ticker(self):
        result = run_filing_analysis("AAPL & whoami", status_only=True)

        self.assertFalse(result["ok"])
        self.assertIn("格式无效", result["error"])

    @patch("src.agent_playground.filing_tool.FilingPipeline")
    def test_runs_pipeline_and_returns_compact_status(self, pipeline_class):
        pipeline_class.return_value.run.return_value = _pipeline_result()

        result = run_filing_analysis(" aapl ", status_only=True)

        self.assertTrue(result["ok"])
        self.assertTrue(result["complete"])
        self.assertEqual(result["ticker"], "AAPL")
        self.assertEqual(
            result["artifacts"]["memo"], "data/sec/AAPL/research_memo.md"
        )
        pipeline_class.return_value.run.assert_called_once_with(
            offline=True, max_new_calls=0, write_status=False
        )

    @patch("src.agent_playground.filing_tool.FilingPipeline")
    def test_full_analysis_runs_all_pending_tasks(self, pipeline_class):
        pipeline_class.return_value.run.return_value = _pipeline_result()

        result = run_filing_analysis("AAPL")

        self.assertTrue(result["complete"])
        pipeline_class.return_value.run.assert_called_once_with(
            offline=False, max_new_calls=None
        )

    @patch("src.agent_playground.filing_tool.FilingPipeline")
    def test_full_analysis_retries_until_complete(self, pipeline_class):
        incomplete = copy.deepcopy(_pipeline_result())
        incomplete["complete"] = False
        incomplete["progress"]["business"] = {
            "cached": 4,
            "total": 5,
            "pending": 1,
        }
        incomplete["map_run"]["selected_new_calls"] = 1
        incomplete["reduce_run"].update(
            {"status": "waiting_for_map", "cached_after": 0, "pending_after": 3}
        )
        incomplete["review_run"].update(
            {"status": "waiting_for_reduce", "cached_after": 0, "pending_after": 3}
        )
        pipeline_class.return_value.run.side_effect = [
            incomplete,
            _pipeline_result(),
        ]

        result = run_filing_analysis("AAPL")

        self.assertTrue(result["complete"])
        self.assertEqual(result["pipeline_passes"], 2)
        self.assertEqual(result["new_tasks_this_run"], 1)
        self.assertEqual(pipeline_class.return_value.run.call_count, 2)

    @patch("src.agent_playground.filing_tool.FilingPipeline")
    def test_full_analysis_stops_when_progress_stalls(self, pipeline_class):
        incomplete = copy.deepcopy(_pipeline_result())
        incomplete["complete"] = False
        incomplete["progress"]["business"] = {
            "cached": 4,
            "total": 5,
            "pending": 1,
        }
        incomplete["map_run"]["selected_new_calls"] = 1
        pipeline_class.return_value.run.return_value = incomplete

        result = run_filing_analysis("AAPL")

        self.assertFalse(result["complete"])
        self.assertEqual(result["pipeline_passes"], 2)
        self.assertEqual(pipeline_class.return_value.run.call_count, 2)
        self.assertIn("没有后台任务", result["next_action"])

    @patch("src.agent_playground.filing_tool.FilingPipeline")
    def test_status_only_incomplete_result_explains_that_no_work_ran(
        self, pipeline_class
    ):
        incomplete = copy.deepcopy(_pipeline_result())
        incomplete["complete"] = False
        incomplete["progress"]["mda"] = {
            "cached": 1,
            "total": 2,
            "pending": 1,
        }
        pipeline_class.return_value.run.return_value = incomplete

        result = run_filing_analysis("AAPL", status_only=True)

        self.assertFalse(result["complete"])
        self.assertEqual(result["diagnostics"], [])
        self.assertIn("仅查询本地状态", result["next_action"])

    @patch("src.agent_playground.filing_tool.FilingPipeline")
    def test_langchain_tool_returns_json(self, pipeline_class):
        pipeline_class.return_value.run.return_value = _pipeline_result()

        raw = analyze_sec_filing.invoke(
            {"ticker": "AAPL", "status_only": True}
        )
        result = json.loads(raw)

        self.assertTrue(result["ok"])
        self.assertEqual(result["semantic_review"]["status"], "complete")
        self.assertFalse(result["running"])

    def test_collects_real_validation_errors_from_cache(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)
            cache_path = cache_dir / "TEST_mda_001.json"
            cache_path.write_text(
                json.dumps(
                    {
                        "chunk_metadata": {"chunk_id": "TEST_mda_001"},
                        "validation": {
                            "structure_passed": False,
                            "structure_errors": ["M001 数值无证据"],
                        },
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            result = {
                "artifacts": {"map_cache": str(cache_dir)},
                "map_run": {"exceptions": []},
                "reduce_run": {"exceptions": []},
                "review_run": {"exceptions": []},
            }

            diagnostics = _collect_failure_diagnostics(result)

            self.assertEqual(diagnostics[0]["task"], "TEST_mda_001")
            self.assertEqual(diagnostics[0]["errors"], ["M001 数值无证据"])


if __name__ == "__main__":
    unittest.main()
