import json
import tempfile
import unittest
from pathlib import Path

from src.agent_playground.filing_pipeline import FilingPipeline


class FilingPipelineSectionTests(unittest.TestCase):
    def test_structural_blocks_preserve_source_positions_and_style(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            pipeline = FilingPipeline(project_root, "TEST")
            html_path = pipeline.company_dir / "sample.htm"
            html_path.write_text(
                """<html><body>
                <div style="font-weight:bold;text-align:center;font-size:16pt">
                  Management Discussion
                </div>
                <p>Operating performance improved during the year.</p>
                <table><tr><td>2025</td><td>100</td></tr></table>
                </body></html>""",
                encoding="utf-8",
            )

            text_path, blocks_path = pipeline._prepare_structural_document(html_path)
            source = text_path.read_text(encoding="utf-8")
            blocks = json.loads(blocks_path.read_text(encoding="utf-8"))["blocks"]

            self.assertGreaterEqual(len(blocks), 4)
            heading = next(block for block in blocks if block["text"] == "Management Discussion")
            self.assertTrue(heading["is_bold"])
            self.assertTrue(heading["is_centered"])
            self.assertEqual(heading["font_size_pt"], 16)
            self.assertTrue(any(block["table_id"] for block in blocks))
            for block in blocks:
                self.assertEqual(
                    source[block["source_start"] : block["source_end"]],
                    block["text"],
                )

    def test_completed_legacy_pipeline_keeps_existing_chunks(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            pipeline = FilingPipeline(project_root, "TEST")
            chunks_path = pipeline.company_dir / "legacy_chunks.json"
            chunks_path.write_text('{"chunks": []}', encoding="utf-8")
            (pipeline.company_dir / "pipeline_status.json").write_text(
                json.dumps(
                    {
                        "complete": True,
                        "artifacts": {"chunks": str(chunks_path)},
                    }
                ),
                encoding="utf-8",
            )

            prepared = pipeline._prepare_input(refresh=False, offline=True)

            self.assertEqual(prepared["mode"], "legacy")
            self.assertEqual(prepared["chunks_path"], chunks_path)


if __name__ == "__main__":
    unittest.main()
