import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.agent_playground.filing_heading_router import (
    HeadingDecision,
    HeadingRouter,
    _parse_heading_output,
    build_heading_analysis_chunks,
    select_heading_candidates,
)


def _block(
    index: int,
    text: str,
    start: int,
    *,
    bold: bool = False,
    table_id: str | None = None,
) -> dict:
    return {
        "block_id": f"B{index:06d}",
        "block_index": index - 1,
        "tag": "div",
        "block_type": "table_content" if table_id else "paragraph",
        "text": text,
        "source_start": start,
        "source_end": start + len(text),
        "char_count": len(text),
        "is_bold": bold,
        "is_centered": False,
        "font_size_pt": 12 if bold else 9,
        "is_all_caps": False,
        "table_id": table_id,
    }


class HeadingRouterTests(unittest.TestCase):
    def test_heading_candidates_use_structure_and_exclude_table_cells(self):
        blocks = [
            _block(1, "Legal Proceedings", 0, bold=True),
            _block(2, "This is an ordinary paragraph.", 18),
            _block(3, "Net revenue", 49, bold=True, table_id="T00001"),
        ]

        candidates = select_heading_candidates(blocks)

        self.assertEqual([item["block_id"] for item in candidates], ["B000001"])

    def test_parses_heading_decisions_keyed_by_block_id(self):
        output = _parse_heading_output(
            """{
              "B000001": {
                "is_heading": true,
                "role": "content_heading",
                "level": 2,
                "section": "business",
                "labels": ["business"],
                "confidence": 0.95,
                "reason": "该文本开启业务正文"
              }
            }"""
        )

        self.assertEqual(output.headings[0].block_id, "B000001")
        self.assertEqual(output.headings[0].heading_level, 2)
        self.assertEqual(output.headings[0].target_analyzers, ["business"])

    def test_normalizes_alternative_heading_field_names(self):
        output = _parse_heading_output(
            """{
              "headings": [{
                "block_id": "B000001",
                "is_title": true,
                "heading_type": "section_heading",
                "level": 2,
                "section_type": "mda",
                "analyzers": ["mda"],
                "confidence_score": 0.9,
                "reasoning": "管理层讨论标题"
              }]
            }"""
        )

        self.assertEqual(output.headings[0].role, "content_heading")
        self.assertEqual(output.headings[0].sec_section, "mda")

    def test_allows_omitted_audit_fields_for_non_heading(self):
        output = _parse_heading_output(
            """{
              "headings": [{
                "block_id": "B000001",
                "is_heading": false,
                "type": "ordinary_text",
                "section": "other"
              }]
            }"""
        )

        self.assertEqual(output.headings[0].role, "not_heading")
        self.assertEqual(output.headings[0].confidence, 0.8)

    def test_normalizes_null_non_heading_section_to_other(self):
        output = _parse_heading_output(
            """{
              "headings": [{
                "block_id": "B000001",
                "role": "not_heading",
                "sec_section": null,
                "confidence": null,
                "reason_cn": null
              }]
            }"""
        )

        self.assertEqual(output.headings[0].sec_section, "other")
        self.assertEqual(output.headings[0].confidence, 0.8)

    def test_derives_is_heading_from_role(self):
        output = _parse_heading_output(
            """{
              "headings": [{
                "block_id": "B000001",
                "heading_role": "content_heading",
                "level": 2,
                "section": "business",
                "labels": ["business"]
              }]
            }"""
        )

        self.assertTrue(output.headings[0].is_heading)

    def test_filters_section_names_that_are_not_available_analyzers(self):
        output = _parse_heading_output(
            """{
              "headings": [{
                "block_id": "B000001",
                "role": "content_heading",
                "level": 2,
                "section": "financial_statements",
                "target_analyzers": ["financial_statements", "mda"]
              }]
            }"""
        )

        self.assertEqual(output.headings[0].target_analyzers, ["mda"])

    def test_heading_boundaries_keep_legal_and_financial_content_separate(self):
        parts = [
            "Legal Proceedings",
            "The company is involved in litigation and regulatory investigations.",
            "Financial Statement Notes",
            "Net revenue was 100 in 2025 and 90 in 2024.",
        ]
        text = "\n".join(parts)
        blocks = []
        position = 0
        for index, part in enumerate(parts, start=1):
            blocks.append(_block(index, part, position, bold=index in {1, 3}))
            position += len(part) + 1
        decisions = {
            "B000001": HeadingDecision(
                block_id="B000001",
                is_heading=True,
                role="content_heading",
                heading_level=2,
                sec_section="legal_proceedings",
                target_analyzers=["risk_factors"],
                confidence=0.95,
                reason_cn="法律章节",
            ).model_dump(),
            "B000003": HeadingDecision(
                block_id="B000003",
                is_heading=True,
                role="content_heading",
                heading_level=2,
                sec_section="financial_notes",
                target_analyzers=["mda"],
                confidence=0.95,
                reason_cn="财务附注章节",
            ).model_dump(),
        }

        chunks = build_heading_analysis_chunks(
            text=text,
            blocks=blocks,
            decisions=decisions,
            ticker="TEST",
            form="10-K",
        )

        risk_text = " ".join(
            chunk["text"] for chunk in chunks if chunk["section"] == "risk_factors"
        )
        mda_text = " ".join(
            chunk["text"] for chunk in chunks if chunk["section"] == "mda"
        )
        self.assertIn("litigation", risk_text)
        self.assertNotIn("Net revenue", risk_text)
        self.assertIn("Net revenue", mda_text)
        self.assertNotIn("litigation", mda_text)

    def test_short_top_level_cross_reference_is_not_sent_to_analyzer(self):
        reference = "Item 7 Management Discussion"
        reference_text = "See the annual report on pages 40-100."
        next_heading = "Item 7A Market Risk"
        parts = [reference, reference_text, next_heading]
        text = "\n".join(parts)
        blocks = []
        position = 0
        for index, part in enumerate(parts, start=1):
            blocks.append(_block(index, part, position, bold=index in {1, 3}))
            position += len(part) + 1
        decisions = {
            "B000001": HeadingDecision(
                block_id="B000001",
                is_heading=True,
                role="content_heading",
                heading_level=1,
                sec_section="mda",
                target_analyzers=["mda"],
                confidence=0.95,
                reason_cn="Item 7",
            ).model_dump(),
            "B000003": HeadingDecision(
                block_id="B000003",
                is_heading=True,
                role="content_heading",
                heading_level=1,
                sec_section="market_risk",
                confidence=0.95,
                reason_cn="Item 7A",
            ).model_dump(),
        }

        chunks = build_heading_analysis_chunks(
            text=text,
            blocks=blocks,
            decisions=decisions,
            ticker="TEST",
            form="10-K",
        )

        self.assertFalse(any(chunk["section"] == "mda" for chunk in chunks))

    def test_caches_heading_decisions(self):
        names = ["Business", "Risk Factors", "Management Discussion"]
        blocks = []
        position = 0
        for index, name in enumerate(names, start=1):
            blocks.append(_block(index, name, position, bold=True))
            position += len(name) + 1

        def fake_classify(batch, **kwargs):
            sections = {
                "B000001": "business",
                "B000002": "risk_factors",
                "B000003": "mda",
            }
            return (
                [
                    HeadingDecision(
                        block_id=block["block_id"],
                        is_heading=True,
                        role="content_heading",
                        heading_level=2,
                        sec_section=sections[block["block_id"]],
                        target_analyzers=[sections[block["block_id"]]],
                        confidence=0.9,
                        reason_cn="测试标题",
                    )
                    for block in batch
                ],
                {"input_tokens": 10},
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            router = HeadingRouter(
                Path(temp_dir) / "headings.json",
                batch_size=2,
                request_delay_seconds=0,
            )
            with patch.object(
                router, "_classify_batch", side_effect=fake_classify
            ) as call:
                first = router.run(
                    blocks, ticker="TEST", form="10-K", max_new_calls=None
                )
                second = router.run(
                    blocks, ticker="TEST", form="10-K", max_new_calls=0
                )

        self.assertTrue(first["complete"])
        self.assertEqual(first["selected_new_calls"], 2)
        self.assertTrue(second["complete"])
        self.assertEqual(second["selected_new_calls"], 0)
        self.assertEqual(call.call_count, 2)


if __name__ == "__main__":
    unittest.main()
