import json
import tempfile
import unittest
from pathlib import Path

from src.agent_playground.filing_review import (
    ClaimReview,
    SectionReview,
    SemanticReviewer,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AAPL_DIR = PROJECT_ROOT / "data" / "sec" / "AAPL"


class SemanticReviewerTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.reviewer = SemanticReviewer(Path(self.temp_dir.name))

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_builds_all_user_facing_claims(self):
        reduce_cache = json.loads(
            (AAPL_DIR / "reduce_v1.json").read_text(encoding="utf-8")
        )

        claims = self.reviewer.build_claims(reduce_cache)

        self.assertEqual(len(claims["business"]), 10)
        self.assertEqual(len(claims["risk_factors"]), 21)
        self.assertEqual(len(claims["mda"]), 25)

    def test_rejects_missing_unknown_and_embedded_ids(self):
        claims = [
            {
                "claim_id": "BUS001",
                "kind": "key_point",
                "title_cn": "title",
                "text_cn": "text",
                "initial_source_ids": ["B001"],
                "declared_status": None,
            },
            {
                "claim_id": "BUS002",
                "kind": "key_point",
                "title_cn": "title",
                "text_cn": "text",
                "initial_source_ids": ["B001"],
                "declared_status": None,
            },
        ]
        output = SectionReview(
            reviews=[
                ClaimReview(
                    claim_id="BUS001",
                    verdict="supported",
                    corrected_text_cn="结论中错误写入 B999",
                    source_ids=["B999"],
                )
            ]
        )

        errors = self.reviewer._validate_output(
            section="business",
            output=output,
            claims=claims,
            allowed_ids={"B001"},
        )

        self.assertTrue(any("BUS002" in error for error in errors))
        self.assertTrue(any("B999" in error for error in errors))
        self.assertTrue(any("corrected_text_cn" in error for error in errors))

    def test_requires_corrected_status_for_key_risk(self):
        claims = [
            {
                "claim_id": "RC001",
                "kind": "key_risk",
                "title_cn": "risk",
                "text_cn": "risk text",
                "initial_source_ids": ["RISK001"],
                "declared_status": "potential",
            }
        ]
        output = SectionReview(
            reviews=[
                ClaimReview(
                    claim_id="RC001",
                    verdict="supported",
                    corrected_text_cn="风险可能发生",
                    source_ids=["RISK001"],
                )
            ]
        )

        errors = self.reviewer._validate_output(
            section="risk_factors",
            output=output,
            claims=claims,
            allowed_ids={"RISK001"},
        )

        self.assertTrue(any("corrected_status" in error for error in errors))

    def test_risk_review_does_not_require_connections_when_none_were_input(self):
        claims = [
            {
                "claim_id": "RISK_SUMMARY",
                "kind": "summary",
                "title_cn": "summary",
                "text_cn": "风险摘要",
                "initial_source_ids": ["RISK001"],
                "declared_status": None,
            }
        ]
        output = SectionReview(
            reviews=[
                ClaimReview(
                    claim_id="RISK_SUMMARY",
                    verdict="supported",
                    corrected_text_cn="风险摘要",
                    source_ids=["RISK001"],
                )
            ]
        )

        errors = self.reviewer._validate_output(
            section="risk_factors",
            output=output,
            claims=claims,
            allowed_ids={"RISK001"},
        )

        self.assertEqual(errors, [])

    def test_supported_null_text_reuses_original_claim(self):
        claims = [
            {
                "claim_id": "BUS001",
                "kind": "key_point",
                "title_cn": "title",
                "text_cn": "原始且受支持的结论",
                "initial_source_ids": ["B001"],
                "declared_status": None,
            }
        ]
        output = SectionReview.model_validate(
            {
                "reviews": [
                    {
                        "claim_id": "BUS001",
                        "verdict": "supported",
                        "corrected_text_cn": None,
                        "source_ids": ["B001"],
                    }
                ]
            }
        )

        self.reviewer._fill_supported_text(output, claims)

        self.assertEqual(
            output.reviews[0].corrected_text_cn, "原始且受支持的结论"
        )

    def test_normalizes_embedded_ids_and_missing_partial_note(self):
        claims = [
            {
                "claim_id": "RC001",
                "kind": "key_risk",
                "title_cn": "risk",
                "text_cn": "原始风险。来源：RISK001。",
                "initial_source_ids": ["RISK001"],
                "declared_status": "mixed",
            }
        ]
        output = SectionReview(
            reviews=[
                ClaimReview(
                    claim_id="RC001",
                    verdict="partially_supported",
                    corrected_text_cn="风险可能发生。来源：RISK001。",
                    source_ids=["RISK001"],
                    corrected_status="potential",
                )
            ]
        )

        self.reviewer._normalize_reviews(output, claims)

        self.assertEqual(output.reviews[0].corrected_text_cn, "风险可能发生。")
        self.assertTrue(output.reviews[0].issues_cn)

    def test_normalizes_state_aid_decision_translation(self):
        claims = [
            {
                "claim_id": "MC001",
                "kind": "key_metric",
                "title_cn": "tax",
                "text_cn": "州援助决定相关准备下降。",
                "initial_source_ids": ["M001"],
                "declared_status": None,
            }
        ]
        output = SectionReview(
            reviews=[
                ClaimReview(
                    claim_id="MC001",
                    verdict="supported",
                    corrected_text_cn="州援助决定相关准备下降。",
                    source_ids=["M001"],
                )
            ]
        )

        self.reviewer._normalize_reviews(output, claims)

        self.assertIn("欧盟国家援助裁定", output.reviews[0].corrected_text_cn)
        self.assertEqual(output.reviews[0].verdict, "partially_supported")


if __name__ == "__main__":
    unittest.main()
