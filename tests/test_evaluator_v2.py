"""Tests for evaluator v2 normalization and groundedness diagnostics."""

from __future__ import annotations

import unittest

from src.evaluator_v2 import evaluate_v2
from src.normalizer import AliasNormalizer, is_supported_by_source, normalize_text


def _record() -> dict:
    return {
        "schema_version": "1.0",
        "job_id": "TEST_001",
        "source": {},
        "input": {
            "job_title": "数据分析实习生",
            "company": "测试公司",
            "city": "上海",
            "responsibilities": "负责数据处理，并开发AI Agent快速识别问题。",
            "requirements": "本科在校生，要求数据处理能力。",
            "preferred_qualifications": "",
            "page_metadata": {
                "location": "上海",
                "salary_yuan_per_day": {"min": 200, "max": 300},
                "education": "本科",
                "work_days_per_week": None,
                "internship_months": None,
            },
        },
        "gold": {
            "job_category": "数据分析",
            "employment_type": "实习",
            "location": "上海",
            "salary_yuan_per_day": {"min": 200, "max": 300},
            "education_min": "本科",
            "student_status": "在校生",
            "graduation_year": None,
            "work_days_per_week": None,
            "internship_months": None,
            "required_hard_skills": ["数据处理", "AI Agent开发"],
            "required_tools": [],
            "preferred_hard_skills": [],
            "preferred_experience": [],
        },
        "evidence": {},
    }


class EvaluatorV2Tests(unittest.TestCase):
    def test_normalize_text_handles_width_case_and_punctuation(self) -> None:
        self.assertEqual(normalize_text(" AI-Agent，开发 "), "aiagent开发")

    def test_alias_normalizer_supports_canonical_reordering(self) -> None:
        normalizer = AliasNormalizer()
        self.assertEqual(
            normalizer.normalized_set("required_hard_skills", ["开发AI Agent"]),
            {normalize_text("AI Agent开发")},
        )

    def test_grounding_requires_source_evidence(self) -> None:
        source = "负责数据处理，并开发AI Agent快速识别问题。"
        self.assertTrue(is_supported_by_source("数据处理能力", source))
        self.assertFalse(is_supported_by_source("Python", source))

    def test_reviewed_grounding_alias_handles_coordinated_phrase(self) -> None:
        normalizer = AliasNormalizer()
        self.assertTrue(
            normalizer.is_supported("B端产品经验", "有B端或策略产品经验者优先。")
        )

    def test_v2_improves_alias_match_without_changing_strict_score(self) -> None:
        prediction = {
            "job_id": "TEST_001",
            "method": "test",
            "prediction": {
                "job_category": "数据分析",
                "employment_type": "实习",
                "location": "上海",
                "salary_yuan_per_day": {"min": 200, "max": 300},
                "education_min": "本科",
                "student_status": "在校生",
                "graduation_year": None,
                "work_days_per_week": None,
                "internship_months": None,
                "required_hard_skills": ["数据处理能力", "开发AI Agent", "Python"],
                "required_tools": [],
                "preferred_hard_skills": [],
                "preferred_experience": [],
            },
        }
        metrics, analysis = evaluate_v2([_record()], [prediction])
        self.assertLess(metrics["strict"]["list_micro"]["f1"], 1.0)
        self.assertEqual(metrics["normalized"]["list_micro"]["recall"], 1.0)
        self.assertEqual(metrics["groundedness"]["potential_unsupported_count"], 1)
        unsupported = analysis[0]["list_analysis"]["required_hard_skills"][
            "potential_unsupported"
        ]
        self.assertEqual(unsupported, ["Python"])


if __name__ == "__main__":
    unittest.main()
