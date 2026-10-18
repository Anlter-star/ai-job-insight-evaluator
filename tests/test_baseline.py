"""Tests for the first rule-based baseline."""

from __future__ import annotations

import unittest

from src.baseline_extractor import RuleBasedExtractor
from src.data_loader import load_jsonl
from src.evaluator import evaluate


class RuleBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.records = load_jsonl()
        cls.by_id = {record["job_id"]: record for record in cls.records}
        cls.extractor = RuleBasedExtractor()

    def test_dataset_contains_nine_unique_jobs(self) -> None:
        self.assertEqual(len(self.records), 9)
        self.assertEqual(len(self.by_id), 9)

    def test_product_job_tools_are_split_by_requirement_level(self) -> None:
        prediction = self.extractor.extract(self.by_id["PM_001"])
        self.assertEqual(prediction["job_category"], "产品经理")
        self.assertIn("Office", prediction["required_tools"])
        self.assertIn("Figma", prediction["preferred_hard_skills"])
        self.assertNotIn("Figma", prediction["required_tools"])

    def test_llm_job_numeric_constraints(self) -> None:
        prediction = self.extractor.extract(self.by_id["LLMEVAL_001"])
        self.assertEqual(prediction["job_category"], "大模型评测")
        self.assertEqual(prediction["graduation_year"], {"min": 2028})
        self.assertEqual(prediction["work_days_per_week"], 4)
        self.assertEqual(prediction["internship_months"], 3)

    def test_page_metadata_is_used_for_conflicting_work_days(self) -> None:
        prediction = self.extractor.extract(self.by_id["DA_002"])
        self.assertEqual(prediction["work_days_per_week"], 4)
        self.assertEqual(prediction["internship_months"], 3)
        self.assertEqual(
            prediction["salary_yuan_per_day"], {"min": 200, "max": 300}
        )

    def test_data_job_extracts_sql_and_prefers_page_work_days(self) -> None:
        prediction = self.extractor.extract(self.by_id["DA_002"])
        self.assertEqual(prediction["job_category"], "数据分析")
        self.assertIn("SQL", prediction["required_hard_skills"])
        self.assertIn("Office", prediction["required_tools"])
        self.assertEqual(prediction["work_days_per_week"], 4)

    def test_evaluator_returns_bounded_metrics(self) -> None:
        predictions = [
            {
                "job_id": record["job_id"],
                "prediction": self.extractor.extract(record),
            }
            for record in self.records
        ]
        metrics = evaluate(self.records, predictions)
        self.assertEqual(metrics["dataset_size"], 9)
        self.assertGreaterEqual(metrics["list_micro"]["f1"], 0.0)
        self.assertLessEqual(metrics["list_micro"]["f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
