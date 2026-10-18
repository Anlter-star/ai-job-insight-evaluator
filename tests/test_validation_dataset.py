"""Integrity checks for the frozen validation split."""

from __future__ import annotations

import unittest
from collections import Counter
from pathlib import Path

from src.data_loader import PROJECT_ROOT, load_jsonl


VALIDATION_PATH = PROJECT_ROOT / "data" / "validation" / "validation_dataset.jsonl"


class ValidationDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.validation_records = load_jsonl(VALIDATION_PATH)
        cls.development_records = load_jsonl()

    def test_validation_file_exists_and_contains_nine_jobs(self) -> None:
        self.assertTrue(Path(VALIDATION_PATH).is_file())
        self.assertEqual(len(self.validation_records), 9)

    def test_validation_split_is_balanced(self) -> None:
        counts = Counter(
            record["gold"]["job_category"] for record in self.validation_records
        )
        self.assertEqual(
            counts,
            Counter({"产品经理": 3, "大模型评测": 3, "数据分析": 3}),
        )

    def test_split_marker_and_ids_are_unique(self) -> None:
        validation_ids = {record["job_id"] for record in self.validation_records}
        development_ids = {record["job_id"] for record in self.development_records}
        self.assertEqual(len(validation_ids), 9)
        self.assertTrue(all(record["split"] == "validation" for record in self.validation_records))
        self.assertTrue(validation_ids.isdisjoint(development_ids))

    def test_every_record_has_source_url_and_key_evidence(self) -> None:
        for record in self.validation_records:
            with self.subTest(job_id=record["job_id"]):
                self.assertTrue(record["source"]["url"].startswith("https://"))
                self.assertEqual(record["source"]["captured_at"], "2026-10-18")
                self.assertIn("location", record["evidence"])
                self.assertIn("salary_yuan_per_day", record["evidence"])
                self.assertIn("work_days_per_week", record["evidence"])
                self.assertIn("internship_months", record["evidence"])


if __name__ == "__main__":
    unittest.main()
