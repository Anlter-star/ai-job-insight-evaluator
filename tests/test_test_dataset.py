"""Integrity checks for the candidate unseen test split."""

from __future__ import annotations

import unittest
from collections import Counter
from hashlib import sha256
from pathlib import Path

from src.data_loader import PROJECT_ROOT, load_jsonl


TEST_PATH = PROJECT_ROOT / "data" / "test" / "test_dataset_v1.jsonl"
CHECKSUM_PATH = PROJECT_ROOT / "data" / "test" / "test_dataset_v1.sha256"
VALIDATION_PATH = PROJECT_ROOT / "data" / "validation" / "validation_dataset.jsonl"


class TestDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.test_records = load_jsonl(TEST_PATH)
        cls.development_records = load_jsonl()
        cls.validation_records = load_jsonl(VALIDATION_PATH)

    def test_frozen_file_exists_and_contains_fifteen_jobs(self) -> None:
        self.assertTrue(Path(TEST_PATH).is_file())
        self.assertEqual(len(self.test_records), 15)

    def test_frozen_file_matches_recorded_checksum(self) -> None:
        expected = CHECKSUM_PATH.read_text(encoding="utf-8").split()[0]
        actual = sha256(TEST_PATH.read_bytes()).hexdigest()
        self.assertEqual(actual, expected)

    def test_candidate_category_counts_are_documented(self) -> None:
        counts = Counter(record["gold"]["job_category"] for record in self.test_records)
        self.assertEqual(
            counts,
            Counter({"数据分析": 5, "大模型评测": 5, "产品经理": 5}),
        )

    def test_candidate_jobs_are_internships_with_top_metadata(self) -> None:
        required_metadata = {
            "location",
            "salary_yuan_per_day",
            "education",
            "work_days_per_week",
            "internship_months",
        }
        for record in self.test_records:
            with self.subTest(job_id=record["job_id"]):
                self.assertEqual(record["split"], "test")
                self.assertEqual(record["gold"]["employment_type"], "实习")
                self.assertEqual(
                    record["input"]["dataset_context"]["employment_type"],
                    "实习",
                )
                self.assertTrue(
                    required_metadata.issubset(record["input"]["page_metadata"])
                )
                self.assertTrue(
                    all(
                        value is not None
                        for value in record["input"]["page_metadata"].values()
                    )
                )

    def test_ids_and_urls_do_not_overlap_existing_splits(self) -> None:
        existing_records = self.development_records + self.validation_records
        existing_ids = {record["job_id"] for record in existing_records}
        existing_urls = {record["source"]["url"] for record in existing_records}
        candidate_ids = {record["job_id"] for record in self.test_records}
        candidate_urls = {record["source"]["url"] for record in self.test_records}

        self.assertEqual(len(candidate_ids), len(self.test_records))
        self.assertEqual(len(candidate_urls), len(self.test_records))
        self.assertTrue(candidate_ids.isdisjoint(existing_ids))
        self.assertTrue(candidate_urls.isdisjoint(existing_urls))


if __name__ == "__main__":
    unittest.main()
