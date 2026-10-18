"""Tests for split-specific validation runners."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.data_loader import PROJECT_ROOT, load_jsonl
from src.run_baseline import run as run_baseline
from src.run_evaluation_v2 import run as run_evaluation_v2


VALIDATION_PATH = PROJECT_ROOT / "data" / "validation" / "validation_dataset.jsonl"


class ValidationRunnerTests(unittest.TestCase):
    def test_baseline_writes_to_isolated_results_directory(self) -> None:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "results") as directory:
            output = Path(directory)
            metrics = run_baseline(VALIDATION_PATH, output)
            self.assertEqual(metrics["dataset_size"], 9)
            self.assertTrue((output / "baseline_predictions.jsonl").is_file())
            self.assertTrue((output / "baseline_metrics.json").is_file())

    def test_evaluator_v2_accepts_split_specific_prediction_paths(self) -> None:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "results") as directory:
            output = Path(directory)
            run_baseline(VALIDATION_PATH, output)
            rules = json.loads(
                (output / "baseline_predictions.jsonl").read_text(encoding="utf-8").splitlines()[0]
            )
            records = load_jsonl(VALIDATION_PATH)
            predictions = []
            for record in records:
                item = dict(rules)
                item["job_id"] = record["job_id"]
                item["prediction"] = {
                    field: record["gold"][field]
                    for field in rules["prediction"]
                }
                predictions.append(item)
            llm_path = output / "llm_predictions.jsonl"
            llm_path.write_text(
                "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in predictions),
                encoding="utf-8",
            )

            metrics = run_evaluation_v2(VALIDATION_PATH, output)
            self.assertEqual(metrics["rules"]["dataset_size"], 9)
            self.assertEqual(metrics["deepseek_flash_v1"]["dataset_size"], 9)
            self.assertTrue((output / "evaluation_v2_metrics.json").is_file())
            self.assertTrue((output / "error_analysis.jsonl").is_file())

    def test_evaluator_v2_can_run_rules_only(self) -> None:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "results") as directory:
            output = Path(directory)
            run_baseline(VALIDATION_PATH, output)
            metrics = run_evaluation_v2(
                VALIDATION_PATH,
                output,
                include_llm=False,
            )
            self.assertEqual(set(metrics), {"rules"})
            self.assertEqual(metrics["rules"]["dataset_size"], 9)


if __name__ == "__main__":
    unittest.main()
