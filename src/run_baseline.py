"""Command-line entry point for the rule baseline experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.baseline_extractor import RuleBasedExtractor
from src.data_loader import DEFAULT_DATASET_PATH, PROJECT_ROOT, load_jsonl
from src.evaluator import evaluate


DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")


def run(dataset_path: Path, results_dir: Path) -> dict[str, Any]:
    records = load_jsonl(dataset_path)
    extractor = RuleBasedExtractor()
    predictions = [
        {
            "job_id": record["job_id"],
            "method": extractor.method_name,
            "prediction": extractor.extract(record),
        }
        for record in records
    ]
    metrics = {
        "method": extractor.method_name,
        "dataset": str(dataset_path.relative_to(PROJECT_ROOT)),
        **evaluate(records, predictions),
    }

    results_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(results_dir / "baseline_predictions.jsonl", predictions)
    _write_json(results_dir / "baseline_metrics.json", metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    args = parser.parse_args()

    metrics = run(args.dataset, args.results_dir)
    print(f"Records: {metrics['dataset_size']}")
    print(f"Scalar macro accuracy: {metrics['scalar_macro_accuracy']:.2%}")
    print(f"List micro precision: {metrics['list_micro']['precision']:.2%}")
    print(f"List micro recall: {metrics['list_micro']['recall']:.2%}")
    print(f"List micro F1: {metrics['list_micro']['f1']:.2%}")
    print(f"Hallucination rate: {metrics['hallucination_rate']:.2%}")


if __name__ == "__main__":
    main()
