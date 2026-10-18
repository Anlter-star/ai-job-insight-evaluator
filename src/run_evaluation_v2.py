"""Evaluate saved rule and LLM predictions with evaluator v2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.data_loader import PROJECT_ROOT, load_jsonl
from src.evaluator_v2 import evaluate_v2


RESULTS_DIR = PROJECT_ROOT / "results"
PREDICTION_FILES = {
    "rules": RESULTS_DIR / "baseline_predictions.jsonl",
    "deepseek_flash_v1": RESULTS_DIR / "llm_predictions.jsonl",
}


def _write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")


def _load_predictions(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")
    predictions: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw_line.strip():
            continue
        item = json.loads(raw_line)
        job_id = item.get("job_id")
        if not isinstance(job_id, str) or not isinstance(item.get("prediction"), dict):
            raise ValueError(f"Invalid prediction at {path}:{line_number}")
        if job_id in seen_ids:
            raise ValueError(f"Duplicate prediction for {job_id} in {path}")
        seen_ids.add(job_id)
        predictions.append(item)
    return predictions


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    gold_records = load_jsonl()
    all_metrics: dict[str, Any] = {}
    all_analyses: list[dict[str, Any]] = []

    for method, path in PREDICTION_FILES.items():
        predictions = _load_predictions(path)
        metrics, analyses = evaluate_v2(gold_records, predictions)
        all_metrics[method] = metrics
        all_analyses.extend({"evaluation_method": method, **item} for item in analyses)

    _write_json(RESULTS_DIR / "evaluation_v2_metrics.json", all_metrics)
    _write_jsonl(RESULTS_DIR / "error_analysis.jsonl", all_analyses)

    print(f"{'Method':<22} {'Strict F1':>11} {'Normalized F1':>15} {'Potential unsupported':>23}")
    print("-" * 75)
    for method, metrics in all_metrics.items():
        print(
            f"{method:<22} "
            f"{metrics['strict']['list_micro']['f1']:>10.2%} "
            f"{metrics['normalized']['list_micro']['f1']:>14.2%} "
            f"{metrics['groundedness']['potential_unsupported_rate']:>22.2%}"
        )


if __name__ == "__main__":
    main()
