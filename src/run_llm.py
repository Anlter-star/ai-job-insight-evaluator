"""Run the LLM extractor and evaluate it against the gold dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.data_loader import DEFAULT_DATASET_PATH, PROJECT_ROOT, load_jsonl
from src.evaluator import evaluate
from src.llm_extractor import (
    LLMConfig,
    LLMConfigurationError,
    LLMResponseError,
    OpenAICompatibleExtractor,
)


DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")


def run(
    dataset_path: Path,
    results_dir: Path,
    limit: int | None = None,
    output_prefix: str = "llm",
) -> dict[str, Any]:
    records = load_jsonl(dataset_path)
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be at least 1")
        records = records[:limit]

    extractor = OpenAICompatibleExtractor(LLMConfig.from_env())
    predictions: list[dict[str, Any]] = []
    results_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = results_dir / f"{output_prefix}_predictions.jsonl"

    for index, record in enumerate(records, start=1):
        print(f"[{index}/{len(records)}] Extracting {record['job_id']} ...")
        predictions.append(
            {
                "job_id": record["job_id"],
                "method": extractor.method_name,
                "prediction": extractor.extract(record),
            }
        )
        # Preserve completed calls if a later request fails.
        _write_jsonl(prediction_path, predictions)

    metrics = {
        "method": extractor.method_name,
        "model": extractor.config.model,
        "base_url": extractor.config.base_url,
        "prompt_version": extractor.prompt_version,
        "dataset": str(dataset_path.relative_to(PROJECT_ROOT)),
        **evaluate(records, predictions),
    }
    _write_json(results_dir / f"{output_prefix}_metrics.json", metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-prefix", default="llm")
    args = parser.parse_args()

    try:
        metrics = run(args.dataset, args.results_dir, args.limit, args.output_prefix)
    except (LLMConfigurationError, LLMResponseError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(f"Records: {metrics['dataset_size']}")
    print(f"Scalar macro accuracy: {metrics['scalar_macro_accuracy']:.2%}")
    print(f"List micro precision: {metrics['list_micro']['precision']:.2%}")
    print(f"List micro recall: {metrics['list_micro']['recall']:.2%}")
    print(f"List micro F1: {metrics['list_micro']['f1']:.2%}")
    print(f"Hallucination rate: {metrics['hallucination_rate']:.2%}")


if __name__ == "__main__":
    main()
