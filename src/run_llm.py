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


def _dataset_label(dataset_path: Path) -> str:
    resolved = dataset_path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")


def _load_resumable_predictions(
    path: Path, method_name: str, valid_job_ids: set[str]
) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    predictions: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        item = json.loads(raw_line)
        job_id = item.get("job_id")
        if (
            item.get("method") != method_name
            or job_id not in valid_job_ids
            or job_id in seen
        ):
            continue
        predictions.append(item)
        seen.add(job_id)
    return predictions


def run(
    dataset_path: Path,
    results_dir: Path,
    limit: int | None = None,
    output_prefix: str = "llm",
    restart: bool = False,
) -> dict[str, Any]:
    records = load_jsonl(dataset_path)
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be at least 1")
        records = records[:limit]

    extractor = OpenAICompatibleExtractor(LLMConfig.from_env())
    results_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = results_dir / f"{output_prefix}_predictions.jsonl"
    predictions = (
        []
        if restart
        else _load_resumable_predictions(
            prediction_path,
            extractor.method_name,
            {record["job_id"] for record in records},
        )
    )
    completed_ids = {item["job_id"] for item in predictions}
    if predictions:
        print(f"Resuming with {len(predictions)} completed record(s).")

    for index, record in enumerate(records, start=1):
        if record["job_id"] in completed_ids:
            print(f"[{index}/{len(records)}] Skipping completed {record['job_id']}")
            continue
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
        "dataset": _dataset_label(dataset_path),
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
    parser.add_argument(
        "--restart", action="store_true", help="Ignore a partial output and call every record again"
    )
    args = parser.parse_args()

    try:
        metrics = run(
            args.dataset,
            args.results_dir,
            args.limit,
            args.output_prefix,
            args.restart,
        )
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
