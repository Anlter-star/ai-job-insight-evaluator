"""Load and validate the annotated job dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "annotated" / "gold_dataset.jsonl"

REQUIRED_TOP_LEVEL_FIELDS = {
    "schema_version",
    "job_id",
    "source",
    "input",
    "gold",
    "evidence",
}

REQUIRED_INPUT_FIELDS = {
    "job_title",
    "company",
    "city",
    "responsibilities",
    "requirements",
    "preferred_qualifications",
}


class DatasetValidationError(ValueError):
    """Raised when an annotated record does not match the expected structure."""


def _validate_record(record: dict[str, Any], line_number: int) -> None:
    missing = REQUIRED_TOP_LEVEL_FIELDS - record.keys()
    if missing:
        raise DatasetValidationError(
            f"Line {line_number} is missing top-level fields: {sorted(missing)}"
        )

    missing_input = REQUIRED_INPUT_FIELDS - record["input"].keys()
    if missing_input:
        raise DatasetValidationError(
            f"Line {line_number} is missing input fields: {sorted(missing_input)}"
        )

    if not isinstance(record["gold"], dict):
        raise DatasetValidationError(f"Line {line_number}: gold must be an object")


def load_jsonl(path: str | Path = DEFAULT_DATASET_PATH) -> list[dict[str, Any]]:
    """Load JSONL records and perform lightweight structural validation."""

    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    with dataset_path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise DatasetValidationError(
                    f"Line {line_number} is not valid JSON: {exc.msg}"
                ) from exc

            _validate_record(record, line_number)
            job_id = record["job_id"]
            if job_id in seen_ids:
                raise DatasetValidationError(f"Duplicate job_id: {job_id}")
            seen_ids.add(job_id)
            records.append(record)

    if not records:
        raise DatasetValidationError("Dataset contains no records")

    return records


def compose_source_text(record: dict[str, Any]) -> str:
    """Combine only model-visible input fields into one extraction text."""

    input_data = record["input"]
    parts = [
        f"职位名称：{input_data['job_title']}",
        f"公司：{input_data['company']}",
        f"城市：{input_data['city']}",
        f"岗位职责：{input_data['responsibilities']}",
        f"任职要求：{input_data['requirements']}",
        f"加分项：{input_data['preferred_qualifications']}",
    ]
    return "\n".join(parts)
