"""Evaluation metrics for structured job extraction."""

from __future__ import annotations

from typing import Any


SCALAR_FIELDS = [
    "job_category",
    "employment_type",
    "location",
    "salary_yuan_per_day",
    "education_min",
    "student_status",
    "graduation_year",
    "work_days_per_week",
    "internship_months",
]

LIST_FIELDS = [
    "required_hard_skills",
    "required_tools",
    "preferred_hard_skills",
    "preferred_experience",
]


def _safe_divide(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _prf(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    f1 = _safe_divide(2 * precision * recall, precision + recall)
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def evaluate(
    gold_records: list[dict[str, Any]], predictions: list[dict[str, Any]]
) -> dict[str, Any]:
    """Compare predictions against gold labels using exact normalized matches."""

    predictions_by_id = {item["job_id"]: item["prediction"] for item in predictions}
    scalar_counts = {field: {"correct": 0, "total": 0} for field in SCALAR_FIELDS}
    list_counts = {field: {"tp": 0, "fp": 0, "fn": 0} for field in LIST_FIELDS}
    job_errors: list[dict[str, Any]] = []

    for record in gold_records:
        job_id = record["job_id"]
        if job_id not in predictions_by_id:
            raise ValueError(f"Missing prediction for {job_id}")

        gold = record["gold"]
        prediction = predictions_by_id[job_id]
        scalar_errors: dict[str, Any] = {}
        list_errors: dict[str, Any] = {}

        for field in SCALAR_FIELDS:
            scalar_counts[field]["total"] += 1
            if prediction.get(field) == gold.get(field):
                scalar_counts[field]["correct"] += 1
            else:
                scalar_errors[field] = {
                    "gold": gold.get(field),
                    "prediction": prediction.get(field),
                }

        for field in LIST_FIELDS:
            gold_values = set(gold.get(field, []))
            predicted_values = set(prediction.get(field, []))
            true_positive = gold_values & predicted_values
            false_positive = predicted_values - gold_values
            false_negative = gold_values - predicted_values

            list_counts[field]["tp"] += len(true_positive)
            list_counts[field]["fp"] += len(false_positive)
            list_counts[field]["fn"] += len(false_negative)

            if false_positive or false_negative:
                list_errors[field] = {
                    "missing": sorted(false_negative),
                    "extra": sorted(false_positive),
                }

        job_errors.append(
            {
                "job_id": job_id,
                "scalar_errors": scalar_errors,
                "list_errors": list_errors,
            }
        )

    scalar_metrics = {
        field: {
            **counts,
            "accuracy": round(_safe_divide(counts["correct"], counts["total"]), 4),
        }
        for field, counts in scalar_counts.items()
    }
    scalar_macro_accuracy = sum(
        metric["accuracy"] for metric in scalar_metrics.values()
    ) / len(scalar_metrics)

    list_metrics = {
        field: _prf(counts["tp"], counts["fp"], counts["fn"])
        for field, counts in list_counts.items()
    }
    total_tp = sum(counts["tp"] for counts in list_counts.values())
    total_fp = sum(counts["fp"] for counts in list_counts.values())
    total_fn = sum(counts["fn"] for counts in list_counts.values())
    list_micro = _prf(total_tp, total_fp, total_fn)
    hallucination_rate = _safe_divide(total_fp, total_tp + total_fp)

    return {
        "dataset_size": len(gold_records),
        "scalar_metrics": scalar_metrics,
        "scalar_macro_accuracy": round(scalar_macro_accuracy, 4),
        "list_metrics": list_metrics,
        "list_micro": list_micro,
        "hallucination_rate": round(hallucination_rate, 4),
        "job_errors": job_errors,
    }
