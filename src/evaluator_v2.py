"""Dual strict/normalized evaluation with groundedness diagnostics."""

from __future__ import annotations

from typing import Any

from src.data_loader import compose_source_text
from src.evaluator import LIST_FIELDS, evaluate
from src.normalizer import AliasNormalizer, normalize_text


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


def evaluate_v2(
    gold_records: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    normalizer: AliasNormalizer | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return dual metrics and a reviewable per-job error analysis."""

    normalizer = normalizer or AliasNormalizer()
    strict = evaluate(gold_records, predictions)
    predictions_by_id = {item["job_id"]: item for item in predictions}
    list_counts = {field: {"tp": 0, "fp": 0, "fn": 0} for field in LIST_FIELDS}
    analyses: list[dict[str, Any]] = []
    total_predictions = 0
    alias_resolved = 0
    supported_extras = 0
    potential_unsupported = 0

    for record in gold_records:
        job_id = record["job_id"]
        if job_id not in predictions_by_id:
            raise ValueError(f"Missing prediction for {job_id}")
        prediction_record = predictions_by_id[job_id]
        prediction = prediction_record["prediction"]
        source_text = compose_source_text(record)
        field_analysis: dict[str, Any] = {}

        for field in LIST_FIELDS:
            gold_values = record["gold"].get(field, [])
            predicted_values = prediction.get(field, [])
            gold_set = {normalize_text(value) for value in gold_values}
            predicted_set = normalizer.normalized_set(field, predicted_values)
            true_positive = gold_set & predicted_set
            false_positive = predicted_set - gold_set
            false_negative = gold_set - predicted_set
            list_counts[field]["tp"] += len(true_positive)
            list_counts[field]["fp"] += len(false_positive)
            list_counts[field]["fn"] += len(false_negative)
            total_predictions += len(predicted_values)

            alias_matches: list[dict[str, Any]] = []
            source_supported_extras: list[str] = []
            unsupported_extras: list[str] = []
            for value in predicted_values:
                targets = normalizer.alias_targets(field, value)
                if targets and any(normalize_text(target) in gold_set for target in targets):
                    alias_matches.append({"prediction": value, "canonical": targets})
                    alias_resolved += 1
                    continue
                expanded = {
                    normalize_text(target)
                    for target in normalizer.expand(field, value)
                }
                if expanded & gold_set:
                    continue
                if normalizer.is_supported(value, source_text):
                    source_supported_extras.append(value)
                    supported_extras += 1
                else:
                    unsupported_extras.append(value)
                    potential_unsupported += 1

            missing_labels = [
                value for value in gold_values if normalize_text(value) in false_negative
            ]
            if alias_matches or source_supported_extras or unsupported_extras or missing_labels:
                field_analysis[field] = {
                    "alias_matches": alias_matches,
                    "source_supported_extras": source_supported_extras,
                    "potential_unsupported": unsupported_extras,
                    "missing_after_normalization": missing_labels,
                }

        strict_job = next(
            item for item in strict["job_errors"] if item["job_id"] == job_id
        )
        analyses.append(
            {
                "job_id": job_id,
                "method": prediction_record.get("method"),
                "scalar_errors": strict_job["scalar_errors"],
                "list_analysis": field_analysis,
            }
        )

    list_metrics = {
        field: _prf(counts["tp"], counts["fp"], counts["fn"])
        for field, counts in list_counts.items()
    }
    total_tp = sum(item["tp"] for item in list_counts.values())
    total_fp = sum(item["fp"] for item in list_counts.values())
    total_fn = sum(item["fn"] for item in list_counts.values())

    metrics = {
        "evaluator_version": "2.0",
        "normalization_schema_version": normalizer.schema_version,
        "dataset_size": len(gold_records),
        "strict": {
            "scalar_macro_accuracy": strict["scalar_macro_accuracy"],
            "list_metrics": strict["list_metrics"],
            "list_micro": strict["list_micro"],
            "strict_extra_rate": strict["hallucination_rate"],
        },
        "normalized": {
            "list_metrics": list_metrics,
            "list_micro": _prf(total_tp, total_fp, total_fn),
        },
        "groundedness": {
            "predicted_item_count": total_predictions,
            "alias_resolved_item_count": alias_resolved,
            "source_supported_extra_count": supported_extras,
            "potential_unsupported_count": potential_unsupported,
            "potential_unsupported_rate": round(
                _safe_divide(potential_unsupported, total_predictions), 4
            ),
        },
    }
    return metrics, analyses
