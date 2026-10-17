"""Compare saved rule-baseline and LLM evaluation metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.data_loader import PROJECT_ROOT


RESULTS_DIR = PROJECT_ROOT / "results"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    baseline = _load(RESULTS_DIR / "baseline_metrics.json")
    llm = _load(RESULTS_DIR / "llm_metrics.json")
    rows = [
        ("Scalar macro accuracy", baseline["scalar_macro_accuracy"], llm["scalar_macro_accuracy"]),
        ("List micro precision", baseline["list_micro"]["precision"], llm["list_micro"]["precision"]),
        ("List micro recall", baseline["list_micro"]["recall"], llm["list_micro"]["recall"]),
        ("List micro F1", baseline["list_micro"]["f1"], llm["list_micro"]["f1"]),
        ("Hallucination rate", baseline["hallucination_rate"], llm["hallucination_rate"]),
    ]
    print(f"{'Metric':<28} {'Rules':>10} {'LLM':>10} {'Delta':>10}")
    print("-" * 62)
    for name, rule_value, llm_value in rows:
        print(f"{name:<28} {rule_value:>9.2%} {llm_value:>9.2%} {llm_value-rule_value:>+9.2%}")


if __name__ == "__main__":
    main()
