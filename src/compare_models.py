"""Compare saved rule-baseline and LLM evaluation metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.data_loader import PROJECT_ROOT


RESULTS_DIR = PROJECT_ROOT / "results"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    baseline = _load(args.results_dir / "baseline_metrics.json")
    llm = _load(args.results_dir / "llm_metrics.json")
    rows = [
        ("Scalar macro accuracy", baseline["scalar_macro_accuracy"], llm["scalar_macro_accuracy"]),
        ("List micro precision", baseline["list_micro"]["precision"], llm["list_micro"]["precision"]),
        ("List micro recall", baseline["list_micro"]["recall"], llm["list_micro"]["recall"]),
        ("List micro F1", baseline["list_micro"]["f1"], llm["list_micro"]["f1"]),
        ("Strict extra rate", baseline["hallucination_rate"], llm["hallucination_rate"]),
    ]
    print(f"{'Metric':<28} {'Rules':>10} {'LLM':>10} {'Delta':>10}")
    print("-" * 62)
    for name, rule_value, llm_value in rows:
        print(f"{name:<28} {rule_value:>9.2%} {llm_value:>9.2%} {llm_value-rule_value:>+9.2%}")

    v2_path = args.results_dir / "evaluation_v2_metrics.json"
    if v2_path.exists():
        v2 = _load(v2_path)
        print("\nEvaluator v2")
        method_width = max(22, *(len(method) for method in v2))
        print(
            f"{'Method':<{method_width}} {'Strict F1':>11} "
            f"{'Normalized F1':>15} {'Unsupported':>13}"
        )
        print("-" * (method_width + 43))
        for method, metrics in v2.items():
            print(
                f"{method:<{method_width}} "
                f"{metrics['strict']['list_micro']['f1']:>10.2%} "
                f"{metrics['normalized']['list_micro']['f1']:>14.2%} "
                f"{metrics['groundedness']['potential_unsupported_rate']:>12.2%}"
            )


if __name__ == "__main__":
    main()
