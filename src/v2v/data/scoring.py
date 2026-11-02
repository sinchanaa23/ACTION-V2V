"""Classification scores shared by Baseline A and the RQ1 error analysis."""

from collections.abc import Sequence
from math import sqrt
from typing import Any

from v2v.contracts import HazardType


def classification_scores(
    gold: Sequence[HazardType], predicted: Sequence[HazardType]
) -> dict[str, Any]:
    """Per-class precision, recall and F1, with macro-F1 over the gold classes."""
    if len(gold) != len(predicted) or not gold:
        raise ValueError("need two equal-length, non-empty label lists")
    pairs = list(zip(gold, predicted, strict=True))
    per_class: dict[str, dict[str, float | int]] = {}
    for hazard in sorted(set(gold)):
        true_pos = sum(g == hazard and p == hazard for g, p in pairs)
        support = sum(g == hazard for g, _ in pairs)
        predicted_n = sum(p == hazard for _, p in pairs)
        precision = true_pos / predicted_n if predicted_n else 0.0
        recall = true_pos / support
        total = precision + recall
        per_class[hazard.value] = {
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / total if total else 0.0,
            "support": support,
        }
    return {
        "n": len(pairs),
        "accuracy": sum(g == p for g, p in pairs) / len(pairs),
        "macro_f1": sum(c["f1"] for c in per_class.values()) / len(per_class),
        "per_class": per_class,
    }


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval, which stays sensible on small subsets."""
    if total <= 0:
        raise ValueError("total must be positive")
    share = successes / total
    denominator = 1 + z**2 / total
    centre = (share + z**2 / (2 * total)) / denominator
    margin = z * sqrt(share * (1 - share) / total + z**2 / (4 * total**2)) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)
