"""Probability calibration: temperature scaling, ECE, reliability diagram. Owner: P2 (P2-04).

Pure Python so it runs (and is tested) without the NLP extras. The temperature
is fitted on the validation split only, never on test.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

DEFAULT_BINS = 15
_MIN_LOG_T = math.log(0.05)
_MAX_LOG_T = math.log(20.0)


def softmax(logits: Sequence[float], temperature: float = 1.0) -> list[float]:
    """Class probabilities from raw scores, softened (T > 1) or sharpened (T < 1)."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    scaled = [z / temperature for z in logits]
    top = max(scaled)
    exps = [math.exp(z - top) for z in scaled]
    total = sum(exps)
    return [e / total for e in exps]


def negative_log_likelihood(
    logits: Sequence[Sequence[float]], labels: Sequence[int], temperature: float
) -> float:
    """Mean NLL of the true labels at ``temperature``."""
    if len(logits) != len(labels) or not labels:
        raise ValueError("need one label per row of logits, and at least one row")
    total = 0.0
    for row, label in zip(logits, labels, strict=True):
        total -= math.log(max(softmax(row, temperature)[label], 1e-12))
    return total / len(labels)


def fit_temperature(
    logits: Sequence[Sequence[float]], labels: Sequence[int], iterations: int = 80
) -> float:
    """Temperature that minimises validation NLL (golden-section search on log T).

    Temperature scaling never changes which class is predicted, only how
    confident the prediction is.
    """
    ratio = (math.sqrt(5) - 1) / 2
    lo, hi = _MIN_LOG_T, _MAX_LOG_T

    def loss(log_t: float) -> float:
        return negative_log_likelihood(logits, labels, math.exp(log_t))

    a, b = hi - ratio * (hi - lo), lo + ratio * (hi - lo)
    fa, fb = loss(a), loss(b)
    for _ in range(iterations):
        if fa < fb:
            hi, b, fb = b, a, fa
            a = hi - ratio * (hi - lo)
            fa = loss(a)
        else:
            lo, a, fa = a, b, fb
            b = lo + ratio * (hi - lo)
            fb = loss(b)
    return math.exp((lo + hi) / 2)


@dataclass(frozen=True)
class ReliabilityBin:
    lower: float
    upper: float
    count: int
    mean_confidence: float
    accuracy: float


def reliability_bins(
    confidences: Sequence[float], correct: Sequence[bool], n_bins: int = DEFAULT_BINS
) -> list[ReliabilityBin]:
    """Equal-width confidence bins; empty bins are left out."""
    if len(confidences) != len(correct):
        raise ValueError("need one correctness flag per confidence")
    grouped: list[list[tuple[float, bool]]] = [[] for _ in range(n_bins)]
    for conf, ok in zip(confidences, correct, strict=True):
        if not 0.0 <= conf <= 1.0:
            raise ValueError(f"confidence {conf} is outside [0, 1]")
        grouped[min(int(conf * n_bins), n_bins - 1)].append((conf, ok))
    return [
        ReliabilityBin(
            lower=i / n_bins,
            upper=(i + 1) / n_bins,
            count=len(items),
            mean_confidence=sum(c for c, _ in items) / len(items),
            accuracy=sum(ok for _, ok in items) / len(items),
        )
        for i, items in enumerate(grouped)
        if items
    ]


def expected_calibration_error(
    confidences: Sequence[float], correct: Sequence[bool], n_bins: int = DEFAULT_BINS
) -> float:
    """ECE: bin-size-weighted mean |accuracy - confidence| over equal-width bins."""
    if not confidences:
        raise ValueError("ECE of an empty set is undefined")
    bins = reliability_bins(confidences, correct, n_bins)
    return sum(b.count * abs(b.accuracy - b.mean_confidence) for b in bins) / len(confidences)


def plot_reliability(bins: Sequence[ReliabilityBin], out_path: Path, title: str) -> None:
    """Save a reliability diagram. Needs matplotlib (not a core dependency)."""
    import matplotlib  # type: ignore[import-not-found, unused-ignore]

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore[import-not-found, unused-ignore]

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="perfect calibration")
    ax.bar(
        [b.lower for b in bins],
        [b.accuracy for b in bins],
        width=[b.upper - b.lower for b in bins],
        align="edge",
        edgecolor="black",
        alpha=0.7,
        label="accuracy",
    )
    ax.plot([b.mean_confidence for b in bins], [b.accuracy for b in bins], "o", color="black")
    ax.set(xlabel="confidence", ylabel="accuracy", title=title, xlim=(0, 1), ylim=(0, 1))
    ax.legend(loc="upper left")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
