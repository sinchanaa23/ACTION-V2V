"""Bootstrap confidence intervals, paired tests, threshold curves. Owner: P3 (P3-07).

Pure Python, seeded, so every reported interval can be regenerated exactly.
Comparisons between configs are paired by (scenario, seed): every config sees
the same scenarios, so differences are taken pair by pair before resampling.
"""

import itertools
import random
import statistics
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

EXACT_PERMUTATION_MAX_N = 16
"""Up to this many pairs the sign-flip test enumerates every flip (2^n)."""


def bootstrap_ci(
    values: Sequence[float], n_resamples: int = 10_000, seed: int = 0, level: float = 0.95
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval of the mean."""
    if not values:
        raise ValueError("bootstrap of an empty sample")
    if not 0.0 < level < 1.0:
        raise ValueError("level must be in (0, 1)")
    rng = random.Random(seed)
    n = len(values)
    means = sorted(statistics.fmean(rng.choices(values, k=n)) for _ in range(n_resamples))
    tail = (1.0 - level) / 2.0
    lo = means[int(tail * (n_resamples - 1))]
    hi = means[int(round((1.0 - tail) * (n_resamples - 1)))]
    return lo, hi


def mean_with_ci(
    values: Sequence[float], n_resamples: int = 10_000, seed: int = 0
) -> dict[str, float]:
    lo, hi = bootstrap_ci(values, n_resamples, seed)
    return {"mean": statistics.fmean(values), "ci_low": lo, "ci_high": hi, "n": float(len(values))}


def paired_permutation_test(
    a: Sequence[float], b: Sequence[float], n_resamples: int = 10_000, seed: int = 0
) -> float:
    """Two-sided p-value for "mean(a - b) = 0" by flipping the sign of each
    paired difference. Exact for up to EXACT_PERMUTATION_MAX_N pairs."""
    if len(a) != len(b) or not a:
        raise ValueError("need two equal-length, non-empty paired samples")
    diffs = [x - y for x, y in zip(a, b, strict=True)]
    observed = abs(statistics.fmean(diffs))
    eps = 1e-12
    if len(diffs) <= EXACT_PERMUTATION_MAX_N:
        flips = list(itertools.product((1, -1), repeat=len(diffs)))
        hits = sum(
            abs(statistics.fmean(s * d for s, d in zip(f, diffs, strict=True))) >= observed - eps
            for f in flips
        )
        return hits / len(flips)
    rng = random.Random(seed)
    hits = sum(
        abs(statistics.fmean(d if rng.random() < 0.5 else -d for d in diffs)) >= observed - eps
        for _ in range(n_resamples)
    )
    return (hits + 1) / (n_resamples + 1)


def paired_comparison(
    a: Sequence[float], b: Sequence[float], n_resamples: int = 10_000, seed: int = 0
) -> dict[str, float]:
    """Mean difference a - b with its bootstrap CI and permutation p-value."""
    diffs = [x - y for x, y in zip(a, b, strict=True)]
    lo, hi = bootstrap_ci(diffs, n_resamples, seed)
    return {
        "mean_diff": statistics.fmean(diffs),
        "ci_low": lo,
        "ci_high": hi,
        "p_value": paired_permutation_test(a, b, n_resamples, seed),
        "n_pairs": float(len(diffs)),
    }


def pair_rows(
    rows: Iterable[Mapping[str, Any]],
    metric: str,
    config_a: str,
    config_b: str,
    pair_keys: Sequence[str] = ("scenario_id", "seed"),
) -> tuple[list[float], list[float]]:
    """Line up one metric for two configs on matching ``pair_keys``. Pairs where
    either value is missing (None) are dropped."""
    by_config: dict[str, dict[tuple[Any, ...], float]] = {config_a: {}, config_b: {}}
    for row in rows:
        if row["config"] in by_config and row.get(metric) is not None:
            by_config[row["config"]][tuple(row[k] for k in pair_keys)] = float(row[metric])
    common = sorted(set(by_config[config_a]) & set(by_config[config_b]), key=str)
    return [by_config[config_a][k] for k in common], [by_config[config_b][k] for k in common]


def tradeoff_curve(
    rows: Iterable[Mapping[str, Any]],
    x: str = "false_alert",
    y: str = "missed_hazard",
    by: Sequence[str] = ("theta_i", "theta_e"),
) -> list[dict[str, Any]]:
    """Mean (x, y) per threshold setting, sorted by x, with Pareto-optimal
    points (no other setting is at least as good on both and better on one)
    flagged."""
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in by), []).append(row)
    points = []
    for key, members in groups.items():
        xs = [float(r[x]) for r in members if r.get(x) is not None]
        ys = [float(r[y]) for r in members if r.get(y) is not None]
        if xs and ys:
            point: dict[str, Any] = dict(zip(by, key, strict=True))
            point.update({x: statistics.fmean(xs), y: statistics.fmean(ys), "n": len(members)})
            points.append(point)
    for p in points:
        p["pareto"] = not any(
            q[x] <= p[x] and q[y] <= p[y] and (q[x] < p[x] or q[y] < p[y]) for q in points
        )
    return sorted(points, key=lambda p: (p[x], p[y]))


def plot_tradeoff(
    curves: Mapping[str, Sequence[Mapping[str, Any]]],
    out_path: Path,
    x: str = "false_alert",
    y: str = "missed_hazard",
) -> None:
    """One line per config through its Pareto points. Needs matplotlib."""
    import matplotlib  # type: ignore[import-not-found, unused-ignore]

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore[import-not-found, unused-ignore]

    fig, ax = plt.subplots(figsize=(5, 4))
    for name, points in curves.items():
        front = [p for p in points if p.get("pareto")]
        ax.plot([p[x] for p in front], [p[y] for p in front], marker="o", label=name)
    ax.set(xlabel=x.replace("_", " "), ylabel=y.replace("_", " "), xlim=(0, 1), ylim=(0, 1))
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
