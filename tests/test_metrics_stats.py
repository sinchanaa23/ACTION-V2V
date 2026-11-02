"""P3-04 metrics and P3-07 statistics. Synthetic numbers only."""

from pathlib import Path

import pytest

from v2v.contracts import HazardType
from v2v.eval import metrics, stats


def _m(**kw: object) -> dict[str, float | None]:
    base: dict[str, object] = {
        "warned": True,
        "event_active": True,
        "needs_warning": True,
        "recipients": ["a", "b", "c"],
        "delivered": ["a", "b"],
        "relevant": ["a", "d"],
        "latencies_ms": [10.0, 30.0],
        "transmissions": 4,
        "location_approx": False,
    }
    base.update(kw)
    return metrics.report_metrics(**base)  # type: ignore[arg-type]


def test_report_metrics_correct_warning() -> None:
    m = _m()
    assert m["false_alert"] == 0.0 and m["missed_hazard"] == 0.0
    assert m["relevant_delivery_rate"] == 0.5  # a delivered, d never addressed
    assert m["irrelevant_recipient_rate"] == pytest.approx(2 / 3)
    assert m["messages_sent"] == 3.0 and m["transmissions"] == 4.0
    assert m["mean_latency_ms"] == 20.0 and m["location_approx"] == 0.0


def test_false_alert_and_miss_are_separate() -> None:
    assert _m(event_active=False)["false_alert"] == 1.0
    silent = _m(warned=False)
    assert silent["missed_hazard"] == 1.0 and silent["false_alert"] == 0.0
    assert silent["messages_sent"] == 0.0
    assert silent["relevant_delivery_rate"] is None  # undefined, not zero
    # A "road clear now" report needs no warning: silence is not a miss.
    assert _m(warned=False, needs_warning=False)["missed_hazard"] == 0.0


def test_classification_reuses_p1_scoring() -> None:
    a, s = HazardType.ACCIDENT, HazardType.SLIPPERY
    scores = metrics.classification([a, a, s], [a, s, s])
    assert scores["accuracy"] == pytest.approx(2 / 3)
    assert set(scores["per_class"]) == {"accident", "slippery_road"}


def test_aggregate_skips_none() -> None:
    rows = [
        {"config": "p", "false_alert": 1.0, "relevant_delivery_rate": None},
        {"config": "p", "false_alert": 0.0, "relevant_delivery_rate": 0.5},
        {"config": "b", "false_alert": 0.0, "relevant_delivery_rate": 1.0},
    ]
    out = {r["config"]: r for r in metrics.aggregate(rows, ["config"])}
    assert out["p"]["false_alert"] == 0.5 and out["p"]["n"] == 2
    assert out["p"]["relevant_delivery_rate"] == 0.5 and out["p"]["n_relevant_delivery_rate"] == 1
    assert out["b"]["missed_hazard"] is None


def test_bootstrap_ci_contains_mean_and_is_seeded() -> None:
    values = [0.0, 1.0] * 20
    lo, hi = stats.bootstrap_ci(values, n_resamples=2000, seed=1)
    assert lo < 0.5 < hi
    assert stats.bootstrap_ci(values, 2000, 1) == (lo, hi)
    assert stats.bootstrap_ci([3.0, 3.0], 100) == (3.0, 3.0)
    with pytest.raises(ValueError):
        stats.bootstrap_ci([])
    summary = stats.mean_with_ci(values, 500)
    assert summary["mean"] == 0.5 and summary["n"] == 40.0


def test_permutation_test_exact_and_sampled() -> None:
    a = [1.0] * 6
    b = [0.0] * 6
    assert stats.paired_permutation_test(a, b) == pytest.approx(2 / 64)  # exact
    assert stats.paired_permutation_test(a, a) == 1.0
    big_a, big_b = [1.0] * 30, [0.0] * 30
    assert stats.paired_permutation_test(big_a, big_b, n_resamples=500) < 0.01
    with pytest.raises(ValueError):
        stats.paired_permutation_test([1.0], [])


def test_paired_comparison_and_pair_rows() -> None:
    rows = [
        {"config": "p", "scenario_id": "s1", "seed": 0, "false_alert": 0.0},
        {"config": "b", "scenario_id": "s1", "seed": 0, "false_alert": 1.0},
        {"config": "p", "scenario_id": "s2", "seed": 0, "false_alert": 0.0},
        {"config": "b", "scenario_id": "s2", "seed": 0, "false_alert": 1.0},
        {"config": "p", "scenario_id": "s3", "seed": 0, "false_alert": None},
        {"config": "b", "scenario_id": "s3", "seed": 0, "false_alert": 1.0},
    ]
    a, b = stats.pair_rows(rows, "false_alert", "p", "b")
    assert a == [0.0, 0.0] and b == [1.0, 1.0]
    result = stats.paired_comparison(a, b, n_resamples=200)
    assert result["mean_diff"] == -1.0 and result["n_pairs"] == 2.0


def test_tradeoff_curve_marks_pareto_points() -> None:
    rows = [
        {"theta_i": 0.5, "theta_e": 0.5, "false_alert": 0.4, "missed_hazard": 0.1},
        {"theta_i": 0.7, "theta_e": 0.7, "false_alert": 0.2, "missed_hazard": 0.2},
        {"theta_i": 0.9, "theta_e": 0.9, "false_alert": 0.3, "missed_hazard": 0.3},  # dominated
    ]
    curve = stats.tradeoff_curve(rows)
    assert [p["false_alert"] for p in curve] == [0.2, 0.3, 0.4]
    assert [p["pareto"] for p in curve] == [True, False, True]


def test_plot_tradeoff(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    curve = stats.tradeoff_curve(
        [{"theta_i": 0.5, "theta_e": 0.5, "false_alert": 0.1, "missed_hazard": 0.2}]
    )
    out = tmp_path / "t.png"
    stats.plot_tradeoff({"proposed": curve}, out)
    assert out.stat().st_size > 0
