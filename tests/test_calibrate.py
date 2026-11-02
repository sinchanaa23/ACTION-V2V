"""P2-04. Synthetic numbers exercise the maths only; no report text involved."""

import math
from pathlib import Path

import pytest

from v2v.nlp import calibrate


def test_softmax_sums_to_one_and_temperature_softens() -> None:
    sharp = calibrate.softmax([2.0, 0.0, -1.0])
    soft = calibrate.softmax([2.0, 0.0, -1.0], temperature=4.0)
    assert math.isclose(sum(sharp), 1.0) and math.isclose(sum(soft), 1.0)
    assert soft[0] < sharp[0]
    with pytest.raises(ValueError):
        calibrate.softmax([1.0], temperature=0.0)


def test_nll_checks_lengths() -> None:
    with pytest.raises(ValueError):
        calibrate.negative_log_likelihood([[1.0, 0.0]], [0, 1], 1.0)


def test_fit_temperature_cools_overconfident_scores() -> None:
    # Scores say "class 0" with huge margins but are right only 60% of the time.
    logits = [[2.0, 0.0]] * 10
    labels = [0] * 6 + [1] * 4
    t = calibrate.fit_temperature(logits, labels)
    assert t > 1.0
    p0 = calibrate.softmax(logits[0], t)[0]
    assert abs(p0 - 0.6) < 0.01
    assert calibrate.negative_log_likelihood(logits, labels, t) < calibrate.negative_log_likelihood(
        logits, labels, 1.0
    )


def test_ece_perfect_and_worst() -> None:
    assert calibrate.expected_calibration_error([1.0, 1.0], [True, True]) == 0.0
    assert calibrate.expected_calibration_error([1.0, 1.0], [False, False]) == 1.0
    with pytest.raises(ValueError):
        calibrate.expected_calibration_error([], [])


def test_ece_known_value() -> None:
    # Two bins: conf 0.9 with 50% accuracy (gap 0.4), conf 0.3 with 0% (gap 0.3).
    ece = calibrate.expected_calibration_error([0.9, 0.9, 0.3, 0.3], [True, False, False, False])
    assert math.isclose(ece, 0.35)


def test_reliability_bins_skip_empty_and_validate() -> None:
    bins = calibrate.reliability_bins([0.05, 0.95, 0.96], [False, True, True], n_bins=10)
    assert [b.count for b in bins] == [1, 2]
    assert bins[1].accuracy == 1.0
    with pytest.raises(ValueError):
        calibrate.reliability_bins([1.5], [True])
    with pytest.raises(ValueError):
        calibrate.reliability_bins([0.5], [])


def test_plot_reliability(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    out = tmp_path / "rel.png"
    calibrate.plot_reliability(calibrate.reliability_bins([0.2, 0.8], [False, True]), out, "t")
    assert out.stat().st_size > 0
