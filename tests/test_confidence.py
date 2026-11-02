"""P3-01/02/03: clustering, evidence and the threshold sweep."""

import math

import pytest

from v2v.confidence import evidence, policy
from v2v.confidence.dedup import CLUSTER_RADIUS_M, CLUSTER_WINDOW_S, HazardTracker
from v2v.contracts import HazardInterpretation, HazardType, ResolvedLocation
from v2v.experiments import PROPOSED


def _interp(
    rid: str,
    hazard: HazardType = HazardType.ACCIDENT,
    negated: bool = False,
    p: float = 0.9,
) -> HazardInterpretation:
    return HazardInterpretation(report_id=rid, hazard=hazard, p_class=p, negated=negated)


def _loc(x: float, approx: bool = False) -> ResolvedLocation:
    return ResolvedLocation(edge_id="e", pos_m=max(0.0, x), x=x, y=0.0, approx=approx)


# --- evidence ------------------------------------------------------------------


def test_evidence_documented_values() -> None:
    assert evidence.event_evidence(1, 0, 0.0) == pytest.approx(0.35)
    assert evidence.event_evidence(3, 0, 0.0) == pytest.approx(0.725375)
    assert evidence.event_evidence(4, 0, 0.0) > 0.75
    assert evidence.event_evidence(3, 1, 0.0) > 0.75
    assert evidence.event_evidence(2, 2, 0.0) > 0.75
    assert evidence.event_evidence(0, 0, 0.0) == 0.0


def test_evidence_decays_and_contradictions_lower_it() -> None:
    fresh = evidence.event_evidence(3, 1, 0.0)
    assert evidence.event_evidence(3, 1, evidence.FRESHNESS_S) == pytest.approx(fresh / math.e)
    assert evidence.event_evidence(3, 1, 0.0, n_contradictions=1) == pytest.approx(fresh / 2)


def test_evidence_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        evidence.event_evidence(-1, 0, 0.0)
    with pytest.raises(ValueError):
        evidence.event_evidence(1, 0, -5.0)


# --- clustering ------------------------------------------------------------------


def test_duplicates_from_different_reporters_corroborate() -> None:
    tracker = HazardTracker(PROPOSED)
    first = tracker.ingest(_interp("r1"), _loc(0), 0, reporter_id="v1", t_s=0)
    assert first.status == "unverified" and first.n_reports == 1
    for i, x in enumerate((50, 120, 200), start=2):
        last = tracker.ingest(_interp(f"r{i}"), _loc(x), 0, reporter_id=f"v{i}", t_s=10 * i)
    assert tracker.clusters == 1
    assert last.cluster_id == first.cluster_id
    assert last.n_reports == 4
    assert last.status == "corroborated"


def test_same_reporter_counts_once() -> None:
    tracker = HazardTracker(PROPOSED)
    for t in (0, 10, 20, 30):
        out = tracker.ingest(_interp(f"r{t}"), _loc(0), 0, reporter_id="v1", t_s=t)
    assert out.n_reports == 1 and out.status == "unverified"


@pytest.mark.parametrize(
    ("x", "t", "hazard"),
    [
        (CLUSTER_RADIUS_M + 50, 10, HazardType.ACCIDENT),  # too far
        (10, CLUSTER_WINDOW_S + 60, HazardType.ACCIDENT),  # too late
        (10, 10, HazardType.SLIPPERY),  # different hazard
    ],
)
def test_new_cluster_when_far_late_or_different(x: float, t: float, hazard: HazardType) -> None:
    tracker = HazardTracker(PROPOSED)
    tracker.ingest(_interp("r1"), _loc(0), 0, reporter_id="v1", t_s=0)
    tracker.ingest(_interp("r2", hazard), _loc(x), 0, reporter_id="v2", t_s=t)
    assert tracker.clusters == 2


def test_contradiction_lowers_evidence_and_is_suppressed() -> None:
    tracker = HazardTracker(PROPOSED)
    for i in range(4):
        tracker.ingest(_interp(f"r{i}"), _loc(i * 10), 0, reporter_id=f"v{i}", t_s=i)
    before = tracker.observe(_interp("r9"), _loc(0), 0, reporter_id="v0", t_s=5)
    out = tracker.ingest(_interp("rc", negated=True), _loc(20), 0, reporter_id="vc", t_s=6)
    assert out.status == "suppressed"
    assert out.cluster_id == before.cluster_id
    after = tracker.observe(_interp("r10"), _loc(0), 0, reporter_id="v1", t_s=7)
    assert after.n_contradictions == 1
    assert after.event_evidence < before.event_evidence


def test_orphan_negation_creates_suppressed_cluster() -> None:
    tracker = HazardTracker(PROPOSED)
    out = tracker.ingest(_interp("r1", negated=True), _loc(0), 0, reporter_id="v1", t_s=0)
    assert out.status == "suppressed"


def test_stated_location_replaces_fallback() -> None:
    tracker = HazardTracker(PROPOSED)
    tracker.ingest(_interp("r1"), _loc(0, approx=True), 0, reporter_id="v1", t_s=0)
    out = tracker.ingest(_interp("r2"), _loc(40), 0, reporter_id="v2", t_s=5)
    assert out.location is not None and not out.location.approx


def test_interp_conf_override_and_max() -> None:
    tracker = HazardTracker(PROPOSED)
    low = tracker.ingest(_interp("r1"), _loc(0), 0, reporter_id="v1", interp_conf=0.4)
    assert low.interp_conf == 0.4 and low.status == "suppressed"
    high = tracker.ingest(_interp("r2"), _loc(0), 0, reporter_id="v2", interp_conf=0.8)
    assert high.interp_conf == 0.8 and high.status == "unverified"


def test_age_since_newest_report() -> None:
    tracker = HazardTracker(PROPOSED)
    tracker.ingest(_interp("r1"), _loc(0), 0, reporter_id="v1", t_s=0)
    view = tracker.observe(_interp("r2"), _loc(0), 0, reporter_id="v1", t_s=100)
    assert view.age_s == 0.0  # the new report refreshes the cluster


# --- policy sweep -----------------------------------------------------------------


def test_threshold_grid_and_sweep() -> None:
    grid = policy.threshold_grid((0.5, 0.9), (0.6, 0.8))
    assert grid == [(0.5, 0.6), (0.5, 0.8), (0.9, 0.6), (0.9, 0.8)]
    statuses = policy.sweep(_interp("r"), 0.7, 0.7, grid)
    assert statuses[(0.5, 0.6)] == "corroborated"
    assert statuses[(0.5, 0.8)] == "unverified"
    assert statuses[(0.9, 0.8)] == "suppressed"
    with pytest.raises(ValueError):
        policy.threshold_grid((1.5,), (0.5,))
