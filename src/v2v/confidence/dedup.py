"""Duplicate clustering and the hazard tracker. Owner: P3 (P3-01).

Reports of the same event are grouped into one cluster when they name the same
kind of hazard, close together, within a time window. Evidence counts one
report per reporter, so a driver repeating themselves cannot corroborate their
own report. A negated report ("accident cleared", "no accident") never starts
a cluster: it counts as a contradiction against every live cluster near it.

The tracker separates two things the experiments need apart:

- ``observe`` updates the clusters and returns a ``ClusterView`` (interpretation
  confidence, evidence, counts) that does not depend on the warning thresholds;
- ``assess`` turns a view into an ``AssessedHazard`` for given thresholds.

So one simulation run can be scored for every (theta_i, theta_e) pair.
"""

import math
from dataclasses import dataclass, field

from v2v.confidence import evidence, policy
from v2v.contracts import (
    AssessedHazard,
    ExperimentConfig,
    HazardInterpretation,
    HazardType,
    ResolvedLocation,
)

CLUSTER_RADIUS_M = 300.0
"""Reports this close (straight-line) can describe the same event."""
CLUSTER_WINDOW_S = 900.0
"""A cluster stops accepting reports this long after its newest one."""


@dataclass(frozen=True)
class ClusterView:
    """A cluster's state after one report, before any threshold is applied."""

    cluster_id: str
    hazard: HazardType
    location: ResolvedLocation | None
    interp_conf: float
    """Highest interpretation confidence among the cluster's reports."""
    event_evidence: float
    n_reports: int
    """Independent reporters supporting the event."""
    n_contradictions: int
    age_s: float
    """Seconds since the newest supporting report."""
    is_contradiction: bool = False
    """True when this report was a negation counted against the cluster."""


@dataclass
class _Cluster:
    cluster_id: str
    hazard: HazardType
    location: ResolvedLocation | None
    interp_conf: float
    first_t: float
    last_t: float
    reporters: set[str] = field(default_factory=set)
    contradictors: set[str] = field(default_factory=set)


def _distance(a: ResolvedLocation | None, b: ResolvedLocation | None) -> float:
    if a is None or b is None:
        return math.inf
    return math.dist((a.x, a.y), (b.x, b.y))


class HazardTracker:
    """Groups reports of the same event and keeps each cluster's assessment."""

    def __init__(self, config: ExperimentConfig) -> None:
        self._config = config
        self._clusters: list[_Cluster] = []

    @property
    def clusters(self) -> int:
        return len(self._clusters)

    def _live_near(self, location: ResolvedLocation | None, t_s: float) -> list[_Cluster]:
        return [
            c
            for c in self._clusters
            if t_s - c.last_t <= CLUSTER_WINDOW_S
            and _distance(c.location, location) <= CLUSTER_RADIUS_M
        ]

    def _view(
        self, c: _Cluster, sensor_flags: int, t_s: float, is_contradiction: bool = False
    ) -> ClusterView:
        age = max(0.0, t_s - c.last_t)
        return ClusterView(
            cluster_id=c.cluster_id,
            hazard=c.hazard,
            location=c.location,
            interp_conf=c.interp_conf,
            event_evidence=evidence.event_evidence(
                len(c.reporters), sensor_flags, age, len(c.contradictors)
            ),
            n_reports=max(1, len(c.reporters)),
            n_contradictions=len(c.contradictors),
            age_s=age,
            is_contradiction=is_contradiction,
        )

    def observe(
        self,
        interp: HazardInterpretation,
        location: ResolvedLocation | None,
        sensor_flags: int,
        *,
        reporter_id: str | None = None,
        t_s: float = 0.0,
        interp_conf: float | None = None,
    ) -> ClusterView:
        """Add one report and return the state of the cluster it landed in."""
        reporter = reporter_id or f"report:{interp.report_id}"
        conf = interp.p_class if interp_conf is None else interp_conf

        if interp.negated:
            near = self._live_near(location, t_s)
            for c in near:
                c.contradictors.add(reporter)
            if near:
                newest = max(near, key=lambda c: c.last_t)
                return self._view(newest, sensor_flags, t_s, is_contradiction=True)
            # Nothing to contradict: a negation on its own is its own, empty cluster.
            orphan = _Cluster(f"c{len(self._clusters)}", interp.hazard, location, conf, t_s, t_s)
            self._clusters.append(orphan)
            return self._view(orphan, sensor_flags, t_s, is_contradiction=True)

        if policy.is_actionable(interp):
            same = [c for c in self._live_near(location, t_s) if c.hazard == interp.hazard]
            if same:
                cluster = min(same, key=lambda c: _distance(c.location, location))
                cluster.reporters.add(reporter)
                cluster.last_t = max(cluster.last_t, t_s)
                cluster.interp_conf = max(cluster.interp_conf, conf)
                if cluster.location is None or (
                    cluster.location.approx and location is not None and not location.approx
                ):
                    cluster.location = location  # prefer a stated location over the fallback
                return self._view(cluster, sensor_flags, t_s)

        cluster = _Cluster(
            f"c{len(self._clusters)}", interp.hazard, location, conf, t_s, t_s, {reporter}
        )
        self._clusters.append(cluster)
        return self._view(cluster, sensor_flags, t_s)

    @staticmethod
    def assess(
        view: ClusterView, interp: HazardInterpretation, theta_i: float, theta_e: float
    ) -> AssessedHazard:
        status = (
            "suppressed"
            if view.is_contradiction
            else policy.decide(interp, view.interp_conf, view.event_evidence, theta_i, theta_e)
        )
        return AssessedHazard(
            cluster_id=view.cluster_id,
            hazard=view.hazard,
            location=view.location,
            interp_conf=min(1.0, max(0.0, view.interp_conf)),
            event_evidence=view.event_evidence,
            n_reports=view.n_reports,
            age_s=view.age_s,
            status=status,
        )

    def ingest(
        self,
        interp: HazardInterpretation,
        location: ResolvedLocation | None,
        sensor_flags: int,
        *,
        reporter_id: str | None = None,
        t_s: float = 0.0,
        interp_conf: float | None = None,
    ) -> AssessedHazard:
        """Add one report and assess its cluster with the config's thresholds."""
        view = self.observe(
            interp,
            location,
            sensor_flags,
            reporter_id=reporter_id,
            t_s=t_s,
            interp_conf=interp_conf,
        )
        return self.assess(view, interp, self._config.theta_i, self._config.theta_e)
