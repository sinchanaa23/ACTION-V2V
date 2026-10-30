"""Duplicate clustering and the hazard tracker. Owner: P3 (P3-01)."""

from v2v.confidence import evidence, policy
from v2v.contracts import AssessedHazard, ExperimentConfig, HazardInterpretation, ResolvedLocation


class HazardTracker:
    """Groups reports of the same event and keeps each cluster's assessment.

    STUB(P3-01): every report becomes its own cluster.
    """

    def __init__(self, config: ExperimentConfig) -> None:
        self._config = config
        self._n_clusters = 0

    def ingest(
        self,
        interp: HazardInterpretation,
        location: ResolvedLocation | None,
        sensor_flags: int,
    ) -> AssessedHazard:
        cluster_id = f"c{self._n_clusters}"
        self._n_clusters += 1
        event = evidence.event_evidence(
            n_independent_reports=1, sensor_flags=sensor_flags, age_s=0.0
        )
        status = policy.decide(
            interp, interp.p_class, event, self._config.theta_i, self._config.theta_e
        )
        return AssessedHazard(
            cluster_id=cluster_id,
            hazard=interp.hazard,
            location=location,
            interp_conf=interp.p_class,
            event_evidence=event,
            n_reports=1,
            age_s=0.0,
            status=status,
        )
