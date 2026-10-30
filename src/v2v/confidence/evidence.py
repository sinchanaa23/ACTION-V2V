"""Event evidence: corroboration, sensors, freshness, contradictions. Owner: P3 (P3-02)."""


def event_evidence(n_independent_reports: int, sensor_flags: int, age_s: float) -> float:
    """Score in [0, 1] for "the reported event really happened".

    Kept separate from interpretation confidence on purpose.

    STUB(P3-02): always 0.0, so every actionable report is at most unverified.
    """
    return 0.0
