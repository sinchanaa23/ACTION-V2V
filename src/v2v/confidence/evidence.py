"""Event evidence: corroboration, sensors, freshness, contradictions. Owner: P3 (P3-02).

How strongly the world supports "the reported event really happened", kept
separate from how sure we are of what the report *says* (interpretation
confidence, P2). Each independent report and each simulated sensor flag is
treated as an independent, imperfect witness and combined with a noisy-OR:

    support = 1 - (1 - q_report)^n_reports * (1 - q_sensor)^sensor_flags

The support then decays with the age of the newest report, exp(-age / F), and
every contradicting report ("road clear now") removes a fixed share of it.

The weights below are starting values, not fitted ones. At the default
theta_e = 0.75 and age 0: one report alone scores 0.35 and three score 0.73,
both unverified; four reports (0.82), three reports plus one sensor flag
(0.79) or two reports plus two flags (0.76) are corroborated. The threshold
sweep in eval.stats shows how results move with theta_e.
"""

import math

Q_REPORT = 0.35
"""Chance that one independent report alone is right, as a noisy-OR witness."""
Q_SENSOR = 0.25
"""Weight of one simulated sensor flag (stopped or crawling vehicle nearby).
Queues at signals also raise flags, so a flag is weaker than a report."""
FRESHNESS_S = 600.0
"""Support falls to 1/e this long after the newest report."""
CONTRADICTION_FACTOR = 0.5
"""Each contradicting report keeps this share of the support."""


def event_evidence(
    n_independent_reports: int,
    sensor_flags: int,
    age_s: float,
    n_contradictions: int = 0,
) -> float:
    """Score in [0, 1] for "the reported event really happened".

    Kept separate from interpretation confidence on purpose.
    """
    if n_independent_reports < 0 or sensor_flags < 0 or n_contradictions < 0:
        raise ValueError("counts must be non-negative")
    if age_s < 0:
        raise ValueError("age must be non-negative")
    doubt = (1 - Q_REPORT) ** n_independent_reports * (1 - Q_SENSOR) ** sensor_flags
    support = 1.0 - doubt
    support *= math.exp(-age_s / FRESHNESS_S)
    support *= CONTRADICTION_FACTOR**n_contradictions
    return min(1.0, max(0.0, support))
