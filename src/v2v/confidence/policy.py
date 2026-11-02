"""Warning policy: corroborated / unverified / suppressed. Owner: P3 (P3-03).

Two thresholds: theta_e on event evidence and theta_i on interpretation
confidence. ``sweep`` scores one report under a whole grid of them, for the
false-alert / missed-hazard trade-off curves.
"""

from collections.abc import Sequence

from v2v.contracts import NON_HAZARD_TYPES, HazardInterpretation, Status


def is_actionable(interp: HazardInterpretation) -> bool:
    """A report can only ever produce a warning if it names a hazard and does
    not negate it."""
    return interp.hazard not in NON_HAZARD_TYPES and not interp.negated


def decide(
    interp: HazardInterpretation,
    interp_conf: float,
    event_evidence: float,
    theta_i: float,
    theta_e: float,
) -> Status:
    """Corroborated if the evidence clears ``theta_e``; unverified if only the
    interpretation clears ``theta_i``; otherwise suppressed."""
    if not is_actionable(interp):
        return "suppressed"
    if event_evidence >= theta_e:
        return "corroborated"
    if interp_conf >= theta_i:
        return "unverified"
    return "suppressed"


DEFAULT_THETA_I: tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
DEFAULT_THETA_E: tuple[float, ...] = (0.5, 0.6, 0.7, 0.75, 0.8, 0.9)


def threshold_grid(
    theta_i: Sequence[float] = DEFAULT_THETA_I, theta_e: Sequence[float] = DEFAULT_THETA_E
) -> list[tuple[float, float]]:
    """Every (theta_i, theta_e) pair for the trade-off sweep."""
    for t in (*theta_i, *theta_e):
        if not 0.0 <= t <= 1.0:
            raise ValueError(f"threshold {t} is outside [0, 1]")
    return [(i, e) for i in theta_i for e in theta_e]


def sweep(
    interp: HazardInterpretation,
    interp_conf: float,
    event_evidence: float,
    grid: Sequence[tuple[float, float]],
) -> dict[tuple[float, float], Status]:
    """The status a report gets under each threshold pair in ``grid``."""
    return {(ti, te): decide(interp, interp_conf, event_evidence, ti, te) for ti, te in grid}
