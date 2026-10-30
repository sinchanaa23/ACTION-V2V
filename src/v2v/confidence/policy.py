"""Warning policy: corroborated / unverified / suppressed. Owner: P3 (P3-03)."""

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
