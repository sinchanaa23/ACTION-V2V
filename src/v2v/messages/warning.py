"""Stage 5: warning schema and uncertainty-aware text. Owner: P2 (P2-07)."""

from collections.abc import Sequence

from v2v.contracts import AssessedHazard, Distribution, HazardType, WarningMessage

# STUB(P2-07): placeholder wording. The real generator is reviewed against the
# proposal's example output.
_PHRASES: dict[HazardType, str] = {
    HazardType.ACCIDENT: "Reported accident ahead",
    HazardType.OBSTRUCTION: "Road obstruction ahead",
    HazardType.SLIPPERY: "Slippery road ahead",
    HazardType.STOPPED: "Stopped vehicle ahead",
    HazardType.SPEEDING: "Speeding vehicle reported ahead",
}


def render_text(hazard: AssessedHazard) -> str:
    """Word a warning so the driver can see how certain it is."""
    parts = [f"{_PHRASES.get(hazard.hazard, 'Hazard reported ahead')}. Reduce speed."]
    if hazard.status == "unverified":
        parts.append("Status: unverified report.")
    if hazard.location is not None and hazard.location.approx:
        parts.append("Location approximate.")
    return " ".join(parts)


def build_warning(
    hazard: AssessedHazard,
    recipients: Sequence[str],
    policy: Distribution,
    msg_id: str,
    t_sent: float,
) -> WarningMessage:
    return WarningMessage(
        msg_id=msg_id,
        cluster_id=hazard.cluster_id,
        text=render_text(hazard),
        status=hazard.status,
        policy=policy,
        recipients=tuple(recipients),
        t_sent=t_sent,
    )
