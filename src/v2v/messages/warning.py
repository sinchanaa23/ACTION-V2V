"""Stage 5: warning schema and uncertainty-aware text. Owner: P2 (P2-07).

The text is built only from templates and structured fields, never by copying
the report, so abuse or personal details in a report cannot reach other
drivers. The one exception is the location phrase, which the extractor copies
from the report as a short place name; it is length-capped here. Every
warning says how certain it is: corroborated, unverified, or approximate
location.
"""

from collections.abc import Sequence

from v2v.contracts import NON_HAZARD_TYPES, AssessedHazard, Distribution, HazardType, WarningMessage

MAX_LOCATION_CHARS = 40
STALE_AFTER_S = 120.0
"""Older warnings say how long ago the hazard was reported."""

# (what, where it is, what to do). "Reported" is added for unverified ones.
_TEMPLATES: dict[HazardType, tuple[str, str, str]] = {
    HazardType.ACCIDENT: ("accident", "ahead", "Slow down and be ready to stop."),
    HazardType.OBSTRUCTION: ("road obstruction", "ahead", "Slow down; a lane may be blocked."),
    HazardType.SLIPPERY: ("slippery road", "ahead", "Reduce speed and avoid hard braking."),
    HazardType.STOPPED: ("stopped vehicle", "ahead", "Slow down and keep a safe gap."),
    HazardType.SPEEDING: ("speeding vehicle", "nearby", "Stay in your lane and keep a safe gap."),
}


def _location_phrase(location_text: str | None) -> str:
    if not location_text:
        return ""
    place = " ".join(location_text.split())[:MAX_LOCATION_CHARS].rstrip()
    return f" near {place}" if place else ""


def render_text(hazard: AssessedHazard, location_text: str | None = None) -> str:
    """Word a warning so the driver can see how certain it is.

    ``location_text`` is the place phrase from the report, if one was stated.
    """
    if hazard.hazard in NON_HAZARD_TYPES or hazard.status == "suppressed":
        raise ValueError(f"no warning text for a {hazard.status} {hazard.hazard} hazard")
    what, where, advice = _TEMPLATES[hazard.hazard]
    if hazard.status == "corroborated":
        head = f"{what.capitalize()} {where}"
    else:
        head = f"Reported {what} {where}"
    parts = [f"{head}{_location_phrase(location_text)}.", advice]

    if hazard.status == "corroborated":
        reports = "1 report" if hazard.n_reports == 1 else f"{hazard.n_reports} reports"
        parts.append(f"Status: confirmed ({reports}).")
    else:
        parts.append("Status: unverified report.")
    if hazard.location is not None and hazard.location.approx:
        parts.append("Location approximate.")
    if hazard.age_s >= STALE_AFTER_S:
        parts.append(f"Reported {int(hazard.age_s // 60)} min ago.")
    return " ".join(parts)


def build_warning(
    hazard: AssessedHazard,
    recipients: Sequence[str],
    policy: Distribution,
    msg_id: str,
    t_sent: float,
    location_text: str | None = None,
) -> WarningMessage:
    return WarningMessage(
        msg_id=msg_id,
        cluster_id=hazard.cluster_id,
        text=render_text(hazard, location_text),
        status=hazard.status,
        policy=policy,
        recipients=tuple(recipients),
        t_sent=t_sent,
    )
