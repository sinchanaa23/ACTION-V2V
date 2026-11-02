"""P2-07: uncertainty-aware warning text."""

import pytest

from v2v.contracts import AssessedHazard, HazardType, ResolvedLocation, Status
from v2v.messages.warning import MAX_LOCATION_CHARS, build_warning, render_text


def _hazard(
    status: Status = "unverified",
    hazard: HazardType = HazardType.ACCIDENT,
    approx: bool = False,
    n_reports: int = 1,
    age_s: float = 0.0,
) -> AssessedHazard:
    return AssessedHazard(
        cluster_id="c0",
        hazard=hazard,
        location=ResolvedLocation(edge_id="e", pos_m=10.0, x=0.0, y=0.0, approx=approx),
        interp_conf=0.9,
        event_evidence=0.8 if status == "corroborated" else 0.1,
        n_reports=n_reports,
        age_s=age_s,
        status=status,
    )


def test_unverified_matches_proposal_example() -> None:
    assert render_text(_hazard(), "bridge") == (
        "Reported accident ahead near bridge. Slow down and be ready to stop. "
        "Status: unverified report."
    )


def test_corroborated_states_report_count() -> None:
    text = render_text(_hazard("corroborated", n_reports=3))
    assert text.startswith("Accident ahead.")
    assert "confirmed (3 reports)" in text
    assert "unverified" not in text


def test_approximate_location_and_staleness_are_stated() -> None:
    text = render_text(_hazard(approx=True, age_s=300.0))
    assert "Location approximate." in text
    assert "Reported 5 min ago." in text


@pytest.mark.parametrize(
    "hazard",
    [
        HazardType.ACCIDENT,
        HazardType.OBSTRUCTION,
        HazardType.SLIPPERY,
        HazardType.STOPPED,
        HazardType.SPEEDING,
    ],
)
def test_every_actionable_class_has_short_text(hazard: HazardType) -> None:
    text = render_text(_hazard(hazard=hazard, approx=True, age_s=600.0), "x" * 200)
    assert len(text) <= 200
    assert "x" * (MAX_LOCATION_CHARS + 1) not in text


@pytest.mark.parametrize("hazard", [HazardType.NON_ACTIONABLE, HazardType.AMBIGUOUS])
def test_non_hazards_get_no_text(hazard: HazardType) -> None:
    with pytest.raises(ValueError):
        render_text(_hazard(hazard=hazard))


def test_suppressed_gets_no_text() -> None:
    with pytest.raises(ValueError):
        render_text(_hazard("suppressed"))


def test_blank_location_is_ignored() -> None:
    assert "near" not in render_text(_hazard(), "   ")


def test_build_warning_carries_fields() -> None:
    msg = build_warning(_hazard(), ["v1", "v2"], "directional", "m0", 12.0, "hebbal flyover")
    assert msg.recipients == ("v1", "v2")
    assert msg.status == "unverified"
    assert "near hebbal flyover" in msg.text
