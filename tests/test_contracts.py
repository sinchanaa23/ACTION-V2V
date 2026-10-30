import pytest
from pydantic import ValidationError

from tests.fixtures import ACCIDENT_HINGLISH
from v2v.contracts import (
    REAL_SOURCES,
    DriverReport,
    HazardInterpretation,
    HazardType,
    VehicleState,
)


def test_fixture_is_not_a_real_source() -> None:
    assert "fixture" not in REAL_SOURCES
    assert {"survey", "reddit", "youtube"} <= REAL_SOURCES


def test_unknown_source_rejected() -> None:
    with pytest.raises(ValidationError):
        DriverReport(report_id="r", text=ACCIDENT_HINGLISH, source="llm")  # type: ignore[arg-type]


def test_probability_bounds() -> None:
    with pytest.raises(ValidationError):
        HazardInterpretation(report_id="r", hazard=HazardType.ACCIDENT, p_class=1.2)


def test_location_defaults_to_unknown() -> None:
    interp = HazardInterpretation(report_id="r", hazard=HazardType.ACCIDENT, p_class=0.9)
    assert interp.location_text is None


def test_contracts_are_frozen_and_strict() -> None:
    report = DriverReport(report_id="r", text=ACCIDENT_HINGLISH, source="fixture")
    with pytest.raises(ValidationError):
        report.text = "changed"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        DriverReport(report_id="r", text="x", source="fixture", extra_field=1)  # type: ignore[call-arg]


def test_heading_range() -> None:
    with pytest.raises(ValidationError):
        VehicleState(
            vehicle_id="v", x=0, y=0, edge_id="e", lane_pos_m=0, heading_deg=360, speed_mps=0
        )


def test_json_round_trip() -> None:
    report = DriverReport(report_id="r", text=ACCIDENT_HINGLISH, source="fixture", t_sim=3.0)
    assert DriverReport.model_validate_json(report.model_dump_json()) == report
