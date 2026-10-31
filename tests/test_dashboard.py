import importlib.util
from pathlib import Path

import pytest

from tests.test_scenarios import _spec
from v2v.sim.recorder import (
    PolicyResult,
    ReportFrame,
    Trace,
    VehicleSnap,
    classify,
    policy_metrics,
)


def _snap(vid: str, relevant: bool) -> VehicleSnap:
    return VehicleSnap(
        vehicle_id=vid,
        lon=77.59,
        lat=13.04,
        heading_deg=0,
        speed_mps=10,
        edge_id="e",
        relevant=relevant,
    )


def _frame(tmp_path: Path) -> ReportFrame:
    spec = _spec(tmp_path)
    vehicles = (
        _snap("needed_and_got", True),
        _snap("needed_addressed_lost", True),
        _snap("needed_missed", True),
        _snap("unneeded_got", False),
        _snap("bystander", False),
    )
    results = tuple(
        PolicyResult(
            policy=policy,  # type: ignore[arg-type]
            location=location,  # type: ignore[arg-type]
            recipients=("needed_and_got", "needed_addressed_lost", "unneeded_got"),
            latency_ms={"needed_and_got": 12.0, "unneeded_got": 30.0},
            hops={"needed_and_got": 1, "unneeded_got": 2},
            transmissions=3,
            receptions=9,
        )
        for policy in ("broadcast", "circular", "directional")
        for location in ("known", "fallback")
    )
    return ReportFrame(
        slot=spec.slots[0],
        t_s=365.0,
        t_pass_s=360.0,
        reporter=_snap("rep_0", False),
        fallback_lon=77.59,
        fallback_lat=13.04,
        sensor_flags=1,
        vehicles=vehicles,
        results=results,
    )


def test_policy_metrics_against_ground_truth(tmp_path: Path) -> None:
    frame = _frame(tmp_path)
    m = policy_metrics(frame, frame.result("directional", "known"))
    assert m["recall"] == pytest.approx(2 / 3)  # addressed 2 of 3 that needed it
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["relevant_reached"] == pytest.approx(1 / 3)
    assert m["irrelevant_reached"] == 1.0
    assert m["mean_latency_ms"] == pytest.approx(21.0)
    assert (m["transmissions"], m["receptions"], m["recipients"]) == (3.0, 9.0, 3.0)


def test_classify_every_vehicle(tmp_path: Path) -> None:
    frame = _frame(tmp_path)
    assert classify(frame, frame.result("circular", "fallback")) == {
        "needed_and_got": "warned, needed it",
        "needed_addressed_lost": "addressed, not delivered",
        "needed_missed": "missed",
        "unneeded_got": "warned, didn't need it",
        "bystander": "not warned",
    }


def _trace(tmp_path: Path) -> Trace:
    return Trace(
        spec=_spec(tmp_path),
        hazard_lon=77.59,
        hazard_lat=13.04,
        radius_m=800,
        comm_range_m=300,
        frames=(_frame(tmp_path),),
        vehicles_passed_hazard=4,
        teleported=0,
    )


def test_trace_round_trip(tmp_path: Path) -> None:
    trace = _trace(tmp_path)
    trace.save(tmp_path / "traces" / "t.json")
    assert Trace.load(tmp_path / "traces" / "t.json") == trace


needs_streamlit = pytest.mark.skipif(
    importlib.util.find_spec("streamlit") is None, reason="needs the [dashboard] extras"
)
APP = str(Path(__file__).resolve().parents[1] / "dashboard" / "app.py")


@needs_streamlit
def test_app_renders_a_trace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from streamlit.testing.v1 import AppTest

    _trace(tmp_path).save(tmp_path / "traces" / "t.json")
    monkeypatch.setenv("V2V_TRACES_DIR", str(tmp_path / "traces"))
    app = AppTest.from_file(APP, default_timeout=60).run()
    assert not app.exception
    text = " ".join(m.value for m in app.markdown)
    assert "pending P2" in text and "event happened" in text
    app.radio[1].set_value("circular").run()  # switch policy
    app.radio[2].set_value("fallback").run()  # switch location mode
    assert not app.exception


@needs_streamlit
def test_app_picks_up_traces_recorded_after_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from streamlit.testing.v1 import AppTest

    traces = tmp_path / "traces"
    _trace(tmp_path).save(traces / "a.json")
    monkeypatch.setenv("V2V_TRACES_DIR", str(traces))
    app = AppTest.from_file(APP, default_timeout=60).run()
    assert len(app.selectbox[0].options) == 1
    _trace(tmp_path).save(traces / "b.json")
    app.run()
    assert len(app.selectbox[0].options) == 2


@needs_streamlit
def test_app_explains_when_there_are_no_traces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("V2V_TRACES_DIR", str(tmp_path / "none"))
    app = AppTest.from_file(APP, default_timeout=60).run()
    assert not app.exception
    assert "v2v.sim.recorder" in app.warning[0].value
