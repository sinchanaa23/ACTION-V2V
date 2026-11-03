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


DASHBOARD = Path(APP).parent


@pytest.fixture
def ui(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the dashboard's sibling ui_* modules importable."""
    monkeypatch.syspath_prepend(str(DASHBOARD))


@needs_streamlit
@pytest.mark.usefixtures("ui")
def test_geometry_helpers() -> None:
    from ui_map import distance_m, ring, ticks
    from ui_pages import cardinal

    lons, lats = ring(77.59, 13.04, 300.0)
    assert (lons[0], lats[0]) == (lons[-1], lats[-1])  # closed outline
    assert all(
        abs(distance_m(77.59, 13.04, x, y) - 300.0) < 1.0 for x, y in zip(lons, lats, strict=True)
    )
    north = _snap("n", False).model_copy(update={"heading_deg": 0.0})
    tick_lons, tick_lats = ticks([north])
    assert tick_lats[1] > tick_lats[0] and tick_lons[1] == pytest.approx(tick_lons[0])
    assert tick_lons[2] is None  # segments are separated
    assert [cardinal(h) for h in (0, 44, 90, 200, 315, 359)] == ["N", "NE", "E", "S", "NW", "N"]


@needs_streamlit
@pytest.mark.usefixtures("ui")
def test_warning_text_comes_from_p2_generator(tmp_path: Path) -> None:
    from ui_data import View, warning_text

    trace = _trace(tmp_path)
    known = warning_text(View(trace, trace.frames[0], "directional", "known"))
    fallback = warning_text(View(trace, trace.frames[0], "directional", "fallback"))
    assert known is not None and "accident" in known and "approximate" not in known
    assert fallback is not None and fallback.endswith("Location approximate.")
    slot = trace.frames[0].slot.model_copy(update={"needs": "non_actionable"})
    frame = trace.frames[0].model_copy(update={"slot": slot})
    assert warning_text(View(trace, frame, "directional", "known")) is None


def _app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, traces: int = 1):  # type: ignore[no-untyped-def]
    from streamlit.testing.v1 import AppTest

    for i in range(traces):
        _trace(tmp_path).save(tmp_path / "traces" / f"t{i}.json")
    monkeypatch.setenv("V2V_TRACES_DIR", str(tmp_path / "traces"))
    return AppTest.from_file(APP, default_timeout=60)


@needs_streamlit
def test_app_renders_a_trace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = _app(tmp_path, monkeypatch).run()
    assert not app.exception
    text = " ".join(m.value for m in app.markdown)
    assert "pending P3" in text and "event happened" in text
    assert "Status: unverified report." in text  # P2's warning text in the trace
    app.radio(key="policy").set_value("circular").run()
    app.radio(key="location").set_value("fallback").run()
    assert not app.exception
    assert "Location approximate." in " ".join(m.value for m in app.markdown)


@needs_streamlit
def test_every_page_renders(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = _app(tmp_path, monkeypatch).run()
    for page in ("overview", "live", "network", "reports", "compare", "settings"):
        app.button(key=f"nav_{page}").click().run()
        assert not app.exception, page
        assert app.session_state["page"] == page


@needs_streamlit
def test_preferences_survive_page_switches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = _app(tmp_path, monkeypatch).run()
    app.button(key="nav_settings").click().run()
    app.radio(key="w_policy_set").set_value("broadcast").run()
    app.button(key="nav_live").click().run()
    assert app.radio(key="policy").value == "broadcast"


@needs_streamlit
def test_app_picks_up_traces_recorded_after_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _app(tmp_path, monkeypatch).run()
    assert len(app.selectbox(key="scenario").options) == 1
    _trace(tmp_path).save(tmp_path / "traces" / "b.json")
    app.run()
    assert len(app.selectbox(key="scenario").options) == 2


@needs_streamlit
def test_app_explains_when_there_are_no_traces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("V2V_TRACES_DIR", str(tmp_path / "none"))
    app = AppTest.from_file(APP, default_timeout=60).run()
    assert not app.exception
    assert "v2v.sim.recorder" in app.warning[0].value


@needs_streamlit
def test_overseer_and_driver_modes_switch_without_losing_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _app(tmp_path, monkeypatch).run()
    assert app.session_state["mode"] == "overseer"
    app.button(key="nav_compare").click().run()
    app.button(key="mode_driver").click().run()
    assert not app.exception
    text = " ".join(m.value for m in app.markdown) + " ".join(w.value for w in app.warning)
    assert "Choose your vehicle" in text
    assert not any(b.key and b.key.startswith("nav_") for b in app.button)  # fleet pages hidden
    app.button(key="mode_overseer").click().run()
    assert not app.exception
    assert app.session_state["page"] == "compare"  # back where the overseer left off


@needs_streamlit
@pytest.mark.usefixtures("ui")
def test_driver_view_helpers() -> None:
    from ui_driver import turn_cue
    from ui_nav import Camera, _path, relevance, where_text

    from v2v.sim.driver import Bearing

    cam = Camera(0.0, 0.0, 0.0, back_m=40.0, height_m=30.0)
    near, far = cam.project(0.0, 10.0), cam.project(0.0, 500.0)
    assert near is not None and far is not None
    assert near[0] == pytest.approx(0.0)
    assert near[1] < far[1] < 0  # farther is higher on screen, up to the horizon
    assert cam.project(0.0, -60.0) is None  # behind the camera
    right = cam.project(20.0, 10.0)
    assert right is not None and right[0] > 0
    east = Camera(0.0, 0.0, 90.0, 40.0, 30.0)
    ahead = east.project(100.0, 0.0)
    assert ahead is not None and ahead[0] == pytest.approx(0.0)  # heading up
    assert _path([0.0, 1.0, None, 2.0, 3.0], [0.0, 1.0, None, 2.0, 3.0], close=True).count("Z") == 2
    assert turn_cue(5)[1] == "Continue straight" and turn_cue(-40)[1] == "Bear left"
    assert turn_cue(90)[1] == "Turn right"
    on_route = Bearing(150.0, 2.0, 160.0)
    assert where_text(on_route) == "160 m ahead on your route"
    assert relevance(on_route)[1] == "on your route"
    assert relevance(Bearing(300.0, 179.0, None))[1] == "passed"
