import importlib.util
import os
from pathlib import Path

import pytest

from tests.test_scenarios import _spec
from v2v.contracts import VehicleState
from v2v.geocast.directional import REACHES, RoadGraph
from v2v.sim.driver import (
    DEMO_FLEET,
    DRIVER_ALERTS,
    along_route,
    bearing_from,
    demo_depart_s,
    route_ahead_xy,
)

# "a" (100 m, east) -> "h" (100 m, north) -> "b" (100 m, north)
GRAPH = RoadGraph.from_edges(
    [("a", 100.0, ["h"]), ("h", 100.0, ["b"]), ("b", 100.0, [])],
    {
        "a": [(-100.0, 0.0), (0.0, 0.0)],
        "h": [(0.0, 0.0), (0.0, 100.0)],
        "b": [(0.0, 100.0), (0.0, 200.0)],
    },
)


def _car(
    edge: str, pos: float, x: float, y: float, heading: float, ahead: tuple[str, ...]
) -> VehicleState:
    return VehicleState(
        vehicle_id="me",
        x=x,
        y=y,
        edge_id=edge,
        lane_pos_m=pos,
        heading_deg=heading,
        speed_mps=10,
        route_ahead=ahead,
    )


def test_the_demo_fleet_is_a_car_and_a_scooter_on_opposite_carriageways() -> None:
    assert {d.kind for d in DEMO_FLEET} == {"car", "scooter"}
    assert {d.route for d in DEMO_FLEET} == {"approach", "opposing"}
    scooter = next(d for d in DEMO_FLEET if d.kind == "scooter")
    assert scooter.vclass == "motorcycle" and scooter.length_m < 2.0
    assert {a.reach for a in DRIVER_ALERTS.values()} == set(REACHES)  # one alert per direction


def test_bearings_are_relative_to_the_heading() -> None:
    north = _car("h", 0, 0, 0, 0.0, ())
    assert bearing_from(north, 0, 50).side == "ahead"
    assert bearing_from(north, 0, -50).side == "behind"
    assert bearing_from(north, 50, 0).side == "right"
    assert bearing_from(north, -50, 0).relative_deg == pytest.approx(-90.0)
    east = _car("a", 50, -50, 0, 90.0, ("h", "b"))
    assert bearing_from(east, 0, 0).side == "ahead"
    assert bearing_from(east, 0, 0).distance_m == pytest.approx(50.0)


def test_distance_along_the_route() -> None:
    me = _car("a", 40, -60, 0, 90.0, ("h", "b"))
    assert along_route(me, GRAPH, "a", 90) == pytest.approx(50.0)
    assert along_route(me, GRAPH, "h", 30) == pytest.approx(90.0)
    assert along_route(me, GRAPH, "b", 0) == pytest.approx(160.0)
    assert along_route(me, GRAPH, "a", 10) is None  # already passed
    assert along_route(me, GRAPH, "elsewhere", 0) is None


def test_route_ahead_follows_the_road_from_where_the_vehicle_is() -> None:
    me = _car("a", 40, -60, 0, 90.0, ("h", "b"))
    pts = route_ahead_xy(me, GRAPH, 120.0)
    assert pts[0] == (-60, 0)
    assert pts[1] == pytest.approx((-60.0, 0.0))
    assert (0.0, 0.0) in [tuple(round(c, 6) for c in p) for p in pts]  # turns north at the corner
    assert pts[-1] == pytest.approx((0.0, 60.0))  # 60 m east + 60 m north = 120 m


def test_demo_vehicles_leave_in_time_to_be_warned(tmp_path: Path) -> None:
    spec = _spec(tmp_path)  # first report due at 365 s, hazard 50 m along "h" after 100 m of "a"
    graph = RoadGraph.from_edges([("a", 100.0, ["h"]), ("h", 100.0, ["b"]), ("b", 100.0, [])])
    t = demo_depart_s(spec, graph)
    assert t == pytest.approx(365.0 - 0.6 * 150.0 / 12.0)
    assert t < 365.0


@pytest.mark.skipif(
    importlib.util.find_spec("traci") is None or not os.environ.get("V2V_NETWORK_TESTS"),
    reason="needs the [sim] extras and network access; set V2V_NETWORK_TESTS=1",
)
def test_driver_session_filters_by_direction_on_the_real_network(tmp_path: Path) -> None:
    from v2v.geocast import directional
    from v2v.sim.driver import DriverSession
    from v2v.sim.osm_network import CORRIDORS
    from v2v.sim.scenarios import generate

    (spec,) = generate(CORRIDORS["hebbal"], ["noisy"], ["low"], [0], tmp_path / "networks")
    session = DriverSession(spec)
    session.open()
    try:
        assert all(session.on_road(d.vehicle_id) for d in DEMO_FLEET)
        while session.time_s < spec.end_s and len(session.live.events) < 2:
            session.advance()
        for e in session.live.events:
            w = e.result.warning
            if w is None or directional.reach_for(e.result.assessed.hazard) != "backward":
                continue
            assert "demo_scooter" not in w.recipients  # the other carriageway
        car = session.state("demo_car")
        assert car is not None and len(session.trails["demo_car"]) > 1
        for kind in DRIVER_ALERTS:
            alert = session.send("demo_car", kind)
            assert "demo_car" not in alert.message.recipients
            assert {r.vehicle_id for r in alert.outcome.records} == set(alert.message.recipients)
        braking = session.sent[0]
        assert all(
            session.where_is("demo_car", *(session.state(v).x, session.state(v).y)).side != "ahead"  # type: ignore[union-attr, misc]
            for v in braking.message.recipients
            if session.state(v) is not None
        )  # hard braking goes to traffic behind, never ahead
    finally:
        session.close()
