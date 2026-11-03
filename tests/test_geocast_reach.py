"""Forward, backward and bidirectional geocast on a divided road, and delivery
without duplicates or a broadcast storm."""

from collections.abc import Iterator

import pytest

from v2v.contracts import (
    AssessedHazard,
    HazardType,
    ResolvedLocation,
    VehicleState,
    WarningMessage,
)
from v2v.geocast import SELECTORS, directional
from v2v.geocast.directional import RoadGraph, in_zone, road_zone, select_reach
from v2v.sim.delivery import DeliveryModel

# A divided road running north. Northbound "n0" -> "n1" -> "n2" at x=0;
# southbound "s2" -> "s1" -> "s0" at x=12 (the other carriageway), with
# "ramp" joining n1 and "exit" leaving n1. Every edge is 400 m except the
# side roads. The hazard is 200 m along n1, at (0, 600).
SHAPES = {
    "n0": [(0.0, 0.0), (0.0, 400.0)],
    "n1": [(0.0, 400.0), (0.0, 800.0)],
    "n2": [(0.0, 800.0), (0.0, 1200.0)],
    "s2": [(12.0, 1200.0), (12.0, 800.0)],
    "s1": [(12.0, 800.0), (12.0, 400.0)],
    "s0": [(12.0, 400.0), (12.0, 0.0)],
    "ramp": [(-150.0, 400.0), (0.0, 400.0)],
    "exit": [(0.0, 800.0), (150.0, 800.0)],
}
GRAPH = RoadGraph.from_edges(
    [
        ("n0", 400.0, ["n1"]),
        ("ramp", 150.0, ["n1"]),
        ("n1", 400.0, ["n2", "exit"]),
        ("n2", 400.0, []),
        ("exit", 150.0, []),
        ("s2", 400.0, ["s1"]),
        ("s1", 400.0, ["s0"]),
        ("s0", 400.0, []),
    ],
    SHAPES,
)
POINT = ResolvedLocation(edge_id="n1", pos_m=200, x=0, y=600, approx=False, heading_deg=0)


def _car(vid: str, edge: str, pos: float) -> VehicleState:
    (x0, y0), (x1, y1) = SHAPES[edge][0], SHAPES[edge][-1]
    f = pos / GRAPH.length_m[edge]
    x, y = x0 + (x1 - x0) * f, y0 + (y1 - y0) * f
    heading = {"n": 0.0, "s": 180.0, "r": 90.0, "e": 90.0}[edge[0]]
    return VehicleState(
        vehicle_id=vid, x=x, y=y, edge_id=edge, lane_pos_m=pos, heading_deg=heading, speed_mps=12
    )


FLEET = [
    _car("behind_far", "n0", 100),  # 500 m before the point
    _car("behind_near", "n1", 50),
    _car("on_ramp", "ramp", 100),
    _car("ahead_near", "n1", 350),
    _car("ahead_far", "n2", 200),
    _car("taking_exit", "exit", 60),
    _car("oncoming_before", "s1", 100),  # southbound, still north of the point: approaching it
    _car("oncoming_past", "s1", 300),  # southbound, already south of the point
    _car("oncoming_far", "s0", 300),
]


def _hazard(kind: HazardType) -> AssessedHazard:
    return AssessedHazard(
        cluster_id="c",
        hazard=kind,
        location=POINT,
        interp_conf=0.9,
        event_evidence=0.9,
        n_reports=2,
        age_s=0.0,
        status="corroborated",
    )


@pytest.fixture
def road_map() -> Iterator[None]:
    directional.use_road_graph(GRAPH)
    yield
    directional.use_road_graph(None)


def test_backward_reaches_traffic_approaching_on_this_carriageway_only() -> None:
    kept = set(select_reach(POINT, FLEET, 800, "backward", GRAPH))
    assert kept == {"behind_far", "behind_near", "on_ramp"}


def test_forward_reaches_traffic_ahead_on_this_carriageway_only() -> None:
    kept = set(select_reach(POINT, FLEET, 800, "forward", GRAPH))
    assert kept == {"ahead_near", "ahead_far", "taking_exit"}


def test_bidirectional_adds_the_approaching_side_of_the_other_carriageway() -> None:
    assert GRAPH.opposite("n1", 0, 600) == ("s1", pytest.approx(200.0))
    kept = set(select_reach(POINT, FLEET, 800, "bidirectional", GRAPH))
    assert kept == {"behind_far", "behind_near", "on_ramp", "oncoming_before"}
    assert not kept & {"oncoming_past", "oncoming_far", "ahead_near"}  # driving away from it


def test_radius_limits_every_direction() -> None:
    assert set(select_reach(POINT, FLEET, 160, "backward", GRAPH)) == {"behind_near"}
    assert set(select_reach(POINT, FLEET, 160, "forward", GRAPH)) == {"ahead_near"}


def test_zone_is_stretches_of_directed_road() -> None:
    zone = road_zone(GRAPH, POINT, 300, "backward")
    assert zone == {"n1": (0.0, 200.0), "n0": (300.0, 400.0), "ramp": (50.0, 150.0)}
    assert in_zone(zone, "n0", 350) and not in_zone(zone, "n0", 250)
    assert not in_zone(zone, "s1", 100)  # the oncoming lanes are another edge


def test_sender_is_never_its_own_recipient() -> None:
    braking = _car("braking", "n1", 200)
    kept = select_reach(POINT, [*FLEET, braking], 800, "backward", GRAPH, exclude="braking")
    assert "braking" not in kept


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        (HazardType.ACCIDENT, {"behind_far", "behind_near", "on_ramp"}),  # example 1
        (HazardType.STOPPED, {"behind_far", "behind_near", "on_ramp"}),  # example 2: braking
        (HazardType.OBSTRUCTION, {"behind_far", "behind_near", "on_ramp"}),  # example 3
        (HazardType.SLIPPERY, {"behind_far", "behind_near", "on_ramp", "oncoming_before"}),  # 4
    ],
)
def test_policy_picks_the_direction_from_the_hazard(
    road_map: None, kind: HazardType, expected: set[str]
) -> None:
    assert set(SELECTORS["directional"](_hazard(kind), FLEET, 800)) == expected


def test_heading_rule_follows_the_direction_without_a_map() -> None:
    hx, hy = 0.0, 600.0
    behind = _car("behind", "n1", 50)
    ahead = _car("ahead", "n1", 350)
    oncoming = _car("oncoming", "s1", 100)
    assert directional.keeps(behind, hx, hy, 0.0, "backward")
    assert not directional.keeps(ahead, hx, hy, 0.0, "backward")
    assert directional.keeps(ahead, hx, hy, 0.0, "forward")
    assert not directional.keeps(behind, hx, hy, 0.0, "forward")
    assert not directional.keeps(oncoming, hx, hy, 0.0, "backward")
    assert directional.keeps(oncoming, hx, hy, 0.0, "bidirectional")


def test_a_closing_world_does_not_strand_another_on_the_heading_rule() -> None:
    a, b = object(), object()
    other = RoadGraph.from_edges([("x", 1.0, [])])
    directional.acquire_road_graph(a, GRAPH)
    directional.acquire_road_graph(b, other)
    directional.release_road_graph(b)
    assert directional.current_road_graph() is GRAPH  # a is still running
    directional.release_road_graph(a)
    assert directional.current_road_graph() is None


class _Line:
    """A dense queue: 40 cars 20 m apart, all addressed."""

    time_s = 0.0

    def __init__(self) -> None:
        self._v = [
            VehicleState(
                vehicle_id=f"v{i}",
                x=0,
                y=i * 20.0,
                edge_id="n1",
                lane_pos_m=0,
                heading_deg=0,
                speed_mps=5,
            )
            for i in range(40)
        ]

    def vehicles(self) -> list[VehicleState]:
        return self._v

    def vehicle(self, vid: str) -> VehicleState:
        return next(v for v in self._v if v.vehicle_id == vid)

    def will_reach(self, vid: str, location: ResolvedLocation) -> bool:
        return True


def _message(ids: list[str]) -> WarningMessage:
    return WarningMessage(
        msg_id="m",
        cluster_id="c",
        text="t",
        status="corroborated",
        policy="directional",
        recipients=tuple(ids),
        t_sent=0,
    )


def test_suppression_cuts_rebroadcasts_without_losing_recipients() -> None:
    world = _Line()
    ids = [v.vehicle_id for v in world.vehicles()]
    point = POINT.model_copy(update={"x": 0, "y": 0})
    flood = DeliveryModel(suppress_after=None).deliver(_message(ids), world, point, "v0")  # type: ignore[arg-type]
    smart = DeliveryModel().deliver(_message(ids), world, point, "v0")  # type: ignore[arg-type]
    assert smart.transmissions < flood.transmissions / 2
    assert smart.suppressed > 0 and flood.suppressed == 0
    delivered = {r.vehicle_id for r in smart.records if r.delivered}
    assert delivered == set(ids) - {"v0"}  # everyone still gets it
    assert len(smart.records) == len(ids)  # one record each: no duplicate delivery
