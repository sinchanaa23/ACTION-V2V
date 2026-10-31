from collections.abc import Iterator

import pytest

from v2v.contracts import AssessedHazard, HazardType, ResolvedLocation, VehicleState
from v2v.geocast import SELECTORS, broadcast, circular, directional
from v2v.geocast.directional import RoadGraph
from v2v.geocast.fallback import fallback_location

# A divided road running north (heading 0) through a hazard at (0, 0).
# The northbound carriageway is x=0; the southbound one is x=10.
HAZARD = AssessedHazard(
    cluster_id="c0",
    hazard=HazardType.ACCIDENT,
    location=ResolvedLocation(edge_id="n", pos_m=500, x=0, y=0, approx=False, heading_deg=0),
    interp_conf=0.9,
    event_evidence=0.0,
    n_reports=1,
    age_s=0.0,
    status="unverified",
)


def _v(vid: str, x: float, y: float, heading: float) -> VehicleState:
    return VehicleState(
        vehicle_id=vid, x=x, y=y, edge_id="e", lane_pos_m=0, heading_deg=heading, speed_mps=10
    )


VEHICLES = [
    _v("approaching", 0, -300, 0),  # northbound, behind the hazard: must be warned
    _v("approaching_curve", 30, -600, 20),  # still approaching on a gentle curve
    _v("at_hazard", 0, -2, 0),
    _v("passed", 0, 200, 0),  # northbound, already past
    _v("opposite", 10, 300, 180),  # southbound carriageway, coming towards the hazard point
    _v("opposite_passed", 10, -300, 180),
    _v("cross_street", -200, -50, 90),  # heading east towards the road
    _v("far", 0, -2000, 0),  # approaching but outside R
]


def _ids(selected: list[str]) -> set[str]:
    return set(selected)


def test_broadcast_takes_everyone() -> None:
    assert _ids(broadcast.select(HAZARD, VEHICLES, 800)) == {v.vehicle_id for v in VEHICLES}


def test_circular_takes_everyone_within_radius_both_carriageways() -> None:
    assert _ids(circular.select(HAZARD, VEHICLES, 800)) == {
        v.vehicle_id for v in VEHICLES if v.vehicle_id != "far"
    }


def test_directional_keeps_only_approaching_same_carriageway() -> None:
    assert _ids(directional.select(HAZARD, VEHICLES, 800)) == {
        "approaching",
        "approaching_curve",
        "at_hazard",
    }


def test_directional_without_road_heading_still_drops_vehicles_past_it() -> None:
    assert HAZARD.location is not None
    no_heading = HAZARD.model_copy(
        update={"location": HAZARD.location.model_copy(update={"heading_deg": None})}
    )
    kept = _ids(directional.select(no_heading, VEHICLES, 800))
    assert {"approaching", "at_hazard"} <= kept
    assert not {"passed", "opposite_passed", "far"} & kept
    assert "opposite" in kept  # can't tell carriageways apart without the road direction


@pytest.mark.parametrize("policy", ["circular", "directional"])
def test_geocast_without_location_is_an_error(policy: str) -> None:
    with pytest.raises(ValueError, match="fallback"):
        SELECTORS[policy](HAZARD.model_copy(update={"location": None}), VEHICLES, 800)  # type: ignore[index]


# Road map: a feeder side street "c" and the northbound "n0" both lead onto
# "n1", which carries the hazard at 100 m and continues to "n2". The
# southbound carriageway "s1" -> "s0" runs alongside, and "x" is an
# unconnected parallel service road.
GRAPH = RoadGraph.from_edges(
    [
        ("n0", 300.0, ["n1"]),
        ("c", 200.0, ["n1"]),
        ("n1", 400.0, ["n2"]),
        ("n2", 300.0, []),
        ("s1", 400.0, ["s0"]),
        ("s0", 300.0, []),
        ("x", 600.0, []),
    ]
)
ROAD_HAZARD = HAZARD.model_copy(
    update={
        "location": ResolvedLocation(edge_id="n1", pos_m=100, x=0, y=0, approx=False, heading_deg=0)
    }
)


def _on(vid: str, edge: str, pos: float, heading: float = 0.0) -> VehicleState:
    return _v(vid, 0, -50, heading).model_copy(update={"edge_id": edge, "lane_pos_m": pos})


@pytest.fixture
def road_map() -> Iterator[None]:
    directional.use_road_graph(GRAPH)
    yield
    directional.use_road_graph(None)


def test_upstream_distances_follow_the_road() -> None:
    assert GRAPH.upstream("n1", 100, 800) == {"n0": 100.0, "c": 100.0}
    assert GRAPH.upstream("n1", 100, 50) == {}  # end of the feeders is 100 m away
    assert GRAPH.upstream("n2", 0, 800) == {"n1": 0.0, "n0": 400.0, "c": 400.0}


def test_road_mode_keeps_feeders_and_drops_the_rest(road_map: None) -> None:
    vehicles = [
        _on("same_edge_before", "n1", 40),
        _on("same_edge_after", "n1", 150),
        _on("on_feeder_street", "c", 50, heading=90),  # turning in from a side street
        _on("on_approach", "n0", 250),
        _on("opposite", "s1", 300, heading=180),
        _on("parallel_service_road", "x", 100),  # same heading, not connected
        _on("too_far_by_road", "n0", 0),  # 300 + 100 = 400 m by road
    ]
    kept = set(directional.select(ROAD_HAZARD, vehicles, 350))
    assert kept == {"same_edge_before", "on_feeder_street", "on_approach"}


def test_road_mode_keeps_oncoming_traffic_that_can_loop_round(road_map: None) -> None:
    # e.g. an interchange: the other carriageway legally leads back to the hazard.
    directional.use_road_graph(RoadGraph.from_edges([("s1", 100.0, ["n1"]), ("n1", 400.0, [])]))
    oncoming = _on("oncoming", "s1", 50, heading=180)
    assert directional.select(ROAD_HAZARD, [oncoming], 800) == ["oncoming"]
    assert directional.select(ROAD_HAZARD, [oncoming], 100) == []  # 150 m by road


def test_road_mode_needs_hazard_edge_on_the_map(road_map: None) -> None:
    off_map = HAZARD  # edge "n" is not in GRAPH: falls back to the heading rule
    assert set(directional.select(off_map, VEHICLES, 800)) == {
        "approaching",
        "approaching_curve",
        "at_hazard",
    }


def test_fallback_puts_hazard_at_reporter_and_flags_it() -> None:
    reporter = _v("rep", 3, 40, 0).model_copy(update={"edge_id": "n", "lane_pos_m": 540})
    loc = fallback_location(reporter)
    assert (loc.edge_id, loc.pos_m, loc.x, loc.y, loc.heading_deg) == ("n", 540, 3, 40, 0)
    assert loc.approx
    # Traffic behind the reporter on its carriageway gets the warning; it doesn't.
    hazard = HAZARD.model_copy(update={"location": loc})
    kept = _ids(directional.select(hazard, [*VEHICLES, reporter], 800))
    assert "approaching" in kept and "opposite" not in kept
