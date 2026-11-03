import math
import sys
from pathlib import Path

import pytest

from v2v.sim.world3d import (
    DECK_HEIGHT_M,
    DEFAULT_BUILDING_M,
    Scene,
    area_kind,
    building_height,
    deck_layers,
    node_heights,
    parse_context,
    way_id,
)

SIM3D = Path(__file__).resolve().parents[1] / "dashboard" / "sim3d"


def test_building_height_from_tags_or_default() -> None:
    assert building_height({"height": "21 m"}) == 21.0
    assert building_height({"building:levels": "4"}) == pytest.approx(12.8)
    assert building_height({"building": "yes"}) == DEFAULT_BUILDING_M


def test_area_kinds() -> None:
    assert area_kind({"building": "house"}) == "building"
    assert area_kind({"natural": "water"}) == "water"
    assert area_kind({"leisure": "park"}) == "park"
    assert area_kind({"highway": "trunk"}) is None


def test_way_id_behind_sumo_edge_ids() -> None:
    assert way_id("-1382087677#1") == "1382087677"
    assert way_id("760685446") == "760685446"


def test_flyover_decks_raise_their_junctions_and_ramp_the_approaches(tmp_path: Path) -> None:
    osm = tmp_path / "r.osm"
    osm.write_text(
        "<osm>"
        '<way id="10"><tag k="bridge" v="yes"/><tag k="layer" v="1"/></way>'
        '<way id="20"><tag k="highway" v="trunk"/></way>'
        '<way id="30"><tag k="bridge" v="yes"/><tag k="layer" v="-1"/></way>'
        "</osm>",
        encoding="utf-8",
    )
    layers = deck_layers(osm)
    assert layers == {"10": 1}  # underpasses stay on the ground
    heights = node_heights([("10#0", "a", "b"), ("-20", "b", "c"), ("30", "c", "d")], layers)
    assert heights == {"a": DECK_HEIGHT_M, "b": DECK_HEIGHT_M, "c": 0.0, "d": 0.0}
    scene = Scene(edge_z={"-20": (DECK_HEIGHT_M, 0.0, 100.0)})
    assert scene.elevation("-20", 50.0) == pytest.approx(DECK_HEIGHT_M / 2)  # ramp down
    assert scene.elevation("-20", math.inf) == 0.0  # junction-internal: end of the edge
    assert scene.elevation("unknown", 5.0) == 0.0


def test_parse_context_keeps_closed_areas_and_known_spots(tmp_path: Path) -> None:
    xml = tmp_path / "c.osm"
    xml.write_text(
        "<osm>"
        '<node id="1" lon="0" lat="0"/><node id="2" lon="1" lat="0"/>'
        '<node id="3" lon="1" lat="1"/><node id="4" lon="0" lat="1"/>'
        '<node id="5" lon="2" lat="2"><tag k="highway" v="traffic_signals"/></node>'
        '<way id="9"><nd ref="1"/><nd ref="2"/><nd ref="3"/><nd ref="4"/><nd ref="1"/>'
        '<tag k="building" v="yes"/><tag k="building:levels" v="2"/></way>'
        '<way id="8"><nd ref="1"/><nd ref="2"/><nd ref="3"/><tag k="natural" v="water"/></way>'
        "</osm>",
        encoding="utf-8",
    )
    areas, spots = parse_context(xml, lambda lon, lat: (lon * 100, lat * 100))
    (building,) = areas  # the open water way is not an area
    assert building.kind == "building" and building.height_m == pytest.approx(6.4)
    assert len(building.outline) == 4 and building.outline[0] == (0.0, 0.0)
    assert [(s.kind, s.x, s.y) for s in spots] == [("traffic_signals", 200.0, 200.0)]


@pytest.fixture
def geom(monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    monkeypatch.syspath_prepend(str(SIM3D))
    import geom

    yield geom
    sys.modules.pop("geom", None)


def test_strips_offset_to_the_left_of_travel(geom) -> None:  # type: ignore[no-untyped-def]
    north = [(0.0, 0.0, 0.0), (0.0, 10.0, 0.0)]
    left, right = geom.strip(north, 2.0, -2.0)
    assert left[0][:2] == pytest.approx((-2.0, 0.0))  # west of a northbound lane
    assert right[1][:2] == pytest.approx((2.0, 10.0))


def test_dashes_and_spacing(geom) -> None:  # type: ignore[no-untyped-def]
    line = [(0.0, 0.0, 0.0), (0.0, 30.0, 0.0)]
    cuts = geom.dashes(line, 3.0, 6.0)
    starts = [round(d[0][1]) for d in cuts]
    assert starts == [0, 9, 18, 27]
    assert all(math.dist(d[0][:2], d[-1][:2]) == pytest.approx(3.0) for d in cuts)
    marks = geom.along(line, 10.0)
    assert [round(p[1]) for p, _ in marks] == [0, 10, 20, 30]
    assert all(h == pytest.approx(0.0) for _, h in marks)  # heading north


def test_polygon_helpers(geom) -> None:  # type: ignore[no-untyped-def]
    square_cw = [(0.0, 0.0), (0.0, 10.0), (10.0, 10.0), (10.0, 0.0)]
    assert geom.signed_area(geom.ccw(square_cw)) == pytest.approx(100.0)
    assert geom.inside(square_cw, 5, 5) and not geom.inside(square_cw, 15, 5)
    trees = geom.scatter(square_cw, 4.0, "park")
    assert trees and all(geom.inside(square_cw, x, y) for x, y in trees)
    assert trees == geom.scatter(square_cw, 4.0, "park")  # deterministic


def test_slow_motion_and_angles(geom) -> None:  # type: ignore[no-untyped-def]
    assert geom.slow_motion(25.0) * 25.0 == pytest.approx(2.5)  # whole flood in 2.5 s
    assert geom.lerp_angle(350.0, 10.0, 0.5) == pytest.approx(0.0)  # the short way round
    assert geom.to_ursina(110.0, 220.0, 7.0, (100.0, 200.0)) == (10.0, 7.0, 20.0)


def test_render_side_keeps_the_latest_snapshot_and_every_event(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The 3D view talks to SUMO in another process; its handle must keep only
    the newest snapshot, every fired event, and the process's state."""
    pytest.importorskip("sumolib")
    import multiprocessing as mp

    from tests.test_scenarios import _spec

    monkeypatch.syspath_prepend(str(SIM3D))
    import worker

    try:
        here, there = mp.Pipe()
        w = worker.SimWorker([_spec(tmp_path)])
        w._conn = here
        for t in (1.0, 2.0):
            snap = worker.Snapshot(t, {}, {}, frozenset(), 0, False, False)
            there.send(("snap", snap))
        there.send(("state", {"playing": True, "status": "running", "speed": 16.0, "error": None}))
        version, snap = w.latest()
        assert snap is not None and snap.t_s == 2.0 and version == 2
        assert w.playing and w.status == "running" and w.speed == 16.0 and w.error is None
        assert w.drain() == [] and w.events() == []
        there.send(("reset", None))
        there.close()
        w.latest()
        assert w.status == "error" and w.error  # the process went away: say so
    finally:
        sys.modules.pop("worker", None)
