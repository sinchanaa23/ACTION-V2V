import importlib.util
import os
from pathlib import Path

import pytest

from v2v.contracts import VehicleState
from v2v.sim.gazetteer import (
    Anchor,
    Gazetteer,
    Landmark,
    kinds_in,
    name_matches,
    normalise,
    place_features,
    places_query,
    road_features,
)

# Hand-written OSM snippets that exercise the parser; not map data.
ROADS_XML = b"""<osm>
 <node id="1" lat="13.0300" lon="77.5900"/>
 <node id="2" lat="13.0310" lon="77.5900"/>
 <node id="3" lat="13.0320" lon="77.5900"/>
 <node id="4" lat="13.0330" lon="77.5900"><tag k="highway" v="traffic_signals"/></node>
 <node id="5" lat="13.0340" lon="77.5900">
  <tag k="junction" v="yes"/><tag k="name" v="Test Circle"/></node>
 <way id="10"><nd ref="1"/><nd ref="2"/><nd ref="3"/>
  <tag k="highway" v="trunk"/><tag k="bridge" v="yes"/><tag k="name" v="Test Flyover"/></way>
 <way id="11"><nd ref="3"/><nd ref="4"/>
  <tag k="highway" v="primary"/><tag k="name" v="Test Main Road"/><tag k="ref" v="NH44"/></way>
 <way id="12"><nd ref="4"/><nd ref="5"/><tag k="highway" v="residential"/></way>
 <way id="13"><nd ref="1"/><nd ref="5"/><tag k="highway" v="service"/><tag k="name" v="Lane"/></way>
</osm>"""

PLACES_XML = b"""<osm>
 <node id="20" lat="13.03" lon="77.59"><tag k="shop" v="mall"/><tag k="name" v="Test Mall"/></node>
 <way id="21"><center lat="13.04" lon="77.60"/>
  <tag k="natural" v="water"/><tag k="name" v="Test Lake"/></way>
 <node id="22" lat="13.03" lon="77.59"><tag k="amenity" v="bench"/><tag k="name" v="Bench"/></node>
 <node id="23" lat="13.03" lon="77.59"><tag k="amenity" v="fuel"/></node>
</osm>"""


def test_road_features_kinds_and_names() -> None:
    by_id = {f.osm_id: f for f in road_features(ROADS_XML)}
    assert by_id["way/10"].kind == "flyover"
    assert by_id["way/10"].way_id == "10"
    assert by_id["way/10"].lat == pytest.approx(13.0310)  # middle node
    assert by_id["way/11"].kind == "road"
    assert by_id["way/11"].names == ("Test Main Road", "NH44")
    assert by_id["node/4"].kind == "signal"
    assert by_id["node/5"].kind == "junction"
    assert "way/12" not in by_id  # unnamed, not a bridge
    assert "way/13" not in by_id  # service roads are not in the network


def test_place_features_use_centre_and_skip_unnamed_or_unknown() -> None:
    by_id = {f.osm_id: f for f in place_features(PLACES_XML)}
    assert by_id["node/20"].kind == "mall"
    assert by_id["way/21"].kind == "lake" and by_id["way/21"].lat == 13.04
    assert "node/22" not in by_id and "node/23" not in by_id


def test_places_query_pinned() -> None:
    query = places_query((13.0, 77.5, 13.1, 77.6))
    assert '[date:"2026-10-01T00:00:00Z"]' in query and "out center tags" in query


def test_text_matching() -> None:
    assert normalise("ORR pe, accident!") == "outer ring road pe accident"
    assert kinds_in("bridge ke paas") == {"bridge", "flyover"}
    assert kinds_in("petrol pump ke saamne") == {"fuel"}
    assert kinds_in("bus stop") == {"bus_stop"} and kinds_in("stop karo") == set()
    text = "hebal flyover pe"
    assert name_matches("Hebbal Flyover", text, text.split())  # small misspelling
    text = "1st cross pe"
    assert not name_matches("1st Cross Road", text, text.split())  # too generic to fuzz


def _anchor(edge: str, x: float, y: float, heading: float) -> Anchor:
    return Anchor(edge_id=edge, pos_m=10.0, x=x, y=y, heading_deg=heading)


def _reporter(
    x: float = 0.0, y: float = 0.0, heading: float = 0.0, edge: str = "n0"
) -> VehicleState:
    return VehicleState(
        vehicle_id="rep", x=x, y=y, edge_id=edge, lane_pos_m=0, heading_deg=heading, speed_mps=10
    )


# A north-south dual carriageway: "n*" edges run north (0), "s*" edges run south (180).
GAZ = Gazetteer(
    corridor="test",
    snapshot="2026-10-01T00:00:00Z",
    landmarks=(
        Landmark(
            landmark_id="way/1",
            names=(),
            kind="bridge",
            x=0,
            y=400,
            anchors=(_anchor("n1", 0, 400, 0), _anchor("s1", 8, 400, 180)),
        ),
        Landmark(
            landmark_id="way/2",
            names=(),
            kind="bridge",
            x=0,
            y=-300,
            anchors=(_anchor("n-1", 0, -300, 0), _anchor("s-1", 8, -300, 180)),
        ),
        Landmark(
            landmark_id="node/3",
            names=("Esteem Mall",),
            kind="mall",
            x=0,
            y=1500,
            anchors=(_anchor("n2", 0, 1500, 0),),
        ),
    ),
)


def test_generic_word_picks_landmark_ahead_on_own_carriageway() -> None:
    loc = GAZ.resolve("bridge ke paas", _reporter(heading=0))
    assert loc is not None and loc.edge_id == "n1" and not loc.approx


def test_heading_south_picks_the_other_bridge_and_carriageway() -> None:
    loc = GAZ.resolve("bridge ke paas", _reporter(x=8, heading=180, edge="s0"))
    assert loc is not None and loc.edge_id == "s-1"


def test_named_landmark_beats_nearer_generic_one() -> None:
    loc = GAZ.resolve("esteem mall wala bridge", _reporter())
    assert loc is not None and loc.edge_id == "n2"


def test_no_match_or_too_far_returns_none() -> None:
    assert GAZ.resolve("kahin pe", _reporter()) is None
    assert GAZ.resolve(None, _reporter()) is None
    assert GAZ.resolve("bridge", _reporter(y=5000)) is None


def test_save_load_round_trip(tmp_path: Path) -> None:
    GAZ.save(tmp_path / "g.json")
    assert Gazetteer.load(tmp_path / "g.json") == GAZ


@pytest.mark.skipif(
    importlib.util.find_spec("sumolib") is None or not os.environ.get("V2V_NETWORK_TESTS"),
    reason="needs the [sim] extras and network access; set V2V_NETWORK_TESTS=1",
)
def test_hebbal_gazetteer(tmp_path: Path) -> None:
    from v2v.sim.gazetteer import angle_diff, build_gazetteer, shape_heading
    from v2v.sim.osm_network import CORRIDORS, read_net

    gazetteer = build_gazetteer(CORRIDORS["hebbal"], tmp_path)
    kinds = {lm.kind for lm in gazetteer.landmarks}
    assert {"flyover", "bridge", "signal", "road"} <= kinds
    assert any("Hebbal Flyover" in lm.names for lm in gazetteer.landmarks)

    net = read_net(tmp_path / "hebbal" / "hebbal.net.xml")
    for edge in [e for e in net.getEdges() if e.getName() == "Bellary Road"][:3]:
        shape = edge.getLanes()[0].getShape()
        heading = shape_heading(shape, 0.0)
        reporter = VehicleState(
            vehicle_id="rep",
            x=shape[0][0],
            y=shape[0][1],
            edge_id=edge.getID(),
            lane_pos_m=0,
            heading_deg=heading,
            speed_mps=10,
        )
        loc = gazetteer.resolve("bridge ke paas", reporter)
        assert loc is not None
        loc_shape = net.getEdge(loc.edge_id).getLanes()[0].getShape()
        assert angle_diff(shape_heading(loc_shape, loc.pos_m), heading) <= 90
        assert gazetteer.resolve("kahin pe", reporter) is None
