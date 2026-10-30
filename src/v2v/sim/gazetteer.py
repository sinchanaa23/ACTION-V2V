"""Landmark list: real named places mapped to network edges. Owner: P4 (P4-02).

    python -m v2v.sim.gazetteer --corridor hebbal

Turns a location phrase from a report ("bridge", "Hebbal flyover", "Esteem
Mall") into a point on the SUMO network near the reporter. Every landmark comes
from OpenStreetMap at the pinned snapshot: bridges, flyovers, named roads and
traffic signals from the road download, plus named places (malls, lakes,
stations, ...) from a second Overpass query. Nothing is hand-placed.

Map data (c) OpenStreetMap contributors, ODbL.
"""

import argparse
import math
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from v2v.contracts import ResolvedLocation, VehicleState
from v2v.sim import osm_network
from v2v.sim.osm_network import CORRIDORS, DEFAULT_OUT_DIR, OSM_SNAPSHOT, ROAD_CLASSES, Corridor

ROAD_ANCHOR_RADIUS_M = 40.0  # signal and junction nodes sit on the road
PLACE_ANCHOR_RADIUS_M = 150.0  # malls, lakes and stations are set back from it
MAX_ANCHORS = 8
NAME_SEARCH_RADIUS_M = 2500.0
KIND_SEARCH_RADIUS_M = 1200.0
BEHIND_PENALTY_M = 600.0  # drivers say "aage" (ahead): prefer landmarks in front
OPPOSITE_PENALTY_M = 300.0  # prefer the reporter's own carriageway over the other one
SIMILAR_TOKEN_RATIO = 0.8

# Generic place words drivers use (English and Hinglish), matched on word
# boundaries, mapped to the landmark kinds they can refer to.
KIND_WORDS: dict[str, tuple[str, ...]] = {
    "bridge": ("bridge", "flyover"),
    "pul": ("bridge", "flyover"),
    "flyover": ("flyover", "bridge"),
    "signal": ("signal",),
    "traffic light": ("signal",),
    "junction": ("junction", "signal"),
    "circle": ("junction", "signal"),
    "chowk": ("junction", "signal"),
    "chauraha": ("junction", "signal"),
    "mall": ("mall",),
    "lake": ("lake",),
    "talab": ("lake",),
    "talaab": ("lake",),
    "park": ("park",),
    "petrol pump": ("fuel",),
    "petrol bunk": ("fuel",),
    "hospital": ("hospital",),
    "school": ("school",),
    "college": ("college",),
    "temple": ("place_of_worship",),
    "mandir": ("place_of_worship",),
    "masjid": ("place_of_worship",),
    "church": ("place_of_worship",),
    "metro": ("station",),
    "station": ("station",),
    "bus stop": ("bus_stop",),
    "police": ("police",),
    "thana": ("police",),
}

# Common spoken short forms, expanded before matching.
ABBREVIATIONS: dict[str, str] = {"orr": "outer ring road"}

# Tokens too common in road names to identify a landmark on their own.
GENERIC_NAME_TOKENS = {"road", "rd", "main", "cross", "street", "st", "the", "of", "and"}

BRIDGE_VALUES = {"yes", "viaduct", "aqueduct"}
AMENITY_KINDS = {
    "fuel": "fuel",
    "hospital": "hospital",
    "clinic": "hospital",
    "school": "school",
    "college": "college",
    "university": "college",
    "place_of_worship": "place_of_worship",
    "police": "police",
    "bus_station": "bus_stop",
    "marketplace": "market",
    "cinema": "cinema",
}
PLACE_VALUES = ("neighbourhood", "suburb", "quarter", "locality", "village")
NAME_TAGS = ("name", "name:en", "alt_name", "short_name", "old_name", "ref")


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class OsmFeature(_Model):
    """A landmark candidate read from OSM, before it is tied to the network."""

    osm_id: str
    names: tuple[str, ...]
    kind: str
    lat: float
    lon: float
    way_id: str | None = None
    """Set for road features: the OSM way the network edges came from."""


class Anchor(_Model):
    """The point on one network edge closest to a landmark."""

    edge_id: str
    pos_m: float
    x: float
    y: float
    heading_deg: float


class Landmark(_Model):
    landmark_id: str
    names: tuple[str, ...]
    kind: str
    x: float
    y: float
    anchors: tuple[Anchor, ...]


class Gazetteer(_Model):
    corridor: str
    snapshot: str
    landmarks: tuple[Landmark, ...]

    def resolve(self, location_text: str | None, reporter: VehicleState) -> ResolvedLocation | None:
        """Map a stated location to the network, or None if nothing matches.

        A landmark named in the text wins over a generic word ("bridge"); among
        candidates the nearest one ahead of the reporter wins. None means the
        caller must use the fallback policy; a location is never guessed.
        """
        text = normalise(location_text or "")
        if not text:
            return None
        tokens = text.split()
        named = [
            lm for lm in self.landmarks if any(name_matches(n, text, tokens) for n in lm.names)
        ]
        if named:
            pool, radius = named, NAME_SEARCH_RADIUS_M
        else:
            kinds = kinds_in(text)
            pool = [lm for lm in self.landmarks if lm.kind in kinds]
            radius = KIND_SEARCH_RADIUS_M
        scored = [
            (score(anchor, reporter), anchor)
            for lm in pool
            if math.dist((lm.x, lm.y), (reporter.x, reporter.y)) <= radius
            for anchor in [pick_anchor(lm, reporter)]
        ]
        if not scored:
            return None
        anchor = min(scored, key=lambda pair: pair[0])[1]
        return ResolvedLocation(
            edge_id=anchor.edge_id, pos_m=anchor.pos_m, x=anchor.x, y=anchor.y, approx=False
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Gazetteer":
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


# --- Text matching -----------------------------------------------------------

_NON_WORD = re.compile(r"[^\w\s]")


def normalise(text: str) -> str:
    words = _NON_WORD.sub(" ", text.lower()).split()
    return " ".join(ABBREVIATIONS.get(w, w) for w in words)


def kinds_in(text: str) -> set[str]:
    return {
        kind
        for phrase, kinds in KIND_WORDS.items()
        if re.search(rf"\b{re.escape(phrase)}\b", text)
        for kind in kinds
    }


def _similar(a: str, b: str) -> bool:
    return a == b or SequenceMatcher(None, a, b).ratio() >= SIMILAR_TOKEN_RATIO


def name_matches(name: str, text: str, tokens: list[str]) -> bool:
    """True if ``name`` is stated in ``text``, allowing small misspellings of
    distinctive words ("hebal flyover")."""
    name_norm = normalise(name)
    if not name_norm:
        return False
    if re.search(rf"\b{re.escape(name_norm)}\b", text):
        return True
    distinctive = [t for t in name_norm.split() if t not in GENERIC_NAME_TOKENS]
    if not distinctive or (len(distinctive) == 1 and len(distinctive[0]) < 5):
        return False  # e.g. "1st Cross Road": only an exact mention counts
    return all(any(_similar(d, t) for t in tokens) for d in distinctive)


# --- Geometry ----------------------------------------------------------------


def bearing_deg(from_x: float, from_y: float, to_x: float, to_y: float) -> float:
    """Compass bearing in SUMO's convention: 0 = north, 90 = east."""
    return math.degrees(math.atan2(to_x - from_x, to_y - from_y)) % 360.0


def angle_diff(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def score(anchor: Anchor, reporter: VehicleState) -> float:
    """Lower is better: distance, plus penalties for being behind the reporter
    or on a carriageway running the other way."""
    distance = math.dist((anchor.x, anchor.y), (reporter.x, reporter.y))
    to_anchor = bearing_deg(reporter.x, reporter.y, anchor.x, anchor.y)
    behind = distance > 1.0 and angle_diff(to_anchor, reporter.heading_deg) > 90.0
    opposite = angle_diff(anchor.heading_deg, reporter.heading_deg) > 90.0
    return (
        distance + (BEHIND_PENALTY_M if behind else 0.0) + (OPPOSITE_PENALTY_M if opposite else 0.0)
    )


def pick_anchor(lm: Landmark, reporter: VehicleState) -> Anchor:
    """Prefer the reporter's own route, then edges running the reporter's way,
    then the nearest edge."""

    def key(a: Anchor) -> tuple[bool, bool, float]:
        on_route = a.edge_id == reporter.edge_id or a.edge_id in reporter.route_ahead
        aligned = angle_diff(a.heading_deg, reporter.heading_deg) <= 90.0
        return (not on_route, not aligned, math.dist((a.x, a.y), (reporter.x, reporter.y)))

    return min(lm.anchors, key=key)


def _shape_heading(shape: list[tuple[float, float]], offset: float) -> float:
    walked = 0.0
    for (x0, y0), (x1, y1) in zip(shape, shape[1:], strict=False):
        length = math.dist((x0, y0), (x1, y1))
        if walked + length >= offset or (x1, y1) == shape[-1]:
            return bearing_deg(x0, y0, x1, y1)
        walked += length
    return 0.0


# --- Reading OSM ---------------------------------------------------------------


def _tags(element: ET.Element) -> dict[str, str]:
    return {t.get("k", ""): t.get("v", "") for t in element.findall("tag")}


def _names(tags: dict[str, str]) -> tuple[str, ...]:
    seen: list[str] = []
    for key in NAME_TAGS:
        for value in tags.get(key, "").split(";"):
            if value.strip() and value.strip() not in seen:
                seen.append(value.strip())
    return tuple(seen)


def road_features(osm_xml: bytes) -> list[OsmFeature]:
    """Bridges, flyovers, named roads, signals and named junctions."""
    root = ET.fromstring(osm_xml)
    coords: dict[str, tuple[float, float]] = {}
    features: list[OsmFeature] = []
    for node in root.iter("node"):
        node_id = node.get("id", "")
        lat, lon = float(node.get("lat", "nan")), float(node.get("lon", "nan"))
        coords[node_id] = (lat, lon)
        tags = _tags(node)
        names = _names(tags)
        if tags.get("highway") == "traffic_signals":
            kind = "signal"
        elif names and ("junction" in tags or tags.get("highway") == "motorway_junction"):
            kind = "junction"
        else:
            continue
        features.append(
            OsmFeature(osm_id=f"node/{node_id}", names=names, kind=kind, lat=lat, lon=lon)
        )
    for way in root.iter("way"):
        tags = _tags(way)
        if tags.get("highway") not in ROAD_CLASSES:
            continue
        names = _names(tags)
        is_bridge = tags.get("bridge") in BRIDGE_VALUES
        if not (is_bridge or names):
            continue
        refs = [nd.get("ref", "") for nd in way.findall("nd")]
        points = [coords[r] for r in refs if r in coords]
        if not points:
            continue
        lat, lon = points[len(points) // 2]
        if is_bridge:
            kind = "flyover" if any("flyover" in n.lower() for n in names) else "bridge"
        else:
            kind = "road"
        way_id = way.get("id", "")
        features.append(
            OsmFeature(
                osm_id=f"way/{way_id}", names=names, kind=kind, lat=lat, lon=lon, way_id=way_id
            )
        )
    return features


def place_kind(tags: dict[str, str]) -> str | None:
    if tags.get("shop") == "mall":
        return "mall"
    if tags.get("natural") == "water" or tags.get("water") in {"lake", "pond", "reservoir"}:
        return "lake"
    if tags.get("leisure") == "park":
        return "park"
    if tags.get("amenity") in AMENITY_KINDS:
        return AMENITY_KINDS[tags["amenity"]]
    if tags.get("railway") in {"station", "halt"} or tags.get("public_transport") == "station":
        return "station"
    if tags.get("highway") == "bus_stop":
        return "bus_stop"
    if tags.get("place") in PLACE_VALUES:
        return "area"
    return None


def place_features(osm_xml: bytes) -> list[OsmFeature]:
    """Named places from ``places_query`` (``out center``: ways carry a centre)."""
    features: list[OsmFeature] = []
    for element in ET.fromstring(osm_xml):
        if element.tag not in {"node", "way", "relation"}:
            continue
        tags = _tags(element)
        names, kind = _names(tags), place_kind(tags)
        point = element if element.tag == "node" else element.find("center")
        if not names or kind is None or point is None:
            continue
        features.append(
            OsmFeature(
                osm_id=f"{element.tag}/{element.get('id')}",
                names=names,
                kind=kind,
                lat=float(point.get("lat", "nan")),
                lon=float(point.get("lon", "nan")),
            )
        )
    return features


def places_query(bbox: tuple[float, float, float, float], snapshot: str = OSM_SNAPSHOT) -> str:
    south, west, north, east = bbox
    amenities = "|".join(AMENITY_KINDS)
    places = "|".join(PLACE_VALUES)
    selectors = [
        'nwr["name"]["shop"="mall"]',
        'nwr["name"]["natural"="water"]',
        'nwr["name"]["leisure"="park"]',
        f'nwr["name"]["amenity"~"^({amenities})$"]',
        'nwr["name"]["railway"~"^(station|halt)$"]',
        'nwr["name"]["public_transport"="station"]',
        'node["name"]["highway"="bus_stop"]',
        f'node["name"]["place"~"^({places})$"]',
    ]
    return (
        f'[out:xml][timeout:180][date:"{snapshot}"][bbox:{south},{west},{north},{east}];'
        f"({';'.join(selectors)};);out center tags;"
    )


# --- Tying landmarks to the network ----------------------------------------------


def edges_by_way(net: Any) -> dict[str, list[Any]]:
    """OSM way id -> network edges built from it (netconvert's ``origId``)."""
    index: dict[str, list[Any]] = defaultdict(list)
    for edge in net.getEdges():
        for way_id in edge.getLanes()[0].getParam("origId", "").split():
            index[way_id.lstrip("-")].append(edge)
    return index


def anchor_on(edge: Any, x: float, y: float) -> tuple[float, Anchor]:
    """(distance, anchor) for the point on ``edge`` closest to (x, y)."""
    from sumolib import geomhelper

    lane = edge.getLanes()[0]
    shape = lane.getShape()
    offset = geomhelper.polygonOffsetWithMinimumDistanceToPoint((x, y), shape)
    px, py = geomhelper.positionAtShapeOffset(shape, offset)
    shape_length = geomhelper.polyLength(shape) or 1.0
    pos = min(offset * lane.getLength() / shape_length, lane.getLength())
    anchor = Anchor(
        edge_id=edge.getID(),
        pos_m=round(pos, 2),
        x=round(px, 2),
        y=round(py, 2),
        heading_deg=round(_shape_heading(shape, offset), 1),
    )
    return math.dist((x, y), (px, py)), anchor


def to_landmark(feature: OsmFeature, net: Any, by_way: dict[str, list[Any]]) -> Landmark | None:
    """Anchor a feature to its edges; None if it doesn't touch the network."""
    x, y = net.convertLonLat2XY(feature.lon, feature.lat)
    if feature.way_id is not None:
        candidates = by_way.get(feature.way_id, [])
    else:
        radius = (
            ROAD_ANCHOR_RADIUS_M
            if feature.kind in {"signal", "junction"}
            else PLACE_ANCHOR_RADIUS_M
        )
        candidates = [edge for edge, _ in net.getNeighboringEdges(x, y, radius)]
    anchored = sorted((anchor_on(edge, x, y) for edge in candidates), key=lambda pair: pair[0])
    if not anchored:
        return None
    return Landmark(
        landmark_id=feature.osm_id,
        names=feature.names,
        kind=feature.kind,
        x=round(x, 2),
        y=round(y, 2),
        anchors=tuple(anchor for _, anchor in anchored[:MAX_ANCHORS]),
    )


def build_gazetteer(
    corridor: Corridor, out_dir: Path = DEFAULT_OUT_DIR, force: bool = False
) -> Gazetteer:
    """Build (or load) ``<corridor>/gazetteer.json`` from the real OSM data."""
    work = out_dir / corridor.name
    path = work / "gazetteer.json"
    if path.exists() and not force:
        return Gazetteer.load(path)
    net_path = osm_network.build_network(corridor, out_dir)
    places_path = work / f"{corridor.name}.places.osm"
    if force or not places_path.exists():
        osm_network.overpass_download(places_query(corridor.bbox), places_path)
    features = road_features((work / f"{corridor.name}.osm").read_bytes())
    features += place_features(places_path.read_bytes())
    net = osm_network.read_net(net_path)
    by_way = edges_by_way(net)
    landmarks = tuple(lm for f in features if (lm := to_landmark(f, net, by_way)) is not None)
    gazetteer = Gazetteer(corridor=corridor.name, snapshot=OSM_SNAPSHOT, landmarks=landmarks)
    gazetteer.save(path)
    return gazetteer


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build the landmark list for a corridor")
    parser.add_argument("--corridor", choices=sorted(CORRIDORS), default="hebbal")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--force", action="store_true", help="re-download and rebuild")
    args = parser.parse_args(argv)
    gazetteer = build_gazetteer(CORRIDORS[args.corridor], args.out, force=args.force)
    print(f"{len(gazetteer.landmarks)} landmarks (OSM snapshot {gazetteer.snapshot})")
    for kind, count in Counter(lm.kind for lm in gazetteer.landmarks).most_common():
        examples = sorted(
            {lm.names[0] for lm in gazetteer.landmarks if lm.kind == kind and lm.names}
        )
        print(f"  {kind:<16} {count:>4}  {', '.join(examples[:4])}")


if __name__ == "__main__":
    main()
