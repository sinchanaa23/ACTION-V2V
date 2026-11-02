"""Scene geometry for the 3D simulation, from the real corridor. Owner: P4.

    python -m v2v.sim.world3d --corridor hebbal     # fetch buildings, water, parks

Roads come from the SUMO network (every lane, with its real width), lifted
onto the flyovers using OSM's ``bridge``/``layer`` tags: bridge edges sit at
``layer * DECK_HEIGHT_M`` and the roads meeting them ramp up to the deck.
Buildings, water, parks, trees and street lamps come from OpenStreetMap for
the same bbox and snapshot date as the road network. Buildings without a
height or level count use ``DEFAULT_BUILDING_M``.

Everything is in SUMO network coordinates (metres), so vehicle positions from
TraCI drop straight in. No rendering engine is imported here.
"""

import argparse
import math
import re
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from v2v.sim import osm_network
from v2v.sim.osm_network import CORRIDORS, OSM_SNAPSHOT, Corridor

DECK_HEIGHT_M = 7.0
"""Height of one OSM layer of flyover deck above the ground."""
DEFAULT_BUILDING_M = 9.0
LEVEL_M = 3.2


@dataclass(frozen=True)
class Lane:
    lane_id: str
    edge_id: str
    road_class: str
    width_m: float
    points: tuple[tuple[float, float, float], ...]
    """(x, y, z) along the lane centre."""


@dataclass(frozen=True)
class Area:
    kind: str
    """"building", "water" or "park"."""
    outline: tuple[tuple[float, float], ...]
    height_m: float = 0.0


@dataclass(frozen=True)
class Spot:
    kind: str
    """"tree", "street_lamp", "traffic_signals" or "bus_stop"."""
    x: float
    y: float


@dataclass
class Scene:
    lanes: list[Lane] = field(default_factory=list)
    junctions: list[tuple[tuple[float, float], ...]] = field(default_factory=list)
    junction_z: list[float] = field(default_factory=list)
    areas: list[Area] = field(default_factory=list)
    spots: list[Spot] = field(default_factory=list)
    bounds: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    """(xmin, ymin, xmax, ymax) of the network."""
    edge_z: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    """Edge id -> (z at start, z at end, length): for placing vehicles."""

    def elevation(self, edge_id: str, pos_m: float) -> float:
        """Road height at ``pos_m`` along ``edge_id`` (0 for unknown edges)."""
        z0, z1, length = self.edge_z.get(edge_id, (0.0, 0.0, 1.0))
        if not math.isfinite(pos_m):
            return z1
        f = min(max(pos_m / length, 0.0), 1.0) if length > 0 else 0.0
        return z0 + (z1 - z0) * f


# --- OSM context -----------------------------------------------------------------------


CONTEXT_PARTS = {
    "buildings": 'way["building"];',
    "land": (
        'way["natural"="water"];way["water"];way["landuse"="reservoir"];'
        'way["leisure"="park"];way["landuse"~"^(grass|forest|recreation_ground)$"];'
    ),
    "spots": 'node["natural"="tree"];node["highway"~"^(street_lamp|traffic_signals|bus_stop)$"];',
}
"""One Overpass query each: a single historical query for all of them runs the
public server out of memory."""
OPTIONAL_PARTS = {"spots"}
"""The public server often cannot run historical node-tag queries; signals and
bus stops then come from the road extract alone."""


def context_query(
    part: str, bbox: tuple[float, float, float, float], snapshot: str = OSM_SNAPSHOT
) -> str:
    south, west, north, east = bbox
    return (
        f'[out:xml][timeout:180][maxsize:536870912][date:"{snapshot}"]'
        f"[bbox:{south},{west},{north},{east}];({CONTEXT_PARTS[part]});(._;>;);out body;"
    )


def context_paths(corridor: Corridor, out_dir: Path = osm_network.DEFAULT_OUT_DIR) -> list[Path]:
    return [out_dir / corridor.name / f"{corridor.name}.{part}.osm" for part in CONTEXT_PARTS]


def download_context(
    corridor: Corridor, out_dir: Path = osm_network.DEFAULT_OUT_DIR, force: bool = False
) -> list[Path]:
    paths = context_paths(corridor, out_dir)
    for part, path in zip(CONTEXT_PARTS, paths, strict=True):
        if path.exists() and not force:
            continue
        body = osm_network.overpass_download(context_query(part, corridor.bbox), path)
        if b"runtime error" in body:
            path.unlink()
            if part in OPTIONAL_PARTS:  # the scene still has the road extract's own nodes
                print(f"skipped {part}: Overpass could not serve it ({body[-160:]!r})")
                continue
            raise RuntimeError(f"Overpass failed on {part}: {body[-300:]!r}")
    return paths


def _tags(element: ET.Element) -> dict[str, str]:
    return {t.get("k", ""): t.get("v", "") for t in element.iter("tag")}


def building_height(tags: dict[str, str]) -> float:
    """Metres from ``height`` or ``building:levels``; DEFAULT_BUILDING_M otherwise."""
    for key, scale in (("height", 1.0), ("building:height", 1.0), ("building:levels", LEVEL_M)):
        match = re.match(r"\s*([0-9]+(?:\.[0-9]+)?)", tags.get(key, ""))
        if match:
            return max(float(match.group(1)) * scale, 2.5)
    return DEFAULT_BUILDING_M


def area_kind(tags: dict[str, str]) -> str | None:
    if "building" in tags:
        return "building"
    if tags.get("natural") == "water" or "water" in tags or tags.get("landuse") == "reservoir":
        return "water"
    if tags.get("leisure") == "park" or tags.get("landuse") in {
        "grass",
        "forest",
        "recreation_ground",
    }:
        return "park"
    return None


def parse_context(xml: Path, to_xy: Any) -> tuple[list[Area], list[Spot]]:
    """Areas and spots from a context download, projected with ``to_xy(lon, lat)``."""
    root = ET.parse(xml).getroot()
    nodes: dict[str, tuple[float, float]] = {}
    spots: list[Spot] = []
    for n in root.iter("node"):
        lon, lat = float(n.get("lon", 0)), float(n.get("lat", 0))
        x, y = to_xy(lon, lat)
        nodes[n.get("id", "")] = (x, y)
        tags = _tags(n)
        kind = "tree" if tags.get("natural") == "tree" else tags.get("highway")
        if kind in {"tree", "street_lamp", "traffic_signals", "bus_stop"}:
            spots.append(Spot(kind, x, y))
    areas: list[Area] = []
    for w in root.iter("way"):
        tags = _tags(w)
        kind = area_kind(tags)
        refs = [nd.get("ref", "") for nd in w.iter("nd")]
        outline = tuple(nodes[r] for r in refs if r in nodes)
        if kind is None or len(outline) < 4 or refs[0] != refs[-1]:
            continue  # only closed ways are areas
        height = building_height(tags) if kind == "building" else 0.0
        areas.append(Area(kind, outline[:-1], height))
    return areas, spots


# --- roads and flyovers ------------------------------------------------------------------


def way_id(edge_id: str) -> str:
    """OSM way id behind a SUMO edge id such as ``-1382087677#1``."""
    return edge_id.lstrip("-").split("#", 1)[0]


def deck_layers(osm: Path) -> dict[str, int]:
    """OSM way id -> deck layer, for ways tagged as bridges (layer 1 if untagged)."""
    out: dict[str, int] = {}
    for w in ET.parse(osm).getroot().iter("way"):
        tags = _tags(w)
        if tags.get("bridge", "no") == "no":
            continue
        try:
            layer = int(tags.get("layer", "1"))
        except ValueError:
            layer = 1
        if layer > 0:
            out[w.get("id", "")] = layer
    return out


def node_heights(edges: Iterable[tuple[str, str, str]], layers: dict[str, int]) -> dict[str, float]:
    """Junction height: the highest deck that meets it, so approaches ramp up."""
    heights: dict[str, float] = {}
    for edge_id, start, end in edges:
        z = layers.get(way_id(edge_id), 0) * DECK_HEIGHT_M
        for node in (start, end):
            heights[node] = max(heights.get(node, 0.0), z)
    return heights


def _ramp(
    shape: list[tuple[float, float]], z0: float, z1: float
) -> tuple[tuple[float, float, float], ...]:
    lengths = [0.0]
    for a, b in zip(shape, shape[1:], strict=False):
        lengths.append(lengths[-1] + math.dist(a, b))
    total = lengths[-1] or 1.0
    return tuple(
        (x, y, z0 + (z1 - z0) * d / total) for (x, y), d in zip(shape, lengths, strict=True)
    )


def build_scene(net_path: Path, osm_path: Path, context: Iterable[Path] = ()) -> Scene:
    """Everything the 3D view draws, in network coordinates."""
    net = osm_network.read_net(net_path)
    layers = deck_layers(osm_path)
    edges = [e for e in net.getEdges() if e.getFunction() != "internal"]
    heights = node_heights(
        ((e.getID(), e.getFromNode().getID(), e.getToNode().getID()) for e in edges), layers
    )
    scene = Scene()
    (xmin, ymin), (xmax, ymax) = (net.getBoundary()[0:2], net.getBoundary()[2:4])
    scene.bounds = (xmin, ymin, xmax, ymax)
    for edge in edges:
        z0 = heights.get(edge.getFromNode().getID(), 0.0)
        z1 = heights.get(edge.getToNode().getID(), 0.0)
        scene.edge_z[edge.getID()] = (z0, z1, edge.getLength())
        road_class = (edge.getType() or "").split(".")[-1]
        for lane in edge.getLanes():
            if not lane.allows("passenger"):
                continue
            scene.lanes.append(
                Lane(
                    lane_id=lane.getID(),
                    edge_id=edge.getID(),
                    road_class=road_class,
                    width_m=lane.getWidth(),
                    points=_ramp([tuple(p[:2]) for p in lane.getShape()], z0, z1),
                )
            )
    for node in net.getNodes():
        shape = node.getShape()
        if len(shape) >= 3:
            scene.junctions.append(tuple((p[0], p[1]) for p in shape))
            scene.junction_z.append(heights.get(node.getID(), 0.0))
    for path in (osm_path, *context):  # the road extract carries signals and bus stops
        if path.exists():
            areas, spots = parse_context(path, net.convertLonLat2XY)
            scene.areas += areas
            scene.spots += spots
    return scene


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download the 3D context for a corridor")
    parser.add_argument("--corridor", choices=sorted(CORRIDORS), default="hebbal")
    parser.add_argument("--out", type=Path, default=osm_network.DEFAULT_OUT_DIR)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    corridor = CORRIDORS[args.corridor]
    paths = download_context(corridor, args.out, args.force)
    root = args.out / corridor.name
    scene = build_scene(root / f"{corridor.name}.net.xml", root / f"{corridor.name}.osm", paths)
    kinds: dict[str, int] = {}
    for a in scene.areas:
        kinds[a.kind] = kinds.get(a.kind, 0) + 1
    for s in scene.spots:
        kinds[s.kind] = kinds.get(s.kind, 0) + 1
    raised = sum(1 for z0, z1, _ in scene.edge_z.values() if max(z0, z1) > 0)
    print(f"{root}: {len(scene.lanes)} lanes, {raised} raised edges, {kinds}")


if __name__ == "__main__":
    main()
