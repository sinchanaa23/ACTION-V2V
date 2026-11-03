"""Proposed distribution: direction-sensitive geocast. Owner: P4 (P4-05).

Uses only what a real V2V receiver shares, its map-matched position and
heading, plus the public road map. A warning covers a *road zone*: stretches
of directed road (SUMO edges are one carriageway, one direction of travel)
within ``radius_m`` of the hazard by driving distance, in one of three
directions:

- ``backward`` (the default): traffic behind the hazard point that is driving
  towards it on its carriageway, plus side streets and ramps that lead onto
  it. For anything that blocks or slows the road ahead of the cars that need
  to know: an accident, a blockage, a stopped or hard-braking vehicle.
- ``forward``: traffic ahead of the point on the same carriageway, driving
  away from it. For a hazard that moves on into that traffic, such as a
  vehicle that needs the road cleared ahead of it.
- ``bidirectional``: ``backward`` on both carriageways at the point, the
  opposite one found from the road geometry. For a hazard on the whole road
  surface (a slippery road).

``select`` (the policy in ``SELECTORS``) picks the direction from the hazard
type (``REACH_BY_HAZARD``). Opposite-direction lanes are separate edges, so
they fall outside a one-carriageway zone unless the road map says they lead
back into it within the radius (e.g. through an interchange): at Hebbal,
dropping everything heading the other way raised precision but cut recall
from 69% to 45%, so legal loops stay in. The contract's ``VehicleState`` has
no lane index, so every lane of a carriageway is covered; a lane block
concerns the whole carriageway anyway, since its traffic must merge round it.

Without a road map the heading rule stands in: a vehicle within radius R is
kept when the hazard lies ahead of it (bearing within AHEAD_TOLERANCE_DEG of
its heading; behind it for ``forward``) and it travels the hazard's way
(heading within SAME_DIRECTION_DEG of the road direction at the hazard, if
known; either way for ``bidirectional``).

Receivers' planned routes are deliberately not used: real vehicles don't
broadcast them, and they are exactly the ground truth this method is scored
against. Vehicles in the zone that will turn off before the hazard are the
method's honest false positives.
"""

import math
import threading
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Literal

from v2v.contracts import AssessedHazard, HazardType, ResolvedLocation, VehicleState
from v2v.sim.gazetteer import angle_diff, bearing_deg

AHEAD_TOLERANCE_DEG = 90.0
SAME_DIRECTION_DEG = 60.0
OPPOSITE_DIRECTION_DEG = 135.0
OPPOSITE_SEARCH_M = 60.0
AT_HAZARD_M = 5.0

Reach = Literal["backward", "forward", "bidirectional"]
REACHES: tuple[Reach, ...] = ("backward", "forward", "bidirectional")

REACH_BY_HAZARD: dict[HazardType, Reach] = {
    HazardType.ACCIDENT: "backward",
    HazardType.OBSTRUCTION: "backward",
    HazardType.STOPPED: "backward",
    HazardType.SPEEDING: "backward",  # relevance is reaching the reported spot
    HazardType.SLIPPERY: "bidirectional",  # the surface, on both carriageways
}

Point = tuple[float, float]


def reach_for(hazard: HazardType) -> Reach:
    return REACH_BY_HAZARD.get(hazard, "backward")


@dataclass(frozen=True, eq=False)
class RoadGraph:
    """The public road map as directed edges: what leads into what, how long,
    and (optionally) where each edge runs."""

    length_m: dict[str, float]
    incoming: dict[str, tuple[str, ...]] = field(default_factory=dict)
    outgoing: dict[str, tuple[str, ...]] = field(default_factory=dict)
    shapes: dict[str, tuple[Point, ...]] = field(default_factory=dict)

    @classmethod
    def from_edges(
        cls,
        edges: Iterable[tuple[str, float, Iterable[str]]],
        shapes: dict[str, Sequence[Point]] | None = None,
    ) -> "RoadGraph":
        """From (edge_id, length_m, outgoing edge ids) triples."""
        length: dict[str, float] = {}
        incoming: dict[str, list[str]] = defaultdict(list)
        outgoing: dict[str, tuple[str, ...]] = {}
        for edge_id, edge_length, nxt in edges:
            length[edge_id] = edge_length
            outgoing[edge_id] = tuple(nxt)
            for n in outgoing[edge_id]:
                incoming[n].append(edge_id)
        return cls(
            length,
            {e: tuple(src) for e, src in incoming.items()},
            outgoing,
            {e: tuple(s) for e, s in (shapes or {}).items()},
        )

    @classmethod
    def from_sumo_net(cls, net: Any) -> "RoadGraph":
        edges = [e for e in net.getEdges() if e.allows("passenger")]
        return cls.from_edges(
            ((e.getID(), e.getLength(), [o.getID() for o in e.getOutgoing()]) for e in edges),
            {e.getID(): [(x, y) for x, y, *_ in e.getShape()] for e in edges},
        )

    def upstream(self, edge_id: str, pos_m: float, max_m: float) -> dict[str, float]:
        """Edges leading to ``(edge_id, pos_m)``, each mapped to the driving
        distance from that edge's *end* to the point, up to ``max_m``. The
        point's own edge is not included."""
        return dict(_upstream(self, edge_id, round(pos_m, 1), max_m))

    def downstream(self, edge_id: str, pos_m: float, max_m: float) -> dict[str, float]:
        """Edges reachable from ``(edge_id, pos_m)``, each mapped to the driving
        distance from the point to that edge's *start*, up to ``max_m``."""
        return dict(_downstream(self, edge_id, round(pos_m, 1), max_m))

    def opposite(self, edge_id: str, x: float, y: float) -> tuple[str, float] | None:
        """The other carriageway beside a point on ``edge_id``: the nearest edge
        within OPPOSITE_SEARCH_M running the other way, and the position on it
        level with the point. None without shapes or when there is none."""
        own = self.shapes.get(edge_id)
        if not own:
            return None
        heading = _closest(own, x, y)[2]
        best: tuple[float, str, float] | None = None
        for other, shape in self.shapes.items():
            if other == edge_id or len(shape) < 2:
                continue
            dist, offset, other_heading = _closest(shape, x, y)
            turned = angle_diff(other_heading, heading)
            if dist > OPPOSITE_SEARCH_M or turned < OPPOSITE_DIRECTION_DEG:
                continue
            if best is None or dist < best[0]:
                scale = self.length_m.get(other, offset) / max(_length(shape), 1e-6)
                best = (dist, other, min(offset * scale, self.length_m.get(other, offset)))
        return None if best is None else (best[1], best[2])


def _length(shape: Sequence[Point]) -> float:
    return sum(math.dist(a, b) for a, b in zip(shape, shape[1:], strict=False))


def _closest(shape: Sequence[Point], x: float, y: float) -> tuple[float, float, float]:
    """(distance to the polyline, offset along it, heading there) for a point."""
    best = (math.inf, 0.0, 0.0)
    walked = 0.0
    for (ax, ay), (bx, by) in zip(shape, shape[1:], strict=False):
        seg = math.hypot(bx - ax, by - ay)
        if seg == 0:
            continue
        t = max(0.0, min(1.0, ((x - ax) * (bx - ax) + (y - ay) * (by - ay)) / (seg * seg)))
        px, py = ax + (bx - ax) * t, ay + (by - ay) * t
        d = math.hypot(x - px, y - py)
        if d < best[0]:
            best = (d, walked + t * seg, bearing_deg(ax, ay, bx, by))
        walked += seg
    return best


@lru_cache(maxsize=256)
def _upstream(
    graph: RoadGraph, edge_id: str, pos_m: float, max_m: float
) -> tuple[tuple[str, float], ...]:
    # Dijkstra backwards: to_point[e] = road distance from the end of e to the point.
    to_point: dict[str, float] = {}
    frontier = [(pos_m, prev) for prev in graph.incoming.get(edge_id, ())]
    while frontier:
        frontier.sort()
        dist, edge = frontier.pop(0)
        if dist > max_m or edge == edge_id or dist >= to_point.get(edge, math.inf):
            continue
        to_point[edge] = dist
        for prev in graph.incoming.get(edge, ()):
            frontier.append((dist + graph.length_m.get(edge, 0.0), prev))
    return tuple(to_point.items())


@lru_cache(maxsize=256)
def _downstream(
    graph: RoadGraph, edge_id: str, pos_m: float, max_m: float
) -> tuple[tuple[str, float], ...]:
    # Dijkstra forwards: from_point[e] = road distance from the point to the start of e.
    from_point: dict[str, float] = {}
    rest = max(0.0, graph.length_m.get(edge_id, pos_m) - pos_m)
    frontier = [(rest, nxt) for nxt in graph.outgoing.get(edge_id, ())]
    while frontier:
        frontier.sort()
        dist, edge = frontier.pop(0)
        if dist > max_m or edge == edge_id or dist >= from_point.get(edge, math.inf):
            continue
        from_point[edge] = dist
        for nxt in graph.outgoing.get(edge, ()):
            frontier.append((dist + graph.length_m.get(edge, 0.0), nxt))
    return tuple(from_point.items())


Zone = dict[str, tuple[float, float]]
"""Edge id -> the stretch of it (from, to metres along the edge) inside the zone."""


def _merge(zone: Zone, edge: str, lo: float, hi: float) -> None:
    if hi < lo:
        return
    if edge in zone:
        a, b = zone[edge]
        lo, hi = min(a, lo), max(b, hi)
    zone[edge] = (lo, hi)


def road_zone(graph: RoadGraph, loc: ResolvedLocation, radius_m: float, reach: Reach) -> Zone:
    """The stretches of road a geocast in direction ``reach`` covers."""
    zone: Zone = {}
    points = [(loc.edge_id, loc.pos_m)]
    if reach == "bidirectional":
        other = graph.opposite(loc.edge_id, loc.x, loc.y)
        if other is not None:
            points.append(other)
    for edge, pos in points:
        length = graph.length_m.get(edge, pos)
        if reach == "forward":
            _merge(zone, edge, pos, min(length, pos + radius_m))
            for nxt, d in graph.downstream(edge, pos, radius_m).items():
                _merge(zone, nxt, 0.0, min(graph.length_m[nxt], radius_m - d))
        else:
            _merge(zone, edge, max(0.0, pos - radius_m), pos)
            for prev, d in graph.upstream(edge, pos, radius_m).items():
                full = graph.length_m[prev]
                _merge(zone, prev, max(0.0, full - (radius_m - d)), full)
    return zone


def in_zone(zone: Zone, edge_id: str, pos_m: float) -> bool:
    span = zone.get(edge_id)
    return span is not None and span[0] - 1e-6 <= pos_m <= span[1] + 1e-6


_road_graph: RoadGraph | None = None
_users: dict[int, RoadGraph] = {}
_lock = threading.Lock()


def use_road_graph(graph: RoadGraph | None) -> None:
    """Set the road map used by ``select``. With None, ``select`` falls back
    to the heading rule (and any registered users are forgotten)."""
    global _road_graph
    with _lock:
        _road_graph = graph
        if graph is None:
            _users.clear()


def acquire_road_graph(owner: object, graph: RoadGraph) -> None:
    """A running world registers the road map it plays on."""
    global _road_graph
    with _lock:
        _users[id(owner)] = graph
        _road_graph = graph


def release_road_graph(owner: object) -> None:
    """A world closing: the map stays while any other world still uses one,
    so one session ending never strands another on the heading rule."""
    global _road_graph
    with _lock:
        _users.pop(id(owner), None)
        _road_graph = next(reversed(_users.values())) if _users else None


def current_road_graph() -> RoadGraph | None:
    return _road_graph


def keeps(
    vehicle: VehicleState,
    hx: float,
    hy: float,
    road_heading: float | None,
    reach: Reach = "backward",
) -> bool:
    """The heading rule for one vehicle (the radius is checked by ``select``)."""
    if road_heading is not None:
        off = angle_diff(vehicle.heading_deg, road_heading)
        same = off <= SAME_DIRECTION_DEG
        opposite = off >= 180.0 - SAME_DIRECTION_DEG
        if not (same or (reach == "bidirectional" and opposite)):
            return False
    if math.dist((vehicle.x, vehicle.y), (hx, hy)) <= AT_HAZARD_M:
        return True
    ahead = angle_diff(bearing_deg(vehicle.x, vehicle.y, hx, hy), vehicle.heading_deg)
    if reach == "forward":
        return ahead >= 180.0 - AHEAD_TOLERANCE_DEG  # the point is behind it
    return ahead <= AHEAD_TOLERANCE_DEG


def select_reach(
    loc: ResolvedLocation,
    vehicles: list[VehicleState],
    radius_m: float,
    reach: Reach,
    graph: RoadGraph | None = None,
    exclude: str | None = None,
) -> list[str]:
    """Vehicles inside the road zone in direction ``reach`` (and within radius
    in a straight line, since that is how far the geocast area extends)."""
    in_range = [
        v
        for v in vehicles
        if v.vehicle_id != exclude and math.dist((v.x, v.y), (loc.x, loc.y)) <= radius_m
    ]
    graph = graph or _road_graph
    if graph is None or loc.edge_id not in graph.length_m:
        return [v.vehicle_id for v in in_range if keeps(v, loc.x, loc.y, loc.heading_deg, reach)]
    zone = road_zone(graph, loc, radius_m, reach)
    return [v.vehicle_id for v in in_range if in_zone(zone, v.edge_id, v.lane_pos_m)]


def select(hazard: AssessedHazard, vehicles: list[VehicleState], radius_m: float) -> list[str]:
    loc = hazard.location
    if loc is None:
        raise ValueError("geocast needs a location; apply the fallback policy first")
    return select_reach(loc, vehicles, radius_m, reach_for(hazard.hazard))
