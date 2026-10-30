"""Proposed distribution: direction-sensitive geocast. Owner: P4 (P4-05).

Uses only what a real V2V receiver shares, its map-matched position and
heading, plus the public road map. A vehicle within radius R is kept when it
is *upstream* of the hazard:

- with a road map (``use_road_graph``): its current road segment leads to the
  hazard point within ``radius_m`` of driving, following the network's legal
  connections. This separates the carriageways of a divided road, drops
  traffic that has passed the hazard, and keeps cars approaching from side
  streets and ramps. Oncoming traffic that can legally loop round (e.g.
  through an interchange) within the radius is kept: at Hebbal, dropping
  everything heading the other way raised precision but cut recall from 69%
  to 45%, so it was rejected;
- without one: the hazard is ahead of it (bearing within AHEAD_TOLERANCE_DEG
  of its heading) and it travels the hazard's way (heading within
  SAME_DIRECTION_DEG of the road direction at the hazard, if known).

Receivers' planned routes are deliberately not used: real vehicles don't
broadcast them, and they are exactly the ground truth this method is scored
against. Vehicles that are upstream but will turn off before the hazard are
the method's honest false positives.
"""

import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from v2v.contracts import AssessedHazard, VehicleState
from v2v.sim.gazetteer import angle_diff, bearing_deg

AHEAD_TOLERANCE_DEG = 90.0
SAME_DIRECTION_DEG = 60.0
AT_HAZARD_M = 5.0


@dataclass(frozen=True, eq=False)
class RoadGraph:
    """The public road map as directed edges: what leads into what, and how long."""

    length_m: dict[str, float]
    incoming: dict[str, tuple[str, ...]] = field(default_factory=dict)

    @classmethod
    def from_edges(cls, edges: Iterable[tuple[str, float, Iterable[str]]]) -> "RoadGraph":
        """From (edge_id, length_m, outgoing edge ids) triples."""
        length: dict[str, float] = {}
        incoming: dict[str, list[str]] = defaultdict(list)
        for edge_id, edge_length, outgoing in edges:
            length[edge_id] = edge_length
            for nxt in outgoing:
                incoming[nxt].append(edge_id)
        return cls(length, {e: tuple(src) for e, src in incoming.items()})

    @classmethod
    def from_sumo_net(cls, net: Any) -> "RoadGraph":
        return cls.from_edges(
            (e.getID(), e.getLength(), [o.getID() for o in e.getOutgoing()])
            for e in net.getEdges()
            if e.allows("passenger")
        )

    def upstream(self, edge_id: str, pos_m: float, max_m: float) -> dict[str, float]:
        """Edges leading to ``(edge_id, pos_m)``, each mapped to the driving
        distance from that edge's *end* to the point, up to ``max_m``. The
        point's own edge is not included."""
        return dict(_upstream(self, edge_id, round(pos_m, 1), max_m))


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


_road_graph: RoadGraph | None = None


def use_road_graph(graph: RoadGraph | None) -> None:
    """Set the road map used by ``select`` (``SumoWorld`` does this on start).
    With None, ``select`` falls back to the heading rule."""
    global _road_graph
    _road_graph = graph


def keeps(vehicle: VehicleState, hx: float, hy: float, road_heading: float | None) -> bool:
    """The heading rule for one vehicle (the radius is checked by ``select``)."""
    if road_heading is not None and (
        angle_diff(vehicle.heading_deg, road_heading) > SAME_DIRECTION_DEG
    ):
        return False
    if math.dist((vehicle.x, vehicle.y), (hx, hy)) <= AT_HAZARD_M:
        return True
    to_hazard = bearing_deg(vehicle.x, vehicle.y, hx, hy)
    return angle_diff(to_hazard, vehicle.heading_deg) <= AHEAD_TOLERANCE_DEG


def select(hazard: AssessedHazard, vehicles: list[VehicleState], radius_m: float) -> list[str]:
    loc = hazard.location
    if loc is None:
        raise ValueError("geocast needs a location; apply the fallback policy first")
    in_range = [v for v in vehicles if math.dist((v.x, v.y), (loc.x, loc.y)) <= radius_m]
    graph = _road_graph
    if graph is None or loc.edge_id not in graph.length_m:
        return [v.vehicle_id for v in in_range if keeps(v, loc.x, loc.y, loc.heading_deg)]
    upstream = graph.upstream(loc.edge_id, loc.pos_m, radius_m)
    kept = []
    for v in in_range:
        if v.edge_id == loc.edge_id:
            by_road = loc.pos_m - v.lane_pos_m  # negative once past the hazard
        elif v.edge_id in upstream:
            by_road = graph.length_m[v.edge_id] - v.lane_pos_m + upstream[v.edge_id]
        else:
            continue
        if 0.0 <= by_road <= radius_m:
            kept.append(v.vehicle_id)
    return kept
