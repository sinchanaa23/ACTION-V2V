"""Driver mode: one vehicle's own view of the live simulation. Owner: P4.

A ``DriverSession`` plays a generated scenario through ``LiveSession`` (so
reports still go through the real pipeline) and adds two demo vehicles to
SUMO: a car on the approach route through the hazard and a scooter on the
opposite carriageway. They are real SUMO vehicles of their own types (sizes,
top speed and acceleration from the makers' published figures), extra to the
scenario's traffic, and they take part in the radio network like any other.

What a driver sees is only what reaches that vehicle: its own position and
route, vehicles close enough to see, and the V2V messages it actually
received (hazard warnings from the pipeline, and alerts other drivers send),
each placed relative to the vehicle. Scenario ground truth is not shown.

Drivers can send three V2V alerts, one per geocast direction, addressed with
the same direction-sensitive selector and delivered by the same radio model
as hazard warnings: hard braking (backward: traffic behind), a speeding
vehicle coming through (forward: traffic ahead), a slippery surface
(bidirectional: traffic approaching on both carriageways).
"""

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from v2v.contracts import ResolvedLocation, VehicleState, WarningMessage
from v2v.geocast import directional
from v2v.geocast.directional import Reach, RoadGraph
from v2v.sim.delivery import DEFAULT_MODEL, DeliveryOutcome
from v2v.sim.gazetteer import angle_diff, bearing_deg
from v2v.sim.live import LiveSession, WarningEvent
from v2v.sim.scenarios import ScenarioSpec
from v2v.sim.traci_loop import net_file, road_graph

VehicleKind = Literal["car", "scooter"]
TRAIL_POINTS = 900
SIGHT_M = 250.0
CRUISE_MPS = 12.0
"""Rough urban speed, for timing the demo vehicles' departure."""


@dataclass(frozen=True)
class DemoVehicle:
    vehicle_id: str
    kind: VehicleKind
    model: str
    label: str
    route: Literal["approach", "opposing"]
    vclass: str
    length_m: float
    width_m: float
    max_speed_mps: float
    accel: float
    decel: float
    powertrain: str


DEMO_FLEET: tuple[DemoVehicle, ...] = (
    DemoVehicle(
        vehicle_id="demo_car",
        kind="car",
        model="Tata Nexon EV",
        label="Demo car",
        route="approach",
        vclass="passenger",
        length_m=3.994,  # maker's figures: 3994 x 1811 mm, 150 km/h, 0-100 km/h in 8.9 s
        width_m=1.811,
        max_speed_mps=150 / 3.6,
        accel=100 / 3.6 / 8.9,
        decel=7.5,
        powertrain="electric",
    ),
    DemoVehicle(
        vehicle_id="demo_scooter",
        kind="scooter",
        model="Ather 450X",
        label="Demo scooter",
        route="opposing",
        vclass="motorcycle",
        length_m=1.837,  # maker's figures: 1837 x 739 mm, 90 km/h, 0-40 km/h in 3.3 s
        width_m=0.739,
        max_speed_mps=90 / 3.6,
        accel=40 / 3.6 / 3.3,
        decel=7.0,
        powertrain="electric",
    ),
)
DEMO_BY_ID = {d.vehicle_id: d for d in DEMO_FLEET}


@dataclass(frozen=True)
class AlertKind:
    key: str
    label: str
    reach: Reach
    radius_m: float
    text: str


DRIVER_ALERTS: dict[str, AlertKind] = {
    a.key: a
    for a in (
        AlertKind(
            "hard_braking",
            "Hard braking",
            "backward",
            300.0,
            "Vehicle ahead braking hard. Slow down.",
        ),
        AlertKind(
            "speeder",
            "Speeding vehicle",
            "forward",
            600.0,
            "A speeding vehicle is coming up behind you. Stay in your lane.",
        ),
        AlertKind(
            "slippery",
            "Slippery surface",
            "bidirectional",
            400.0,
            "Slippery road surface ahead. Slow down.",
        ),
    )
}


@dataclass(frozen=True)
class SentAlert:
    """A V2V alert a driver sent, as addressed and delivered."""

    kind: AlertKind
    sender: str
    t_s: float
    location: ResolvedLocation
    message: WarningMessage
    outcome: DeliveryOutcome

    @property
    def delivered(self) -> dict[str, float]:
        return {
            r.vehicle_id: r.latency_ms
            for r in self.outcome.records
            if r.delivered and r.latency_ms is not None
        }


@dataclass(frozen=True)
class Incoming:
    """A V2V message this vehicle received."""

    msg_id: str
    t_s: float
    title: str
    text: str
    status: str
    reach: Reach | None
    sender: str
    latency_ms: float
    hop: int
    relay: str
    at: tuple[float, float]
    """Where the message is about: the hazard point, or the sender's position."""


@dataclass(frozen=True)
class Bearing:
    """Where a point is, seen from a vehicle."""

    distance_m: float
    """Straight line."""
    relative_deg: float
    """-180..180: 0 dead ahead, positive to the right."""
    by_road_m: float | None
    """Driving distance if the point is on the vehicle's route ahead."""

    @property
    def side(self) -> str:
        a = abs(self.relative_deg)
        if a <= 35:
            return "ahead"
        if a >= 145:
            return "behind"
        return "right" if self.relative_deg > 0 else "left"


def bearing_from(v: VehicleState, x: float, y: float, by_road_m: float | None = None) -> Bearing:
    rel = (bearing_deg(v.x, v.y, x, y) - v.heading_deg + 540.0) % 360.0 - 180.0
    return Bearing(math.dist((v.x, v.y), (x, y)), rel, by_road_m)


def along_route(v: VehicleState, graph: RoadGraph, edge_id: str, pos_m: float) -> float | None:
    """Driving distance from the vehicle to (edge_id, pos_m) along its route, or None."""
    if edge_id == v.edge_id:
        return pos_m - v.lane_pos_m if pos_m >= v.lane_pos_m else None
    dist = graph.length_m.get(v.edge_id, v.lane_pos_m) - v.lane_pos_m
    for edge in v.route_ahead:
        if edge == edge_id:
            return dist + pos_m
        dist += graph.length_m.get(edge, 0.0)
    return None


def _cut(
    shape: tuple[tuple[float, float], ...], start: float, end: float
) -> list[tuple[float, float]]:
    """The part of a polyline between two distances along it."""
    out: list[tuple[float, float]] = []
    walked = 0.0
    for a, b in zip(shape, shape[1:], strict=False):
        seg = math.dist(a, b)
        if seg == 0:
            continue
        lo, hi = max(start, walked), min(end, walked + seg)
        if lo <= hi:
            for d in (lo, hi):
                f = (d - walked) / seg
                p = (a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f)
                if not out or math.dist(out[-1], p) > 0.01:
                    out.append(p)
        walked += seg
    return out


def route_ahead_xy(
    v: VehicleState, graph: RoadGraph, max_m: float = 1500.0
) -> list[tuple[float, float]]:
    """The road the vehicle will drive next, as a polyline, from where it is now."""
    pts: list[tuple[float, float]] = [(v.x, v.y)]
    left = max_m
    for k, edge in enumerate((v.edge_id, *v.route_ahead)):
        if left <= 0:
            break
        shape = graph.shapes.get(edge)
        length = graph.length_m.get(edge, 0.0)
        if not shape:
            continue
        geo = sum(math.dist(a, b) for a, b in zip(shape, shape[1:], strict=False)) or 1.0
        scale = geo / max(length, 1e-6)
        start = v.lane_pos_m * scale if k == 0 else 0.0
        end = min(geo, start + left * scale)
        pts += _cut(shape, start, end)
        left -= (end - start) / scale
    return pts


def demo_depart_s(spec: ScenarioSpec, graph: RoadGraph) -> float:
    """When to add the demo vehicles: early enough that the car is still
    approaching the hazard, within the geocast radius, when the first report
    is due."""
    first = min((s.expected_report_s for s in spec.slots), default=spec.event_start_s)
    route = spec.approach_route
    upto = route.index(spec.hazard.edge_id) if spec.hazard.edge_id in route else len(route)
    to_hazard = sum(graph.length_m.get(e, 0.0) for e in route[:upto]) + spec.hazard.pos_m
    return max(0.0, first - 0.6 * to_hazard / CRUISE_MPS)


@dataclass
class DriverSession:
    """A live scenario with the demo fleet in it, steppable from the dashboard."""

    spec: ScenarioSpec
    config_name: str = "proposed"
    live: LiveSession = field(init=False)
    graph: RoadGraph = field(init=False)
    trails: dict[str, list[tuple[float, float]]] = field(default_factory=dict)
    sent: list[SentAlert] = field(default_factory=list)
    depart_s: float = 0.0

    def __post_init__(self) -> None:
        self.live = LiveSession(self.spec, config_name=self.config_name)
        self.graph = road_graph(net_file(Path(self.spec.sumocfg)))

    def open(self) -> None:
        """Start SUMO, play up to the demo departure (reports fire as usual on
        the way) and add the demo vehicles."""
        self.live.open()
        self.trails = {d.vehicle_id: [] for d in DEMO_FLEET}
        self.sent = []
        self.depart_s = demo_depart_s(self.spec, self.graph)
        while self.live.time_s < self.depart_s and not self.live.finished:
            self.live.advance()
        world = self.live.world
        for d in DEMO_FLEET:
            edges = self.spec.approach_route if d.route == "approach" else self.spec.opposing_route
            world.add_vehicle(
                d.vehicle_id,
                edges,
                f"demo_{d.kind}",
                d.vclass,
                d.length_m,
                d.width_m,
                d.max_speed_mps,
                d.accel,
                d.decel,
            )
        self.advance()

    def close(self) -> None:
        self.live.close()

    @property
    def time_s(self) -> float:
        return self.live.time_s

    def advance(self, steps: int = 1) -> list[WarningEvent]:
        fired: list[WarningEvent] = []
        for _ in range(steps):
            if self.live.finished:
                break
            fired += self.live.advance()
            for d in DEMO_FLEET:
                v = self.state(d.vehicle_id)
                if v is not None:
                    trail = self.trails[d.vehicle_id]
                    if not trail or math.dist(trail[-1], (v.x, v.y)) > 0.5:
                        trail.append((v.x, v.y))
                        del trail[:-TRAIL_POINTS]
        return fired

    def state(self, vehicle_id: str) -> VehicleState | None:
        try:
            return self.live.world.vehicle(vehicle_id)
        except KeyError:
            return None

    def on_road(self, vehicle_id: str) -> bool:
        return any(v.vehicle_id == vehicle_id for v in self.live.vehicles())

    def nearby(
        self, vehicle_id: str, radius_m: float = SIGHT_M
    ) -> list[tuple[VehicleState, Bearing]]:
        """Vehicles close enough to see (stopped hazard vehicles included), nearest first."""
        me = self.state(vehicle_id)
        if me is None:
            return []
        out = [
            (v, bearing_from(me, v.x, v.y))
            for v in (*self.live.vehicles(), *self.live.blocking())
            if v.vehicle_id != vehicle_id and math.dist((me.x, me.y), (v.x, v.y)) <= radius_m
        ]
        return sorted(out, key=lambda pair: pair[1].distance_m)

    def neighbours_in_radio_range(self, vehicle_id: str) -> int:
        me = self.state(vehicle_id)
        if me is None:
            return 0
        reach = DEFAULT_MODEL.comm_range_m
        return sum(
            1
            for v in self.live.vehicles()
            if v.vehicle_id != vehicle_id and math.dist((me.x, me.y), (v.x, v.y)) <= reach
        )

    def incoming(self, vehicle_id: str) -> list[Incoming]:
        """Every message this vehicle received, oldest first."""
        out: list[Incoming] = []
        for e in self.live.events:
            got = e.delivered.get(vehicle_id)
            if got is None or e.outcome is None or e.result.warning is None:
                continue
            w = e.result.warning
            out.append(
                Incoming(
                    msg_id=w.msg_id,
                    t_s=e.t_s,
                    title=e.result.assessed.hazard.value.replace("_", " "),
                    text=w.text,
                    status=w.status,
                    reach=directional.reach_for(e.result.assessed.hazard)
                    if w.policy == "directional"
                    else None,
                    sender=e.slot.reporter_id,
                    latency_ms=got,
                    hop=e.outcome.hops.get(vehicle_id, 1),
                    relay=e.outcome.relays.get(vehicle_id, e.slot.reporter_id),
                    at=e.centre_xy,
                )
            )
        for a in self.sent:
            got = a.delivered.get(vehicle_id)
            if got is None:
                continue
            out.append(
                Incoming(
                    msg_id=a.message.msg_id,
                    t_s=a.t_s,
                    title=a.kind.label.lower(),
                    text=a.kind.text,
                    status="driver alert",
                    reach=a.kind.reach,
                    sender=a.sender,
                    latency_ms=got,
                    hop=a.outcome.hops.get(vehicle_id, 1),
                    relay=a.outcome.relays.get(vehicle_id, a.sender),
                    at=(a.location.x, a.location.y),
                )
            )
        return sorted(out, key=lambda m: m.t_s)

    def send(self, vehicle_id: str, kind: str) -> SentAlert:
        """Send a driver alert from this vehicle: addressed by direction-sensitive
        geocast around it, delivered over the radio model."""
        me = self.state(vehicle_id)
        if me is None:
            raise KeyError(f"{vehicle_id} is not on the road")
        k = DRIVER_ALERTS[kind]
        world = self.live.world
        where = ResolvedLocation(
            edge_id=me.edge_id,
            pos_m=me.lane_pos_m,
            x=me.x,
            y=me.y,
            approx=False,
            heading_deg=me.heading_deg,
        )
        recipients = directional.select_reach(
            where, world.vehicles(), k.radius_m, k.reach, self.graph, exclude=vehicle_id
        )
        message = WarningMessage(
            msg_id=f"{vehicle_id}-{kind}-{world.time_s:.0f}-{len(self.sent)}",
            cluster_id=f"driver-{vehicle_id}",
            text=k.text,
            status="unverified",
            policy="directional",
            recipients=tuple(recipients),
            t_sent=world.time_s,
        )
        outcome = DEFAULT_MODEL.deliver(message, world, where, sender_id=vehicle_id)
        alert = SentAlert(k, vehicle_id, world.time_s, where, message, outcome)
        self.sent.append(alert)
        return alert

    def where_is(
        self, vehicle_id: str, x: float, y: float, edge_id: str | None = None, pos_m: float = 0.0
    ) -> Bearing | None:
        me = self.state(vehicle_id)
        if me is None:
            return None
        by_road = along_route(me, self.graph, edge_id, pos_m) if edge_id else None
        return bearing_from(me, x, y, by_road)

    def heading_change(self, vehicle_id: str) -> float:
        """Degrees the route turns over the next ~150 m (positive: right), for the turn cue."""
        me = self.state(vehicle_id)
        if me is None:
            return 0.0
        pts = route_ahead_xy(me, self.graph, 150.0)
        if len(pts) < 3:
            return 0.0
        end = bearing_deg(pts[-2][0], pts[-2][1], pts[-1][0], pts[-1][1])
        sign = 1.0 if (end - me.heading_deg + 540) % 360 - 180 >= 0 else -1.0
        return sign * angle_diff(end, me.heading_deg)
