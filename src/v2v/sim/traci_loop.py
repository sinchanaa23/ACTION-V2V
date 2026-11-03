"""SUMO-backed ``World``, driven through TraCI. Owner: P4 (P4-04).

    python -m v2v.sim.traci_loop sim_out/scenarios/hebbal/noisy-medium-s0

``SumoWorld`` plays one generated scenario (see ``scenarios.py``) and gives the
pipeline what the ``World`` protocol promises:

- vehicle states (position, edge, heading, speed, remaining route), read with
  TraCI subscriptions so a step stays cheap at high density;
- ground truth: ``will_reach`` is exact from each vehicle's fixed route, and
  ``hazard_passes`` records when every vehicle actually passed the hazard;
- report triggers: each reporter's slot falls due ``delay_s`` after that
  vehicle really passes the hazard, so traffic delays move reports in time;
- simulated sensors: ``sensor_flags`` counts stopped or crawling vehicles near
  a location (queues at a signal can produce false flags, as real sensors do);
- the hazard's effect on traffic: lane-blocking hazards are a stopped vehicle
  from the routes file; a slippery road lowers the hazard edge's speed limit
  for the event.

The stopped ``hazard_*`` vehicle is an obstacle, not a V2V participant, so it
is never returned by ``vehicles()``.
"""

import argparse
import math
import uuid
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from v2v.contracts import ResolvedLocation, VehicleState
from v2v.geocast import directional
from v2v.geocast.directional import RoadGraph
from v2v.sim import osm_network
from v2v.sim.gazetteer import Gazetteer
from v2v.sim.scenarios import ReportSlot, ScenarioSpec

HAZARD_PREFIX = "hazard_"
SLOW_SPEED_MPS = 1.0  # "stopped or crawling" for the simulated sensors
SLIPPERY_SPEED_FACTOR = 0.4

# TraCI variable ids (traci.constants), kept here so this module imports
# without the [sim] extras installed.
VAR_ROAD_ID = 0x50
VAR_LANEPOSITION = 0x56
VAR_POSITION = 0x42
VAR_ANGLE = 0x43
VAR_SPEED = 0x40
VAR_ROUTE_INDEX = 0x69
SUBSCRIBED = (VAR_ROAD_ID, VAR_LANEPOSITION, VAR_POSITION, VAR_ANGLE, VAR_SPEED, VAR_ROUTE_INDEX)


def passed_point(
    route: tuple[str, ...], route_index: int, lane_pos_m: float, edge_id: str, pos_m: float
) -> bool | None:
    """Whether a vehicle has passed (edge_id, pos_m) on its route; None if the
    point is not on its route at all."""
    if edge_id not in route:
        return None
    target = route.index(edge_id)
    return route_index > target or (route_index == target and lane_pos_m >= pos_m)


def net_file(sumocfg: Path) -> Path:
    """The network a .sumocfg points at."""
    net = ET.parse(sumocfg).getroot().find("input/net-file")
    if net is None:
        raise ValueError(f"{sumocfg} names no net-file")
    return (sumocfg.parent / net.get("value", "")).resolve()


@lru_cache(maxsize=4)
def road_graph(net_path: Path) -> RoadGraph:
    return RoadGraph.from_sumo_net(osm_network.read_net(net_path))


@dataclass
class ReportClock:
    """Turns "reporter passed the hazard at t" into "report due at t + delay"."""

    slots: tuple[ReportSlot, ...]
    passed_at: dict[str, float] = field(default_factory=dict)
    fired: set[str] = field(default_factory=set)

    def mark_passed(self, vehicle_id: str, t: float) -> None:
        self.passed_at.setdefault(vehicle_id, t)

    def due(self, now: float) -> list[tuple[ReportSlot, float]]:
        """Slots whose report time has come, each returned once, with the
        time the reporter passed the hazard."""
        ready = []
        for slot in self.slots:
            t_pass = self.passed_at.get(slot.reporter_id)
            if slot.slot_id in self.fired or t_pass is None or now < t_pass + slot.delay_s:
                continue
            self.fired.add(slot.slot_id)
            ready.append((slot, t_pass))
        return sorted(ready, key=lambda pair: pair[1] + pair[0].delay_s)


@dataclass
class _Tracked:
    route: tuple[str, ...]
    values: dict[int, Any] = field(default_factory=dict)
    """Latest subscription values from while the vehicle was on a road."""
    on_road: bool = False
    """False while SUMO is teleporting it out of a jam (no road, no position)."""


class SumoWorld:
    """One running SUMO scenario that satisfies ``v2v.contracts.World``."""

    def __init__(
        self,
        spec: ScenarioSpec,
        gazetteer: Gazetteer | None = None,
        gui: bool = False,
        extra_args: Iterable[str] = (),
    ) -> None:
        import traci

        self.spec = spec
        self.gazetteer = gazetteer or Gazetteer.load(Path(spec.gazetteer))
        self._label = f"v2v-{uuid.uuid4().hex[:8]}"
        cmd = [
            osm_network.sumo_binary("sumo-gui" if gui else "sumo"),
            "-c", spec.sumocfg,
            "--seed", str(spec.seed),
            *extra_args,
        ]  # fmt: skip
        # Direction-sensitive geocast works on the public road map of this network.
        graph = road_graph(net_file(Path(spec.sumocfg)))
        directional.acquire_road_graph(self, graph)
        self.opposite_point = self._opposite_point(graph)
        """For a hazard on both carriageways: (edge, pos) level with it on the
        other one. It is part of the event there too (slowed, and ground truth)."""
        traci.start(cmd, label=self._label)
        self._t: Any = traci.getConnection(self._label)
        self._tracked: dict[str, _Tracked] = {}
        self._left: dict[str, _Tracked] = {}
        """Last known state of vehicles that have driven out of the network."""
        self._slow_zone_saved: dict[str, float] = {}
        self.clock = ReportClock(spec.slots)
        self.hazard_passes: dict[str, float] = {}
        """Vehicle id -> time it actually passed the hazard point."""
        self.teleported: set[str] = set()
        self._closed = False

    # --- World protocol -------------------------------------------------------

    @property
    def time_s(self) -> float:
        return float(self._t.simulation.getTime())

    def step(self) -> None:
        self._t.simulationStep()
        sim = self._t.simulation
        for vid in sim.getDepartedIDList():
            self._t.vehicle.subscribe(vid, SUBSCRIBED)
            self._tracked[vid] = _Tracked(route=tuple(self._t.vehicle.getRoute(vid)))
        for vid in sim.getArrivedIDList():
            gone = self._tracked.pop(vid, None)
            if gone is not None and gone.values:
                self._left[vid] = gone
        self.teleported.update(sim.getStartingTeleportIDList())
        for vid, values in self._t.vehicle.getAllSubscriptionResults().items():
            tracked = self._tracked.get(vid)
            if tracked is None:
                continue
            tracked.on_road = values.get(VAR_ROAD_ID, "") != ""
            if tracked.on_road:  # keep the last on-road state through a teleport
                tracked.values = values
        self._update_passes()
        self._apply_effects()

    def vehicles(self) -> list[VehicleState]:
        return [
            self._state(vid, tracked)
            for vid, tracked in self._tracked.items()
            if tracked.on_road and not vid.startswith(HAZARD_PREFIX)
        ]

    def vehicle(self, vehicle_id: str) -> VehicleState:
        """Current state. For a vehicle that has left the network (a late
        reporter) or is mid-teleport, its last on-road state; such vehicles
        are never in ``vehicles()``."""
        tracked = self._tracked.get(vehicle_id) or self._left.get(vehicle_id)
        if tracked is None or not tracked.values:
            raise KeyError(f"{vehicle_id} has not entered the simulation")
        return self._state(vehicle_id, tracked)

    def resolve_location(
        self, location_text: str | None, reporter_id: str
    ) -> ResolvedLocation | None:
        return self.gazetteer.resolve(location_text, self.vehicle(reporter_id))

    def sensor_flags(self, location: ResolvedLocation, radius_m: float) -> int:
        """Stopped or crawling vehicles within ``radius_m`` (hazard vehicle included)."""
        count = 0
        for tracked in self._tracked.values():
            v = tracked.values
            if not tracked.on_road or v[VAR_SPEED] >= SLOW_SPEED_MPS:
                continue
            x, y = v[VAR_POSITION]
            if math.dist((x, y), (location.x, location.y)) <= radius_m:
                count += 1
        return count

    def will_reach(self, vehicle_id: str, location: ResolvedLocation) -> bool:
        tracked = self._tracked.get(vehicle_id)
        if tracked is None or not tracked.values:
            return False
        points = [(location.edge_id, location.pos_m)]
        hazard = self.spec.hazard
        if self.opposite_point is not None and (location.edge_id, location.pos_m) == (
            hazard.edge_id,
            hazard.pos_m,
        ):
            points.append(self.opposite_point)  # the event covers both carriageways
        return any(
            passed_point(
                tracked.route,
                tracked.values[VAR_ROUTE_INDEX],
                self._lane_pos(tracked),
                edge,
                pos,
            )
            is False
            for edge, pos in points
        )

    def close(self) -> None:
        if not self._closed:
            self._t.close()
            directional.release_road_graph(self)
            self._closed = True

    # --- Beyond the protocol ----------------------------------------------------

    def lonlat(self, x: float, y: float) -> tuple[float, float]:
        """Network coordinates to (longitude, latitude)."""
        lon, lat = self._t.simulation.convertGeo(x, y)
        return float(lon), float(lat)

    def blocking_vehicles(self) -> list[VehicleState]:
        """The scenario's hazard vehicles (left out of ``vehicles()``), for drawing."""
        return [
            self._state(vid, tracked)
            for vid, tracked in self._tracked.items()
            if tracked.on_road and vid.startswith(HAZARD_PREFIX)
        ]

    def add_vehicle(
        self,
        vehicle_id: str,
        edges: Iterable[str],
        type_id: str,
        vclass: str,
        length_m: float,
        width_m: float,
        max_speed_mps: float,
        accel: float,
        decel: float,
    ) -> None:
        """Insert one extra vehicle now (a demo vehicle): a real SUMO vehicle
        of its own type, on a route of real edges, tracked like any other."""
        t = self._t
        if type_id not in t.vehicletype.getIDList():
            t.vehicletype.copy("DEFAULT_VEHTYPE", type_id)
            t.vehicletype.setVehicleClass(type_id, vclass)
            t.vehicletype.setShapeClass(type_id, vclass)
            t.vehicletype.setLength(type_id, length_m)
            t.vehicletype.setWidth(type_id, width_m)
            t.vehicletype.setMaxSpeed(type_id, max_speed_mps)
            t.vehicletype.setAccel(type_id, accel)
            t.vehicletype.setDecel(type_id, decel)
        route_id = f"{vehicle_id}_route"
        t.route.add(route_id, list(edges))
        t.vehicle.add(vehicle_id, route_id, typeID=type_id, depart="now", departLane="best")

    def run_until(self, t: float) -> None:
        while self.time_s < t:
            self.step()

    def due_reports(self) -> list[tuple[ReportSlot, float]]:
        """Report slots that fall due now: (slot, time the reporter passed)."""
        return self.clock.due(self.time_s)

    def __enter__(self) -> "SumoWorld":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # --- Internals ----------------------------------------------------------------

    def _lane_pos(self, tracked: _Tracked) -> float:
        """Lane position on the current route edge; on a junction's internal
        lane the vehicle counts as at the end of the edge it just left."""
        road = tracked.values[VAR_ROAD_ID]
        if road.startswith(":"):
            return math.inf
        return float(tracked.values[VAR_LANEPOSITION])

    def _state(self, vid: str, tracked: _Tracked) -> VehicleState:
        v = tracked.values
        index = v[VAR_ROUTE_INDEX]
        edge, lane_pos = v[VAR_ROAD_ID], float(v[VAR_LANEPOSITION])
        if edge.startswith(":"):
            # Inside a junction: map-match to the road it is turning onto, at its start.
            index = min(index + 1, len(tracked.route) - 1)
            edge, lane_pos = tracked.route[index], 0.0
        x, y = v[VAR_POSITION]
        return VehicleState(
            vehicle_id=vid,
            x=x,
            y=y,
            edge_id=edge,
            lane_pos_m=max(0.0, lane_pos),
            heading_deg=float(v[VAR_ANGLE]) % 360.0,
            speed_mps=max(0.0, float(v[VAR_SPEED])),
            route_ahead=tracked.route[index + 1 :],
        )

    def _update_passes(self) -> None:
        hz = self.spec.hazard
        now = self.time_s
        for vid, tracked in self._tracked.items():
            if not tracked.values or vid in self.hazard_passes:
                continue
            passed = passed_point(
                tracked.route,
                tracked.values[VAR_ROUTE_INDEX],
                self._lane_pos(tracked),
                hz.edge_id,
                hz.pos_m,
            )
            if passed:
                self.hazard_passes[vid] = now
                self.clock.mark_passed(vid, now)

    def _opposite_point(self, graph: RoadGraph) -> tuple[str, float] | None:
        spec = self.spec
        if directional.reach_for(spec.hazard_class) != "bidirectional":
            return None
        return graph.opposite(spec.hazard.edge_id, spec.hazard.x, spec.hazard.y)

    def _apply_effects(self) -> None:
        spec = self.spec
        if spec.effect != "slow_zone" or not spec.event_occurred:
            return
        active = spec.event_start_s <= self.time_s < spec.event_end_s
        edges = [spec.hazard.edge_id]
        if self.opposite_point is not None:
            edges.append(self.opposite_point[0])
        lanes = [f"{e}_{i}" for e in edges for i in range(self._t.edge.getLaneNumber(e))]
        if active and not self._slow_zone_saved:
            for lane in lanes:
                self._slow_zone_saved[lane] = self._t.lane.getMaxSpeed(lane)
                self._t.lane.setMaxSpeed(lane, self._slow_zone_saved[lane] * SLIPPERY_SPEED_FACTOR)
        elif not active and self._slow_zone_saved:
            for lane, speed in self._slow_zone_saved.items():
                self._t.lane.setMaxSpeed(lane, speed)
            self._slow_zone_saved = {}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Play a generated scenario and log what happens")
    parser.add_argument("scenario_dir", type=Path)
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args(argv)
    spec = ScenarioSpec.load(args.scenario_dir / "spec.json")
    where = spec.landmark_names or spec.landmark_kind
    print(f"{spec.scenario_id}: {spec.hazard_class.value} at {where}")
    print(
        f"event {'happens' if spec.event_occurred else 'does NOT happen'}, "
        f"{spec.event_start_s:.0f}-{spec.event_end_s:.0f} s, effect {spec.effect}"
    )
    with SumoWorld(spec, gui=args.gui) as world:
        while world.time_s < spec.end_s:
            world.step()
            for slot, t_pass in world.due_reports():
                vehicles = world.vehicles()
                relevant = sum(world.will_reach(v.vehicle_id, spec.hazard) for v in vehicles)
                flags = world.sensor_flags(spec.hazard, 200.0)
                print(
                    f"  t={world.time_s:6.0f}  {slot.reporter_id} ({slot.role}, "
                    f"needs {slot.needs}) passed at {t_pass:.0f}; "
                    f"{len(vehicles)} vehicles, {relevant} will reach "
                    f"the hazard; {flags} sensor flags"
                )
        missing = [s.reporter_id for s in spec.slots if s.slot_id not in world.clock.fired]
        print(
            f"done at {world.time_s:.0f} s; {len(world.hazard_passes)} vehicles passed the hazard; "
            f"{len(world.teleported)} teleported; reports not fired: {missing or 'none'}"
        )


if __name__ == "__main__":
    main()
