"""A pure-Python stand-in for SUMO so the pipeline runs end to end in CI.

Two straight one-way edges side by side: ``fwd`` runs west to east (heading 90)
and ``rev`` runs east to west (heading 270). Vehicles drive at constant speed
and leave at the end of their edge. This is test scaffolding, not a result
source: experiments run on the real OSM network through ``traci_loop``.
"""

import random

from v2v.contracts import ResolvedLocation, Scenario, VehicleState

EDGE_LENGTH_M = 2000.0
VEHICLES_PER_EDGE = 18
SPACING_M = 100.0
STEP_S = 1.0
_LANE_Y = {"fwd": 0.0, "rev": 8.0}
_HEADING = {"fwd": 90.0, "rev": 270.0}


class ToyWorld:
    def __init__(self, scenario: Scenario, seed: int = 0) -> None:
        rng = random.Random(seed)
        self._scenario = scenario
        self._time_s = 0.0
        self._pos: dict[str, float] = {}
        self._speed: dict[str, float] = {}
        self._edge: dict[str, str] = {}
        for edge in ("fwd", "rev"):
            for i in range(VEHICLES_PER_EDGE):
                vid = f"{edge}_{i}"
                self._pos[vid] = i * SPACING_M + rng.uniform(0.0, 20.0)
                self._speed[vid] = rng.uniform(12.0, 16.0)
                self._edge[vid] = edge

    @property
    def time_s(self) -> float:
        return self._time_s

    def step(self) -> None:
        self._time_s += STEP_S
        for vid in list(self._pos):
            self._pos[vid] += self._speed[vid] * STEP_S
            if self._pos[vid] >= EDGE_LENGTH_M:
                del self._pos[vid], self._speed[vid], self._edge[vid]

    def vehicles(self) -> list[VehicleState]:
        return [self.vehicle(vid) for vid in self._pos]

    def vehicle(self, vehicle_id: str) -> VehicleState:
        edge = self._edge[vehicle_id]
        pos = self._pos[vehicle_id]
        x = pos if edge == "fwd" else EDGE_LENGTH_M - pos
        return VehicleState(
            vehicle_id=vehicle_id,
            x=x,
            y=_LANE_Y[edge],
            edge_id=edge,
            lane_pos_m=pos,
            heading_deg=_HEADING[edge],
            speed_mps=self._speed[vehicle_id],
        )

    def resolve_location(
        self, location_text: str | None, reporter_id: str
    ) -> ResolvedLocation | None:
        """Any stated location resolves to the scenario's hazard; the real
        landmark lookup is P4-02."""
        if location_text is None:
            return None
        return self._scenario.hazard_location

    def sensor_flags(self, location: ResolvedLocation, radius_m: float) -> int:
        return 0

    def will_reach(self, vehicle_id: str, location: ResolvedLocation) -> bool:
        if vehicle_id not in self._pos:
            return False
        return self._edge[vehicle_id] == location.edge_id and self._pos[vehicle_id] < location.pos_m

    def close(self) -> None:
        pass
