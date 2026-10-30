"""SUMO-backed ``World`` driven through TraCI/libsumo. Owner: P4 (P4-04)."""

from v2v.contracts import ResolvedLocation, Scenario, VehicleState


class SumoWorld:
    """STUB(P4-04): implements the ``World`` protocol on the real OSM network."""

    def __init__(self, scenario: Scenario, sumocfg: str, seed: int = 0) -> None:
        raise NotImplementedError("P4-04: TraCI loop")

    @property
    def time_s(self) -> float:
        raise NotImplementedError

    def step(self) -> None:
        raise NotImplementedError

    def vehicles(self) -> list[VehicleState]:
        raise NotImplementedError

    def vehicle(self, vehicle_id: str) -> VehicleState:
        raise NotImplementedError

    def resolve_location(
        self, location_text: str | None, reporter_id: str
    ) -> ResolvedLocation | None:
        raise NotImplementedError

    def sensor_flags(self, location: ResolvedLocation, radius_m: float) -> int:
        raise NotImplementedError

    def will_reach(self, vehicle_id: str, location: ResolvedLocation) -> bool:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError
