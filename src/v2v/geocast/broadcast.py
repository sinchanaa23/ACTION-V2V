"""Baseline A distribution: every vehicle in range of the network. Owner: P4 (P4-05)."""

from v2v.contracts import AssessedHazard, VehicleState


def select(hazard: AssessedHazard, vehicles: list[VehicleState], radius_m: float) -> list[str]:
    """Everyone, regardless of location or radius."""
    return [v.vehicle_id for v in vehicles]
