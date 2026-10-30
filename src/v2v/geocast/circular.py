"""Baseline B distribution: every vehicle within radius R. Owner: P4 (P4-05)."""

from v2v.contracts import AssessedHazard, VehicleState


def select(hazard: AssessedHazard, vehicles: list[VehicleState], radius_m: float) -> list[str]:
    """STUB(P4-05): returns every vehicle."""
    return [v.vehicle_id for v in vehicles]
