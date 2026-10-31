"""Baseline B distribution: every vehicle within radius R of the hazard. Owner: P4 (P4-05)."""

import math

from v2v.contracts import AssessedHazard, VehicleState


def select(hazard: AssessedHazard, vehicles: list[VehicleState], radius_m: float) -> list[str]:
    if hazard.location is None:
        raise ValueError("geocast needs a location; apply the fallback policy first")
    centre = (hazard.location.x, hazard.location.y)
    return [v.vehicle_id for v in vehicles if math.dist((v.x, v.y), centre) <= radius_m]
