"""Proposed distribution: direction-sensitive geocast. Owner: P4 (P4-05).

Keeps a vehicle within radius R if the hazard edge is in its remaining route,
or it is upstream on the same edge heading within 45 degrees of the edge.
"""

from v2v.contracts import AssessedHazard, VehicleState


def select(hazard: AssessedHazard, vehicles: list[VehicleState], radius_m: float) -> list[str]:
    """STUB(P4-05): returns every vehicle."""
    return [v.vehicle_id for v in vehicles]
