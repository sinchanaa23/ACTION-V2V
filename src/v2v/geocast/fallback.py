"""Fallback when a report gives no usable location. Owner: P4 (P4-05)."""

from v2v.contracts import ResolvedLocation, VehicleState


def fallback_location(reporter: VehicleState) -> ResolvedLocation:
    """Approximate the hazard location from the reporter.

    STUB(P4-05): the reporter's current position. The real policy also covers
    the next k edges in the reporter's direction of travel.
    """
    return ResolvedLocation(
        edge_id=reporter.edge_id,
        pos_m=reporter.lane_pos_m,
        x=reporter.x,
        y=reporter.y,
        approx=True,
    )
