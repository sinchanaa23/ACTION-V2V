"""Fallback when a report gives no usable location. Owner: P4 (P4-05).

The report says what happened but not where, so the best evidence of where is
the reporter itself: it is on, or has just driven past, the hazard. The
fallback places the hazard at the reporter's position with the reporter's
direction of travel, flagged ``approx=True`` so the warning says "location
approximate" and metrics can count these cases separately. Direction-sensitive
geocast then warns traffic behind the reporter on the same carriageway.
"""

from v2v.contracts import ResolvedLocation, VehicleState


def fallback_location(reporter: VehicleState) -> ResolvedLocation:
    return ResolvedLocation(
        edge_id=reporter.edge_id,
        pos_m=reporter.lane_pos_m,
        x=reporter.x,
        y=reporter.y,
        approx=True,
        heading_deg=reporter.heading_deg,
    )
