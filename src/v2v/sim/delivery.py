"""Simplified V2V delivery model. Owner: P4 (P4-06).

Replaces OMNeT++/Veins in the core build: range, distance-based loss, per-hop
latency and optional multi-hop forwarding.
"""

from v2v.contracts import DeliveryRecord, ResolvedLocation, WarningMessage, World


def deliver(
    message: WarningMessage, world: World, hazard: ResolvedLocation
) -> list[DeliveryRecord]:
    """One record per addressed recipient.

    STUB(P4-06): every recipient receives the message instantly.
    """
    return [
        DeliveryRecord(
            msg_id=message.msg_id,
            vehicle_id=vid,
            delivered=True,
            latency_ms=0.0,
            relevant=world.will_reach(vid, hazard),
        )
        for vid in message.recipients
    ]
