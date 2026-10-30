"""Per-run warning metrics. Owner: P3 (P3-04).

False alerts (warnings for events that did not happen) are kept separate from
irrelevant recipients (correct warnings sent to vehicles that won't reach the
hazard). Classification metrics (macro-F1, ECE) are added in P3-04.
"""

from v2v.contracts import RunResult


def summarise(result: RunResult) -> dict[str, float | None]:
    recipients = {d.vehicle_id for d in result.deliveries}
    delivered = {d.vehicle_id for d in result.deliveries if d.delivered}
    relevant = set(result.relevant_vehicle_ids)
    warned = result.warning is not None
    latencies = [d.latency_ms for d in result.deliveries if d.latency_ms is not None]
    return {
        "false_alert": float(warned and not result.event_occurred),
        "missed_hazard": float(result.event_occurred and not warned),
        "relevant_delivery_rate": len(delivered & relevant) / len(relevant) if relevant else None,
        "irrelevant_recipient_rate": (
            len(recipients - relevant) / len(recipients) if recipients else None
        ),
        "messages_sent": float(len(recipients)),
        "mean_latency_ms": sum(latencies) / len(latencies) if latencies else None,
    }
