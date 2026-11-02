"""Warning metrics, per report and aggregated. Owner: P3 (P3-04).

False alerts (warnings for events that are not happening) are kept separate
from irrelevant recipients (correct warnings sent to vehicles that won't reach
the hazard). Classification scores (macro-F1, per-class precision and recall)
reuse P1's ``v2v.data.scoring`` so Baseline A and the NLP models are scored the
same way.
"""

import statistics
from collections import defaultdict
from collections.abc import Collection, Iterable, Mapping, Sequence
from typing import Any

from v2v.contracts import HazardType, RunResult
from v2v.data.scoring import classification_scores

RATE_METRICS: tuple[str, ...] = (
    "false_alert",
    "missed_hazard",
    "relevant_delivery_rate",
    "irrelevant_recipient_rate",
    "messages_sent",
    "transmissions",
    "mean_latency_ms",
    "location_approx",
)
"""Per-report metrics averaged by ``aggregate``."""


def report_metrics(
    *,
    warned: bool,
    event_active: bool,
    needs_warning: bool,
    recipients: Collection[str],
    delivered: Collection[str],
    relevant: Collection[str],
    latencies_ms: Sequence[float],
    transmissions: int | None = None,
    location_approx: bool | None = None,
) -> dict[str, float | None]:
    """Metrics for one report.

    ``event_active``: the hazard is really there when the warning would go out.
    ``needs_warning``: the report is a genuine report of that hazard, so staying
    silent is a miss. Rates with an empty denominator are None, never 0.
    """
    sent_to = set(recipients) if warned else set()
    got = set(delivered) & sent_to
    need = set(relevant)
    return {
        "warned": float(warned),
        "false_alert": float(warned and not event_active),
        "missed_hazard": float(needs_warning and event_active and not warned),
        "relevant_delivery_rate": len(got & need) / len(need) if warned and need else None,
        "irrelevant_recipient_rate": len(sent_to - need) / len(sent_to) if sent_to else None,
        "messages_sent": float(len(sent_to)),
        "transmissions": float(transmissions) if warned and transmissions is not None else None,
        "mean_latency_ms": statistics.mean(latencies_ms) if warned and latencies_ms else None,
        "location_approx": None if location_approx is None else float(location_approx),
    }


def summarise(result: RunResult) -> dict[str, float | None]:
    """Metrics for one pipeline run of a single report."""
    warned = result.warning is not None
    location = result.assessed.location
    metrics = report_metrics(
        warned=warned,
        event_active=result.event_occurred,
        needs_warning=True,
        recipients=[d.vehicle_id for d in result.deliveries],
        delivered=[d.vehicle_id for d in result.deliveries if d.delivered],
        relevant=result.relevant_vehicle_ids,
        latencies_ms=[d.latency_ms for d in result.deliveries if d.latency_ms is not None],
        location_approx=None if location is None else location.approx,
    )
    del metrics["transmissions"]  # a RunResult does not carry overhead counts
    return metrics


def classification(gold: Sequence[HazardType], predicted: Sequence[HazardType]) -> dict[str, Any]:
    """Macro-F1 and per-class precision/recall/F1 over the gold classes."""
    return classification_scores(gold, predicted)


def aggregate(
    rows: Iterable[Mapping[str, Any]],
    keys: Sequence[str],
    metrics: Sequence[str] = RATE_METRICS,
) -> list[dict[str, Any]]:
    """Mean of each metric per group of ``keys``. None values are left out of a
    mean (and counted in ``n_<metric>``), so a rate is averaged only over the
    reports where it is defined."""
    groups: defaultdict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    out = []
    for group_key, members in sorted(groups.items(), key=lambda kv: str(kv[0])):
        summary: dict[str, Any] = dict(zip(keys, group_key, strict=True))
        summary["n"] = len(members)
        for m in metrics:
            values = [r[m] for r in members if r.get(m) is not None]
            summary[m] = statistics.mean(values) if values else None
            summary[f"n_{m}"] = len(values)
        out.append(summary)
    return out
