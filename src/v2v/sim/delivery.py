"""Simplified V2V delivery model. Owner: P4 (P4-06).

Replaces OMNeT++/Veins in the core build with a small, documented Python model
of how a warning spreads between vehicles. It is a model, not a radio
simulator: its numbers support comparisons between distribution policies, not
claims about real V2V performance.

Channel. A transmission at distance d is received with the Nakagami-m (m=3)
probability of Killat & Hartenstein (2009), "An empirical model for
probability of packet reception in vehicular ad hoc networks", EURASIP J.
Wireless Comm. and Networking:

    P(d) = exp(-3 (d/R)^2) * (1 + 3 (d/R)^2 + 4.5 (d/R)^4)

with R the communication range (``comm_range_m``; 300 m is an assumption for
urban 5.9 GHz). P(0) = 1, P(R) ~ 0.42, P(2R) ~ 0.002.

Latency. Each hop costs a fixed processing time, a random channel-access
delay, and a busy-channel term that grows with the number of vehicles in
range of the transmitter.

Forwarding. The sender (the reporter's vehicle, or the hazard point if no
sender is given) transmits first. With ``multi_hop``, an *addressed* vehicle
may rebroadcast once, up to ``max_hops`` hops, so only the geocast's own
recipients relay and a wider policy also pays for a wider flood. Plain
flooding would have every one of them repeat it; instead each waits a random
assessment delay (up to ``assessment_ms``) after first hearing it and stays
silent if by then it has heard the warning ``suppress_after`` times: the
counter-based scheme of Ni et al. (1999), "The broadcast storm problem in a
mobile ad hoc network", MobiCom. On the Hebbal scenarios a threshold of 4
halves the transmissions (356 -> 171) for 333 instead of 335 of 392
recipients reached; 3 cut deeper but lost more. Vehicles never deliver a warning twice: only
the first reception counts. Positions are frozen at the send time (vehicles
move ~1-2 m during a delivery that takes tens of milliseconds).

Every transmission and every successful reception (including duplicates and
vehicles that were not addressed) is counted, for the overhead metric.
"""

import heapq
import math
import random
from collections.abc import Iterable
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict

from v2v.contracts import DeliveryRecord, ResolvedLocation, VehicleState, WarningMessage, World

CUTOFF_FACTOR = 2.5  # beyond 2.5 R, P(d) < 1e-5: not worth evaluating


def reception_probability(distance_m: float, comm_range_m: float) -> float:
    """Nakagami-m (m=3) packet reception probability (Killat & Hartenstein 2009)."""
    q = (distance_m / comm_range_m) ** 2
    return math.exp(-3.0 * q) * (1.0 + 3.0 * q + 4.5 * q * q)


class DeliveryOutcome(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    records: tuple[DeliveryRecord, ...]
    """One per addressed recipient."""
    transmissions: int
    """Sender plus every rebroadcast."""
    receptions: int
    """Every successful reception, including duplicates and unaddressed vehicles."""
    hops: dict[str, int]
    """Addressed vehicle -> hop count at first reception."""
    relays: dict[str, str] = {}
    """Addressed vehicle -> the transmitter it first heard the warning from
    (the sender's id, or "origin" for the hazard point)."""
    suppressed: int = 0
    """Rebroadcasts cancelled because the vehicle had already heard it enough."""


@dataclass(frozen=True)
class DeliveryModel:
    comm_range_m: float = 300.0
    multi_hop: bool = True
    max_hops: int = 5
    processing_ms: float = 2.0
    access_jitter_ms: float = 8.0
    busy_ms_per_neighbour: float = 0.05
    assessment_ms: float = 4.0
    suppress_after: int | None = 4
    """Cancel a rebroadcast once the warning has been heard this many times
    (None: plain flooding)."""

    def deliver(
        self,
        message: WarningMessage,
        world: World,
        hazard: ResolvedLocation,
        sender_id: str | None = None,
        seed: int = 0,
    ) -> DeliveryOutcome:
        rng = random.Random(f"{message.msg_id}-{seed}")
        vehicles = {v.vehicle_id: v for v in world.vehicles()}
        addressed = set(message.recipients) & set(vehicles)
        origin = self._origin(world, hazard, sender_id)
        received_ms: dict[str, float] = {}
        hops: dict[str, int] = {}
        heard_from: dict[str, str] = {}
        heard = dict.fromkeys(vehicles, 0)
        transmissions = receptions = suppressed = 0
        # (time the transmission starts, tie-break counter, x, y, hop count, transmitter id)
        queue: list[tuple[float, int, float, float, int, str]] = [
            (0.0, 0, origin[0], origin[1], 0, sender_id or "origin")
        ]
        counter = 1
        while queue:
            t_ms, _, x, y, hop, sender = heapq.heappop(queue)
            if hop > 0 and self.suppress_after is not None and heard[sender] >= self.suppress_after:
                suppressed += 1  # its neighbourhood already has it
                continue
            transmissions += 1
            in_range = self._in_range(vehicles.values(), x, y, exclude=sender)
            hop_ms = (
                self.processing_ms
                + rng.uniform(0.0, self.access_jitter_ms)
                + self.busy_ms_per_neighbour * len(in_range)
            )
            for v, dist in in_range:
                if rng.random() >= reception_probability(dist, self.comm_range_m):
                    continue
                receptions += 1
                heard[v.vehicle_id] += 1
                if v.vehicle_id in received_ms or v.vehicle_id == sender_id:
                    continue  # it has it already (the originator wrote it)
                received_ms[v.vehicle_id] = t_ms + hop_ms
                hops[v.vehicle_id] = hop + 1
                heard_from[v.vehicle_id] = sender
                if self.multi_hop and v.vehicle_id in addressed and hop + 1 < self.max_hops:
                    wait = rng.uniform(0.0, self.assessment_ms)
                    heapq.heappush(
                        queue, (t_ms + hop_ms + wait, counter, v.x, v.y, hop + 1, v.vehicle_id)
                    )
                    counter += 1
        records = tuple(
            DeliveryRecord(
                msg_id=message.msg_id,
                vehicle_id=vid,
                delivered=vid in received_ms,
                latency_ms=round(received_ms[vid], 3) if vid in received_ms else None,
                relevant=world.will_reach(vid, hazard),
            )
            for vid in message.recipients
        )
        return DeliveryOutcome(
            records=records,
            transmissions=transmissions,
            receptions=receptions,
            hops={vid: h for vid, h in hops.items() if vid in addressed},
            relays={vid: s for vid, s in heard_from.items() if vid in addressed},
            suppressed=suppressed,
        )

    def _origin(
        self, world: World, hazard: ResolvedLocation, sender_id: str | None
    ) -> tuple[float, float]:
        if sender_id is not None:
            try:
                sender = world.vehicle(sender_id)
                return sender.x, sender.y
            except KeyError:
                pass
        return hazard.x, hazard.y

    def _in_range(
        self, vehicles: Iterable[VehicleState], x: float, y: float, exclude: str
    ) -> list[tuple[VehicleState, float]]:
        cutoff = CUTOFF_FACTOR * self.comm_range_m
        out = []
        for v in vehicles:
            if v.vehicle_id == exclude:
                continue
            d = math.dist((v.x, v.y), (x, y))
            if d <= cutoff:
                out.append((v, d))
        return out


DEFAULT_MODEL = DeliveryModel()


def deliver(
    message: WarningMessage,
    world: World,
    hazard: ResolvedLocation,
    sender_id: str | None = None,
) -> list[DeliveryRecord]:
    """One record per addressed recipient, using the default model. Use
    ``DEFAULT_MODEL.deliver`` for transmission and reception counts."""
    return list(DEFAULT_MODEL.deliver(message, world, hazard, sender_id).records)
