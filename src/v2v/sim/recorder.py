"""Record what each distribution policy does at every report. Owner: P4 (P4-07).

    python -m v2v.sim.recorder sim_out/scenarios/hebbal/*-s0

Plays scenarios through ``SumoWorld``. At each report that falls due it saves
a snapshot of every vehicle (position, heading, ground-truth relevance), and
for each policy and each way of knowing the location (the true hazard point,
or the no-location fallback) the recipients and the delivery outcome. Traces
go to ``sim_out/traces/<scenario_id>.json`` for the dashboard.

Report text, the NLP reading and the confidence verdict are not recorded:
they need P1's real test split and P2/P3's models, and arrive with P3's
experiment runner. Until then every recorded warning is treated as sent.
"""

import argparse
import math
import statistics
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from v2v.contracts import (
    AssessedHazard,
    Distribution,
    ResolvedLocation,
    VehicleState,
    WarningMessage,
)
from v2v.geocast import SELECTORS
from v2v.geocast.fallback import fallback_location
from v2v.sim.delivery import DEFAULT_MODEL, DeliveryModel
from v2v.sim.scenarios import ReportSlot, ScenarioSpec
from v2v.sim.traci_loop import SumoWorld

LocationMode = Literal["known", "fallback"]
DEFAULT_TRACE_DIR = Path("sim_out/traces")
RADIUS_M = 800.0


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class VehicleSnap(_Model):
    vehicle_id: str
    lon: float
    lat: float
    heading_deg: float
    speed_mps: float
    edge_id: str
    relevant: bool
    """Ground truth: this vehicle will reach the hazard."""


class PolicyResult(_Model):
    policy: Distribution
    location: LocationMode
    recipients: tuple[str, ...]
    latency_ms: dict[str, float]
    """Delivered recipients and their latency; missing = not delivered."""
    hops: dict[str, int]
    transmissions: int
    receptions: int


class ReportFrame(_Model):
    slot: ReportSlot
    t_s: float
    t_pass_s: float
    reporter: VehicleSnap
    fallback_lon: float
    fallback_lat: float
    sensor_flags: int
    vehicles: tuple[VehicleSnap, ...]
    results: tuple[PolicyResult, ...]

    def result(self, policy: str, location: LocationMode) -> PolicyResult:
        return next(r for r in self.results if r.policy == policy and r.location == location)


class Trace(_Model):
    spec: ScenarioSpec
    hazard_lon: float
    hazard_lat: float
    radius_m: float
    comm_range_m: float
    frames: tuple[ReportFrame, ...]
    vehicles_passed_hazard: int
    teleported: int

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Trace":
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


def policy_metrics(frame: ReportFrame, result: PolicyResult) -> dict[str, float | None]:
    """Recipient selection and delivery against ground truth, for one report."""
    relevant = {v.vehicle_id for v in frame.vehicles if v.relevant}
    chosen = set(result.recipients)
    delivered = set(result.latency_ms)
    latencies = list(result.latency_ms.values())
    return {
        "recall": len(chosen & relevant) / len(relevant) if relevant else None,
        "precision": len(chosen & relevant) / len(chosen) if chosen else None,
        "relevant_reached": len(delivered & relevant) / len(relevant) if relevant else None,
        "irrelevant_reached": float(len(delivered - relevant)),
        "transmissions": float(result.transmissions),
        "receptions": float(result.receptions),
        "mean_latency_ms": statistics.mean(latencies) if latencies else None,
        "recipients": float(len(chosen)),
    }


def classify(frame: ReportFrame, result: PolicyResult) -> dict[str, str]:
    """Each vehicle's outcome under a policy: what the map colours mean."""
    chosen, delivered = set(result.recipients), set(result.latency_ms)
    out: dict[str, str] = {}
    for v in frame.vehicles:
        if v.relevant and v.vehicle_id in delivered:
            out[v.vehicle_id] = "warned, needed it"
        elif v.relevant and v.vehicle_id in chosen:
            out[v.vehicle_id] = "addressed, not delivered"
        elif v.relevant:
            out[v.vehicle_id] = "missed"
        elif v.vehicle_id in delivered:
            out[v.vehicle_id] = "warned, didn't need it"
        else:
            out[v.vehicle_id] = "not warned"
    return out


def _snap(world: SumoWorld, v: VehicleState, relevant: bool) -> VehicleSnap:
    lon, lat = world.lonlat(v.x, v.y)
    return VehicleSnap(
        vehicle_id=v.vehicle_id,
        lon=round(lon, 7),
        lat=round(lat, 7),
        heading_deg=round(v.heading_deg, 1),
        speed_mps=round(v.speed_mps, 2),
        edge_id=v.edge_id,
        relevant=relevant,
    )


def record(spec: ScenarioSpec, model: DeliveryModel = DEFAULT_MODEL) -> Trace:
    frames: list[ReportFrame] = []
    with SumoWorld(spec) as world:
        hazard_lon, hazard_lat = world.lonlat(spec.hazard.x, spec.hazard.y)
        while world.time_s < spec.end_s and len(world.clock.fired) < len(spec.slots):
            world.step()
            for slot, t_pass in world.due_reports():
                frames.append(_frame(world, spec, slot, t_pass, model))
        return Trace(
            spec=spec,
            hazard_lon=round(hazard_lon, 7),
            hazard_lat=round(hazard_lat, 7),
            radius_m=RADIUS_M,
            comm_range_m=model.comm_range_m,
            frames=tuple(frames),
            vehicles_passed_hazard=len(world.hazard_passes),
            teleported=len(world.teleported),
        )


def _frame(
    world: SumoWorld, spec: ScenarioSpec, slot: ReportSlot, t_pass: float, model: DeliveryModel
) -> ReportFrame:
    reporter = world.vehicle(slot.reporter_id)
    vehicles = [v for v in world.vehicles() if v.vehicle_id != slot.reporter_id]
    fallback = fallback_location(reporter)
    results: list[PolicyResult] = []
    modes: tuple[tuple[LocationMode, ResolvedLocation], ...] = (
        ("known", spec.hazard),
        ("fallback", fallback),
    )
    for mode, location in modes:
        hazard = AssessedHazard(
            cluster_id=slot.slot_id,
            hazard=spec.hazard_class,
            location=location,
            interp_conf=1.0,
            event_evidence=0.0,
            n_reports=1,
            age_s=0.0,
            status="unverified",
        )
        for policy, select in SELECTORS.items():
            message = WarningMessage(
                msg_id=f"{spec.scenario_id}-{slot.slot_id}-{mode}-{policy}",
                cluster_id=slot.slot_id,
                text="(recorded without report text)",
                status="unverified",
                policy=policy,
                recipients=tuple(select(hazard, vehicles, RADIUS_M)),
                t_sent=world.time_s,
            )
            outcome = model.deliver(message, world, spec.hazard, sender_id=slot.reporter_id)
            results.append(
                PolicyResult(
                    policy=policy,
                    location=mode,
                    recipients=message.recipients,
                    latency_ms={
                        r.vehicle_id: r.latency_ms
                        for r in outcome.records
                        if r.latency_ms is not None
                    },
                    hops=outcome.hops,
                    transmissions=outcome.transmissions,
                    receptions=outcome.receptions,
                )
            )
    fb_lon, fb_lat = world.lonlat(fallback.x, fallback.y)
    return ReportFrame(
        slot=slot,
        t_s=world.time_s,
        t_pass_s=t_pass,
        reporter=_snap(world, reporter, relevant=False),
        fallback_lon=round(fb_lon, 7),
        fallback_lat=round(fb_lat, 7),
        sensor_flags=world.sensor_flags(spec.hazard, 200.0),
        vehicles=tuple(
            _snap(world, v, world.will_reach(v.vehicle_id, spec.hazard)) for v in vehicles
        ),
        results=tuple(results),
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Record scenario runs for the dashboard")
    parser.add_argument("scenario_dirs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=DEFAULT_TRACE_DIR)
    args = parser.parse_args(argv)
    for scenario_dir in args.scenario_dirs:
        spec = ScenarioSpec.load(scenario_dir / "spec.json")
        trace = record(spec)
        path = args.out / f"{spec.scenario_id}.json"
        trace.save(path)
        best = [
            policy_metrics(f, f.result("directional", "known"))["relevant_reached"]
            for f in trace.frames
        ]
        reached = [b for b in best if b is not None]
        print(
            f"{spec.scenario_id:<32} {len(trace.frames)} reports -> {path} "
            f"(directional reached {statistics.mean(reached) if reached else math.nan:.0%})"
        )


if __name__ == "__main__":
    main()
