"""A live SUMO session for the 3D simulation. Owner: P4.

Steps one generated scenario through ``SumoWorld``. When a report falls due it
runs through the real pipeline (P2's reading, P3's confidence, P4's geocast and
delivery), and the session keeps what the 3D view needs: every vehicle's
state, each warning's recipients, who relayed it to whom and when, and the
ground truth of who will reach the hazard.

Report text comes from a ``ReportPool``. Until P1's real test split exists that
is P3's fixture pool (``source="fixture"``), and every event says so.
"""

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from v2v.confidence.dedup import HazardTracker
from v2v.contracts import DriverReport, ExperimentConfig, RunResult, VehicleState
from v2v.eval.runner import DEMO_POOL, ReportPool
from v2v.experiments import CONFIGS
from v2v.geocast import directional
from v2v.pipeline import process_report
from v2v.sim.delivery import DEFAULT_MODEL, DeliveryModel, DeliveryOutcome
from v2v.sim.scenarios import ReportSlot, ScenarioSpec
from v2v.sim.traci_loop import SumoWorld, net_file, road_graph

ORIGIN = "origin"
"""Relay id for a warning sent from the hazard point (no sender vehicle)."""


@dataclass(frozen=True)
class Pulse:
    """One V2V hop: ``src`` transmits and ``dst`` first receives the warning."""

    src: str
    dst: str
    start_ms: float
    end_ms: float
    hop: int


@dataclass(frozen=True)
class WarningEvent:
    """One report and what the system did with it."""

    t_s: float
    t_pass_s: float
    slot: ReportSlot
    text: str
    text_source: str
    result: RunResult
    outcome: DeliveryOutcome | None
    """Delivery with hop counts and relays; None when the warning was suppressed."""
    sender_xy: tuple[float, float]
    centre_xy: tuple[float, float]
    """Where the system placed the hazard (the geocast centre)."""
    radius_m: float
    pulses: tuple[Pulse, ...] = ()
    zone_edges: dict[str, float] = field(default_factory=dict)
    """Directional geocast on the road map: each covered edge -> metres of it
    inside the zone (backward, forward or both carriageways, as the hazard
    type decides). Empty for circular and broadcast, and when the heading
    rule had to stand in."""

    @property
    def sent(self) -> bool:
        return self.result.warning is not None

    @property
    def delivered(self) -> dict[str, float]:
        return {
            r.vehicle_id: r.latency_ms
            for r in self.result.deliveries
            if r.delivered and r.latency_ms is not None
        }

    @property
    def false_alarm(self) -> bool:
        """A warning went out although the event never happened (misleading reports)."""
        return self.sent and not self.result.event_occurred

    @property
    def relevant(self) -> frozenset[str]:
        """Vehicles that needed this warning: those that will reach a hazard that
        is really there. When the event never happened, nobody needed it."""
        if not self.result.event_occurred:
            return frozenset()
        return frozenset(self.result.relevant_vehicle_ids)

    @property
    def missed(self) -> frozenset[str]:
        """Will reach the hazard but the warning never arrived."""
        return self.relevant - set(self.delivered)

    @property
    def unneeded(self) -> frozenset[str]:
        """Warned but did not need it (every recipient, for a false alarm)."""
        return frozenset(self.delivered) - self.relevant


def pulses(outcome: DeliveryOutcome, sender: str) -> tuple[Pulse, ...]:
    """The relay tree as timed hops, earliest first."""
    latency = {r.vehicle_id: r.latency_ms for r in outcome.records if r.latency_ms is not None}
    out = []
    for vid, hop in outcome.hops.items():
        if vid not in latency:
            continue
        src = outcome.relays.get(vid, ORIGIN)
        src = sender if src == ORIGIN and sender != ORIGIN else src
        out.append(Pulse(src, vid, latency.get(src, 0.0), latency[vid], hop))
    return tuple(sorted(out, key=lambda p: (p.end_ms, p.dst)))


def scenario_specs(root: Path) -> list[ScenarioSpec]:
    """Every generated scenario under ``root`` (``sim_out/scenarios/<corridor>``)."""
    return [ScenarioSpec.load(p) for p in sorted(root.glob("*/spec.json"))]


@dataclass
class LiveSession:
    """One scenario playing in SUMO, steppable from a render loop."""

    spec: ScenarioSpec
    config_name: str = "proposed"
    pool: ReportPool = field(default_factory=lambda: ReportPool(DEMO_POOL, allow_reuse=True))
    seed: int = 0
    model: DeliveryModel = DEFAULT_MODEL
    events: list[WarningEvent] = field(default_factory=list)
    _world: SumoWorld | None = None
    _tracker: HazardTracker | None = None
    _used: set[str] = field(default_factory=set)

    @property
    def config(self) -> ExperimentConfig:
        return CONFIGS[self.config_name]

    @property
    def world(self) -> SumoWorld:
        if self._world is None:
            self.open()
        assert self._world is not None
        return self._world

    @property
    def time_s(self) -> float:
        return self.world.time_s

    @property
    def finished(self) -> bool:
        return self.time_s >= self.spec.end_s

    @property
    def event_active(self) -> bool:
        """Ground truth: the hazard is physically there right now."""
        t = self.time_s
        return self.spec.event_occurred and self.spec.event_start_s <= t < self.spec.event_end_s

    def open(self) -> None:
        self.close()
        self._world = SumoWorld(self.spec)
        self._tracker = HazardTracker(self.config)
        self.events = []
        self._used = set()

    def close(self) -> None:
        if self._world is not None:
            self._world.close()
            self._world = None

    def reset(self, config_name: str | None = None) -> None:
        """Back to t=0, optionally with another experiment config."""
        if config_name is not None:
            self.config_name = config_name
        self.open()

    def advance(self) -> list[WarningEvent]:
        """One SUMO step; returns the reports that fired in it."""
        world = self.world
        if self.finished:
            return []
        world.step()
        fired = [self._fire(slot, t_pass) for slot, t_pass in world.due_reports()]
        new = [e for e in fired if e is not None]
        self.events.extend(new)
        return new

    def vehicles(self) -> list[VehicleState]:
        return self.world.vehicles()

    def blocking(self) -> list[VehicleState]:
        return self.world.blocking_vehicles()

    def will_reach(self, ids: Iterable[str]) -> set[str]:
        """Ground truth now: which of ``ids`` will reach the hazard."""
        world, hazard = self.world, self.spec.hazard
        return {vid for vid in ids if world.will_reach(vid, hazard)}

    def _fire(self, slot: ReportSlot, t_pass: float) -> WarningEvent | None:
        world, spec = self.world, self.spec
        record = self.pool.pick(
            slot.needs, slot.hazard_class, f"{spec.scenario_id}-{slot.slot_id}", self._used
        )
        if record is None:
            return None  # no text of the kind this slot needs
        self._used.add(record.report_id)
        report = DriverReport(
            report_id=slot.slot_id,
            text=record.text,
            source=record.source,
            reporter_id=slot.reporter_id,
            t_sim=world.time_s,
        )
        assert self._tracker is not None
        result = process_report(
            report, world, spec.to_scenario(), self.config, self._tracker, self.seed
        )
        reporter = world.vehicle(slot.reporter_id)
        location = result.assessed.location
        centre = (location.x, location.y) if location is not None else (reporter.x, reporter.y)
        outcome = None
        hops: tuple[Pulse, ...] = ()
        if result.warning is not None:
            # The pipeline's delivery is seeded by the message id, so this replays
            # exactly the same outcome, with the hop counts and relays kept.
            outcome = self.model.deliver(
                result.warning, world, spec.hazard, sender_id=slot.reporter_id
            )
            hops = pulses(outcome, slot.reporter_id)
        zone: dict[str, float] = {}
        graph = road_graph(net_file(Path(spec.sumocfg)))
        if (
            result.warning is not None
            and self.config.distribution == "directional"
            and location is not None
            and location.edge_id in graph.length_m
        ):
            reach = directional.reach_for(result.assessed.hazard)
            zone = {
                edge: span[1] - span[0]
                for edge, span in directional.road_zone(
                    graph, location, self.config.radius_m, reach
                ).items()
            }
        return WarningEvent(
            t_s=world.time_s,
            t_pass_s=t_pass,
            slot=slot,
            text=record.text,
            text_source=record.source,
            result=result,
            outcome=outcome,
            sender_xy=(reporter.x, reporter.y),
            centre_xy=centre,
            radius_m=self.config.radius_m,
            pulses=hops,
            zone_edges=zone,
        )

    def __enter__(self) -> "LiveSession":
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
