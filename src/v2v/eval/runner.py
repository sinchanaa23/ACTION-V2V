"""Experiment runner: every config on every scenario, seed and threshold. Owner: P3 (P3-05).

    python -m v2v.eval.runner sim_out/scenarios/hebbal/*-s* --reports data/splits/test/reports.jsonl
    python -m v2v.eval.runner --toy --seeds 0 1 2          # no SUMO, fixture text
    python -m v2v.eval.runner sim_out/scenarios/hebbal/clean-low-s0 --fixtures --max-time 600

One SUMO run per scenario serves all five configs and the whole threshold
grid. Warnings do not change how vehicles drive, so at each report that falls
due the runner reads the report both ways (keyword and NLP), updates one
hazard tracker per reading, picks recipients under each distribution policy
and plays the delivery once; the configs and thresholds then only decide
whether that warning would have gone out. Every row records the git SHA, a
hash of the trained models and the seed, so each number can be traced.

Report text comes from P1's frozen split. Each scenario slot asks for a kind
of text (a report of its hazard class, a negated report, or a non-actionable
one) and a seeded draw picks one real record that fits. If no record fits, the
scenario is skipped and logged rather than filled with a wrong-class text.
"""

import argparse
import hashlib
import json
import platform
import random
import subprocess
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from v2v import pipeline
from v2v.confidence import policy
from v2v.confidence.dedup import ClusterView, HazardTracker
from v2v.contracts import (
    NON_HAZARD_TYPES,
    AssessedHazard,
    DriverReport,
    ExperimentConfig,
    HazardInterpretation,
    HazardProcessing,
    HazardType,
    ResolvedLocation,
    VehicleState,
    WarningMessage,
    World,
)
from v2v.data.annotate import GoldRecord, read_gold
from v2v.eval import metrics
from v2v.experiments import BASELINE_A, CONFIGS, PROPOSED
from v2v.geocast import SELECTORS
from v2v.geocast.fallback import fallback_location
from v2v.messages.warning import build_warning
from v2v.sim.delivery import DEFAULT_MODEL, DeliveryModel

DEFAULT_RESULTS_DIR = Path("experiments/results")
DEFAULT_REPORTS = Path("data/splits/test/reports.jsonl")
MODELS_DIR = Path("models")
SENSOR_RADIUS_M = pipeline.SENSOR_RADIUS_M

TextNeed = str  # "hazard" | "negated_hazard" | "non_actionable", as in sim.scenarios


# --- Report text ---------------------------------------------------------------

DEMO_POOL: tuple[GoldRecord, ...] = tuple(
    GoldRecord(
        report_id=f"demo-{i}",
        text=text,
        source="fixture",
        hazard=hazard,
        negated=negated,
        hedged=False,
    )
    for i, (text, hazard, negated) in enumerate(
        [
            (pipeline.DEMO_REPORT, HazardType.ACCIDENT, False),
            ("Bada gaddha hai flyover ke paas, dhyan se", HazardType.OBSTRUCTION, False),
            ("Oil spill near the signal, road is slippery", HazardType.SLIPPERY, False),
            ("Truck breakdown ho gaya left lane mein", HazardType.STOPPED, False),
            ("Bike speeding on the wrong side near the junction", HazardType.SPEEDING, False),
            ("Accident clear ho gaya, road is open now", HazardType.ACCIDENT, True),
            ("Traffic is fine today, reached office on time.", HazardType.NON_ACTIONABLE, False),
        ]
    )
)
"""Hand-written strings for the toy run and CI only (source="fixture"). They
exercise code paths; results made with them are labelled and never reported."""


class ReportPool:
    """Real records to fill scenario slots, drawn reproducibly."""

    def __init__(self, records: Sequence[GoldRecord]) -> None:
        if not records:
            raise ValueError("empty report pool")
        self.records = tuple(records)

    @classmethod
    def load(cls, path: Path) -> "ReportPool":
        return cls(read_gold(path))

    def candidates(self, needs: TextNeed, hazard_class: HazardType) -> list[GoldRecord]:
        if needs == "hazard":
            return [r for r in self.records if r.hazard == hazard_class and not r.negated]
        if needs == "negated_hazard":
            negated = [r for r in self.records if r.negated]
            same = [r for r in negated if r.hazard == hazard_class]
            return same or negated
        if needs == "non_actionable":
            return [
                r for r in self.records if r.hazard == HazardType.NON_ACTIONABLE and not r.negated
            ]
        raise ValueError(f"unknown text need {needs!r}")

    def pick(
        self, needs: TextNeed, hazard_class: HazardType, key: str, used: set[str]
    ) -> GoldRecord | None:
        """Seeded choice, avoiding records already used in this scenario."""
        options = sorted(
            (r for r in self.candidates(needs, hazard_class) if r.report_id not in used),
            key=lambda r: r.report_id,
        )
        if not options:
            return None
        return random.Random(key).choice(options)


# --- Provenance ----------------------------------------------------------------


def git_sha() -> str:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{sha}-dirty" if dirty else sha


def model_hash(models_dir: Path = MODELS_DIR) -> str:
    """SHA-256 over every file under ``models_dir`` (names and bytes), or
    "none" when no trained model exists (keyword fallback)."""
    files = sorted(p for p in models_dir.rglob("*") if p.is_file()) if models_dir.exists() else []
    if not files:
        return "none"
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(models_dir).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


# --- One report, all configs ------------------------------------------------------


@dataclass(frozen=True)
class ReportEvent:
    """A report falling due in a run: which slot, what text it needs."""

    slot_id: str
    role: str
    needs: TextNeed
    hazard_class: HazardType
    reporter_id: str


@dataclass(frozen=True)
class RunInfo:
    scenario_id: str
    condition: str
    density: str
    seed: int
    hazard: ResolvedLocation
    event_occurred: bool
    event_start_s: float
    event_end_s: float

    def event_active(self, t_s: float) -> bool:
        return self.event_occurred and self.event_start_s <= t_s <= self.event_end_s


@dataclass
class _Reading:
    interp: HazardInterpretation
    location: ResolvedLocation
    flags: int
    view: ClusterView


_PROCESSING_CONFIG: dict[HazardProcessing, ExperimentConfig] = {
    "keyword": BASELINE_A,
    "nlp": PROPOSED,
}


class Evaluator:
    """Scores every config and threshold pair at each report of one run."""

    def __init__(
        self,
        world: World,
        run: RunInfo,
        configs: Sequence[ExperimentConfig],
        grid: Sequence[tuple[float, float]],
        delivery_model: DeliveryModel = DEFAULT_MODEL,
        provenance: dict[str, str] | None = None,
    ) -> None:
        self.world = world
        self.run = run
        self.configs = list(configs)
        self.grid = list(grid)
        self.delivery_model = delivery_model
        self.provenance = provenance or {}
        processing = {c.hazard_processing for c in self.configs}
        self.trackers = {p: HazardTracker(_PROCESSING_CONFIG[p]) for p in processing}

    def _read(self, report: DriverReport, processing: HazardProcessing) -> _Reading:
        assert report.reporter_id is not None
        interp = pipeline.interpret(report, _PROCESSING_CONFIG[processing])
        reporter = self.world.vehicle(report.reporter_id)
        location = self.world.resolve_location(
            interp.location_text, reporter.vehicle_id
        ) or fallback_location(reporter)
        flags = self.world.sensor_flags(location, SENSOR_RADIUS_M)
        view = self.trackers[processing].observe(
            interp, location, flags, reporter_id=report.reporter_id, t_s=self.world.time_s
        )
        return _Reading(interp, location, flags, view)

    def evaluate(self, event: ReportEvent, record: GoldRecord) -> list[dict[str, Any]]:
        t = self.world.time_s
        report = DriverReport(
            report_id=f"{self.run.scenario_id}-{event.slot_id}",
            text=record.text,
            source=record.source,
            reporter_id=event.reporter_id,
            t_sim=t,
        )
        readings = {p: self._read(report, p) for p in self.trackers}
        vehicles = [v for v in self.world.vehicles() if v.vehicle_id != event.reporter_id]
        relevant = sorted(
            v.vehicle_id for v in vehicles if self.world.will_reach(v.vehicle_id, self.run.hazard)
        )
        cache: dict[tuple[Any, ...], dict[str, Any]] = {}
        rows: list[dict[str, Any]] = []
        base = {
            "scenario_id": self.run.scenario_id,
            "condition": self.run.condition,
            "density": self.run.density,
            "seed": self.run.seed,
            "slot_id": event.slot_id,
            "role": event.role,
            "t_s": round(t, 1),
            "text_source": record.source,
            "gold_report_id": record.report_id,
            "gold_hazard": record.hazard.value,
            "gold_negated": record.negated,
            "event_active": self.run.event_active(t),
            "n_vehicles": len(vehicles),
            "n_relevant": len(relevant),
            **self.provenance,
        }
        for config in self.configs:
            reading = readings[config.hazard_processing]
            location = reading.view.location if config.confidence else reading.location
            outcome = self._distribute(
                cache, config, reading, location or reading.location, vehicles, event
            )
            thresholds = (
                self.grid if config.confidence else sorted({(ti, 0.0) for ti, _ in self.grid})
            )
            for theta_i, theta_e in thresholds:
                status = self._status(config, reading, theta_i, theta_e)
                warned = status != "suppressed"
                rows.append(
                    {
                        **base,
                        "config": config.name,
                        "theta_i": theta_i,
                        "theta_e": theta_e if config.confidence else None,
                        "pred_hazard": reading.interp.hazard.value,
                        "p_class": reading.interp.p_class,
                        "pred_negated": reading.interp.negated,
                        "pred_hedged": reading.interp.hedged,
                        "location_text": reading.interp.location_text,
                        "interp_conf": reading.view.interp_conf,
                        "event_evidence": reading.view.event_evidence,
                        "n_reports": reading.view.n_reports,
                        "sensor_flags": reading.flags,
                        "status": status,
                        "warning_text": outcome["text"] if warned else None,
                        **metrics.report_metrics(
                            warned=warned,
                            event_active=self.run.event_active(t),
                            needs_warning=event.needs == "hazard",
                            recipients=outcome["recipients"],
                            delivered=outcome["delivered"],
                            relevant=relevant,
                            latencies_ms=outcome["latencies"],
                            transmissions=outcome["transmissions"],
                            location_approx=(location or reading.location).approx,
                        ),
                    }
                )
        return rows

    @staticmethod
    def _status(config: ExperimentConfig, reading: _Reading, theta_i: float, theta_e: float) -> str:
        if config.confidence:
            return HazardTracker.assess(reading.view, reading.interp, theta_i, theta_e).status
        # Baselines: the class threshold alone decides, as in pipeline.assess.
        actionable = policy.is_actionable(reading.interp) and reading.interp.p_class >= theta_i
        return "unverified" if actionable else "suppressed"

    def _distribute(
        self,
        cache: dict[tuple[Any, ...], dict[str, Any]],
        config: ExperimentConfig,
        reading: _Reading,
        location: ResolvedLocation,
        vehicles: list[VehicleState],
        event: ReportEvent,
    ) -> dict[str, Any]:
        """Recipients and delivery for a warning, shared by every threshold."""
        key = (config.hazard_processing, config.confidence, config.distribution)
        if key in cache:
            return cache[key]
        hazard = _as_assessed(reading, location, config.confidence)
        recipients = SELECTORS[config.distribution](hazard, vehicles, config.radius_m)
        warning: WarningMessage = build_warning(
            hazard,
            recipients,
            config.distribution,
            msg_id=f"{self.run.scenario_id}-{event.slot_id}-{'-'.join(map(str, key))}",
            t_sent=self.world.time_s,
        )
        outcome = self.delivery_model.deliver(
            warning, self.world, self.run.hazard, sender_id=event.reporter_id, seed=self.run.seed
        )
        cache[key] = {
            "text": warning.text,
            "recipients": recipients,
            "delivered": [r.vehicle_id for r in outcome.records if r.delivered],
            "latencies": [r.latency_ms for r in outcome.records if r.latency_ms is not None],
            "transmissions": outcome.transmissions,
        }
        return cache[key]


def _as_assessed(reading: _Reading, location: ResolvedLocation, clustered: bool) -> AssessedHazard:
    """The hazard as geocast and the wording see it. Its status is a placeholder
    ("unverified"): recipients don't depend on it, and the per-threshold status
    is recorded separately in each row."""
    view = reading.view
    hazard = view.hazard if clustered else reading.interp.hazard
    if hazard in NON_HAZARD_TYPES:
        hazard = HazardType.ACCIDENT  # wording only; such reports are always suppressed
    return AssessedHazard(
        cluster_id=view.cluster_id,
        hazard=hazard,
        location=location,
        interp_conf=min(1.0, max(0.0, view.interp_conf)),
        event_evidence=view.event_evidence,
        n_reports=view.n_reports if clustered else 1,
        age_s=view.age_s if clustered else 0.0,
        status="unverified",
    )


# --- Driving a world -----------------------------------------------------------------


def _fill(pool: ReportPool, event: ReportEvent, run: RunInfo, used: set[str]) -> GoldRecord | None:
    record = pool.pick(
        event.needs, event.hazard_class, f"{run.scenario_id}-{event.slot_id}-{run.seed}", used
    )
    if record is not None:
        used.add(record.report_id)
    return record


def run_sumo_scenario(
    spec_path: Path,
    pool: ReportPool,
    configs: Sequence[ExperimentConfig],
    grid: Sequence[tuple[float, float]],
    provenance: dict[str, str],
    log: Callable[[str], None] = print,
    max_time_s: float | None = None,
) -> list[dict[str, Any]]:
    """Play one generated scenario through SumoWorld and score every report."""
    from v2v.sim.scenarios import ScenarioSpec
    from v2v.sim.traci_loop import SumoWorld

    spec = ScenarioSpec.load(spec_path)
    run = RunInfo(
        scenario_id=spec.scenario_id,
        condition=spec.condition,
        density=spec.density,
        seed=spec.seed,
        hazard=spec.hazard,
        event_occurred=spec.event_occurred,
        event_start_s=spec.event_start_s,
        event_end_s=spec.event_end_s,
    )
    # Pick every slot's text up front: a scenario runs only if all slots fit.
    used: set[str] = set()
    records: dict[str, GoldRecord] = {}
    for slot in spec.slots:
        event = ReportEvent(
            slot.slot_id, slot.role, slot.needs, slot.hazard_class, slot.reporter_id
        )
        record = _fill(pool, event, run, used)
        if record is None:
            log(f"skip {spec.scenario_id}: no {slot.needs} text for {slot.hazard_class.value}")
            return []
        records[slot.slot_id] = record
    end = spec.end_s if max_time_s is None else min(spec.end_s, max_time_s)
    rows: list[dict[str, Any]] = []
    with SumoWorld(spec) as world:
        evaluator = Evaluator(world, run, configs, grid, provenance=provenance)
        while world.time_s < end and len(world.clock.fired) < len(spec.slots):
            world.step()
            for slot, _ in world.due_reports():
                event = ReportEvent(
                    slot.slot_id, slot.role, slot.needs, slot.hazard_class, slot.reporter_id
                )
                rows.extend(evaluator.evaluate(event, records[slot.slot_id]))
    unfired = len(spec.slots) - len({r["slot_id"] for r in rows})
    if unfired:
        log(f"{spec.scenario_id}: {unfired} report(s) never fell due before t={end:.0f} s")
    return rows


def run_toy(
    seed: int,
    pool: ReportPool,
    configs: Sequence[ExperimentConfig],
    grid: Sequence[tuple[float, float]],
    provenance: dict[str, str],
) -> list[dict[str, Any]]:
    """The same scoring on the toy world: one genuine accident report."""
    from v2v.sim.scenarios import toy_scenario
    from v2v.sim.toy_world import ToyWorld

    scenario = toy_scenario(seed)
    run = RunInfo(
        scenario_id=scenario.scenario_id,
        condition="clean",
        density="toy",
        seed=seed,
        hazard=scenario.hazard_location,
        event_occurred=True,
        event_start_s=0.0,
        event_end_s=float("inf"),
    )
    event = ReportEvent("slot_0", "original", "hazard", HazardType.ACCIDENT, scenario.reporter_id)
    record = _fill(pool, event, run, set())
    if record is None:
        return []
    world = ToyWorld(scenario, seed=seed)
    try:
        while world.time_s < scenario.report_time_s:
            world.step()
        return Evaluator(world, run, configs, grid, provenance=provenance).evaluate(event, record)
    finally:
        world.close()


def run_grid(
    configs: Iterable[ExperimentConfig], seeds: Iterable[int], report_text: str
) -> list[dict[str, object]]:
    """One metrics row per (config, seed) for a single report on the toy world,
    at each config's own thresholds. Kept for quick checks and the tests."""
    rows: list[dict[str, object]] = []
    seed_list = list(seeds)
    for config in configs:
        for seed in seed_list:
            result = pipeline.run(config, report_text, seed=seed)
            rows.append({"config": config.name, "seed": seed, **metrics.summarise(result)})
    return rows


def write_results(
    rows: Sequence[dict[str, Any]], meta: dict[str, Any], out_dir: Path, name: str
) -> Path:
    """Rows as JSONL plus a meta file. Returns the rows path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / f"{name}.jsonl"
    rows_path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    (out_dir / f"{name}.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return rows_path


def _spec_paths(paths: Iterable[Path]) -> Iterator[Path]:
    for path in paths:
        yield path / "spec.json" if path.is_dir() else path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenarios", nargs="*", type=Path, help="scenario dirs or spec.json files")
    parser.add_argument("--reports", type=Path, default=DEFAULT_REPORTS)
    parser.add_argument("--configs", nargs="+", choices=sorted(CONFIGS), default=sorted(CONFIGS))
    parser.add_argument("--theta-i", nargs="+", type=float, default=list(policy.DEFAULT_THETA_I))
    parser.add_argument("--theta-e", nargs="+", type=float, default=list(policy.DEFAULT_THETA_E))
    parser.add_argument("--toy", action="store_true", help="toy world with fixture text (no SUMO)")
    parser.add_argument(
        "--fixtures", action="store_true", help="smoke test: fixture text instead of --reports"
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0], help="toy run seeds")
    parser.add_argument("--max-time", type=float, help="stop each scenario at this sim time")
    parser.add_argument("--out", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--name", default=time.strftime("run-%Y%m%d-%H%M%S"))
    args = parser.parse_args(argv)

    configs = [CONFIGS[name] for name in args.configs]
    grid = policy.threshold_grid(args.theta_i, args.theta_e)
    provenance = {"git_sha": git_sha(), "model_hash": model_hash()}
    fixtures = args.toy or args.fixtures
    pool = ReportPool(DEMO_POOL) if fixtures else ReportPool.load(args.reports)
    rows: list[dict[str, Any]] = []
    if args.toy:
        for seed in args.seeds:
            rows.extend(run_toy(seed, pool, configs, grid, provenance))
    else:
        if not args.scenarios:
            parser.error("give scenario directories, or --toy")
        for spec_path in _spec_paths(args.scenarios):
            print(f"running {spec_path}")
            rows.extend(
                run_sumo_scenario(
                    spec_path, pool, configs, grid, provenance, max_time_s=args.max_time
                )
            )
    meta = {
        **provenance,
        "configs": args.configs,
        "grid": grid,
        "reports": "fixture (smoke test, not results)" if fixtures else str(args.reports),
        "n_rows": len(rows),
        "python": platform.python_version(),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    path = write_results(rows, meta, args.out, args.name)
    print(f"wrote {len(rows)} rows to {path}")


if __name__ == "__main__":
    main()
