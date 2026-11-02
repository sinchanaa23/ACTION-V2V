"""Orchestrator: one report through all five stages. Owner: P3.

python -m v2v.pipeline --config proposed --seed 0
"""

import argparse
import json

from v2v.confidence import policy
from v2v.confidence.dedup import HazardTracker
from v2v.contracts import (
    AssessedHazard,
    DeliveryRecord,
    DriverReport,
    ExperimentConfig,
    HazardInterpretation,
    ResolvedLocation,
    RunResult,
    Scenario,
    Source,
    World,
)
from v2v.eval import metrics
from v2v.experiments import CONFIGS
from v2v.geocast import SELECTORS, fallback
from v2v.messages.warning import build_warning
from v2v.nlp import classifier, keyword_baseline, preprocess
from v2v.sim import delivery, scenarios
from v2v.sim.toy_world import ToyWorld

DEMO_REPORT = "Bhai, aage accident hua hai, bridge ke paas. Slow down!"
"""The proposal's worked example. Demo input only; never training or evaluation data."""

SENSOR_RADIUS_M = 200.0


def interpret(report: DriverReport, config: ExperimentConfig) -> HazardInterpretation:
    text = preprocess.normalise(report.text)
    if config.hazard_processing == "keyword":
        return keyword_baseline.classify(report.report_id, text)
    return classifier.interpret(report.report_id, text)


def interp_confidence(interp: HazardInterpretation, config: ExperimentConfig) -> float:
    """Confidence in the reading for theta_i. NLP readings use P2's
    interpretation_confidence (lower for hedged or location-less reports);
    the keyword baseline has only its fixed p_class."""
    if config.hazard_processing == "nlp":
        return classifier.interpretation_confidence(interp)
    return interp.p_class


def assess(
    interp: HazardInterpretation,
    location: ResolvedLocation,
    world: World,
    config: ExperimentConfig,
    tracker: HazardTracker,
    reporter_id: str | None = None,
) -> AssessedHazard:
    if config.confidence:
        return tracker.ingest(
            interp,
            location,
            world.sensor_flags(location, SENSOR_RADIUS_M),
            reporter_id=reporter_id,
            t_s=world.time_s,
            interp_conf=interp_confidence(interp, config),
        )
    # Baselines: the class threshold alone decides; no evidence stage.
    actionable = policy.is_actionable(interp) and interp.p_class >= config.theta_i
    return AssessedHazard(
        cluster_id=f"r-{interp.report_id}",
        hazard=interp.hazard,
        location=location,
        interp_conf=interp.p_class,
        event_evidence=0.0,
        n_reports=1,
        age_s=0.0,
        status="unverified" if actionable else "suppressed",
    )


def process_report(
    report: DriverReport,
    world: World,
    scenario: Scenario,
    config: ExperimentConfig,
    tracker: HazardTracker,
    seed: int,
) -> RunResult:
    if report.reporter_id is None:
        raise ValueError("a report injected into a scenario needs a reporter_id")
    interp = interpret(report, config)
    reporter = world.vehicle(report.reporter_id)
    location = world.resolve_location(
        interp.location_text, reporter.vehicle_id
    ) or fallback.fallback_location(reporter)
    assessed = assess(interp, location, world, config, tracker, report.reporter_id)

    vehicles = world.vehicles()
    relevant = tuple(
        sorted(
            v.vehicle_id
            for v in vehicles
            if world.will_reach(v.vehicle_id, scenario.hazard_location)
        )
    )
    warning = None
    deliveries: tuple[DeliveryRecord, ...] = ()
    if assessed.status != "suppressed":
        recipients = SELECTORS[config.distribution](assessed, vehicles, config.radius_m)
        warning = build_warning(
            assessed,
            recipients,
            config.distribution,
            msg_id=f"{report.report_id}-m0",
            t_sent=world.time_s,
            location_text=interp.location_text,
        )
        deliveries = tuple(
            delivery.deliver(warning, world, scenario.hazard_location, report.reporter_id)
        )

    return RunResult(
        config_name=config.name,
        seed=seed,
        scenario_id=scenario.scenario_id,
        event_occurred=scenario.event_occurred,
        report=report,
        interpretation=interp,
        assessed=assessed,
        warning=warning,
        deliveries=deliveries,
        relevant_vehicle_ids=relevant,
    )


def run(
    config: ExperimentConfig, report_text: str, seed: int = 0, source: Source = "fixture"
) -> RunResult:
    """Run one report through the toy world (CLI demo and CI smoke run).

    Experiments on the real network go through ``v2v.eval.runner``, which
    plays generated scenarios in SumoWorld and scores every config at once.
    """
    scenario = scenarios.toy_scenario(seed)
    world = ToyWorld(scenario, seed=seed)
    try:
        while world.time_s < scenario.report_time_s:
            world.step()
        report = DriverReport(
            report_id=f"{scenario.scenario_id}-r0",
            text=report_text,
            source=source,
            reporter_id=scenario.reporter_id,
            t_sim=world.time_s,
        )
        return process_report(report, world, scenario, config, HazardTracker(config), seed)
    finally:
        world.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", choices=sorted(CONFIGS), default="proposed")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--text", default=DEMO_REPORT)
    args = parser.parse_args(argv)

    result = run(CONFIGS[args.config], args.text, seed=args.seed)
    summary = {
        "config": result.config_name,
        "hazard": result.interpretation.hazard.value,
        "status": result.assessed.status,
        "warning": result.warning.text if result.warning else None,
        **metrics.summarise(result),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
