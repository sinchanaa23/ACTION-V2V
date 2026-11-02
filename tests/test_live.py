import importlib.util
import os
from pathlib import Path

import pytest

from v2v.contracts import DeliveryRecord
from v2v.sim.delivery import DeliveryOutcome
from v2v.sim.live import ORIGIN, pulses


def _outcome(latency: dict[str, float | None], hops: dict[str, int], relays: dict[str, str]):  # type: ignore[no-untyped-def]
    return DeliveryOutcome(
        records=tuple(
            DeliveryRecord(
                msg_id="m",
                vehicle_id=vid,
                delivered=ms is not None,
                latency_ms=ms,
                relevant=False,
            )
            for vid, ms in latency.items()
        ),
        transmissions=3,
        receptions=4,
        hops=hops,
        relays=relays,
    )


def test_pulses_follow_the_relay_tree_in_time_order() -> None:
    outcome = _outcome(
        {"a": 5.0, "b": 12.0, "c": None},
        hops={"a": 1, "b": 2},
        relays={"a": "rep", "b": "a"},
    )
    out = pulses(outcome, "rep")
    assert [(p.src, p.dst, p.start_ms, p.end_ms, p.hop) for p in out] == [
        ("rep", "a", 0.0, 5.0, 1),
        ("a", "b", 5.0, 12.0, 2),
    ]  # "c" was addressed but never received it: no pulse


def test_pulses_from_the_hazard_point_when_there_is_no_sender() -> None:
    outcome = _outcome({"a": 4.0}, hops={"a": 1}, relays={"a": ORIGIN})
    (pulse,) = pulses(outcome, ORIGIN)
    assert pulse.src == ORIGIN


@pytest.mark.skipif(
    importlib.util.find_spec("traci") is None or not os.environ.get("V2V_NETWORK_TESTS"),
    reason="needs the [sim] extras and network access; set V2V_NETWORK_TESTS=1",
)
def test_live_session_runs_reports_through_the_pipeline(tmp_path: Path) -> None:
    from v2v.sim.live import LiveSession
    from v2v.sim.osm_network import CORRIDORS
    from v2v.sim.scenarios import generate

    (spec,) = generate(CORRIDORS["hebbal"], ["noisy"], ["low"], [0], tmp_path / "networks")
    with LiveSession(spec) as session:
        while not session.finished and len(session.events) < len(spec.slots):
            session.advance()
        events = list(session.events)
        assert [e.slot.role for e in events][0] == "original"
        for e in events:
            assert e.text_source == "fixture"  # labelled until P1's real split exists
            if e.outcome is not None:  # replay matches the pipeline's own delivery
                assert e.outcome.records == e.result.deliveries
                assert {p.dst for p in e.pulses} == set(e.delivered)
        session.reset("baseline_a")
        assert session.time_s == 0.0 and session.events == []
        assert session.config.distribution == "broadcast"


def test_nobody_needs_a_warning_about_an_event_that_never_happened() -> None:
    from v2v.experiments import CONFIGS
    from v2v.pipeline import run
    from v2v.sim.live import WarningEvent
    from v2v.sim.scenarios import ReportSlot

    result = run(CONFIGS["baseline_a"], "Accident ahead near the bridge, slow down")
    assert result.warning is not None and result.relevant_vehicle_ids
    slot = ReportSlot(
        slot_id="s0",
        role="false",
        needs="hazard",
        hazard_class=result.interpretation.hazard,
        reporter_id="rep",
        route=("a",),
        depart_s=0.0,
        delay_s=0.0,
        expected_report_s=0.0,
    )

    def event(occurred: bool) -> WarningEvent:
        return WarningEvent(
            t_s=0.0,
            t_pass_s=0.0,
            slot=slot,
            text="t",
            text_source="fixture",
            result=result.model_copy(update={"event_occurred": occurred}),
            outcome=None,
            sender_xy=(0.0, 0.0),
            centre_xy=(0.0, 0.0),
            radius_m=800.0,
        )

    real, false = event(True), event(False)
    assert real.relevant and not real.false_alarm
    assert false.false_alarm and false.relevant == frozenset() and false.missed == frozenset()
    assert false.unneeded == frozenset(false.delivered)  # every delivery was unneeded
