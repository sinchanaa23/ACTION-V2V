import importlib.util
import os
from pathlib import Path

import pytest

from tests.fixtures import ACCIDENT_HINGLISH
from v2v.contracts import HazardType
from v2v.sim.scenarios import ReportSlot
from v2v.sim.traci_loop import ReportClock, passed_point

ROUTE = ("a", "h", "b")


@pytest.mark.parametrize(
    ("index", "lane_pos", "expected"),
    [
        (0, 500.0, False),  # still upstream
        (1, 49.0, False),  # on the hazard edge, short of the point
        (1, 50.0, True),  # at the point
        (2, 0.0, True),  # beyond it
    ],
)
def test_passed_point_along_route(index: int, lane_pos: float, expected: bool) -> None:
    assert passed_point(ROUTE, index, lane_pos, "h", 50.0) is expected


def test_point_off_route_is_none() -> None:
    assert passed_point(ROUTE, 0, 0.0, "x", 50.0) is None


def _slot(i: int, delay: float) -> ReportSlot:
    return ReportSlot(
        slot_id=f"slot_{i}",
        role="original",
        needs="hazard",
        hazard_class=HazardType.ACCIDENT,
        reporter_id=f"rep_{i}",
        route=ROUTE,
        depart_s=0.0,
        delay_s=delay,
        expected_report_s=0.0,
    )


def test_report_clock_fires_once_after_real_pass_plus_delay() -> None:
    clock = ReportClock((_slot(0, 5.0), _slot(1, 600.0)))
    assert clock.due(100.0) == []  # nobody has passed yet
    clock.mark_passed("rep_0", 100.0)
    clock.mark_passed("rep_1", 90.0)
    clock.mark_passed("rep_0", 120.0)  # first pass counts
    assert clock.due(104.0) == []
    assert [(s.slot_id, t) for s, t in clock.due(105.0)] == [("slot_0", 100.0)]
    assert clock.due(106.0) == []  # never twice
    assert [s.slot_id for s, _ in clock.due(700.0)] == ["slot_1"]


@pytest.mark.skipif(
    importlib.util.find_spec("traci") is None or not os.environ.get("V2V_NETWORK_TESTS"),
    reason="needs the [sim] extras and network access; set V2V_NETWORK_TESTS=1",
)
def test_noisy_scenario_through_sumo_and_pipeline(tmp_path: Path) -> None:
    from v2v.confidence.dedup import HazardTracker
    from v2v.contracts import DriverReport, World
    from v2v.experiments import CONFIGS
    from v2v.pipeline import process_report
    from v2v.sim.osm_network import CORRIDORS
    from v2v.sim.scenarios import generate
    from v2v.sim.traci_loop import SumoWorld

    (spec,) = generate(CORRIDORS["hebbal"], ["noisy"], ["low"], [0], tmp_path / "networks")
    config = CONFIGS["proposed"]
    fired: list[str] = []
    with SumoWorld(spec) as sumo:
        world: World = sumo  # SumoWorld satisfies the contract's protocol
        while sumo.time_s < spec.end_s and len(fired) < len(spec.slots):
            sumo.step()
            for slot, t_pass in sumo.due_reports():
                fired.append(slot.role)
                assert sumo.time_s >= t_pass + slot.delay_s
                if slot.role != "original":
                    continue
                # Event in progress: the blocking vehicle shows up on the sensors.
                assert sumo.sensor_flags(spec.hazard, 200.0) >= 1
                approaching = [
                    v.vehicle_id
                    for v in world.vehicles()
                    if world.will_reach(v.vehicle_id, spec.hazard)
                ]
                assert approaching and slot.reporter_id not in approaching
                assert not any(v.startswith("hazard_") for v in approaching)
                # Code-path check only: fixture text, never a reported result.
                report = DriverReport(
                    report_id=slot.slot_id,
                    text=ACCIDENT_HINGLISH,
                    source="fixture",
                    reporter_id=slot.reporter_id,
                    t_sim=sumo.time_s,
                )
                result = process_report(
                    report, world, spec.to_scenario(), config, HazardTracker(config), seed=0
                )
                assert set(result.relevant_vehicle_ids) == set(approaching)
                assert result.warning is not None and result.deliveries
    assert sorted(fired) == sorted(s.role for s in spec.slots)
    assert fired.index("contradiction") < fired.index("stale")  # stale arrives last
    assert sumo.hazard_passes and not sumo.teleported


@pytest.mark.skipif(
    importlib.util.find_spec("traci") is None or not os.environ.get("V2V_NETWORK_TESTS"),
    reason="needs the [sim] extras and network access; set V2V_NETWORK_TESTS=1",
)
def test_slippery_road_lowers_speed_only_during_event(tmp_path: Path) -> None:
    from v2v.sim.osm_network import CORRIDORS
    from v2v.sim.scenarios import generate
    from v2v.sim.traci_loop import SLIPPERY_SPEED_FACTOR, SumoWorld

    (spec,) = generate(
        CORRIDORS["hebbal"], ["clean"], ["low"], [1], tmp_path / "networks", validate=False
    )
    assert spec.effect == "slow_zone"  # seed 1's clean scenario is a slippery road
    spec = spec.model_copy(update={"event_end_s": 700.0})
    lane = f"{spec.hazard.edge_id}_0"
    with SumoWorld(spec) as world:
        world.run_until(200.0)
        normal = world._t.lane.getMaxSpeed(lane)
        world.run_until(400.0)
        assert world._t.lane.getMaxSpeed(lane) == pytest.approx(normal * SLIPPERY_SPEED_FACTOR)
        world.run_until(800.0)
        assert world._t.lane.getMaxSpeed(lane) == pytest.approx(normal)
