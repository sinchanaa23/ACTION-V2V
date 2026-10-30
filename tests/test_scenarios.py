import importlib.util
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from v2v.contracts import HazardType, ResolvedLocation
from v2v.sim.scenarios import (
    CONGESTION_FACTOR,
    ReportSlot,
    ScenarioSpec,
    depart_time,
    plan_slots,
    reachable,
    write_scenario_routes,
    write_sumocfg,
)


def test_clean_is_one_real_report() -> None:
    (slot,) = plan_slots("clean", 300, 1500)
    assert (slot.role, slot.needs) == ("original", "hazard")


def test_noisy_has_duplicates_stale_after_clear_and_late_contradiction() -> None:
    slots = plan_slots("noisy", 300, 900)
    roles = [s.role for s in slots]
    assert roles[0] == "original" and roles.count("duplicate") == 2
    stale = next(s for s in slots if s.role == "stale")
    assert stale.pass_s < 900 < stale.pass_s + stale.delay_s  # seen during, sent after clear
    contradiction = next(s for s in slots if s.role == "contradiction")
    assert contradiction.needs == "negated_hazard" and contradiction.pass_s > 900


def test_misleading_has_no_true_report() -> None:
    slots = plan_slots("misleading", 300, 1500)
    assert [s.role for s in slots] == ["false", "non_actionable"]


def test_depart_time_allows_for_congestion_and_never_negative() -> None:
    assert depart_time(360, 50) == pytest.approx(360 - 50 * CONGESTION_FACTOR)
    assert depart_time(10, 100) == 0.0


class _Edge:
    """Just enough of a sumolib edge for the network search."""

    def __init__(self, eid: str, length: float) -> None:
        self.eid, self.length = eid, length
        self.out: list[_Edge] = []
        self.inc: list[_Edge] = []

    def getID(self) -> str:  # noqa: N802 - sumolib's API
        return self.eid

    def getLength(self) -> float:  # noqa: N802
        return self.length

    def getOutgoing(self) -> list["_Edge"]:  # noqa: N802
        return self.out

    def getIncoming(self) -> list["_Edge"]:  # noqa: N802
        return self.inc

    def allows(self, vclass: str) -> bool:
        return True

    def getFunction(self) -> str:  # noqa: N802
        return ""


def _chain(*lengths: float) -> list[_Edge]:
    edges = [_Edge(f"e{i}", length) for i, length in enumerate(lengths)]
    for a, b in zip(edges, edges[1:], strict=False):
        a.out.append(b)
        b.inc.append(a)
    return edges


def test_reachable_measures_along_the_road() -> None:
    e = _chain(400, 400, 400, 400)  # e0 -> e1 -> e2 -> e3
    assert [x.getID() for x in reachable(e[3], False, 600, 1000)] == ["e1"]  # 800 m upstream
    assert [x.getID() for x in reachable(e[0], True, 700, 900)] == ["e2"]


def _spec(tmp_path: Path, effect: str = "lane_block", occurred: bool = True) -> ScenarioSpec:
    slot = ReportSlot(
        slot_id="slot_0",
        role="original",
        needs="hazard",
        hazard_class=HazardType.ACCIDENT,
        reporter_id="rep_0",
        route=("a", "h", "b"),
        depart_s=250.0,
        delay_s=5.0,
        expected_report_s=365.0,
    )
    return ScenarioSpec(
        scenario_id="t-clean-low-s0",
        corridor="t",
        condition="clean",
        density="low",
        seed=0,
        hazard_class=HazardType.ACCIDENT,
        effect=effect,  # type: ignore[arg-type]
        hazard=ResolvedLocation(edge_id="h", pos_m=50.0, x=1.0, y=2.0, approx=False),
        landmark_id="way/1",
        landmark_names=("Test Flyover",),
        landmark_kind="flyover",
        opposite_edge="-h",
        event_occurred=occurred,
        event_start_s=300.0,
        event_end_s=1500.0,
        end_s=1500.0,
        approach_route=("a", "h", "b"),
        opposing_route=("c", "-h", "d"),
        slots=(slot,),
        sumocfg=str(tmp_path / "scenario.sumocfg"),
    )


def test_routes_file_has_streams_reporter_and_blocking_hazard(tmp_path: Path) -> None:
    write_scenario_routes(_spec(tmp_path), tmp_path / "r.xml", "low")
    root = ET.parse(tmp_path / "r.xml").getroot()
    flows = {f.get("id"): f for f in root.iter("flow")}
    assert set(flows) == {"approach", "opposing"} and flows["approach"].get("period") == "20.0"
    vehicles = [v.get("id") for v in root.iter("vehicle")]
    assert vehicles == ["rep_0", "hazard_0"]  # sorted by departure time
    stop = root.find("vehicle[@id='hazard_0']/stop")
    assert stop is not None and stop.get("lane") == "h_0" and stop.get("until") == "1500.0"


@pytest.mark.parametrize(("effect", "occurred"), [("slow_zone", True), ("lane_block", False)])
def test_no_blocking_vehicle_unless_a_real_lane_block(
    tmp_path: Path, effect: str, occurred: bool
) -> None:
    write_scenario_routes(_spec(tmp_path, effect, occurred), tmp_path / "r.xml", "low")
    root = ET.parse(tmp_path / "r.xml").getroot()
    assert root.find("vehicle[@id='hazard_0']") is None


def test_sumocfg_uses_relative_paths(tmp_path: Path) -> None:
    (tmp_path / "net").mkdir()
    work = tmp_path / "scen"
    work.mkdir()
    spec = _spec(work)
    write_sumocfg(work / "scenario.sumocfg", tmp_path / "net" / "x.net.xml", [work / "r.xml"], spec)
    cfg = ET.parse(work / "scenario.sumocfg").getroot()
    assert cfg.find("input/net-file").get("value") == "../net/x.net.xml"  # type: ignore[union-attr]
    assert cfg.find("time/end").get("value") == "1500"  # type: ignore[union-attr]


def test_contract_view_and_round_trip(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    scenario = spec.to_scenario()
    assert scenario.reporter_id == "rep_0" and scenario.report_time_s == 365.0
    spec.save(tmp_path / "s.json")
    assert ScenarioSpec.load(tmp_path / "s.json") == spec


@pytest.mark.skipif(
    importlib.util.find_spec("sumolib") is None or not os.environ.get("V2V_NETWORK_TESTS"),
    reason="needs the [sim] extras and network access; set V2V_NETWORK_TESTS=1",
)
def test_hebbal_noisy_scenario_plays_out(tmp_path: Path) -> None:
    from v2v.sim.osm_network import CORRIDORS
    from v2v.sim.scenarios import check_run, generate

    specs = generate(CORRIDORS["hebbal"], ["noisy"], ["medium"], [0], tmp_path / "networks")
    (spec,) = specs
    assert check_run(spec) == []
    assert spec.event_occurred and len(spec.slots) == 5
    assert spec.opposite_edge != spec.hazard.edge_id
    assert spec.hazard.edge_id in spec.approach_route
    assert spec.opposite_edge in spec.opposing_route
