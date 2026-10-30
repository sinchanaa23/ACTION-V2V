"""Scenario generation on the real network. Owner: P4 (P4-03).

    python -m v2v.sim.scenarios --corridor hebbal --seeds 0 1 2

A scenario puts one hazard at a real OSM landmark on a main road and writes
everything SUMO needs to play it out:

- background traffic at one of three densities (randomTrips, routed);
- an approaching stream that drives through the hazard and an opposing stream
  on the other carriageway (so direction-sensitive geocast has vehicles to
  include and to leave out);
- dedicated reporter vehicles timed to pass the hazard, one per report slot;
- for lane-blocking hazards, a stopped vehicle on the kerb lane while the
  event lasts.

Report *text* is not chosen here: each slot states what kind of real report it
needs (from P1's frozen test split), and the pipeline fills it in.

Three report-quality conditions, as in the experiment plan:
  clean       one genuine report of a real event
  noisy       the same, plus duplicates, a stale report sent after the event
              cleared, and a "road is clear now" contradiction
  misleading  no event at all; one hazard report and one non-actionable report
"""

import argparse
import math
import random
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from v2v.contracts import NON_HAZARD_TYPES, HazardType, ResolvedLocation, Scenario
from v2v.sim import gazetteer, osm_network
from v2v.sim.osm_network import CORRIDORS, DEFAULT_OUT_DIR, Corridor

Condition = Literal["clean", "noisy", "misleading"]
Density = Literal["low", "medium", "high"]
SlotRole = Literal["original", "duplicate", "stale", "contradiction", "false", "non_actionable"]
TextNeed = Literal["hazard", "negated_hazard", "non_actionable"]
Effect = Literal["lane_block", "slow_zone", "none"]

CONDITIONS: tuple[Condition, ...] = ("clean", "noisy", "misleading")
DENSITIES: tuple[Density, ...] = ("low", "medium", "high")

# Seconds between background trip insertions across the whole network. "high"
# is the busiest level that stays clear of gridlock teleports, which would
# break the "will this vehicle reach the hazard" ground truth.
BACKGROUND_PERIOD_S: dict[Density, float] = {"low": 2.0, "medium": 1.0, "high": 0.7}
# Seconds between vehicles in each of the approaching and opposing streams.
STREAM_PERIOD_S: dict[Density, float] = {"low": 20.0, "medium": 10.0, "high": 6.0}

WARMUP_S = 300.0  # let the network fill before the event starts
EVENT_DURATION_S = 600.0  # noisy scenarios clear the event after this
END_S = 1500.0  # leaves the last reporter ~10 min of slack in heavy traffic
UPSTREAM_M = (600.0, 1500.0)  # where reporters and the approaching stream start
DOWNSTREAM_M = 300.0
CONGESTION_FACTOR = 1.4  # free-flow travel time x this, to time reporters
MIN_LANES = 2
MIN_SPEED_MPS = 11.0  # ~40 km/h: main roads only
OPPOSITE_SEARCH_M = 60.0
MAX_TELEPORT_SHARE = 0.05
MAX_ATTEMPTS = 8
EFFECTS: dict[HazardType, Effect] = {
    HazardType.ACCIDENT: "lane_block",
    HazardType.OBSTRUCTION: "lane_block",
    HazardType.STOPPED: "lane_block",
    HazardType.SLIPPERY: "slow_zone",
    HazardType.SPEEDING: "none",
}
ACTIONABLE = tuple(h for h in HazardType if h not in NON_HAZARD_TYPES)


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class SlotPlan(_Model):
    """When a report should happen, before a vehicle and route are attached."""

    role: SlotRole
    needs: TextNeed
    pass_s: float
    """Target time for the reporter to pass the hazard point."""
    delay_s: float
    """The report is sent this long after the reporter actually passes."""


class ReportSlot(_Model):
    slot_id: str
    role: SlotRole
    needs: TextNeed
    hazard_class: HazardType
    """The class a "hazard"/"negated_hazard" text must carry (ignored for non_actionable)."""
    reporter_id: str
    route: tuple[str, ...]
    depart_s: float
    delay_s: float
    expected_report_s: float


class ScenarioSpec(_Model):
    scenario_id: str
    corridor: str
    condition: Condition
    density: Density
    seed: int
    hazard_class: HazardType
    effect: Effect
    hazard: ResolvedLocation
    landmark_id: str
    landmark_names: tuple[str, ...]
    landmark_kind: str
    opposite_edge: str
    event_occurred: bool
    event_start_s: float
    event_end_s: float
    end_s: float
    approach_route: tuple[str, ...]
    opposing_route: tuple[str, ...]
    slots: tuple[ReportSlot, ...]
    sumocfg: str

    def to_scenario(self) -> Scenario:
        """The contract view: ground truth plus the first report slot."""
        first = self.slots[0]
        return Scenario(
            scenario_id=self.scenario_id,
            hazard_location=self.hazard,
            event_occurred=self.event_occurred,
            reporter_id=first.reporter_id,
            report_time_s=first.expected_report_s,
        )

    def save(self, path: Path) -> None:
        path.write_text(self.model_dump_json(indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "ScenarioSpec":
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


# --- Timing ----------------------------------------------------------------------


def plan_slots(condition: Condition, event_start_s: float, event_end_s: float) -> list[SlotPlan]:
    """Report slots for a condition. The first slot is the primary report."""
    start = event_start_s
    if condition == "clean":
        return [SlotPlan(role="original", needs="hazard", pass_s=start + 60, delay_s=5)]
    if condition == "noisy":
        return [
            SlotPlan(role="original", needs="hazard", pass_s=start + 60, delay_s=5),
            SlotPlan(role="duplicate", needs="hazard", pass_s=start + 105, delay_s=10),
            SlotPlan(role="duplicate", needs="hazard", pass_s=start + 150, delay_s=20),
            # Saw it early, only reported after the event had cleared.
            SlotPlan(
                role="stale", needs="hazard", pass_s=start + 30, delay_s=event_end_s - start + 30
            ),
            SlotPlan(
                role="contradiction", needs="negated_hazard", pass_s=event_end_s + 20, delay_s=5
            ),
        ]
    return [
        SlotPlan(role="false", needs="hazard", pass_s=start + 60, delay_s=5),
        SlotPlan(role="non_actionable", needs="non_actionable", pass_s=start + 120, delay_s=5),
    ]


def depart_time(pass_s: float, travel_s: float) -> float:
    return round(max(0.0, pass_s - travel_s * CONGESTION_FACTOR), 1)


def heading_at(shape: Sequence[tuple[float, float]], offset: float) -> float:
    return gazetteer.shape_heading(list(shape), offset)


# --- Network search ----------------------------------------------------------------


def _drivable(edge: Any) -> bool:
    return bool(edge.allows("passenger")) and edge.getFunction() != "internal"


def _main_road(edge: Any) -> bool:
    return (
        _drivable(edge) and edge.getLaneNumber() >= MIN_LANES and edge.getSpeed() >= MIN_SPEED_MPS
    )


def reachable(start: Any, forward: bool, min_m: float, max_m: float) -> list[Any]:
    """Edges whose distance from ``start`` (along the network, forwards or
    backwards) falls in [min_m, max_m]."""
    found: list[Any] = []
    best: dict[str, float] = {start.getID(): 0.0}
    frontier: list[tuple[float, Any]] = [(0.0, start)]
    while frontier:
        frontier.sort(key=lambda pair: pair[0])
        dist, edge = frontier.pop(0)
        if dist > best.get(edge.getID(), math.inf):
            continue
        if min_m <= dist <= max_m:
            found.append(edge)
        if dist > max_m:
            continue
        neighbours = edge.getOutgoing() if forward else edge.getIncoming()
        for nxt in neighbours:
            if not _drivable(nxt):
                continue
            step = (edge if forward else nxt).getLength()
            if dist + step < best.get(nxt.getID(), math.inf):
                best[nxt.getID()] = dist + step
                frontier.append((dist + step, nxt))
    return sorted(found, key=lambda e: e.getID())


def _prefer_main(edges: list[Any]) -> list[Any]:
    return [e for e in edges if _main_road(e)] or edges


def route_through(net: Any, start: Any, via: Any, end: Any) -> tuple[str, ...] | None:
    first, _ = net.getShortestPath(start, via, vClass="passenger")
    second, _ = net.getShortestPath(via, end, vClass="passenger")
    if not first or not second:
        return None
    return tuple(e.getID() for e in [*first, *second[1:]])


def travel_time_to(net: Any, route: Sequence[str], edge_id: str, pos_m: float) -> float:
    total = 0.0
    for eid in route:
        edge = net.getEdge(eid)
        if eid == edge_id:
            return total + pos_m / float(edge.getSpeed())
        total += float(edge.getLength()) / float(edge.getSpeed())
    raise ValueError(f"{edge_id} not on route")


def opposite_edge(net: Any, hazard: ResolvedLocation, heading: float) -> Any | None:
    """The nearest edge beside the hazard running roughly the other way."""
    best: tuple[float, Any] | None = None
    for edge, dist in net.getNeighboringEdges(hazard.x, hazard.y, OPPOSITE_SEARCH_M):
        if edge.getID() == hazard.edge_id or not _drivable(edge):
            continue
        shape = edge.getLanes()[0].getShape()
        from sumolib import geomhelper

        offset = geomhelper.polygonOffsetWithMinimumDistanceToPoint((hazard.x, hazard.y), shape)
        if gazetteer.angle_diff(heading_at(shape, offset), heading) >= 135.0:
            if best is None or dist < best[0]:
                best = (dist, edge)
    return None if best is None else best[1]


class _Site(_Model):
    landmark: gazetteer.Landmark
    anchor: gazetteer.Anchor
    opposite: str


def candidate_sites(net: Any, gz: gazetteer.Gazetteer) -> list[_Site]:
    """Landmark anchors on main roads, away from edge ends, with an opposite
    carriageway and enough road upstream for reporters to approach."""
    sites: list[_Site] = []
    cache: dict[str, bool] = {}

    def has_upstream(edge: Any) -> bool:
        if edge.getID() not in cache:
            cache[edge.getID()] = bool(reachable(edge, False, *UPSTREAM_M))
        return cache[edge.getID()]

    for lm in gz.landmarks:
        for anchor in lm.anchors:
            edge = net.getEdge(anchor.edge_id)
            if not _main_road(edge) or not 10.0 <= anchor.pos_m <= edge.getLength() - 10.0:
                continue
            if not has_upstream(edge):
                continue
            loc = ResolvedLocation(
                edge_id=anchor.edge_id, pos_m=anchor.pos_m, x=anchor.x, y=anchor.y, approx=False
            )
            opp = opposite_edge(net, loc, anchor.heading_deg)
            if opp is None or not has_upstream(opp):
                continue
            sites.append(_Site(landmark=lm, anchor=anchor, opposite=opp.getID()))
    return sites


# --- Writing SUMO inputs --------------------------------------------------------------


def _xml(root: ET.Element, path: Path) -> None:
    ET.indent(root)
    path.write_bytes(ET.tostring(root, encoding="utf-8", xml_declaration=True))


def write_scenario_routes(spec: ScenarioSpec, path: Path, density: Density) -> None:
    routes = ET.Element("routes")
    ET.SubElement(routes, "vType", id="car", vClass="passenger")
    ET.SubElement(routes, "vType", id="reporter", vClass="passenger", color="0,160,255")
    ET.SubElement(routes, "vType", id="hazard", vClass="passenger", color="255,0,0")
    ET.SubElement(routes, "route", id="approach", edges=" ".join(spec.approach_route))
    ET.SubElement(routes, "route", id="opposing", edges=" ".join(spec.opposing_route))
    for name in ("approach", "opposing"):
        ET.SubElement(
            routes,
            "flow",
            id=name,
            type="car",
            route=name,
            begin="0",
            end=f"{spec.end_s:.0f}",
            period=f"{STREAM_PERIOD_S[density]:.1f}",
            departLane="best",
            departSpeed="max",
        )
    vehicles: list[tuple[float, ET.Element]] = []
    if spec.event_occurred and spec.effect == "lane_block":
        hazard = ET.Element(
            "vehicle",
            id="hazard_0",
            type="hazard",
            depart=f"{spec.event_start_s:.1f}",
            departLane="0",
            departPos=f"{max(0.0, spec.hazard.pos_m - 10.0):.2f}",
            departSpeed="0",
            insertionChecks="none",  # an accident appears even in a queue
        )
        ET.SubElement(hazard, "route", edges=spec.hazard.edge_id)
        ET.SubElement(
            hazard,
            "stop",
            lane=f"{spec.hazard.edge_id}_0",
            endPos=f"{spec.hazard.pos_m:.2f}",
            until=f"{spec.event_end_s:.1f}",
        )
        vehicles.append((spec.event_start_s, hazard))
    for slot in spec.slots:
        vehicle = ET.Element(
            "vehicle",
            id=slot.reporter_id,
            type="reporter",
            depart=f"{slot.depart_s:.1f}",
            departLane="best",
            departSpeed="max",
        )
        ET.SubElement(vehicle, "route", edges=" ".join(slot.route))
        vehicles.append((slot.depart_s, vehicle))
    for _, vehicle in sorted(vehicles, key=lambda pair: pair[0]):  # SUMO wants depart order
        routes.append(vehicle)
    _xml(routes, path)


def write_background(net_path: Path, out: Path, density: Density, end_s: float, seed: int) -> None:
    import sumo

    tool = Path(sumo.SUMO_HOME) / "tools" / "randomTrips.py"
    args = [
        sys.executable, str(tool),
        "-n", str(net_path),
        "-o", str(out.with_suffix(".trips.xml")),
        "-r", str(out),
        "--begin", "0", "--end", f"{end_s:.0f}",
        "--period", f"{BACKGROUND_PERIOD_S[density]}",
        "--seed", str(seed),
        "--fringe-factor", "5",
        "--min-distance", "300",
        "--lanes",
        "--vclass", "passenger",
        "--prefix", "bg_",
        "--validate",
        "--trip-attributes", 'departLane="best" departSpeed="max"',
    ]  # fmt: skip
    done = subprocess.run(args, capture_output=True, text=True)
    if done.returncode != 0:
        raise RuntimeError(f"randomTrips failed:\n{done.stderr[-2000:]}")


def write_sumocfg(
    path: Path, net_path: Path, route_files: Sequence[Path], spec: ScenarioSpec
) -> None:
    cfg = ET.Element("configuration")
    inputs = ET.SubElement(cfg, "input")
    ET.SubElement(inputs, "net-file", value=_rel(net_path, path.parent))
    ET.SubElement(inputs, "route-files", value=",".join(_rel(r, path.parent) for r in route_files))
    time = ET.SubElement(cfg, "time")
    ET.SubElement(time, "begin", value="0")
    ET.SubElement(time, "end", value=f"{spec.end_s:.0f}")
    proc = ET.SubElement(cfg, "processing")
    ET.SubElement(proc, "time-to-teleport", value="300")
    # Unsignalised OSM junctions can deadlock on give-way; let a vehicle stuck
    # behind a junction blocker for 15 s proceed instead of being teleported.
    ET.SubElement(proc, "ignore-junction-blocker", value="15")
    ET.SubElement(cfg, "random_number").append(ET.Element("seed", value=str(spec.seed)))
    report = ET.SubElement(cfg, "report")
    ET.SubElement(report, "no-step-log", value="true")
    ET.SubElement(report, "no-warnings", value="true")
    _xml(cfg, path)


def _rel(target: Path, base: Path) -> str:
    import os

    return os.path.relpath(target.resolve(), base.resolve()).replace("\\", "/")


# --- Building scenarios ---------------------------------------------------------------


def build_spec(
    net: Any,
    sites: Sequence[_Site],
    corridor: Corridor,
    condition: Condition,
    density: Density,
    seed: int,
    sumocfg: Path,
    attempt: int = 0,
) -> ScenarioSpec:
    """Choose a site and hazard for (condition, seed) and plan every vehicle.

    The random stream ignores density, so all three densities of a
    (condition, seed, attempt) share the same site, hazard and routes.
    """
    rng = random.Random(f"{corridor.name}-{condition}-{seed}-{attempt}")
    site = rng.choice(list(sites))
    hazard_class = rng.choice(ACTIONABLE)
    anchor = site.anchor
    hazard = ResolvedLocation(
        edge_id=anchor.edge_id, pos_m=anchor.pos_m, x=anchor.x, y=anchor.y, approx=False
    )
    hazard_edge = net.getEdge(anchor.edge_id)
    opposite = net.getEdge(site.opposite)
    event_start = WARMUP_S
    event_occurred = condition != "misleading"
    event_end = WARMUP_S + EVENT_DURATION_S if condition == "noisy" else END_S

    def through(via: Any) -> tuple[str, ...]:
        # Start and end on main roads where possible, so streams don't queue
        # behind a side-street give-way.
        starts = _prefer_main(reachable(via, False, *UPSTREAM_M))
        ends = _prefer_main(reachable(via, True, DOWNSTREAM_M, DOWNSTREAM_M * 4))
        for _ in range(20):
            route = route_through(net, rng.choice(starts), via, rng.choice(ends) if ends else via)
            if route is not None:
                return route
        raise RuntimeError(f"no route through {via.getID()}")

    slots: list[ReportSlot] = []
    for i, plan in enumerate(plan_slots(condition, event_start, event_end)):
        route = through(hazard_edge)
        travel = travel_time_to(net, route, anchor.edge_id, anchor.pos_m)
        depart = depart_time(plan.pass_s, travel)
        slots.append(
            ReportSlot(
                slot_id=f"slot_{i}",
                role=plan.role,
                needs=plan.needs,
                hazard_class=hazard_class,
                reporter_id=f"rep_{i}",
                route=route,
                depart_s=depart,
                delay_s=plan.delay_s,
                expected_report_s=round(plan.pass_s + plan.delay_s, 1),
            )
        )
    return ScenarioSpec(
        scenario_id=f"{corridor.name}-{condition}-{density}-s{seed}",
        corridor=corridor.name,
        condition=condition,
        density=density,
        seed=seed,
        hazard_class=hazard_class,
        effect=EFFECTS[hazard_class],
        hazard=hazard,
        landmark_id=site.landmark.landmark_id,
        landmark_names=site.landmark.names,
        landmark_kind=site.landmark.kind,
        opposite_edge=site.opposite,
        event_occurred=event_occurred,
        event_start_s=event_start,
        event_end_s=event_end,
        end_s=END_S,
        approach_route=through(hazard_edge),
        opposing_route=through(opposite),
        slots=tuple(slots),
        sumocfg=str(sumocfg),
    )


def check_run(spec: ScenarioSpec) -> list[str]:
    """Play the scenario in SUMO and list what makes it unusable: a reporter
    that never passes the hazard, a report that would arrive after the run
    ends, or more than MAX_TELEPORT_SHARE of vehicles teleported out of jams."""
    work = Path(spec.sumocfg).parent
    vehroute, stats = work / "check.vehroute.xml", work / "check.stats.xml"
    done = subprocess.run(
        [
            osm_network.sumo_binary("sumo"), "-c", spec.sumocfg,
            "--vehroute-output", str(vehroute), "--vehroute-output.exit-times", "true",
            "--statistic-output", str(stats),
        ],
        capture_output=True, text=True,
    )  # fmt: skip
    if done.returncode != 0:
        return [f"sumo failed: {done.stderr[-500:]}"]
    passed: dict[str, float] = {}
    for vehicle in ET.parse(vehroute).getroot().iter("vehicle"):
        route = vehicle.find("route")
        if route is None or not vehicle.get("id", "").startswith("rep_"):
            continue
        times = dict(
            zip(route.get("edges", "").split(), route.get("exitTimes", "").split(), strict=False)
        )
        if spec.hazard.edge_id in times:
            passed[vehicle.get("id", "")] = float(times[spec.hazard.edge_id])
    problems = []
    for slot in spec.slots:
        t = passed.get(slot.reporter_id)
        if t is None:
            problems.append(f"{slot.reporter_id} never passed the hazard")
        elif t + slot.delay_s > spec.end_s:
            problems.append(f"{slot.reporter_id} would report after the run ends")
    root = ET.parse(stats).getroot()
    inserted = int(root.find("vehicles").get("inserted", "0"))  # type: ignore[union-attr]
    teleports = int(root.find("teleports").get("total", "0"))  # type: ignore[union-attr]
    if inserted and teleports / inserted > MAX_TELEPORT_SHARE:
        problems.append(f"{teleports} teleports out of {inserted} vehicles")
    for path in (vehroute, stats):
        path.unlink(missing_ok=True)
    return problems


def generate(
    corridor: Corridor,
    conditions: Sequence[Condition] = CONDITIONS,
    densities: Sequence[Density] = DENSITIES,
    seeds: Sequence[int] = (0,),
    out_dir: Path = DEFAULT_OUT_DIR,
    validate: bool = True,
    log: Any = print,
) -> list[ScenarioSpec]:
    """Write every (condition, density, seed) scenario and return their specs.

    With ``validate`` each (condition, seed) is played at every density; if any
    run fails ``check_run`` a different site is drawn, up to MAX_ATTEMPTS.
    """
    gz = gazetteer.build_gazetteer(corridor, out_dir)
    net_path = osm_network.build_network(corridor, out_dir)
    net = osm_network.read_net(net_path)
    sites = candidate_sites(net, gz)
    if not sites:
        raise RuntimeError(f"no main-road landmark sites in {corridor.name}")
    root = out_dir.parent / "scenarios" / corridor.name
    specs: list[ScenarioSpec] = []
    for seed in seeds:
        backgrounds: dict[Density, Path] = {}
        for density in densities:
            backgrounds[density] = root / "background" / f"{density}-s{seed}.rou.xml"
            backgrounds[density].parent.mkdir(parents=True, exist_ok=True)
            if not backgrounds[density].exists():
                write_background(net_path, backgrounds[density], density, END_S, seed)
        for condition in conditions:
            for attempt in range(MAX_ATTEMPTS):
                batch: list[ScenarioSpec] = []
                for density in densities:
                    work = root / f"{condition}-{density}-s{seed}"
                    work.mkdir(parents=True, exist_ok=True)
                    cfg = work / "scenario.sumocfg"
                    spec = build_spec(net, sites, corridor, condition, density, seed, cfg, attempt)
                    write_scenario_routes(spec, work / "scenario.rou.xml", density)
                    write_sumocfg(
                        cfg, net_path, [backgrounds[density], work / "scenario.rou.xml"], spec
                    )
                    spec.save(work / "spec.json")
                    batch.append(spec)
                problems = [p for s in batch for p in check_run(s)] if validate else []
                if not problems:
                    specs.extend(batch)
                    break
                log(f"{condition}-s{seed} attempt {attempt}: {'; '.join(problems)}; new site")
            else:
                raise RuntimeError(f"no usable site for {condition}-s{seed}")
    return specs


def toy_scenario(seed: int = 0) -> Scenario:
    """A hazard on the toy world's ``fwd`` edge at 1200 m, reported at t=10 s by
    ``fwd_13``, which starts just past it. Used by tests and the CI smoke run."""
    return Scenario(
        scenario_id=f"toy-{seed}",
        hazard_location=ResolvedLocation(
            edge_id="fwd", pos_m=1200.0, x=1200.0, y=0.0, approx=False
        ),
        event_occurred=True,
        reporter_id="fwd_13",
        report_time_s=10.0,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate SUMO scenarios on a real corridor")
    parser.add_argument("--corridor", choices=sorted(CORRIDORS), default="hebbal")
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--densities", nargs="+", choices=DENSITIES, default=list(DENSITIES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--no-validate", action="store_true", help="skip the SUMO check run of each scenario"
    )
    args = parser.parse_args(argv)
    specs = generate(
        CORRIDORS[args.corridor],
        args.conditions,
        args.densities,
        args.seeds,
        args.out,
        validate=not args.no_validate,
    )
    for spec in specs:
        where = spec.landmark_names[0] if spec.landmark_names else spec.landmark_kind
        print(
            f"{spec.scenario_id:<32} {spec.hazard_class.value:<17} at {where} "
            f"({spec.hazard.edge_id} @ {spec.hazard.pos_m:.0f} m), {len(spec.slots)} reports"
        )


if __name__ == "__main__":
    main()
