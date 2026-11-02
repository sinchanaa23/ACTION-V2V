"""Loading recorded traces and turning them into tables. Owner: P4.

Everything here is computed from ``v2v.sim.recorder`` traces; nothing is
estimated or filled in for display.
"""

import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import streamlit as st

from v2v.contracts import AssessedHazard, HazardType, ResolvedLocation
from v2v.messages.warning import render_text
from v2v.sim.recorder import PolicyResult, ReportFrame, Trace, classify, policy_metrics

REPO_ROOT = Path(__file__).resolve().parents[1]
POLICIES = ("broadcast", "circular", "directional")
LOCATIONS = ("known", "fallback")
METRIC_LABELS = {
    "recall": "Needed it: addressed",
    "precision": "Addressed: needed it",
    "relevant_reached": "Needed it: delivered",
    "irrelevant_reached": "Unneeded warnings delivered",
    "transmissions": "Transmissions",
    "receptions": "Receptions (channel load)",
    "mean_latency_ms": "Mean latency (ms)",
    "recipients": "Recipients",
}


def traces_dir() -> Path:
    return Path(os.environ.get("V2V_TRACES_DIR", REPO_ROOT / "sim_out" / "traces"))


def trace_files(directory: Path) -> tuple[tuple[str, int], ...]:
    """(path, mtime) of every trace; the cache key, so re-recorded runs show up."""
    return tuple((str(p), p.stat().st_mtime_ns) for p in sorted(directory.glob("*.json")))


@st.cache_data(show_spinner=False)
def load(files: tuple[tuple[str, int], ...]) -> dict[str, Trace]:
    return {Path(path).stem: Trace.load(Path(path)) for path, _ in files}


@st.cache_data(show_spinner=False)
def metrics(files: tuple[tuple[str, int], ...]) -> pd.DataFrame:
    """One row per (scenario, report, policy, location mode) with every metric."""
    rows = []
    for trace in load(files).values():
        spec = trace.spec
        for i, frame in enumerate(trace.frames):
            for location in LOCATIONS:
                for policy in POLICIES:
                    m = policy_metrics(frame, frame.result(policy, location))
                    rows.append(
                        {
                            "scenario": spec.scenario_id,
                            "condition": spec.condition,
                            "density": spec.density,
                            "frame": i,
                            "report": frame.slot.role,
                            "location": location,
                            "policy": policy,
                            **m,
                        }
                    )
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def reports(files: tuple[tuple[str, int], ...]) -> pd.DataFrame:
    """One row per report that fired in a recorded run."""
    rows = []
    for trace in load(files).values():
        spec = trace.spec
        for frame in trace.frames:
            slot = frame.slot
            known = policy_metrics(frame, frame.result("directional", "known"))
            rows.append(
                {
                    "scenario": spec.scenario_id,
                    "condition": spec.condition,
                    "density": spec.density,
                    "role": slot.role,
                    "reporter": slot.reporter_id,
                    "text needed": slot.needs,
                    "class": slot.hazard_class.value,
                    "passed (s)": round(frame.t_pass_s),
                    "sent (s)": round(frame.t_s),
                    "delay (s)": round(frame.t_s - frame.t_pass_s),
                    "sensor flags": frame.sensor_flags,
                    "vehicles": len(frame.vehicles),
                    "needed it": sum(v.relevant for v in frame.vehicles),
                    "directional delivered": known["relevant_reached"],
                }
            )
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class View:
    """What the user is looking at: one report of one run, one policy, one location mode."""

    trace: Trace
    frame: ReportFrame
    policy: str
    location: str

    @property
    def result(self) -> PolicyResult:
        return self.frame.result(self.policy, self.location)  # type: ignore[arg-type]

    @property
    def outcome(self) -> dict[str, str]:
        return classify(self.frame, self.result)

    def metrics(self, policy: str | None = None) -> dict[str, float | None]:
        frame = self.frame
        return policy_metrics(frame, frame.result(policy or self.policy, self.location))  # type: ignore[arg-type]


def warning_text(view: View) -> str | None:
    """The text P2's generator words for this hazard, as the recorder sends it.

    The recorder has no report text or confidence verdict yet, so the status is
    its default ("unverified"); None when a correct reading would send nothing.
    """
    slot, spec = view.frame.slot, view.trace.spec
    if slot.needs != "hazard" or slot.hazard_class in (
        HazardType.NON_ACTIONABLE,
        HazardType.AMBIGUOUS,
    ):
        return None
    location: ResolvedLocation = spec.hazard
    if view.location == "fallback":
        location = spec.hazard.model_copy(update={"approx": True})
    hazard = AssessedHazard(
        cluster_id=slot.slot_id,
        hazard=slot.hazard_class,
        location=location,
        interp_conf=1.0,
        event_evidence=0.0,
        n_reports=1,
        age_s=max(0.0, view.frame.t_s - view.frame.t_pass_s),
        status="unverified",
    )
    return render_text(hazard)


def alerts(traces: dict[str, Trace]) -> list[tuple[str, str]]:
    """(severity, message) facts about the recorded runs worth a look."""
    out: list[tuple[str, str]] = []
    for name, trace in traces.items():
        if not trace.frames:
            out.append(("red", f"{name}: no report fired in this run"))
        if trace.teleported:
            out.append(("orange", f"{name}: {trace.teleported} vehicle(s) teleported"))
    out.append(("blue", "Report text, NLP reading and confidence verdict: pending P3's runner"))
    return out
