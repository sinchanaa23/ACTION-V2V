"""Streamlit dashboard: map, per-report trace, policy comparison. Owner: P4 (P4-07).

    python -m v2v.sim.recorder sim_out/scenarios/hebbal/*-s0     # record runs first
    streamlit run dashboard/app.py

Reads traces written by ``v2v.sim.recorder`` (``V2V_TRACES_DIR``, default
``sim_out/traces``). Report text, the NLP reading and the confidence verdict
are shown as pending until P3's experiment runner records them; nothing here
is placeholder data presented as results.

Map data (c) OpenStreetMap contributors, ODbL.
"""

import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from v2v.sim.recorder import Trace, classify, policy_metrics

REPO_ROOT = Path(__file__).resolve().parents[1]
TRACES_DIR = Path(os.environ.get("V2V_TRACES_DIR", REPO_ROOT / "sim_out" / "traces"))
POLICIES = ("broadcast", "circular", "directional")
OUTCOME_COLOURS = {
    "warned, needed it": "#1a9850",
    "addressed, not delivered": "#fdae61",
    "missed": "#d73027",
    "warned, didn't need it": "#4575b4",
    "not warned": "#bdbdbd",
}
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


def load_traces(directory: Path) -> dict[str, Trace]:
    """All traces, re-read whenever a file is added or re-recorded."""
    files = sorted(directory.glob("*.json"))
    return _load(tuple((str(p), p.stat().st_mtime_ns) for p in files))


@st.cache_data(show_spinner=False)
def _load(files: tuple[tuple[str, int], ...]) -> dict[str, Trace]:
    return {Path(path).stem: Trace.load(Path(path)) for path, _ in files}


def metrics_table(traces: list[Trace], location: str) -> pd.DataFrame:
    rows = []
    for trace in traces:
        spec = trace.spec
        for frame in trace.frames:
            for policy in POLICIES:
                m = policy_metrics(frame, frame.result(policy, location))  # type: ignore[arg-type]
                rows.append(
                    {
                        "scenario": spec.scenario_id,
                        "condition": spec.condition,
                        "density": spec.density,
                        "report": frame.slot.role,
                        "policy": policy,
                        **m,
                    }
                )
    return pd.DataFrame(rows)


def vehicle_map(trace: Trace, frame_idx: int, policy: str, location: str) -> go.Figure:
    frame = trace.frames[frame_idx]
    outcome = classify(frame, frame.result(policy, location))  # type: ignore[arg-type]
    result = frame.result(policy, location)  # type: ignore[arg-type]
    fig = go.Figure()
    for label, colour in OUTCOME_COLOURS.items():
        vs = [v for v in frame.vehicles if outcome[v.vehicle_id] == label]
        if not vs:
            continue
        fig.add_trace(
            go.Scattermap(
                lon=[v.lon for v in vs],
                lat=[v.lat for v in vs],
                mode="markers",
                marker={"size": 10 if label != "not warned" else 7, "color": colour},
                name=f"{label} ({len(vs)})",
                text=[
                    f"{v.vehicle_id}<br>{v.speed_mps * 3.6:.0f} km/h, heading {v.heading_deg:.0f}°"
                    + (
                        f"<br>latency {result.latency_ms[v.vehicle_id]:.1f} ms, "
                        f"{result.hops.get(v.vehicle_id, '?')} hop(s)"
                        if v.vehicle_id in result.latency_ms
                        else ""
                    )
                    for v in vs
                ],
                hoverinfo="text",
            )
        )
    fig.add_trace(
        go.Scattermap(
            lon=[trace.hazard_lon],
            lat=[trace.hazard_lat],
            mode="markers",
            marker={"size": 18, "color": "black"},
            name="hazard (true position)",
            hoverinfo="name",
        )
    )
    reporter = frame.reporter
    fig.add_trace(
        go.Scattermap(
            lon=[reporter.lon],
            lat=[reporter.lat],
            mode="markers",
            marker={"size": 14, "color": "#762a83"},
            name=f"reporter {reporter.vehicle_id}",
            hoverinfo="name",
        )
    )
    if location == "fallback":
        fig.add_trace(
            go.Scattermap(
                lon=[frame.fallback_lon],
                lat=[frame.fallback_lat],
                mode="markers",
                marker={"size": 12, "color": "#ff7f00"},
                name="fallback location (approx)",
                hoverinfo="name",
            )
        )
    fig.update_layout(
        map={
            "style": "open-street-map",
            "center": {"lon": trace.hazard_lon, "lat": trace.hazard_lat},
            "zoom": 14.6,
        },
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        height=560,
        legend={
            "bgcolor": "rgba(255,255,255,0.9)",
            "font": {"color": "#1d2330"},  # readable in light and dark themes
            "x": 0.01,
            "y": 0.99,
        },
    )
    return fig


def render_trace(trace: Trace, frame_idx: int, location: str) -> None:
    spec = trace.spec
    frame = trace.frames[frame_idx]
    slot = frame.slot
    st.subheader("Report trace")
    where = ", ".join(spec.landmark_names) or spec.landmark_kind
    cols = st.columns(5)
    cols[0].markdown(
        f"**1 · Report**  \n{slot.reporter_id} ({slot.role})  \npassed {frame.t_pass_s:.0f} s, "
        f"sent {frame.t_s:.0f} s  \nneeds a real *{slot.needs}* text"
        + (f" of class *{slot.hazard_class.value}*" if slot.needs != "non_actionable" else "")
        + "  \n:grey[text from P1's test split: pending]"
    )
    cols[1].markdown("**2 · NLP reading**  \n:grey[pending P2's models]")
    cols[2].markdown(
        f"**3 · Confidence**  \n{frame.sensor_flags} sensor flag(s) within 200 m  \n"
        ":grey[verdict pending P3]"
    )
    cols[3].markdown(
        "**4 · Location**  \n"
        + (
            f"true hazard: {where}"
            if location == "known"
            else "fallback: reporter's position (approx)"
        )
    )
    relevant = sum(v.relevant for v in frame.vehicles)
    cols[4].markdown(
        f"**Ground truth**  \nevent {'happened' if spec.event_occurred else 'did NOT happen'}  \n"
        f"{relevant} of {len(frame.vehicles)} vehicles will reach the hazard"
    )
    rows = []
    for policy in POLICIES:
        m = policy_metrics(frame, frame.result(policy, location))  # type: ignore[arg-type]
        rows.append({"policy": policy, **{METRIC_LABELS[k]: v for k, v in m.items()}})
    st.dataframe(
        pd.DataFrame(rows).set_index("policy"),
        use_container_width=True,
        column_config={
            METRIC_LABELS[k]: st.column_config.NumberColumn(format="percent")
            for k in ("recall", "precision", "relevant_reached")
        },
    )


def render_comparison(traces: dict[str, Trace], location: str) -> None:
    st.subheader("Policy comparison across recorded runs")
    df = metrics_table(list(traces.values()), location)
    if df.empty:
        st.info("No reports recorded yet.")
        return
    c1, c2 = st.columns(2)
    conditions = c1.multiselect("Condition", sorted(df["condition"].unique()), default=None)
    densities = c2.multiselect("Density", sorted(df["density"].unique()), default=None)
    if conditions:
        df = df[df["condition"].isin(conditions)]
    if densities:
        df = df[df["density"].isin(densities)]
    st.caption(
        f"{df['scenario'].nunique()} scenario(s), {len(df) // len(POLICIES)} report(s); "
        "means over reports. Delivery model: Nakagami-m (m=3), multi-hop."
    )
    summary = df.groupby("policy")[list(METRIC_LABELS)].mean().reindex(POLICIES)
    left, right = st.columns(2)
    rates = summary[["recall", "precision", "relevant_reached"]].rename(columns=METRIC_LABELS)
    left.plotly_chart(
        px.bar(
            rates.reset_index().melt(id_vars="policy"),
            x="variable",
            y="value",
            color="policy",
            barmode="group",
            labels={"variable": "", "value": "share"},
            range_y=[0, 1],
        ),
        use_container_width=True,
    )
    load = summary[["transmissions", "irrelevant_reached"]].rename(columns=METRIC_LABELS)
    right.plotly_chart(
        px.bar(
            load.reset_index().melt(id_vars="policy"),
            x="variable",
            y="value",
            color="policy",
            barmode="group",
            labels={"variable": "", "value": "per report"},
            log_y=True,
        ),
        use_container_width=True,
    )
    st.dataframe(summary.rename(columns=METRIC_LABELS), use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="A.C.T.I.O.N-V2V", layout="wide")
    st.title("A.C.T.I.O.N-V2V: hazard warnings on real Hebbal roads")
    traces = load_traces(TRACES_DIR)
    if not traces:
        st.warning(
            f"No traces in `{TRACES_DIR}`. Record some first:\n\n"
            "```\npython -m v2v.sim.recorder sim_out/scenarios/hebbal/*-s0\n```"
        )
        return
    with st.sidebar:
        name = st.selectbox("Scenario", list(traces))
        trace = traces[name]
        spec = trace.spec
        st.markdown(
            f"**{spec.hazard_class.value}** ({spec.effect})  \n"
            f"at {', '.join(spec.landmark_names) or spec.landmark_kind}  \n"
            f"condition **{spec.condition}**, density **{spec.density}**, seed {spec.seed}  \n"
            f"event {spec.event_start_s:.0f}–{spec.event_end_s:.0f} s  \n"
            f"{trace.vehicles_passed_hazard} vehicles passed the hazard, "
            f"{trace.teleported} teleported"
        )
        if not trace.frames:
            st.error("No report fired in this run.")
            return
        frame_idx = st.radio(
            "Report",
            range(len(trace.frames)),
            format_func=lambda i: (
                f"t={trace.frames[i].t_s:.0f}s · {trace.frames[i].slot.role} "
                f"({trace.frames[i].slot.reporter_id})"
            ),
        )
        policy = st.radio("Policy", POLICIES, index=2)
        location = st.radio(
            "Location",
            ("known", "fallback"),
            format_func=lambda m: (
                "known (stated in report)" if m == "known" else "fallback (none stated)"
            ),
        )
        st.caption(
            f"Geocast radius {trace.radius_m:.0f} m · radio range {trace.comm_range_m:.0f} m  \n"
            "Map © OpenStreetMap contributors"
        )
    map_tab, compare_tab = st.tabs(["Map & trace", "Compare policies"])
    with map_tab:
        st.plotly_chart(vehicle_map(trace, frame_idx, policy, location), use_container_width=True)
        render_trace(trace, frame_idx, location)
    with compare_tab:
        render_comparison(traces, location)


main()
