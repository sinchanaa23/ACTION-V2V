# ruff: noqa: E501 - long inline HTML/CSS strings
"""The dashboard's pages. Owner: P4.

Overview, Live Simulation, Road Network, Reports, Policy Comparison and
Settings. Every number shown is computed from recorded SUMO traces; stages
that need P1/P2/P3 output that is not recorded yet say so.
"""

import math
from collections import Counter
from collections.abc import Callable
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from ui_data import (
    LOCATIONS,
    METRIC_LABELS,
    POLICIES,
    View,
    alerts,
    load,
    metrics,
    reports,
    trace_files,
    traces_dir,
    warning_text,
)
from ui_map import (
    BASEMAPS,
    MAP_CONFIG,
    SENSOR_RADIUS_M,
    distance_m,
    live_map,
    sites_map,
    traffic_map,
)
from ui_theme import (
    BLUE,
    BORDER,
    CONDITION_COLOURS,
    FAINT,
    GREY,
    LIME,
    MUTED,
    ORANGE,
    OUTCOME_COLOURS,
    POLICY_COLOURS,
    PURPLE,
    RAISED,
    RED,
    SURFACE,
    TEXT,
    badge,
    bar,
    html,
    kpi,
    pct,
    style_fig,
)

from v2v.sim.recorder import Trace

PREFS = {
    "pref_location": "known",
    "pref_basemap": "Dark",
    "pref_zones": True,
    "pref_ticks": True,
    "pref_policy": "directional",
}
ROLE_COLOURS = {
    "original": LIME,
    "duplicate": BLUE,
    "stale": ORANGE,
    "contradiction": PURPLE,
    "false": RED,
    "non_actionable": GREY,
}
LOCATION_LABELS = {"known": "Known location", "fallback": "Fallback (none stated)"}
CHART_CONFIG = {"displaylogo": False, "displayModeBar": False}


def init_prefs() -> None:
    for key, value in PREFS.items():
        st.session_state.setdefault(key, value)


def _sync(widget: str, pref: str) -> None:
    st.session_state[pref] = st.session_state[widget]


def pref_radio(
    label: str, options: tuple[str, ...], pref: str, key: str, fmt: Callable[[str], str] = str
) -> str:
    """A radio whose choice outlives page switches (widgets forget when not drawn)."""
    value = st.session_state[pref]
    return str(
        st.radio(
            label,
            options,
            index=options.index(value) if value in options else 0,
            key=key,
            format_func=fmt,
            horizontal=True,
            on_change=_sync,
            args=(key, pref),
        )
    )


def pref_toggle(label: str, pref: str, key: str) -> bool:
    return bool(
        st.toggle(label, value=st.session_state[pref], key=key, on_change=_sync, args=(key, pref))
    )


def title(text: str, subtitle: str) -> None:
    html(f'<div class="v2v-title">{text}</div><div class="v2v-subtitle">{subtitle}</div>')


def cardinal(heading: float) -> str:
    return ("N", "NE", "E", "SE", "S", "SW", "W", "NW")[round(heading / 45) % 8]


def compass_svg(heading: float, size: int = 118) -> str:
    """Compass dial with the vehicle's heading (degrees clockwise from north)."""
    c = size / 2
    ticks = "".join(
        f'<line x1="{c}" y1="{6 if i % 3 == 0 else 9}" x2="{c}" y2="14" '
        f'stroke="{TEXT if i % 9 == 0 else FAINT}" stroke-width="{1.6 if i % 9 == 0 else 1}" '
        f'transform="rotate({i * 10} {c} {c})"/>'
        for i in range(36)
    )
    letters = "".join(
        f'<text x="{c + (c - 24) * math.sin(math.radians(a))}" '
        f'y="{c - (c - 24) * math.cos(math.radians(a)) + 3.5}" fill="{MUTED}" font-size="9.5" '
        f'text-anchor="middle" font-weight="600">{t}</text>'
        for a, t in ((0, "N"), (90, "E"), (180, "S"), (270, "W"))
    )
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">'
        f'<circle cx="{c}" cy="{c}" r="{c - 2}" fill="{SURFACE}" stroke="{BORDER}"/>'
        f'<circle cx="{c}" cy="{c}" r="{c - 34}" fill="#141414" stroke="{BORDER}"/>{ticks}{letters}'
        f'<g transform="rotate({heading} {c} {c})">'
        f'<polygon points="{c},{18} {c - 5},{c} {c + 5},{c}" fill="{RED}"/>'
        f'<polygon points="{c},{size - 18} {c - 5},{c} {c + 5},{c}" fill="{FAINT}"/></g>'
        f'<circle cx="{c}" cy="{c}" r="15" fill="#101010" stroke="{BORDER}"/>'
        f'<text x="{c}" y="{c + 4}" fill="{TEXT}" font-size="11" font-weight="700" '
        f'text-anchor="middle">{cardinal(heading)}</text></svg>'
    )


def speed_gauge(kmh: float) -> go.Figure:
    top = max(80.0, math.ceil(kmh / 20) * 20)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=kmh,
            number={"suffix": " km/h", "font": {"size": 18, "color": TEXT}},
            gauge={
                "axis": {"range": [0, top], "tickcolor": FAINT, "tickfont": {"size": 8}},
                "bar": {"color": LIME, "thickness": 0.28},
                "bgcolor": RAISED,
                "borderwidth": 0,
                "steps": [{"range": [top * 0.75, top], "color": "rgba(255,90,82,0.25)"}],
            },
        )
    )
    style_fig(fig, height=128, legend=False)
    fig.update_layout(margin={"l": 14, "r": 14, "t": 14, "b": 0})
    return fig


# --- live simulation -----------------------------------------------------------------


def _telemetry(view: View) -> None:
    frame, trace = view.frame, view.trace
    outcome, result = view.outcome, view.result
    hz = (trace.hazard_lon, trace.hazard_lat)
    reporter = frame.reporter
    ordered = [
        reporter,
        *sorted(frame.vehicles, key=lambda v: (not v.relevant, distance_m(v.lon, v.lat, *hz))),
    ]
    by_id = {v.vehicle_id: v for v in ordered}
    ids = list(by_id)
    tracked = st.selectbox(
        "Track vehicle",
        ids,
        key="track",
        format_func=lambda i: (
            f"{i} · reporter" if i == reporter.vehicle_id else f"{i} · {outcome[i]}"
        ),
    )
    v = by_id[str(tracked)]
    is_reporter = v.vehicle_id == reporter.vehicle_id
    label = "reporter" if is_reporter else outcome[v.vehicle_id]
    colour = PURPLE if is_reporter else OUTCOME_COLOURS[label]
    approach = v.edge_id in trace.spec.approach_route
    opposing = v.edge_id in trace.spec.opposing_route
    road = "approach route" if approach else "opposing route" if opposing else "other road"
    dist = distance_m(v.lon, v.lat, *hz)
    html(
        f'<div class="v2v-row"><div><div class="v2v-mid">{v.vehicle_id}</div>'
        f'<div class="v2v-muted">edge {v.edge_id} · {road}</div></div>{badge(label, colour)}</div>'
    )
    html(
        f'<div class="v2v-row" style="margin-top:4px"><span class="v2v-muted">Straight line to hazard'
        f'</span><b style="font-size:13px">{dist:,.0f} m</b></div>'
        + bar(1 - min(dist / trace.radius_m, 1.0), RED if dist < SENSOR_RADIUS_M else ORANGE)
        + f'<div class="v2v-muted" style="margin-top:4px;font-size:11px">geocast radius '
        f"{trace.radius_m:.0f} m · sensor zone {SENSOR_RADIUS_M:.0f} m</div>"
    )
    left, right = st.columns([1.15, 1])
    with left:
        html('<div class="v2v-label">Speed</div>')
        st.plotly_chart(speed_gauge(v.speed_mps * 3.6), config=CHART_CONFIG, key="gauge")
    with right:
        html(
            f'<div class="v2v-label">Heading {v.heading_deg:.0f}°</div>'
            f'<div style="display:grid;place-items:center;margin-top:6px">{compass_svg(v.heading_deg)}</div>'
        )
    if is_reporter:
        note, tone = (
            f"Reporter · passed the hazard at {frame.t_pass_s:.0f} s, sent at {frame.t_s:.0f} s",
            PURPLE,
        )
    elif v.vehicle_id in result.latency_ms:
        hops = result.hops.get(v.vehicle_id, "?")
        tone = LIME if v.relevant else BLUE
        note = (
            f"Warning delivered in {result.latency_ms[v.vehicle_id]:.1f} ms over {hops} hop(s)"
            + (" · will reach the hazard" if v.relevant else " · won't reach the hazard")
        )
    elif v.relevant:
        tone = ORANGE if v.vehicle_id in result.recipients else RED
        note = (
            "Addressed but the radio lost it · will reach the hazard"
            if v.vehicle_id in result.recipients
            else "Not warned · will reach the hazard"
        )
    else:
        note, tone = "Not addressed · won't reach the hazard", MUTED
    html(
        f'<div class="v2v-alert" style="background:{tone}1a;border:1px solid {tone}40;color:{tone}">'
        f"<span>●</span><span>{note}</span></div>"
    )


def _hazard_panel(view: View) -> None:
    spec, frame = view.trace.spec, view.frame
    where = ", ".join(spec.landmark_names[:2]) or spec.landmark_kind
    happened = spec.event_occurred
    html(
        f'<div class="v2v-row"><span class="v2v-label">Active hazard</span>'
        f"{badge(spec.effect.replace('_', ' '), RED)}</div>"
        f'<div class="v2v-mid" style="margin-top:2px">{spec.hazard_class.value.replace("_", " ").capitalize()}</div>'
        f'<div class="v2v-muted">{where} · {spec.landmark_kind}</div>'
        f'<div class="v2v-row" style="margin-top:8px"><span class="v2v-muted">Event window</span>'
        f'<b style="font-size:12.5px">{spec.event_start_s:.0f}–{spec.event_end_s:.0f} s</b></div>'
        f'<div class="v2v-row"><span class="v2v-muted">Ground truth</span>'
        + badge(
            "event happened" if happened else "event did NOT happen", LIME if happened else BLUE
        )
        + "</div>"
    )
    html('<div style="height:1px;background:rgba(255,255,255,0.07);margin:4px 0"></div>')
    flags = frame.sensor_flags
    html(
        '<div class="v2v-label">Verification</div>'
        f'<div class="v2v-row"><span class="v2v-muted">Sensor flags ≤ {SENSOR_RADIUS_M:.0f} m</span>'
        f'<b style="color:{LIME if flags else MUTED};font-size:12.5px">{flags}</b></div>'
        f'<div class="v2v-row"><span class="v2v-muted">Confidence verdict</span>'
        f"{badge('pending P3', ORANGE)}</div>"
        f'<div class="v2v-row"><span class="v2v-muted">Sent as</span>{badge("unverified", MUTED)}</div>'
    )
    html('<div style="height:1px;background:rgba(255,255,255,0.07);margin:4px 0"></div>')
    counts = Counter(view.outcome.values())
    total = max(len(frame.vehicles), 1)
    rows = "".join(
        f'<div class="v2v-row" style="font-size:12px;margin-top:5px"><span style="color:{MUTED}">'
        f'<i style="display:inline-block;width:7px;height:7px;border-radius:50%;background:{c};'
        f'margin-right:7px"></i>{label}</span><b>{counts.get(label, 0)}</b></div>'
        + bar(counts.get(label, 0) / total, c)
        for label, c in OUTCOME_COLOURS.items()
    )
    html(f'<div class="v2v-label">Recipients · {view.policy}</div>{rows}')


def _legend(location: str, show_zones: bool) -> str:
    items = [(c, label) for label, c in OUTCOME_COLOURS.items()]
    items += [(RED, "hazard"), (PURPLE, "reporter"), ("#ffffff", "tracked")]
    if location == "fallback":
        items.append((ORANGE, "fallback location"))
    spans = "".join(f'<span><i style="background:{c}"></i>{t}</span>' for c, t in items)
    if show_zones:
        spans += f'<span><i style="background:transparent;border:1px solid {LIME}"></i>geocast area</span>'
        spans += (
            f'<span><i style="background:{RED}44;border:1px solid {RED}"></i>sensor zone</span>'
        )
    return f'<div class="v2v-legend">{spans}</div>'


def _timeline(view: View, frames_idx: int) -> go.Figure:
    trace, spec = view.trace, view.trace.spec
    fig = go.Figure()
    fig.add_vrect(
        x0=spec.event_start_s,
        x1=spec.event_end_s,
        fillcolor=RED if spec.event_occurred else BLUE,
        opacity=0.09,
        line_width=0,
        annotation_text="event" if spec.event_occurred else "no event (misleading)",
        annotation_font={"color": MUTED, "size": 10},
        annotation_position="top left",
    )
    for i, f in enumerate(trace.frames):
        selected = i == frames_idx
        colour = ROLE_COLOURS.get(f.slot.role, MUTED)
        fig.add_trace(
            go.Scatter(
                x=[f.t_pass_s, f.t_s],
                y=[i, i],
                mode="lines+markers",
                line={"color": colour, "width": 3 if selected else 1.5},
                marker={
                    "size": [8, 13 if selected else 9],
                    "color": [SURFACE, colour],
                    "line": {"color": colour, "width": 2},
                },
                hovertext=[
                    f"{f.slot.reporter_id} passed the hazard at {f.t_pass_s:.0f} s",
                    f"{f.slot.role} report sent at {f.t_s:.0f} s",
                ],
                hoverinfo="text",
                name=f.slot.role,
            )
        )
    fig.update_yaxes(
        tickvals=list(range(len(trace.frames))),
        ticktext=[f.slot.role for f in trace.frames],
        autorange="reversed",
        showgrid=False,
    )
    fig.update_xaxes(
        range=[0, spec.end_s], title={"text": "simulation time (s)", "font": {"size": 10}}
    )
    style_fig(fig, height=110 + 34 * max(len(trace.frames), 1), legend=False)
    fig.update_layout(margin={"l": 8, "r": 8, "t": 22, "b": 36})
    return fig


def _trace_steps(view: View) -> None:
    frame, spec = view.frame, view.trace.spec
    slot, result = frame.slot, view.result
    m = view.metrics()
    text = warning_text(view)
    where = ", ".join(spec.landmark_names[:2]) or spec.landmark_kind
    needs = f"a real <b>{slot.needs.replace('_', ' ')}</b> text" + (
        f" of class <b>{slot.hazard_class.value}</b>" if slot.needs != "non_actionable" else ""
    )
    warn = (
        f"“{text}”<br><span style='color:{MUTED}'>P2 generator · status unverified until P3</span>"
        if text
        else f"<span style='color:{MUTED}'>A correct reading sends nothing here; the recorder "
        "still distributes it to measure the cost.</span>"
    )
    steps = [
        (
            "Report",
            PURPLE,
            f"{slot.reporter_id} · {slot.role}<br>passed {frame.t_pass_s:.0f} s, sent "
            f"{frame.t_s:.0f} s<br>needs {needs}<br><span style='color:{MUTED}'>text from P1's test "
            "split: pending</span>",
        ),
        (
            "NLP reading",
            BLUE,
            f"<span style='color:{MUTED}'>pending P2's models on the report "
            "text (recorded by P3's runner)</span>",
        ),
        (
            "Confidence",
            ORANGE,
            f"{frame.sensor_flags} sensor flag(s) within {SENSOR_RADIUS_M:.0f} m"
            f"<br><span style='color:{MUTED}'>verdict pending P3</span>",
        ),
        (
            "Location",
            LIME,
            f"true hazard: {where}"
            if view.location == "known"
            else "fallback: reporter's position (approx)",
        ),
        ("Warning", RED, warn),
        (
            "Distribution",
            LIME,
            f"{view.policy} · {len(result.recipients)} addressed, "
            f"{len(result.latency_ms)} delivered<br>{result.transmissions} transmissions · "
            f"needed it delivered {pct(m['relevant_reached'])}",
        ),
    ]
    cols = st.columns(len(steps))
    for i, (col, (name, colour, body)) in enumerate(zip(cols, steps, strict=True), start=1):
        col.markdown(
            f'<div class="v2v-step"><div style="margin-bottom:6px"><span class="n" '
            f'style="background:{colour}22;color:{colour}">{i}</span><b>{name}</b></div>{body}</div>',
            unsafe_allow_html=True,
        )
    relevant = sum(v.relevant for v in frame.vehicles)
    html(
        f'<div class="v2v-muted" style="margin-top:4px">Ground truth: event '
        f"{'happened' if spec.event_occurred else 'did NOT happen'} · {relevant} of "
        f"{len(frame.vehicles)} vehicles will reach the hazard</div>"
    )


def _policy_table(view: View) -> str:
    keys = ("recall", "precision", "relevant_reached", "transmissions", "mean_latency_ms")
    head = "".join(
        f'<th style="text-align:right;font-weight:500">{METRIC_LABELS[k]}</th>' for k in keys
    )
    body = ""
    for policy in POLICIES:
        m = view.metrics(policy)
        cells = ""
        for k in keys:
            value = m[k]
            if k in ("recall", "precision", "relevant_reached"):
                shown = pct(value)
            elif value is None:
                shown = "—"
            else:
                shown = f"{value:,.1f}" if k == "mean_latency_ms" else f"{value:,.0f}"
            cells += f'<td style="text-align:right;color:{TEXT}">{shown}</td>'
        weight = "600" if policy == view.policy else "400"
        body += (
            f'<tr style="border-top:1px solid {BORDER}"><td style="font-weight:{weight};white-space:nowrap;padding:7px 0">'
            f'<i style="display:inline-block;width:8px;height:8px;border-radius:50%;'
            f'background:{POLICY_COLOURS[policy]};margin-right:8px"></i>{policy}</td>{cells}</tr>'
        )
    return (
        f'<table style="width:100%;border-collapse:collapse;font-size:12px;color:{MUTED}">'
        f'<tr><th style="text-align:left;font-weight:500">Policy</th>{head}</tr>{body}</table>'
    )


def _network_activity(view: View) -> go.Figure:
    result, frame = view.result, view.frame
    relevant = {v.vehicle_id for v in frame.vehicles if v.relevant}
    hops = sorted(set(result.hops.values())) or [1]
    fig = go.Figure()
    for name, members, colour in (
        ("needed it", relevant, LIME),
        ("didn't need it", None, BLUE),
    ):
        counts = [
            sum(
                1
                for vid, h in result.hops.items()
                if h == hop
                and vid in result.latency_ms
                and ((vid in relevant) if members is not None else (vid not in relevant))
            )
            for hop in hops
        ]
        fig.add_trace(
            go.Bar(x=[f"hop {h}" for h in hops], y=counts, name=name, marker_color=colour)
        )
    fig.update_layout(barmode="stack")
    return style_fig(fig, height=210)


def page_live(traces: dict[str, Trace], trace: Trace) -> None:
    spec = trace.spec
    if not trace.frames:
        title("Live Simulation", spec.scenario_id)
        st.error("No report fired in this run.")
        return
    head, toggles = st.columns([3, 2], vertical_alignment="bottom")
    with toggles:
        t1, t2 = st.columns(2)
        with t1:
            show_zones = pref_toggle("Hazard & geocast zones", "pref_zones", "w_zones")
        with t2:
            show_ticks = pref_toggle("Heading ticks", "pref_ticks", "w_ticks")
    c_report, c_policy, c_loc = st.columns([3.2, 2.4, 2.2], vertical_alignment="bottom")
    with c_report:
        frame_idx = int(
            st.radio(
                "Report",
                range(len(trace.frames)),
                key="report",
                horizontal=True,
                format_func=lambda i: f"{trace.frames[i].t_s:.0f}s · {trace.frames[i].slot.role}",
            )
        )
    with c_policy:
        policy = pref_radio("Policy", POLICIES, "pref_policy", "policy")
    with c_loc:
        location = pref_radio(
            "Location", LOCATIONS, "pref_location", "location", lambda m: LOCATION_LABELS[m]
        )
    frame = trace.frames[frame_idx]
    view = View(trace, frame, policy, location)
    with head:
        title(
            "Live Simulation",
            f"{spec.corridor.capitalize()} corridor · {', '.join(spec.landmark_names[:2])} · "
            f"{spec.condition} traffic, {spec.density} density, seed {spec.seed} · "
            f"report at t = {frame.t_s:.0f} s",
        )
    with st.container(key="stage"):
        tracked = st.session_state.get("track", frame.reporter.vehicle_id)
        st.plotly_chart(
            live_map(
                trace,
                frame,
                view.result,
                view.outcome,
                location=location,
                tracked=tracked,
                show_zones=show_zones,
                show_ticks=show_ticks,
                style=BASEMAPS[st.session_state["pref_basemap"]],
            ),
            config=MAP_CONFIG,
            key="live_map",
        )
        with st.container(key="float_left"):
            _telemetry(view)
        with st.container(key="float_right"):
            _hazard_panel(view)
        html(_legend(location, show_zones))
    html(
        '<div class="v2v-foot"><span>Scroll to zoom · drag to pan · hover a vehicle for details'
        f"</span><span>Recorded SUMO run · geocast radius {trace.radius_m:.0f} m · radio range "
        f"{trace.comm_range_m:.0f} m · Nakagami-m (m=3) multi-hop · © OpenStreetMap, © CARTO</span></div>"
    )
    m = view.metrics()
    left, mid, right = st.columns([1.05, 1, 1.25])
    with left, st.container(key="card_delivery"):
        html(
            f'<div class="v2v-row"><span class="v2v-h">Delivery statistics</span>'
            f"{badge(policy, POLICY_COLOURS[policy])}</div>"
        )
        a, b = st.columns(2)
        a.markdown(
            kpi(
                "Needed it: delivered",
                pct(m["relevant_reached"]),
                "of vehicles that will reach it",
                LIME,
            ),
            unsafe_allow_html=True,
        )
        b.markdown(
            kpi("Needed it: addressed", pct(m["recall"]), "recall of recipient selection"),
            unsafe_allow_html=True,
        )
        a, b = st.columns(2)
        lat = m["mean_latency_ms"]
        a.markdown(
            kpi("Transmissions", f"{m['transmissions']:.0f}", f"{m['receptions']:.0f} receptions"),
            unsafe_allow_html=True,
        )
        b.markdown(
            kpi(
                "Mean latency",
                "—" if lat is None else f"{lat:.1f} ms",
                f"{m['irrelevant_reached']:.0f} unneeded deliveries",
                BLUE,
            ),
            unsafe_allow_html=True,
        )
    with mid, st.container(key="card_network"):
        html(
            '<div class="v2v-h">Network activity</div><div class="v2v-muted">'
            "Delivered recipients by hop count</div>"
        )
        st.plotly_chart(_network_activity(view), config=CHART_CONFIG, key="hops")
    with right, st.container(key="card_perf"):
        html(
            '<div class="v2v-h">Performance · this report</div><div class="v2v-muted">'
            f"All three policies, {LOCATION_LABELS[location].lower()}</div>"
        )
        html(_policy_table(view))
    with st.container(key="card_timeline"):
        html(
            '<div class="v2v-row"><span class="v2v-h">Simulation timeline</span><span class="v2v-muted">'
            "○ reporter passes the hazard · ● report sent</span></div>"
        )
        st.plotly_chart(_timeline(view, frame_idx), config=CHART_CONFIG, key="timeline")
    with st.container(key="card_trace"):
        html('<div class="v2v-h">Report-to-warning trace</div>')
        _trace_steps(view)


# --- overview --------------------------------------------------------------------------


def page_overview(traces: dict[str, Trace], trace: Trace) -> None:
    files = trace_files(traces_dir())
    df = metrics(files)
    loc = st.session_state["pref_location"]
    title(
        "Overview",
        f"{len(traces)} recorded SUMO runs · {datetime.now():%A, %d %B %Y} · "
        f"{LOCATION_LABELS[loc].lower()}",
    )
    d = df[df["location"] == loc]
    by_policy = d.groupby("policy")[list(METRIC_LABELS)].mean()
    n_reports = sum(len(t.frames) for t in traces.values())
    snaps = sum(len(f.vehicles) for t in traces.values() for f in t.frames)
    tele = sum(t.teleported for t in traces.values())
    passed = sum(t.vehicles_passed_hazard for t in traces.values())
    cut = None
    if not by_policy.empty and by_policy.loc["directional", "transmissions"]:
        cut = (
            by_policy.loc["broadcast", "transmissions"]
            / by_policy.loc["directional", "transmissions"]
        )
    cards = [
        kpi(
            "Scenarios",
            f"{len(traces)}",
            f"{len({t.spec.condition for t in traces.values()})} conditions",
        ),
        kpi("Reports fired", f"{n_reports}", f"{snaps:,} vehicle snapshots"),
        kpi(
            "Directional delivered",
            pct(by_policy["relevant_reached"].get("directional") if not by_policy.empty else None),
            f"broadcast {pct(by_policy['relevant_reached'].get('broadcast') if not by_policy.empty else None)}",
            LIME,
        ),
        kpi(
            "Directional precision",
            pct(by_policy["precision"].get("directional") if not by_policy.empty else None),
            f"broadcast {pct(by_policy['precision'].get('broadcast') if not by_policy.empty else None)}",
        ),
        kpi(
            "Fewer transmissions",
            "—" if cut is None else f"{cut:.1f}×",
            "broadcast ÷ directional",
            BLUE,
        ),
        kpi("Passed the hazard", f"{passed:,}", f"{tele} teleported", ORANGE if tele else TEXT),
    ]
    for col, card in zip(st.columns(len(cards)), cards, strict=True):
        col.markdown(card, unsafe_allow_html=True)
    left, right = st.columns([1.6, 1])
    with left, st.container(key="card_sites"):
        html(
            '<div class="v2v-row"><span class="v2v-h">Hazard sites</span><span class="v2v-muted">'
            + " ".join(
                f'<span style="color:{c}">●</span> {k}' for k, c in CONDITION_COLOURS.items()
            )
            + "</span></div>"
        )
        st.plotly_chart(
            sites_map(list(traces.values()), BASEMAPS[st.session_state["pref_basemap"]]),
            config=MAP_CONFIG,
            key="sites_map",
        )
    with right, st.container(key="card_heat"):
        html(
            '<div class="v2v-h">Directional: needed it delivered</div><div class="v2v-muted">'
            "Mean over reports, by condition and density</div>"
        )
        heat = (
            d[d["policy"] == "directional"]
            .pivot_table(
                index="condition", columns="density", values="relevant_reached", aggfunc="mean"
            )
            .reindex(columns=[c for c in ("low", "medium", "high") if c in set(d["density"])])
        )
        if not heat.empty:
            fig = go.Figure(
                go.Heatmap(
                    z=heat.values,
                    x=list(heat.columns),
                    y=list(heat.index),
                    colorscale=[[0, "#262626"], [1, LIME]],
                    zmin=0,
                    zmax=1,
                    text=[[pct(v) if v == v else "—" for v in row] for row in heat.values],
                    texttemplate="%{text}",
                    textfont={"color": "#111", "size": 13},
                    showscale=False,
                    xgap=4,
                    ygap=4,
                    hovertemplate="%{y} · %{x}: %{text}<extra></extra>",
                )
            )
            st.plotly_chart(
                style_fig(fig, height=200, legend=False), config=CHART_CONFIG, key="heat"
            )
        html('<div class="v2v-h" style="margin-top:6px">Alerts</div>')
        for severity, message in alerts(traces)[:6]:
            colour = {"red": RED, "orange": ORANGE, "blue": BLUE}[severity]
            html(
                f'<div class="v2v-alert" style="background:{colour}14;border:1px solid {colour}33;'
                f'color:{TEXT};margin-top:6px"><span style="color:{colour}">●</span>{message}</div>'
            )
    with st.container(key="card_runs"):
        html('<div class="v2v-h">Recorded runs</div>')
        rows = []
        for t in traces.values():
            s = t.spec
            dd = d[(d["scenario"] == s.scenario_id) & (d["policy"] == "directional")]
            rows.append(
                {
                    "scenario": s.scenario_id,
                    "condition": s.condition,
                    "density": s.density,
                    "hazard": s.hazard_class.value,
                    "effect": s.effect,
                    "landmark": ", ".join(s.landmark_names[:2]),
                    "event": "happened" if s.event_occurred else "did not happen",
                    "reports": len(t.frames),
                    "passed hazard": t.vehicles_passed_hazard,
                    "teleported": t.teleported,
                    "directional delivered": dd["relevant_reached"].mean() if len(dd) else None,
                }
            )
        st.dataframe(
            pd.DataFrame(rows).set_index("scenario"),
            column_config={
                "directional delivered": st.column_config.ProgressColumn(
                    min_value=0, max_value=1, format="percent"
                )
            },
        )


# --- road network ----------------------------------------------------------------------


def page_network(traces: dict[str, Trace], trace: Trace) -> None:
    spec = trace.spec
    head, tog = st.columns([3, 1], vertical_alignment="bottom")
    with head:
        title(
            "Road Network",
            f"{spec.corridor.capitalize()} corridor · real OpenStreetMap roads in SUMO · every "
            "vehicle position recorded in this run, coloured by speed",
        )
    with tog:
        show_zones = pref_toggle("Zones", "pref_zones", "w_zones_net")
    with st.container(key="stage"):
        st.plotly_chart(
            traffic_map(trace, BASEMAPS[st.session_state["pref_basemap"]], show_zones),
            config=MAP_CONFIG,
            key="traffic_map",
        )
        with st.container(key="float_right"):
            snaps = sum(len(f.vehicles) + 1 for f in trace.frames)
            on_routes = sum(
                v.edge_id in spec.approach_route or v.edge_id in spec.opposing_route
                for f in trace.frames
                for v in f.vehicles
            )
            rows = [
                ("Landmark", ", ".join(spec.landmark_names[:3]) or "—"),
                ("Kind", spec.landmark_kind),
                ("OSM id", spec.landmark_id),
                ("Hazard edge", f"{spec.hazard.edge_id} @ {spec.hazard.pos_m:.0f} m"),
                ("Opposite edge", spec.opposite_edge),
                ("Approach route", f"{len(spec.approach_route)} edges"),
                ("Opposing route", f"{len(spec.opposing_route)} edges"),
                ("Snapshots", f"{snaps:,} ({on_routes:,} on the two routes)"),
                ("Passed hazard", f"{trace.vehicles_passed_hazard}"),
                ("Teleported", f"{trace.teleported}"),
            ]
            html(
                '<div class="v2v-label">Scenario geometry</div>'
                f'<div class="v2v-mid">{spec.hazard_class.value.replace("_", " ").capitalize()}</div>'
            )
            html(
                "".join(
                    f'<div class="v2v-row" style="font-size:12px;margin-top:6px"><span style="color:{MUTED}">'
                    f'{k}</span><span style="text-align:right;color:{TEXT}">{v}</span></div>'
                    for k, v in rows
                )
            )
        legend = (
            f'<span><i style="background:{LIME}"></i>free flow</span>'
            f'<span><i style="background:{ORANGE}"></i>slow</span>'
            f'<span><i style="background:{RED}"></i>stopped</span>'
            "<span>larger dots: approach / opposing route</span>"
        )
        if show_zones:
            legend += (
                f'<span><i style="border:1px solid {LIME}"></i>geocast {trace.radius_m:.0f} m</span>'
                f'<span><i style="border:1px solid {BLUE}"></i>radio {trace.comm_range_m:.0f} m</span>'
                f'<span><i style="border:1px solid {RED}"></i>sensor {SENSOR_RADIUS_M:.0f} m</span>'
            )
        html(f'<div class="v2v-legend">{legend}</div>')
    left, right = st.columns(2)
    snaps_all = [v for f in trace.frames for v in f.vehicles]
    with left, st.container(key="card_speed"):
        html(
            '<div class="v2v-h">Speed distribution</div><div class="v2v-muted">All snapshots in this run</div>'
        )
        fig = go.Figure(
            go.Histogram(x=[v.speed_mps * 3.6 for v in snaps_all], nbinsx=30, marker_color=LIME)
        )
        fig.update_xaxes(title={"text": "km/h", "font": {"size": 10}})
        st.plotly_chart(
            style_fig(fig, height=230, legend=False), config=CHART_CONFIG, key="speed_hist"
        )
    with right, st.container(key="card_dist"):
        html(
            '<div class="v2v-h">Distance to hazard</div><div class="v2v-muted">'
            "Straight line, split by ground truth</div>"
        )
        fig = go.Figure()
        for name, keep, colour in (("will reach it", True, RED), ("won't", False, BLUE)):
            fig.add_trace(
                go.Histogram(
                    x=[
                        distance_m(v.lon, v.lat, trace.hazard_lon, trace.hazard_lat)
                        for v in snaps_all
                        if v.relevant is keep
                    ],
                    name=name,
                    marker_color=colour,
                    nbinsx=30,
                    opacity=0.85,
                )
            )
        fig.update_layout(barmode="overlay")
        fig.update_xaxes(title={"text": "metres", "font": {"size": 10}})
        st.plotly_chart(style_fig(fig, height=230), config=CHART_CONFIG, key="dist_hist")


# --- reports ---------------------------------------------------------------------------


def page_reports(traces: dict[str, Trace], trace: Trace) -> None:
    df = reports(trace_files(traces_dir()))
    title("Reports", f"{len(df)} driver reports fired across {len(traces)} recorded runs")
    if df.empty:
        st.info("No reports recorded yet.")
        return
    a, b = st.columns(2)
    roles = a.multiselect("Role", sorted(df["role"].unique()), key="rep_roles")
    conditions = b.multiselect("Condition", sorted(df["condition"].unique()), key="rep_cond")
    if roles:
        df = df[df["role"].isin(roles)]
    if conditions:
        df = df[df["condition"].isin(conditions)]
    left, right = st.columns([1, 1.4])
    with left, st.container(key="card_roles"):
        html('<div class="v2v-h">Reports by role</div>')
        counts = df["role"].value_counts()
        fig = go.Figure(
            go.Bar(
                x=counts.values,
                y=counts.index,
                orientation="h",
                marker_color=[ROLE_COLOURS.get(r, MUTED) for r in counts.index],
                text=counts.values,
                textposition="outside",
            )
        )
        fig.update_yaxes(autorange="reversed", showgrid=False)
        fig.update_xaxes(showgrid=False, visible=False)
        st.plotly_chart(style_fig(fig, height=250, legend=False), config=CHART_CONFIG, key="roles")
    with right, st.container(key="card_delay"):
        html(
            '<div class="v2v-h">Pass → send delay</div><div class="v2v-muted">'
            "When each reporter passed the hazard and when the report went out</div>"
        )
        fig = px.scatter(
            df,
            x="passed (s)",
            y="delay (s)",
            color="role",
            color_discrete_map=ROLE_COLOURS,
            hover_data=["scenario", "reporter", "sent (s)"],
        )
        fig.update_traces(marker={"size": 10, "line": {"width": 0}})
        st.plotly_chart(style_fig(fig, height=250), config=CHART_CONFIG, key="delay")
    with st.container(key="card_reports"):
        html('<div class="v2v-h">All reports</div>')
        st.dataframe(
            df,
            hide_index=True,
            column_config={
                "directional delivered": st.column_config.ProgressColumn(
                    min_value=0, max_value=1, format="percent"
                )
            },
        )


# --- policy comparison -----------------------------------------------------------------


def page_compare(traces: dict[str, Trace], trace: Trace) -> None:
    head, right = st.columns([3, 2], vertical_alignment="bottom")
    with head:
        title(
            "Policy Comparison",
            "Broadcast vs circular vs directional geocast, across every recorded report",
        )
    with right:
        location = pref_radio(
            "Location", LOCATIONS, "pref_location", "location_cmp", lambda m: LOCATION_LABELS[m]
        )
    df = metrics(trace_files(traces_dir()))
    df = df[df["location"] == location]
    if df.empty:
        st.info("No reports recorded yet.")
        return
    c1, c2, c3 = st.columns(3)
    conditions = c1.multiselect("Condition", sorted(df["condition"].unique()), key="cmp_cond")
    densities = c2.multiselect("Density", sorted(df["density"].unique()), key="cmp_dens")
    roles = c3.multiselect("Report role", sorted(df["report"].unique()), key="cmp_role")
    if conditions:
        df = df[df["condition"].isin(conditions)]
    if densities:
        df = df[df["density"].isin(densities)]
    if roles:
        df = df[df["report"].isin(roles)]
    if df.empty:
        st.info("No reports match these filters.")
        return
    html(
        f'<div class="v2v-muted">{df["scenario"].nunique()} scenario(s), '
        f"{len(df) // len(POLICIES)} report(s); means over reports. Delivery model: Nakagami-m "
        "(m=3), multi-hop.</div>"
    )
    summary = df.groupby("policy")[list(METRIC_LABELS)].mean().reindex(POLICIES)
    for col, policy in zip(st.columns(3), POLICIES, strict=True):
        s = summary.loc[policy]
        colour = POLICY_COLOURS[policy]
        lat = s["mean_latency_ms"]
        col.markdown(
            f'<div class="v2v-kpi" style="border-color:{colour}33">'
            f'<div class="v2v-row"><span class="v2v-label">{policy}</span>'
            f'<i style="width:8px;height:8px;border-radius:50%;background:{colour}"></i></div>'
            f'<div class="v" style="color:{colour}">{pct(s["relevant_reached"])}</div>'
            f'<div class="d">needed it: delivered</div>{bar(s["relevant_reached"], colour)}'
            f'<div class="v2v-row" style="margin-top:10px;font-size:12px"><span class="v2v-muted">'
            f"precision</span><b>{pct(s['precision'])}</b></div>"
            f'<div class="v2v-row" style="font-size:12px"><span class="v2v-muted">transmissions'
            f"</span><b>{s['transmissions']:.1f}</b></div>"
            f'<div class="v2v-row" style="font-size:12px"><span class="v2v-muted">receptions'
            f"</span><b>{s['receptions']:,.0f}</b></div>"
            f'<div class="v2v-row" style="font-size:12px"><span class="v2v-muted">mean latency'
            f"</span><b>{'—' if pd.isna(lat) else f'{lat:.1f} ms'}</b></div></div>",
            unsafe_allow_html=True,
        )
    left, right = st.columns(2)
    with left, st.container(key="card_rates"):
        html(
            '<div class="v2v-h">Selection and delivery</div><div class="v2v-muted">Share, higher is better</div>'
        )
        rates = summary[["recall", "precision", "relevant_reached"]].rename(columns=METRIC_LABELS)
        fig = px.bar(
            rates.reset_index().melt(id_vars="policy"),
            x="variable",
            y="value",
            color="policy",
            barmode="group",
            color_discrete_map=POLICY_COLOURS,
            labels={"variable": "", "value": "share"},
            range_y=[0, 1],
        )
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(style_fig(fig, height=300), config=CHART_CONFIG, key="rates")
    with right, st.container(key="card_load"):
        html(
            '<div class="v2v-h">Channel load</div><div class="v2v-muted">Per report, log scale, lower is better</div>'
        )
        load_df = summary[["transmissions", "receptions", "irrelevant_reached"]].rename(
            columns=METRIC_LABELS
        )
        fig = px.bar(
            load_df.reset_index().melt(id_vars="policy"),
            x="variable",
            y="value",
            color="policy",
            barmode="group",
            color_discrete_map=POLICY_COLOURS,
            labels={"variable": "", "value": "per report"},
            log_y=True,
        )
        st.plotly_chart(style_fig(fig, height=300), config=CHART_CONFIG, key="load")
    left, right = st.columns(2)
    for col, by, key in ((left, "density", "by_density"), (right, "condition", "by_condition")):
        with col, st.container(key=f"card_{key}"):
            html(f'<div class="v2v-h">Needed it: delivered, by {by}</div>')
            g = df.groupby([by, "policy"])["relevant_reached"].mean().reset_index()
            order = ["low", "medium", "high"] if by == "density" else sorted(g[by].unique())
            fig = px.bar(
                g,
                x=by,
                y="relevant_reached",
                color="policy",
                barmode="group",
                color_discrete_map=POLICY_COLOURS,
                category_orders={by: order, "policy": list(POLICIES)},
                labels={by: "", "relevant_reached": "share"},
                range_y=[0, 1],
            )
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(style_fig(fig, height=260), config=CHART_CONFIG, key=key)
    with st.container(key="card_summary"):
        html('<div class="v2v-h">Summary table</div>')
        st.dataframe(
            summary.rename(columns=METRIC_LABELS),
            column_config={
                METRIC_LABELS[k]: st.column_config.NumberColumn(format="percent")
                for k in ("recall", "precision", "relevant_reached")
            },
        )


# --- settings --------------------------------------------------------------------------


def page_settings(traces: dict[str, Trace], trace: Trace) -> None:
    title("Settings", "Display preferences for this browser session and where the data comes from")
    left, right = st.columns(2)
    with left, st.container(key="card_display"):
        html('<div class="v2v-h">Display</div>')
        pref_radio("Basemap", tuple(BASEMAPS), "pref_basemap", "w_basemap")
        pref_radio(
            "Default location mode",
            LOCATIONS,
            "pref_location",
            "w_location_set",
            lambda m: LOCATION_LABELS[m],
        )
        pref_radio("Default policy", POLICIES, "pref_policy", "w_policy_set")
        pref_toggle("Show hazard and geocast zones", "pref_zones", "w_zones_set")
        pref_toggle("Show heading ticks", "pref_ticks", "w_ticks_set")
    with right, st.container(key="card_data"):
        directory = traces_dir()
        files = sorted(directory.glob("*.json"))
        newest = max((p.stat().st_mtime for p in files), default=None)
        rows = [
            ("Traces folder", str(directory)),
            ("Recorded runs", str(len(files))),
            (
                "Last recorded",
                "—" if newest is None else f"{datetime.fromtimestamp(newest):%d %b %Y, %H:%M}",
            ),
            ("Geocast radius", f"{trace.radius_m:.0f} m"),
            ("Radio range", f"{trace.comm_range_m:.0f} m"),
            ("Delivery model", "Nakagami-m (m=3), multi-hop ≤ 5"),
            ("Sensor zone", f"{SENSOR_RADIUS_M:.0f} m"),
        ]
        html(
            '<div class="v2v-h">Data source</div>'
            + "".join(
                f'<div class="v2v-row" style="font-size:12px;margin-top:8px"><span style="color:{MUTED}">'
                f'{k}</span><span style="color:{TEXT};text-align:right;word-break:break-all">{v}</span></div>'
                for k, v in rows
            )
        )
        if st.button("Reload traces", icon=":material/refresh:", key="reload"):
            load.clear()
            metrics.clear()
            reports.clear()
            st.rerun()
        html(
            '<div class="v2v-muted" style="margin-top:8px">Record runs with '
            "<code>python -m v2v.sim.recorder sim_out/scenarios/hebbal/*-s0</code>. "
            "Report text, the NLP reading and the confidence verdict arrive with P3's experiment "
            "runner. Map data © OpenStreetMap contributors (ODbL), basemap © CARTO.</div>"
        )


PAGES: dict[str, tuple[str, str, Callable[[dict[str, Trace], Trace], None]]] = {
    "overview": ("Overview", ":material/space_dashboard:", page_overview),
    "live": ("Live Simulation", ":material/sensors:", page_live),
    "network": ("Road Network", ":material/route:", page_network),
    "reports": ("Reports", ":material/description:", page_reports),
    "compare": ("Policy Comparison", ":material/leaderboard:", page_compare),
    "settings": ("Settings", ":material/settings:", page_settings),
}
