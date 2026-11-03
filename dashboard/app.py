# ruff: noqa: E501 - long inline HTML/CSS strings
"""ACTION V2V operations dashboard. Owner: P4 (P4-07).

    python -m v2v.sim.recorder sim_out/scenarios/hebbal/*-s0     # record runs first
    streamlit run dashboard/app.py

Reads traces written by ``v2v.sim.recorder`` (``V2V_TRACES_DIR``, default
``sim_out/traces``). Pages: Overview, Live Simulation (map, telemetry,
hazard, delivery, timeline and report-to-warning trace), Road Network,
Reports, Policy Comparison and Settings: the Overseer view, fleet-wide.
Driver mode (``ui_driver``) is one demo vehicle's own dashboard on a live SUMO
run; the two buttons at the top of the rail switch between them. Report text,
the NLP reading and the confidence verdict are shown as pending until P3's
experiment runner records them; nothing here is placeholder data presented as
results.

Map data (c) OpenStreetMap contributors, ODbL; basemap (c) CARTO.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling ui_* modules

import streamlit as st  # noqa: E402
from ui_data import alerts, load, metrics, trace_files, traces_dir  # noqa: E402
from ui_driver import page_driver  # noqa: E402
from ui_pages import PAGES, init_prefs  # noqa: E402
from ui_theme import BLUE, ORANGE, RED, TEXT, chip, html, inject_css, pct  # noqa: E402

from v2v.sim.recorder import Trace  # noqa: E402

CONDITIONS = ("all", "clean", "noisy", "misleading")


def _go(page: str) -> None:
    st.session_state["page"] = page


def _mode(mode: str) -> None:
    st.session_state["mode"] = mode


def rail() -> str:
    """Slim icon navigation; the label shows as a tooltip. The two buttons at
    the top switch between Overseer (fleet-wide) and Driver mode."""
    page = st.session_state.setdefault("page", "live")
    mode = st.session_state.setdefault("mode", "overseer")
    with st.sidebar, st.container(key="rail"):
        html('<div class="v2v-logo">V2V</div>')
        for mid, name, icon in (
            ("overseer", "Overseer mode: fleet-wide monitoring", ":material/monitoring:"),
            ("driver", "Driver mode: one vehicle's dashboard", ":material/directions_car:"),
        ):
            st.button(
                name,
                icon=icon,
                key=f"mode_{mid}",
                help=name,
                type="primary" if mid == mode else "secondary",
                on_click=_mode,
                args=(mid,),
            )
        html('<div style="height:1px;width:28px;background:rgba(255,255,255,0.1)"></div>')
        if mode == "driver":
            return "driver"
        for pid, (name, icon, _) in PAGES.items():
            if pid == "settings":
                html('<div style="height:34vh"></div>')
            st.button(
                name,
                icon=icon,
                key=f"nav_{pid}",
                help=name,
                type="primary" if pid == page else "secondary",
                on_click=_go,
                args=(pid,),
            )
    return page


def top_bar(traces: dict[str, Trace]) -> Trace:
    files = trace_files(traces_dir())
    df = metrics(files)
    known = df[(df["location"] == "known") & (df["policy"] == "directional")]
    n_reports = sum(len(t.frames) for t in traces.values())
    condition = st.session_state.get("cond_filter", "all")
    names = [n for n, t in traces.items() if condition == "all" or t.spec.condition == condition]
    names = names or list(traces)
    brand, chips, pick, bell, gear = st.columns(
        [1.7, 6.2, 2.4, 0.5, 0.5], vertical_alignment="center"
    )
    with pick:
        name = st.selectbox(
            "Scenario",
            names,
            key="scenario",
            label_visibility="collapsed",
            placeholder="Search scenarios",
        )
    trace = traces[str(name)]
    with brand:
        html(
            '<div class="v2v-brand"><div class="v2v-logo" style="margin:0">V2V</div><div>'
            '<div class="name">ACTION V2V</div><div class="sub">Hazard warning operations</div>'
            "</div></div>"
        )
    with chips:
        html(
            f'<span class="v2v-live"><span class="v2v-dot"></span>SUMO replay · '
            f"{trace.spec.corridor}</span> "
            + chip("Runs", f"{len(traces)}")
            + chip("Reports", f"{n_reports}")
            + chip("Passed hazard", f"{sum(t.vehicles_passed_hazard for t in traces.values()):,}")
            + chip(
                "Directional delivered",
                pct(known["relevant_reached"].mean() if len(known) else None),
            )
        )
    found = alerts(traces)
    with bell, st.popover(f"{len(found)}", icon=":material/notifications:"):
        colours = {"red": RED, "orange": ORANGE, "blue": BLUE}
        for severity, message in found:
            html(
                f'<div class="v2v-alert" style="color:{TEXT}"><span style="color:'
                f'{colours[severity]}">●</span>{message}</div>'
            )
    with gear:
        st.button(
            "Settings",
            icon=":material/tune:",
            key="top_settings",
            help="Settings",
            on_click=_go,
            args=("settings",),
        )
    return trace


def condition_filter(traces: dict[str, Trace]) -> None:
    counts = {
        c: sum(c == "all" or t.spec.condition == c for t in traces.values()) for c in CONDITIONS
    }
    st.radio(
        "Scenarios",
        [c for c in CONDITIONS if counts[c]],
        key="cond_filter",
        horizontal=True,
        format_func=lambda c: f"{c.capitalize()} · {counts[c]}",
        label_visibility="collapsed",
    )


def main() -> None:
    st.set_page_config(
        page_title="ACTION V2V · Operations",
        page_icon="🛰️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()
    init_prefs()
    page = rail()
    if page == "driver":
        page_driver()
        return
    directory = traces_dir()
    traces = load(trace_files(directory)) if directory.is_dir() else {}
    if not traces:
        st.warning(
            f"No traces in `{directory}`. Record some first:\n\n"
            "```\npython -m v2v.sim.recorder sim_out/scenarios/hebbal/*-s0\n```"
        )
        return
    trace = top_bar(traces)
    if page in ("live", "network"):
        condition_filter(traces)
    PAGES[page][2](traces, trace)
    html(
        '<div style="height:12px;border-top:1px solid rgba(255,255,255,0.04);margin-top:18px"></div>'
        '<div class="v2v-foot"><span>ACTION V2V · real Hebbal roads in SUMO</span>'
        "<span>© OpenStreetMap contributors · © CARTO</span></div>"
    )


main()
