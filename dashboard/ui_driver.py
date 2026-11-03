# ruff: noqa: E501 - long inline HTML/SVG strings
"""Driver mode: one demo vehicle's dashboard on a live SUMO run. Owner: P4.

Pick a scenario and a demo vehicle (a car on the approach route, a scooter on
the opposite carriageway), then drive: a heading-up perspective
navigation view of the real roads and buildings (``ui_nav``), with the road
ahead and the path already travelled, speed and heading, the traffic in
sight, and only the V2V messages this vehicle actually received, each placed
relative to it. The driver can send V2V alerts of their own; they are
addressed by the same direction-sensitive geocast as hazard warnings.

Everything comes from ``v2v.sim.driver.DriverSession``: live SUMO and the
real pipeline. Scenario ground truth stays in Overseer mode.
"""

import math
import threading
import time
from pathlib import Path

import streamlit as st
from ui_nav import nav_view, relevance, where_text
from ui_theme import BLUE, BORDER, FAINT, LIME, MUTED, ORANGE, RAISED, RED, SURFACE, TEXT, html

from v2v.experiments import CONFIGS
from v2v.sim.driver import (
    DEMO_BY_ID,
    DEMO_FLEET,
    DRIVER_ALERTS,
    Bearing,
    DriverSession,
    Incoming,
)
from v2v.sim.live import scenario_specs
from v2v.sim.scenarios import ScenarioSpec

REPO = Path(__file__).resolve().parents[1]
SCENARIO_DIR = REPO / "sim_out" / "scenarios" / "hebbal"
CYAN = "#22d3ee"
SPEEDS = (1, 2, 4, 8)
REACH_TEXT = {
    "backward": ("Backward geocast", "to traffic approaching from behind"),
    "forward": ("Forward geocast", "to traffic ahead"),
    "bidirectional": ("Bidirectional geocast", "to traffic approaching on both carriageways"),
}
MAP_CONFIG = {"displaylogo": False, "displayModeBar": False, "scrollZoom": False}

_CSS = f"""
<style>
.v2v-cockpit {{ display:flex; align-items:center; justify-content:space-between; gap:14px; }}
.v2v-vehicle {{ display:flex; align-items:center; gap:12px; }}
.v2v-vehicle .icon {{ width:44px; height:44px; border-radius:14px; display:grid; place-items:center;
  background:linear-gradient(145deg, rgba(34,211,238,.22), rgba(34,211,238,.04));
  border:1px solid rgba(34,211,238,.35); font-size:22px; }}
.v2v-vehicle .model {{ font-size:18px; font-weight:650; letter-spacing:-.01em; color:{TEXT}; }}
.v2v-vehicle .meta {{ font-size:11.5px; color:{MUTED}; }}
.v2v-banner {{ border-radius:16px; padding:12px 16px; display:flex; gap:14px; align-items:center;
  backdrop-filter: blur(10px); }}
.v2v-banner .t {{ font-size:15px; font-weight:650; }}
.v2v-banner .s {{ font-size:12px; color:{MUTED}; margin-top:2px; }}
.v2v-msg {{ border:1px solid {BORDER}; background:{SURFACE}; border-radius:14px; padding:10px 12px;
  display:flex; gap:12px; align-items:flex-start; }}
.v2v-msg .t {{ font-size:13.5px; font-weight:600; color:{TEXT}; text-transform:capitalize; }}
.v2v-msg .s {{ font-size:11.5px; color:{MUTED}; margin-top:2px; line-height:1.45; }}
.v2v-spec {{ display:grid; grid-template-columns: 1fr auto; gap:5px 10px; font-size:12px; }}
.v2v-spec span:nth-child(odd) {{ color:{MUTED}; }}
.v2v-spec span:nth-child(even) {{ color:{TEXT}; text-align:right; font-weight:500; }}
.v2v-pick {{ border:1px solid {BORDER}; border-radius:20px; padding:18px; background:
  radial-gradient(120% 80% at 0% 0%, rgba(34,211,238,.10), transparent 60%), {SURFACE}; }}
.v2v-pick .big {{ font-size:46px; line-height:1; }}
.v2v-digital {{ font-variant-numeric: tabular-nums; font-size:76px; font-weight:700; line-height:.95;
  letter-spacing:-.04em; color:{TEXT}; }}
.v2v-seg {{ display:flex; gap:3px; margin-top:8px; }}
.v2v-seg div {{ flex:1; height:10px; border-radius:3px; background:{RAISED}; }}
</style>
"""


# --- small graphics -------------------------------------------------------------------


def speed_ring(kmh: float, top: float, size: int = 210) -> str:
    """A 270-degree speed dial."""
    c, r = size / 2, size / 2 - 14
    frac = max(0.0, min(1.0, kmh / top))

    def arc(f: float) -> str:
        a0, a1 = math.radians(135), math.radians(135 + 270 * f)
        x0, y0 = c + r * math.cos(a0), c + r * math.sin(a0)
        x1, y1 = c + r * math.cos(a1), c + r * math.sin(a1)
        return f"M {x0:.1f} {y0:.1f} A {r} {r} 0 {1 if 270 * f > 180 else 0} 1 {x1:.1f} {y1:.1f}"

    ticks = "".join(
        f'<line x1="{c + (r - 16) * math.cos(math.radians(135 + 27 * i)):.1f}" y1="{c + (r - 16) * math.sin(math.radians(135 + 27 * i)):.1f}" '
        f'x2="{c + (r - 9) * math.cos(math.radians(135 + 27 * i)):.1f}" y2="{c + (r - 9) * math.sin(math.radians(135 + 27 * i)):.1f}" stroke="{FAINT}" stroke-width="1.4"/>'
        for i in range(11)
    )
    colour = CYAN if frac < 0.75 else ORANGE
    return (
        f'<svg width="100%" viewBox="0 0 {size} {size}"><defs><linearGradient id="g" x1="0" x2="1">'
        f'<stop offset="0" stop-color="{BLUE}"/><stop offset="1" stop-color="{colour}"/></linearGradient></defs>'
        f'<path d="{arc(1.0)}" stroke="{RAISED}" stroke-width="12" fill="none" stroke-linecap="round"/>'
        + (
            f'<path d="{arc(frac)}" stroke="url(#g)" stroke-width="12" fill="none" stroke-linecap="round"/>'
            if frac > 0.004
            else ""
        )
        + ticks
        + f'<text x="{c}" y="{c + 8}" text-anchor="middle" font-size="54" font-weight="700" fill="{TEXT}" letter-spacing="-2">{kmh:.0f}</text>'
        f'<text x="{c}" y="{c + 32}" text-anchor="middle" font-size="12" fill="{MUTED}">km/h</text></svg>'
    )


def arrow(rel_deg: float, colour: str, size: int = 38) -> str:
    """An arrow pointing where a message's subject is, relative to the vehicle (up = ahead)."""
    c = size / 2
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}" style="flex:none">'
        f'<circle cx="{c}" cy="{c}" r="{c - 1}" fill="{colour}1f" stroke="{colour}66"/>'
        f'<g transform="rotate({rel_deg:.0f} {c} {c})"><polygon points="{c},{c - 11} {c + 7},{c + 7} {c},{c + 3} {c - 7},{c + 7}" fill="{colour}"/></g></svg>'
    )


def turn_cue(change: float) -> tuple[str, str]:
    a = abs(change)
    side = "right" if change > 0 else "left"
    if a < 20:
        return "↑", "Continue straight"
    if a < 60:
        return ("↗" if change > 0 else "↖"), f"Bear {side}"
    return ("↱" if change > 0 else "↰"), f"Turn {side}"


# --- panels ---------------------------------------------------------------------------------


def message_card(session: DriverSession, vid: str, m: Incoming) -> str:
    b = session.where_is(vid, *m.at, *_hazard_point(session, m))
    if b is None:
        return ""
    colour, word = relevance(b)
    reach = REACH_TEXT.get(m.reach or "", ("Geocast", ""))[0]
    return (
        f'<div class="v2v-msg">{arrow(b.relative_deg, colour)}<div style="flex:1">'
        f'<div class="v2v-row"><span class="t">{m.title}</span><span style="color:{colour};font-size:11px;font-weight:600">{word}</span></div>'
        f'<div class="s">{where_text(b)}</div>'
        f'<div class="s" style="color:{TEXT}">{m.text}</div>'
        f'<div class="s">{m.status} · {reach} · {m.latency_ms:.0f} ms, hop {m.hop} via {m.relay} · t {m.t_s:.0f} s</div>'
        "</div></div>"
    )


def _hazard_point(session: DriverSession, m: Incoming) -> tuple[str | None, float]:
    """Where on the road a hazard warning is about, for distance along the route."""
    for e in session.live.events:
        w = e.result.warning
        loc = e.result.assessed.location
        if w is not None and w.msg_id == m.msg_id and loc is not None:
            return loc.edge_id, loc.pos_m
    for a in session.sent:
        if a.message.msg_id == m.msg_id:
            return a.location.edge_id, a.location.pos_m
    return None, 0.0


def headline(session: DriverSession, vid: str) -> str:
    """The one thing the driver most needs to know now."""
    best: tuple[float, Incoming, Bearing] | None = None
    for m in session.incoming(vid):
        b = session.where_is(vid, *m.at, *_hazard_point(session, m))
        if b is not None and b.by_road_m is not None and (best is None or b.by_road_m < best[0]):
            best = (b.by_road_m, m, b)
    if best is None:
        return (
            f'<div class="v2v-banner" style="background:rgba(197,242,58,.07);border:1px solid rgba(197,242,58,.25)">'
            f'<div style="font-size:24px">✓</div><div><div class="t" style="color:{LIME}">Road ahead clear</div>'
            '<div class="s">No V2V warning about your route</div></div></div>'
        )
    _, m, b = best
    return (
        f'<div class="v2v-banner" style="background:rgba(255,90,82,.10);border:1px solid rgba(255,90,82,.4)">'
        f"{arrow(b.relative_deg, RED, 44)}<div><div class='t' style='color:{RED}'>{m.title.capitalize()} · {b.by_road_m:,.0f} m ahead</div>"
        f'<div class="s">{m.text}</div></div></div>'
    )


def vehicle_info(session: DriverSession, vid: str) -> str:
    d = DEMO_BY_ID[vid]
    me = session.state(vid)
    status = (
        "on the road"
        if session.on_road(vid)
        else ("arrived" if me is not None else "waiting to depart")
    )
    if me is not None and session.on_road(vid) and me.speed_mps < 1.0:
        status = "stopped"
    rows = [
        ("Vehicle ID", vid),
        ("Type", f"{d.kind} · {d.powertrain}"),
        ("Model", d.model),
        ("Size", f"{d.length_m:.2f} × {d.width_m:.2f} m"),
        ("Top speed", f"{d.max_speed_mps * 3.6:.0f} km/h"),
        ("Status", status),
        ("Road", me.edge_id if me else "—"),
    ]
    return (
        '<div class="v2v-spec">'
        + "".join(f"<span>{k}</span><span>{v}</span>" for k, v in rows)
        + "</div>"
    )


def comm_status(session: DriverSession, vid: str) -> str:
    n = session.neighbours_in_radio_range(vid)
    got = session.incoming(vid)
    last = got[-1] if got else None
    link = (LIME, "V2V link active") if n else (ORANGE, "no vehicle in radio range")
    return (
        f'<div class="v2v-row"><span style="color:{link[0]};font-weight:600;font-size:13px">● {link[1]}</span></div>'
        f'<div class="v2v-muted" style="margin-top:4px">{n} vehicles within 300 m radio range · '
        f"{len(got)} messages received"
        + (
            f" · last {last.latency_ms:.0f} ms over {last.hop} hop{'s' if last.hop > 1 else ''}"
            if last
            else ""
        )
        + "</div>"
    )


# --- the page -----------------------------------------------------------------------------


IDLE_S = 180.0
_open: dict[int, tuple[DriverSession, float]] = {}
"""Every drive with a SUMO process, and when it last ticked. A browser tab
closed mid-drive never says so; the next drive to start closes such ones."""
_open_lock = threading.Lock()


def _touch(session: DriverSession) -> None:
    with _open_lock:
        _open[id(session)] = (session, time.monotonic())


def _reap() -> None:
    now = time.monotonic()
    with _open_lock:
        stale = [k for k, (_, seen) in _open.items() if now - seen > IDLE_S]
        for k in stale:
            session, _ = _open.pop(k)
            session.close()


def _close_drive() -> None:
    session = st.session_state.pop("drive", None)
    if session is not None:
        with _open_lock:
            _open.pop(id(session), None)
        session.close()
    for k in ("drive_vid", "drive_note", "drive_due"):
        st.session_state.pop(k, None)


def _start(spec: ScenarioSpec, config: str, vid: str) -> None:
    _close_drive()
    _reap()
    with st.spinner("Starting SUMO and driving the demo vehicles onto the road…"):
        session = DriverSession(spec, config_name=config)
        session.open()
    _touch(session)
    st.session_state["drive"] = session
    st.session_state["drive_vid"] = vid
    st.session_state.setdefault("drive_play", True)


def picker() -> None:
    html(
        '<div class="v2v-title">Choose your vehicle</div><div class="v2v-subtitle">Two demo vehicles join a live SUMO run of the real Hebbal corridor. You see only what reaches your vehicle.</div>'
    )
    specs = {Path(s.sumocfg).parent.name: s for s in scenario_specs(SCENARIO_DIR)}
    if not specs:
        st.warning(
            f"No scenarios in `{SCENARIO_DIR}`. Generate them with `python -m v2v.sim.scenarios --corridor hebbal`."
        )
        return
    a, b = st.columns([3, 2])
    with a:
        name = st.selectbox(
            "Scenario", list(specs), key="drive_scenario",
            format_func=lambda n: f"{n} · {specs[n].hazard_class.value.replace('_', ' ')}",
        )  # fmt: skip
    with b:
        config = st.selectbox(
            "Experiment config",
            list(CONFIGS),
            index=list(CONFIGS).index("proposed"),
            key="drive_config",
        )
    spec = specs[str(name)]
    cols = st.columns(len(DEMO_FLEET))
    for col, d in zip(cols, DEMO_FLEET, strict=True):
        with col, st.container(key=f"card_pick_{d.vehicle_id}"):
            where = (
                "the approach to the hazard"
                if d.route == "approach"
                else "the opposite carriageway"
            )
            html(
                f'<div class="v2v-pick"><div class="big">{"🚗" if d.kind == "car" else "🛵"}</div>'
                f'<div class="v2v-mid" style="margin-top:10px">{d.model}</div>'
                f'<div class="v2v-muted">{d.label} · {d.vehicle_id} · rides {where}</div>'
                f'<div style="height:10px"></div><div class="v2v-spec">'
                f"<span>Size</span><span>{d.length_m:.2f} × {d.width_m:.2f} m</span>"
                f"<span>Top speed</span><span>{d.max_speed_mps * 3.6:.0f} km/h</span>"
                f"<span>Acceleration</span><span>{d.accel:.1f} m/s²</span>"
                f"<span>Powertrain</span><span>{d.powertrain}</span></div></div>"
            )
            st.button(
                f"Drive the {d.kind}",
                key=f"w_drive_{d.vehicle_id}",
                type="primary",
                use_container_width=True,
                on_click=_start,
                args=(spec, str(config), d.vehicle_id),
            )
    html(
        f'<div class="v2v-muted" style="margin-top:6px">Sizes and performance are the makers\' published figures. Hazard: {spec.hazard_class.value.replace("_", " ")} — {"backward geocast (traffic approaching on its carriageway)" if spec.hazard_class.value != "slippery_road" else "bidirectional geocast (both carriageways)"}. Report text is fixture text until P1\'s real reports exist.</div>'
    )


def _send(kind: str) -> None:
    session: DriverSession = st.session_state["drive"]
    vid = st.session_state["drive_vid"]
    try:
        a = session.send(vid, kind)
    except KeyError:
        st.session_state["drive_note"] = "Your vehicle is not on the road."
        return
    label, to = REACH_TEXT[a.kind.reach]
    st.session_state["drive_note"] = (
        f"{a.kind.label}: {label.lower()} {to} — addressed {len(a.message.recipients)}, delivered {len(a.delivered)} in {a.outcome.transmissions} transmissions."
    )


def cockpit() -> None:
    session: DriverSession = st.session_state["drive"]
    vid: str = st.session_state["drive_vid"]
    d = DEMO_BY_ID[vid]
    playing = bool(st.session_state.get("drive_play", True))
    speed = int(st.session_state.get("drive_speed") or 2)  # clicking the chosen speed clears it
    scooter = d.kind == "scooter"

    head, ctrl = st.columns([3, 2], vertical_alignment="center")
    with head:
        html(
            f'<div class="v2v-vehicle"><div class="icon">{"🛵" if scooter else "🚗"}</div><div>'
            f'<div class="model">{d.model}</div><div class="meta">{vid} · {d.kind} · {session.spec.scenario_id} · config {session.config_name}</div></div></div>'
        )
    with ctrl:
        c1, c2, c3, c4 = st.columns([1, 1.3, 1.2, 1])
        c1.button(
            "Pause" if playing else "Play",
            key="w_drive_play",
            icon=":material/pause:" if playing else ":material/play_arrow:",
            on_click=lambda: st.session_state.update(drive_play=not playing),
            use_container_width=True,
        )
        c2.segmented_control(
            "Speed",
            SPEEDS,
            key="drive_speed",
            default=2,
            format_func=lambda s: f"{s}×",
            label_visibility="collapsed",
        )
        others = [x.vehicle_id for x in DEMO_FLEET if x.vehicle_id != vid]
        c3.button(
            "Switch vehicle",
            key="w_drive_switch",
            icon=":material/swap_horiz:",
            on_click=lambda: st.session_state.update(drive_vid=others[0]),
            use_container_width=True,
        )
        c4.button(
            "End",
            key="w_drive_end",
            icon=":material/stop:",
            on_click=_close_drive,
            use_container_width=True,
        )

    @st.fragment(run_every=0.5 if playing else None)
    def live() -> None:
        _touch(session)
        if st.session_state.get("drive_play", True):
            # half a second per tick; the remainder carries over so 1x stays real time
            due = st.session_state.get("drive_due", 0.0) + speed * 0.5
            st.session_state["drive_due"] = due - int(due)
            session.advance(int(due))
        me = session.state(vid)
        if me is None:
            html(
                f'<div class="v2v-muted">{vid} has not departed yet (t {session.time_s:.0f} s).</div>'
            )
            return
        on_road = session.on_road(vid)
        icon, cue = turn_cue(session.heading_change(vid)) if on_road else ("■", "Arrived")
        msgs = session.incoming(vid)
        if scooter:
            left, centre = st.columns([1.05, 2.2])
            with left, st.container(key="card_drive_speed"):
                kmh = me.speed_mps * 3.6
                lit = round(12 * min(1.0, kmh / (d.max_speed_mps * 3.6)))
                html(
                    f'<div class="v2v-label">Speed</div><div class="v2v-digital">{kmh:.0f}<span style="font-size:16px;color:{MUTED};font-weight:500"> km/h</span></div>'
                    '<div class="v2v-seg">'
                    + "".join(
                        f'<div style="background:{(CYAN if i < 9 else ORANGE) if i < lit else RAISED}"></div>'
                        for i in range(12)
                    )
                    + "</div>"
                    f'<div class="v2v-row" style="margin-top:14px"><span class="v2v-mid">{icon} {cue}</span><span class="v2v-muted">{me.heading_deg:.0f}°</span></div>'
                )
            with left, st.container(key="card_drive_msgs"):
                html('<div class="v2v-label">V2V messages</div>')
                html(
                    "".join(message_card(session, vid, m) for m in reversed(msgs[-3:]))
                    or '<div class="v2v-muted">None received yet.</div>'
                )
            with centre:
                html(headline(session, vid))
                st.plotly_chart(
                    nav_view(session, vid, True, 560), config=MAP_CONFIG, key="drive_map"
                )
        else:
            left, centre, right = st.columns([1.05, 2.5, 1.35])
            with left, st.container(key="card_drive_speed"):
                html(speed_ring(me.speed_mps * 3.6, 140.0))
                html(
                    f'<div class="v2v-row"><span class="v2v-mid">{icon} {cue}</span><span class="v2v-muted">{me.heading_deg:.0f}°</span></div>'
                )
            with left, st.container(key="card_drive_info"):
                html('<div class="v2v-label">Vehicle</div>' + vehicle_info(session, vid))
            with centre:
                html(headline(session, vid))
                st.plotly_chart(
                    nav_view(session, vid, False, 600), config=MAP_CONFIG, key="drive_map"
                )
            with right, st.container(key="card_drive_msgs"):
                html('<div class="v2v-label">V2V messages</div>')
                html(
                    "".join(message_card(session, vid, m) for m in reversed(msgs[-4:]))
                    or '<div class="v2v-muted">None received yet. Warnings reach you only when you are in their geocast area.</div>'
                )
        with st.container(key="card_drive_comm"):
            a, b = st.columns([2, 3], vertical_alignment="center")
            with a:
                html(
                    comm_status(session, vid)
                    + f'<div class="v2v-muted">sim time {session.time_s:.0f} s · {len(session.nearby(vid))} vehicles in sight</div>'
                )
            with b:
                cols = st.columns(len(DRIVER_ALERTS))
                for col, (key, k) in zip(cols, DRIVER_ALERTS.items(), strict=True):
                    col.button(
                        k.label,
                        key=f"w_alert_{key}",
                        help=f"{REACH_TEXT[k.reach][0]} {REACH_TEXT[k.reach][1]}, {k.radius_m:.0f} m",
                        on_click=_send,
                        args=(key,),
                        use_container_width=True,
                        disabled=not on_road,
                    )
                note = st.session_state.get("drive_note")
                if note:
                    html(f'<div class="v2v-muted" style="margin-top:4px">{note}</div>')
        if scooter:
            with st.container(key="card_drive_info"):
                html('<div class="v2v-label">Vehicle</div>' + vehicle_info(session, vid))

    live()


def page_driver() -> None:
    html(_CSS)
    if "drive" in st.session_state:
        cockpit()
    else:
        picker()
