# ruff: noqa: E501 - long inline HTML strings
"""Voice report: say "hey ACTION", and warn the drivers behind you. Owner: P4.

The microphone listens on this machine (``v2v.voice.listener``). After the wake
phrase the words appear live; when the speaker pauses, the report runs through
the real pipeline from the chosen demo vehicle in a live SUMO run. The page
shows each step: what was heard, the cleaned text, the NLP reading, whether it
was safe to send, and the template warning that went out by geocast, with a
radar of the road around the vehicle and who received it.
"""

import html as html_lib
import math
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st
from ui_driver import forget, reap, touch
from ui_theme import BORDER, FAINT, LIME, MUTED, ORANGE, RAISED, RED, SURFACE, TEXT, html

from v2v.geocast import directional
from v2v.sim.driver import DEMO_BY_ID, DEMO_FLEET, DriverSession, VoiceReport, verdict
from v2v.sim.live import scenario_specs
from v2v.voice.listener import Listener, models_ready

SCENARIO_DIR = Path(__file__).resolve().parents[1] / "sim_out" / "scenarios" / "hebbal"
CYAN = "#22d3ee"
RADAR_M = 900.0
MAP_CONFIG = {"displaylogo": False, "displayModeBar": False, "scrollZoom": False}
DEFAULT_SCENARIO = "noisy-medium-s0"
"""Enough traffic that there is usually someone behind you."""

_CSS = f"""
<style>
.vr-orb-wrap {{ display:flex; flex-direction:column; align-items:center; padding:14px 0 6px; }}
.vr-orb {{ width:112px; height:112px; border-radius:50%; display:grid; place-items:center; font-size:38px;
  background:radial-gradient(circle at 35% 30%, rgba(34,211,238,.35), rgba(34,211,238,.05) 70%);
  border:1px solid rgba(34,211,238,.4); }}
.vr-orb.waiting {{ animation:vr-breathe 3.2s ease-in-out infinite; }}
.vr-orb.listening {{ background:radial-gradient(circle at 35% 30%, rgba(255,90,82,.55), rgba(255,90,82,.08) 70%);
  border-color:rgba(255,90,82,.8); animation:vr-pulse .9s ease-in-out infinite; }}
.vr-orb.off {{ background:{RAISED}; border-color:{BORDER}; filter:grayscale(1); }}
@keyframes vr-breathe {{ 50% {{ box-shadow:0 0 34px rgba(34,211,238,.28); transform:scale(1.03); }} }}
@keyframes vr-pulse {{ 50% {{ box-shadow:0 0 48px rgba(255,90,82,.5); transform:scale(1.07); }} }}
.vr-state {{ font-size:17px; font-weight:650; color:{TEXT}; margin-top:12px; text-align:center; }}
.vr-hint {{ font-size:12px; color:{MUTED}; text-align:center; margin-top:2px; }}
.vr-live {{ min-height:64px; font-size:21px; line-height:1.35; font-weight:500; color:{TEXT}; text-align:center; padding:10px 6px; }}
.vr-live.dim {{ color:{FAINT}; font-size:14px; font-weight:400; }}
.vr-step {{ display:flex; gap:12px; padding:9px 0; border-bottom:1px solid {BORDER}; }}
.vr-step:last-child {{ border-bottom:none; }}
.vr-step .n {{ flex:none; width:24px; height:24px; border-radius:50%; display:grid; place-items:center;
  font-size:12px; font-weight:700; color:#0b0f17; }}
.vr-step .k {{ font-size:11px; letter-spacing:.06em; text-transform:uppercase; color:{MUTED}; }}
.vr-step .v {{ font-size:14px; color:{TEXT}; margin-top:1px; line-height:1.4; }}
.vr-step .s {{ font-size:12px; color:{MUTED}; margin-top:1px; }}
.vr-msg {{ border:1px solid rgba(34,211,238,.4); background:rgba(34,211,238,.07); border-radius:12px;
  padding:9px 12px; margin-top:5px; font-size:14.5px; font-weight:550; color:{TEXT}; line-height:1.4; }}
.vr-blocked {{ border:1px solid rgba(255,90,82,.4); background:rgba(255,90,82,.08); border-radius:12px;
  padding:9px 12px; margin-top:5px; font-size:14px; color:{TEXT}; }}
.vr-row {{ display:flex; justify-content:space-between; gap:10px; font-size:12px; padding:3px 0; color:{TEXT}; }}
.vr-row span:last-child {{ color:{MUTED}; }}
.vr-card {{ border:1px solid {BORDER}; background:{SURFACE}; border-radius:14px; padding:12px; }}
</style>
"""


@st.cache_resource(show_spinner=False)
def listener() -> Listener:
    """One microphone per machine, whichever tab is open."""
    return Listener()


def esc(text: str) -> str:
    return html_lib.escape(text)


# --- the radar --------------------------------------------------------------------------


def radar(
    session: DriverSession, vid: str, last: VoiceReport | None, height: int = 430
) -> go.Figure:
    """Top-down, heading up: road ahead above you, drivers behind you below.

    Centred on the vehicle as it moves. After a report the view stays turned the
    way the vehicle faced when it reported, so "behind you" stays true for that
    warning even if the vehicle has since turned a corner."""
    me = session.state(vid)
    if me is None:
        raise KeyError(vid)
    h = math.radians(last.heading_deg if last is not None else me.heading_deg)
    fx, fy = math.sin(h), math.cos(h)

    def to_view(x: float, y: float) -> tuple[float, float]:
        dx, dy = x - me.x, y - me.y
        return dx * fy - dy * fx, dx * fx + dy * fy

    roads: list[str] = []
    lit: list[str] = []
    zone = last.event.zone_edges if last is not None else {}
    for edge, shape in session.graph.shapes.items():
        if (
            not shape
            or min(
                math.dist((me.x, me.y), p) for p in (shape[0], shape[len(shape) // 2], shape[-1])
            )
            > RADAR_M * 1.5
        ):
            continue
        pts = [to_view(*p) for p in shape]
        path = "M" + " L".join(f"{a:.1f},{b:.1f}" for a, b in pts)
        (lit if edge in zone else roads).append(path)
    shapes = [
        {
            "type": "circle",
            "x0": -r,
            "x1": r,
            "y0": -r,
            "y1": r,
            "line": {"color": "rgba(255,255,255,0.07)", "width": 1},
            "layer": "below",
        }
        for r in (300, 600, 900)
    ]
    if roads:
        shapes.append(
            {
                "type": "path",
                "path": " ".join(roads),
                "line": {"color": "#2c3748", "width": 2.5},
                "layer": "below",
            }
        )
    if lit:
        shapes.append(
            {
                "type": "path",
                "path": " ".join(lit),
                "line": {"color": "rgba(34,211,238,0.3)", "width": 11},
                "layer": "below",
            }
        )
        shapes.append(
            {
                "type": "path",
                "path": " ".join(lit),
                "line": {"color": CYAN, "width": 3},
                "layer": "below",
            }
        )
    fig = go.Figure()

    now = {v.vehicle_id: v for v in session.live.vehicles()}
    delivered = last.delivered if last is not None else {}
    addressed = set(last.recipients) if last is not None else set()
    groups: dict[str, list[tuple[float, float, str]]] = {"other": [], "warned": [], "lost": []}
    for other, v in now.items():
        if other == vid:
            continue
        x, y = to_view(v.x, v.y)
        if max(abs(x), abs(y)) > RADAR_M * 1.2:
            continue
        kind = "warned" if other in delivered else "lost" if other in addressed else "other"
        groups[kind].append((x, y, other))
    if last is not None and last.event.pulses:
        lx: list[float | None] = []
        ly: list[float | None] = []
        for p in last.event.pulses:
            a, b = now.get(p.src), now.get(p.dst)
            src = (a.x, a.y) if a is not None else last.positions.get(p.src)
            dst = (b.x, b.y) if b is not None else last.positions.get(p.dst)
            if src is None or dst is None:
                continue
            ax, ay = to_view(*src)
            bx, by = to_view(*dst)
            lx += [ax, bx, None]
            ly += [ay, by, None]
        fig.add_trace(
            go.Scatter(
                x=lx,
                y=ly,
                mode="lines",
                line={"color": "rgba(197,242,58,0.4)", "width": 1.2},
                hoverinfo="skip",
            )
        )
    for kind, colour, size, label in (
        ("other", "#6b7686", 6, "not addressed"),
        ("lost", ORANGE, 10, "addressed, not reached"),
        ("warned", LIME, 12, "warned"),
    ):
        pts3 = groups[kind]
        if pts3:
            fig.add_trace(
                go.Scatter(
                    x=[p[0] for p in pts3], y=[p[1] for p in pts3], mode="markers",
                    marker={"size": size, "color": colour}, text=[f"{p[2]}: {label}" for p in pts3],
                    hovertemplate="%{text}<extra></extra>",
                )
            )  # fmt: skip
    if last is not None and last.event.sent:
        x, y = to_view(*last.event.centre_xy)
        fig.add_trace(
            go.Scatter(
                x=[x],
                y=[y],
                mode="markers",
                marker={"size": 15, "color": RED, "symbol": "x", "line": {"width": 0}},
                hovertemplate="Reported hazard<extra></extra>",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[0],
            y=[0],
            mode="markers",
            marker={"size": 30, "color": "rgba(34,211,238,0.2)"},
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0],
            y=[0],
            mode="markers",
            marker={"size": 15, "color": "#ffffff", "symbol": "triangle-up"},
            hovertemplate="You<extra></extra>",
        )
    )
    fig.update_layout(
        height=height, margin={"l": 0, "r": 0, "t": 0, "b": 0}, showlegend=False,
        paper_bgcolor="#0b0f17", plot_bgcolor="#0b0f17", shapes=shapes,
        xaxis={"visible": False, "range": [-RADAR_M, RADAR_M], "fixedrange": True},
        yaxis={"visible": False, "range": [-RADAR_M, RADAR_M], "fixedrange": True, "scaleanchor": "x"},
        annotations=[
            {"x": 0, "y": RADAR_M * 0.93, "text": "AHEAD", "showarrow": False, "font": {"color": MUTED, "size": 10}},
            {"x": 0, "y": -RADAR_M * 0.93, "text": "BEHIND YOU" if last is None else "BEHIND WHEN YOU REPORTED", "showarrow": False, "font": {"color": CYAN, "size": 10}},
            {"x": 300, "y": 0, "text": "300 m", "showarrow": False, "font": {"color": FAINT, "size": 9}},
            {"x": 600, "y": 0, "text": "600 m", "showarrow": False, "font": {"color": FAINT, "size": 9}},
        ],
    )  # fmt: skip
    return fig


# --- the steps -------------------------------------------------------------------------


def step(n: int, colour: str, key: str, value: str, sub: str = "") -> str:
    return (
        f'<div class="vr-step"><div class="n" style="background:{colour}">{n}</div><div style="flex:1;min-width:0">'
        f'<div class="k">{key}</div><div class="v">{value}</div>'
        + (f'<div class="s">{sub}</div>' if sub else "")
        + "</div></div>"
    )


def steps(report: VoiceReport) -> str:
    e = report.event
    r = e.result
    i, a = r.interpretation, r.assessed
    word, why = verdict(e)
    verdict_colour = {"Confirmed": LIME, "Sent as unverified": ORANGE, "Blocked": RED}[word]
    flags = [x for x, on in (("negated", i.negated), ("hedged", i.hedged)) if on]
    loc = a.location
    if loc is None:
        where = "no location"
    elif loc.approx:
        where = "no place named, so your position (approximate)"
    else:
        where = (
            f"{esc(i.location_text)}, found on the road map"
            if i.location_text
            else "joined an earlier report of this hazard nearby, at its spot"
        )
    out = [
        step(1, CYAN, "Heard", esc(report.heard) or "&nbsp;"),
        step(2, CYAN, "Cleaned", esc(report.cleaned), "Respelled for the NLP, personal details removed"),
        step(
            3, CYAN, "Understood",
            f"{i.hazard.value.replace('_', ' ').capitalize()}" + (f" · {', '.join(flags)}" if flags else ""),
            f"Location: {where} · read by {esc(i.details.get('model', '?'))}",
        ),
        step(
            4, verdict_colour, "Safe to send?",
            f'<span style="color:{verdict_colour};font-weight:650">{word}</span>: {why}',
            f"Interpretation confidence {a.interp_conf:.2f} · event evidence {a.event_evidence:.2f} · {a.n_reports} report{'s' if a.n_reports != 1 else ''}",
        ),
    ]  # fmt: skip
    if r.warning is None:
        out.append(
            step(
                5,
                RED,
                "Sent",
                '<div class="vr-blocked">Nothing was sent. No driver was disturbed.</div>',
            )
        )
    else:
        reach = directional.reach_for(a.hazard) if r.warning.policy == "directional" else None
        to = {
            "backward": "drivers behind you, approaching this spot on your carriageway",
            "bidirectional": "drivers approaching this spot on both carriageways",
            "forward": "drivers ahead of you",
        }.get(reach or "", f"{r.warning.policy} geocast")
        out.append(
            step(
                5, LIME, "Sent by geocast",
                f'<div class="vr-msg">{esc(r.warning.text)}</div>',
                f"To {to}: addressed {len(report.recipients)}, reached {len(report.delivered)}"
                + (f" in {e.outcome.transmissions} radio transmissions" if e.outcome else ""),
            )
        )  # fmt: skip
    return "".join(out)


def reached(session: DriverSession, vid: str, report: VoiceReport) -> str:
    """The drivers this message reached, nearest first."""
    if not report.delivered:
        return ""
    rows = []
    me = report.positions.get(vid)
    for other, ms in sorted(
        report.delivered.items(),
        key=lambda kv: math.dist(report.positions.get(kv[0], (0, 0)), me or (0, 0)),
    )[:6]:
        where = report.positions.get(other)
        d = f"{math.dist(where, me):.0f} m" if where and me else ""
        hop = report.event.outcome.hops.get(other, 1) if report.event.outcome else 1
        rows.append(
            f'<div class="vr-row"><span>{esc(other)}</span><span>{d} · {ms:.0f} ms · hop {hop}</span></div>'
        )
    more = len(report.delivered) - len(rows)
    return (
        '<div class="v2v-label" style="margin-top:8px">Drivers reached</div>'
        + "".join(rows)
        + (
            f'<div class="vr-row"><span></span><span>and {more} more</span></div>'
            if more > 0
            else ""
        )
    )


# --- the page --------------------------------------------------------------------------


def _start(name: str, vid: str) -> None:
    _end_drive()
    reap()
    specs = {Path(s.sumocfg).parent.name: s for s in scenario_specs(SCENARIO_DIR)}
    session = DriverSession(specs[name])
    session.open()
    touch(session)
    st.session_state["voice_drive"] = session
    st.session_state["voice_vid"] = vid
    st.session_state["voice_history"] = []
    st.session_state["voice_due"] = 0.0


def _end_drive() -> None:
    session = st.session_state.pop("voice_drive", None)
    if session is not None:
        forget(session)
        session.close()


def _stop() -> None:
    _end_drive()
    listener().stop()


def _submit(session: DriverSession, vid: str, heard: str, roman: str) -> None:
    if not roman.strip():
        return
    session.respawn(vid)
    if not session.on_road(vid):
        session.advance(1)  # a respawned vehicle joins at the next step
    try:
        report = session.voice_report(vid, heard, roman)
    except KeyError:
        st.session_state["voice_note"] = (
            "Your vehicle is not on the road yet. Try again in a moment."
        )
        return
    st.session_state.setdefault("voice_history", []).append(report)


def _typed(session: DriverSession, vid: str) -> None:
    text = str(st.session_state.get("voice_typed", "")).strip()
    st.session_state["voice_typed"] = ""
    _submit(session, vid, text, text)


def setup() -> None:
    html(
        '<div class="v2v-title">Voice report</div><div class="v2v-subtitle">Speak a hazard from your vehicle. Your words become a checked, cleaned warning for the drivers behind you.</div>'
    )
    specs = {Path(s.sumocfg).parent.name: s for s in scenario_specs(SCENARIO_DIR)}
    if not specs:
        st.warning(
            f"No scenarios in `{SCENARIO_DIR}`. Generate them with `python -m v2v.sim.scenarios --corridor hebbal`."
        )
        return
    a, b = st.columns([3, 2])
    names = sorted(specs)
    name = a.selectbox(
        "Traffic",
        names,
        index=names.index(DEFAULT_SCENARIO) if DEFAULT_SCENARIO in names else 0,
        key="voice_scenario",
    )
    vid = b.selectbox(
        "Your vehicle",
        [d.vehicle_id for d in DEMO_FLEET],
        format_func=lambda v: f"{DEMO_BY_ID[v].model} ({DEMO_BY_ID[v].kind})",
        key="voice_vehicle",
    )
    st.button(
        "Start driving",
        type="primary",
        key="w_voice_start",
        on_click=_start,
        args=(str(name), str(vid)),
        use_container_width=True,
    )
    if not models_ready():
        st.info(
            "The speech models are not downloaded yet, so only typed reports will work. Get them (about 80 MB) with `python -m v2v.voice.listener --download`."
        )


def console() -> None:
    session: DriverSession = st.session_state["voice_drive"]
    vid: str = st.session_state["voice_vid"]
    d = DEMO_BY_ID[vid]
    mic = listener()

    head, ctl = st.columns([3, 2], vertical_alignment="center")
    head.markdown(
        f'<div class="v2v-vehicle"><div class="icon">{"🛵" if d.kind == "scooter" else "🚗"}</div><div><div class="model">{d.model}</div>'
        f'<div class="meta">{vid} · live SUMO traffic · {session.spec.scenario_id}</div></div></div>',
        unsafe_allow_html=True,
    )
    c1, c2 = ctl.columns(2)
    c1.button(
        "Stop microphone" if mic.running else "Start microphone",
        key="w_voice_mic",
        icon=":material/mic:",
        disabled=not models_ready(),
        on_click=mic.stop if mic.running else mic.start,
        use_container_width=True,
    )
    c2.button(
        "End", key="w_voice_end", icon=":material/stop:", on_click=_stop, use_container_width=True
    )

    @st.fragment(run_every=0.4)
    def live() -> None:
        touch(session)
        due = (
            st.session_state.get("voice_due", 0.0) + 0.4
        )  # real time: 1 simulated second per second
        st.session_state["voice_due"] = due - int(due)
        session.advance(int(due))
        session.respawn(vid)
        for spoken in mic.take():
            _submit(session, vid, spoken.devanagari, spoken.roman)
        history: list[VoiceReport] = st.session_state.get("voice_history", [])
        last = history[-1] if history else None

        left, middle, right = st.columns([1.05, 1.5, 1.45])
        with left, st.container(key="card_voice_mic"):
            phase = mic.phase
            orb, title, hint = {
                "off": ("off", "Microphone off", "Press Start microphone"),
                "loading": ("off", "Loading speech models…", "A few seconds"),
                "waiting": (
                    "waiting",
                    'Say "Hey ACTION"',
                    "Listening for the wake phrase only. Nothing leaves this laptop.",
                ),
                "listening": ("listening", "Go ahead, I'm listening", "Pause when you're done"),
            }[phase]
            live_text = mic.partial
            html(
                f'<div class="vr-orb-wrap"><div class="vr-orb {orb}">🎙️</div><div class="vr-state">{title}</div><div class="vr-hint">{hint}</div></div>'
                + (
                    f'<div class="vr-live">{esc(live_text)}</div>'
                    if live_text
                    else '<div class="vr-live dim">Your words appear here as you speak</div>'
                )
            )
            if mic.error:
                st.error(f"Microphone problem: {mic.error}")
            if mic.running and phase == "waiting":
                st.button(
                    "Tap to talk",
                    key="w_voice_tap",
                    icon=":material/touch_app:",
                    on_click=mic.tap,
                    use_container_width=True,
                )
            st.text_input(
                "Or type a report",
                key="voice_typed",
                placeholder="mujhe ek bandar road pe dikh raha hai",
                on_change=_typed,
                args=(session, vid),
            )
            note = st.session_state.pop("voice_note", None)
            if note:
                st.warning(note)
        with middle, st.container(key="card_voice_steps"):
            html('<div class="v2v-label">What happened to your report</div>')
            if last is None:
                html(
                    '<div class="v2v-muted" style="padding:14px 0">Nothing reported yet. Try: <i>"Hey ACTION, mujhe ek bandar road pe naacha hua dikh raha hai."</i></div>'
                )
            else:
                html(steps(last))
                if last.event.text_source == "fixture":
                    html(
                        f'<div class="v2v-muted" style="margin-top:6px;color:{FAINT}">Live demo input, never used as training data or results.</div>'
                    )
        with right, st.container(key="card_voice_radar"):
            html('<div class="v2v-label">The road around you</div>')
            if session.state(vid) is None:
                html('<div class="v2v-muted">Your vehicle is joining the traffic…</div>')
            else:
                st.plotly_chart(radar(session, vid, last), config=MAP_CONFIG, key="voice_radar")
            if last is not None:
                html(reached(session, vid, last))

    live()


def page_voice() -> None:
    html(_CSS)
    if "voice_drive" in st.session_state:
        console()
    else:
        setup()
