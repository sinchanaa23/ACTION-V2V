# ruff: noqa: E501 - long figure expressions
"""Driver mode's navigation view. Owner: P4.

A vector perspective view, as in-car navigation draws it: the real road map
(the SUMO network) and OSM buildings, seen heading-up from a camera behind and
above the vehicle, with the route ahead, the path already travelled, traffic
in sight and what each received V2V message is about. Drawn as plain Plotly
shapes so it renders instantly on every live tick; a tile basemap would
reload each time Streamlit replaces the chart.
"""

import math
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st
from ui_theme import BLUE, MUTED, ORANGE, RAISED, RED, TEXT

from v2v.sim.driver import DEMO_BY_ID, Bearing, DriverSession, route_ahead_xy

REPO = Path(__file__).resolve().parents[1]
CYAN = "#22d3ee"
NAV_BG = "#0b0f17"
CELL_M = 200.0
ROAD_REACH_M = 900.0
BUILDING_REACH_M = 650.0

Footprint = tuple[tuple[float, float], ...]


def relevance(b: Bearing) -> tuple[str, str]:
    """Colour and a word for how much a message matters to this vehicle now."""
    if b.by_road_m is not None:
        return (RED, "on your route") if b.by_road_m < 400 else (ORANGE, "on your route")
    if b.side == "behind":
        return MUTED, "passed"
    return BLUE, "nearby"


def where_text(b: Bearing) -> str:
    if b.by_road_m is not None:
        return f"{b.by_road_m:,.0f} m ahead on your route"
    if b.side == "behind":
        return f"{b.distance_m:,.0f} m behind you"
    if b.side == "ahead":
        return f"{b.distance_m:,.0f} m ahead, off your route"
    return f"{b.distance_m:,.0f} m to your {b.side}"


@st.cache_resource(show_spinner=False)
def city() -> dict[tuple[int, int], list[tuple[Footprint, float]]]:
    """OSM building footprints with heights, in a coarse grid for fast lookup.
    Empty (roads only) if the building context has not been downloaded."""
    from v2v.sim.osm_network import CORRIDORS
    from v2v.sim.world3d import build_scene, context_paths

    corridor = CORRIDORS["hebbal"]
    net_dir = REPO / "sim_out" / "networks" / corridor.name
    grid: dict[tuple[int, int], list[tuple[Footprint, float]]] = {}
    try:
        scene = build_scene(
            net_dir / f"{corridor.name}.net.xml",
            net_dir / f"{corridor.name}.osm",
            [p for p in context_paths(corridor) if p.exists()],
        )
    except (OSError, ValueError):
        return grid
    for a in scene.areas:
        if a.kind == "building" and a.outline:
            x, y = a.outline[0]
            grid.setdefault((int(x // CELL_M), int(y // CELL_M)), []).append(
                (a.outline, a.height_m)
            )
    return grid


class Camera:
    """World (SUMO x/y, metres) to screen: heading-up, perspective from behind and above."""

    def __init__(self, x: float, y: float, heading: float, back_m: float, height_m: float) -> None:
        self.x, self.y = x, y
        h = math.radians(heading)
        self.fx, self.fy = math.sin(h), math.cos(h)
        self.back, self.height = back_m, height_m

    def depth(self, px: float, py: float) -> tuple[float, float]:
        """(metres to the right, metres in front of the camera)."""
        dx, dy = px - self.x, py - self.y
        return dx * self.fy - dy * self.fx, dx * self.fx + dy * self.fy + self.back

    def project(self, px: float, py: float, pz: float = 0.0) -> tuple[float, float] | None:
        right, z = self.depth(px, py)
        if z < 4.0:
            return None
        return right / z, -(self.height - pz) / z

    def line(
        self, pts: list[tuple[float, float]], pz: float = 0.0
    ) -> tuple[list[float | None], list[float | None]]:
        """A projected polyline, broken where it passes behind the camera."""
        xs: list[float | None] = []
        ys: list[float | None] = []
        for px, py in pts:
            q = self.project(px, py, pz)
            if q is None:
                if xs and xs[-1] is not None:
                    xs.append(None)
                    ys.append(None)
                continue
            xs.append(q[0])
            ys.append(q[1])
        return xs, ys


def nav_view(session: DriverSession, vid: str, scooter: bool, height: int) -> go.Figure:
    me = session.state(vid)
    if me is None:
        raise KeyError(vid)
    cam = Camera(me.x, me.y, me.heading_deg, 34.0 if scooter else 46.0, 24.0 if scooter else 30.0)
    fig = go.Figure()

    # the road network nearby
    rx: list[float | None] = []
    ry: list[float | None] = []
    for edge, shape in session.graph.shapes.items():
        if not shape or edge.startswith(":"):
            continue
        if (
            min(math.dist((me.x, me.y), p) for p in (shape[0], shape[len(shape) // 2], shape[-1]))
            > ROAD_REACH_M
        ):
            continue
        xs, ys = cam.line(list(shape))
        if xs:
            rx += [*xs, None]
            ry += [*ys, None]
    roads = _path(rx, ry, close=False)

    # buildings, far to near so nearer ones cover farther ones
    cx, cy = int(me.x // CELL_M), int(me.y // CELL_M)
    k = int(BUILDING_REACH_M // CELL_M) + 1
    blocks: list[tuple[float, Footprint, float]] = []
    for gx in range(cx - k, cx + k + 1):
        for gy in range(cy - k, cy + k + 1):
            for outline, h in city().get((gx, gy), []):
                right, z = cam.depth(*outline[0])
                if 8.0 < z < BUILDING_REACH_M and abs(right) < z:
                    blocks.append((z, outline, h))
    bx: list[float | None] = []
    by: list[float | None] = []
    for _, outline, h in sorted(blocks, key=lambda b: -b[0]):
        ring = [*outline, outline[0]]
        faces = [
            [
                cam.project(x1, y1),
                cam.project(x1, y1, h),
                cam.project(x2, y2, h),
                cam.project(x2, y2),
            ]
            for (x1, y1), (x2, y2) in zip(ring, ring[1:], strict=False)
        ]
        faces.append([cam.project(x, y, h) for x, y in ring])
        for face in faces:
            if all(q is not None for q in face):
                bx += [q[0] for q in face if q is not None] + [None]
                by += [q[1] for q in face if q is not None] + [None]
    buildings = _path(bx, by, close=True)

    # the path already travelled, and the route ahead
    trail = session.trails.get(vid, [])[-400:]
    if len(trail) > 1:
        xs, ys = cam.line(trail)
        fig.add_trace(
            go.Scatter(
                x=xs, y=ys, mode="lines", line={"color": "#3b5a86", "width": 7}, hoverinfo="skip"
            )
        )
    ahead = route_ahead_xy(me, session.graph, 1400.0)
    if len(ahead) > 1:
        xs, ys = cam.line(ahead)
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line={"color": "rgba(34,211,238,0.2)", "width": 28},
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(x=xs, y=ys, mode="lines", line={"color": CYAN, "width": 8}, hoverinfo="skip")
        )

    # traffic in sight
    seen = []
    for v, b in session.nearby(vid, 400.0):
        q = cam.project(v.x, v.y, 0.8)
        if q is not None:
            seen.append((q, v, b))
    if seen:
        fig.add_trace(
            go.Scatter(
                x=[q[0] for q, _, _ in seen], y=[q[1] for q, _, _ in seen], mode="markers",
                marker={
                    "size": [max(5.0, min(22.0, 420.0 / max(b.distance_m, 1.0))) for _, _, b in seen],
                    "color": [RED if v.vehicle_id.startswith("hazard_") else ("#a78bfa" if v.vehicle_id in DEMO_BY_ID else "#c4ccd8") for _, v, _ in seen],
                    "symbol": "square",
                },
                text=[f"{v.vehicle_id} · {v.speed_mps * 3.6:.0f} km/h · {b.distance_m:.0f} m {b.side}" for _, v, b in seen],
                hovertemplate="%{text}<extra></extra>",
            )
        )  # fmt: skip

    # what each recent V2V message is about: a beacon where it is
    for m in session.incoming(vid)[-3:]:
        b = session.where_is(vid, *m.at)
        q = cam.project(*m.at)
        top = cam.project(*m.at, 26.0)
        if b is None or q is None:
            continue
        colour = relevance(b)[0]
        if top is not None:
            fig.add_trace(
                go.Scatter(
                    x=[q[0], top[0]],
                    y=[q[1], top[1]],
                    mode="lines",
                    line={"color": colour, "width": 3},
                    hoverinfo="skip",
                )
            )
        fig.add_trace(
            go.Scatter(
                x=[q[0]], y=[q[1]], mode="markers+text",
                marker={"size": 16, "color": colour, "line": {"color": "#ffffff", "width": 1.5}},
                text=[f"<b>{m.title.capitalize()}</b> · {b.distance_m:,.0f} m"], textposition="top center",
                textfont={"color": TEXT, "size": 13}, hovertemplate=f"{m.title}<br>{m.text}<extra></extra>",
            )
        )  # fmt: skip

    # this vehicle: a chevron pointing up the road
    car = cam.project(me.x, me.y)
    if car is None:
        raise ValueError("the camera must sit behind the vehicle")
    w, ln = (0.02, 0.055) if scooter else (0.032, 0.075)
    x0, y0 = car
    fig.add_trace(
        go.Scatter(
            x=[x0],
            y=[y0],
            mode="markers",
            marker={"size": 70, "color": "rgba(34,211,238,0.14)"},
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[x0 - w, x0, x0 + w, x0, x0 - w],
            y=[y0 - ln * 0.35, y0 + ln * 0.65, y0 - ln * 0.35, y0 - ln * 0.1, y0 - ln * 0.35],
            mode="lines", fill="toself", fillcolor="#ffffff", line={"color": CYAN, "width": 2},
            hovertemplate="You<extra></extra>",
        )
    )  # fmt: skip

    aspect = 1.5 if scooter else 1.75
    span = 0.9
    fig.update_layout(
        height=height, margin={"l": 0, "r": 0, "t": 0, "b": 0}, showlegend=False,
        paper_bgcolor=NAV_BG, plot_bgcolor=NAV_BG,
        xaxis={"visible": False, "range": [-span * aspect / 2, span * aspect / 2], "fixedrange": True},
        yaxis={"visible": False, "range": [y0 - 0.16, y0 - 0.16 + span], "fixedrange": True},
        hoverlabel={"bgcolor": RAISED, "font": {"color": TEXT}},
        # ground, then roads, then solid buildings painted far to near; the route,
        # traffic and messages are traces, so they draw on top of all of it
        shapes=[
            {"type": "rect", "xref": "paper", "yref": "paper", "x0": 0, "x1": 1, "y0": 0.8, "y1": 1,
             "fillcolor": "rgba(34,211,238,0.035)", "line": {"width": 0}, "layer": "below"},
            {"type": "path", "path": roads, "line": {"color": "#222c3b", "width": 7}, "layer": "below"},
            {"type": "path", "path": roads, "line": {"color": "#34425a", "width": 1.2}, "layer": "below"},
            {"type": "path", "path": buildings, "fillcolor": "#1a2232", "line": {"color": "#36445c", "width": 0.8}, "layer": "below"},
        ] if roads or buildings else [],
    )  # fmt: skip
    return fig


def _path(xs: list[float | None], ys: list[float | None], close: bool) -> str:
    """SVG path data for gap-separated polylines (None breaks a line)."""
    out: list[str] = []
    start = True
    for x, y in zip(xs, ys, strict=True):
        if x is None or y is None:
            if close and not start:
                out.append("Z")
            start = True
            continue
        out.append(f"{'M' if start else 'L'}{x:.4f},{y:.4f}")
        start = False
    if close and not start:
        out.append("Z")
    return " ".join(out)
