"""Map figures for the dashboard. Owner: P4.

Token-free MapLibre basemaps (CARTO dark by default). Vehicles are drawn as a
dot plus a short heading tick so direction of travel reads at a glance; the
hazard gets a glowing marker and its 200 m sensor zone, and the geocast area
is outlined. Everything plotted comes from recorded SUMO traces.

Map data (c) OpenStreetMap contributors, ODbL; basemap (c) CARTO.
"""

import math
from collections.abc import Iterable, Sequence

import plotly.graph_objects as go
from ui_theme import (
    BG,
    BLUE,
    BORDER_STRONG,
    CONDITION_COLOURS,
    FONT,
    LIME,
    ORANGE,
    OUTCOME_COLOURS,
    PURPLE,
    RAISED,
    RED,
    TEXT,
)

from v2v.sim.recorder import PolicyResult, ReportFrame, Trace, VehicleSnap

BASEMAPS = {
    "Dark": "carto-darkmatter",
    "Dark, no labels": "carto-darkmatter-nolabels",
    "Streets": "open-street-map",
}
SENSOR_RADIUS_M = 200.0
"""Radius used for the hazard's sensor flags (``SumoWorld.sensor_flags``)."""
TICK_M = 38.0
M_PER_DEG = 111_320.0
MAP_CONFIG = {
    "scrollZoom": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": ["select2d", "lasso2d", "toImage"],
}


def ring(lon: float, lat: float, radius_m: float, n: int = 96) -> tuple[list[float], list[float]]:
    """A circle of ``radius_m`` around a point, as lon/lat lists."""
    lons, lats = [], []
    for i in range(n + 1):
        a = 2 * math.pi * i / n
        lats.append(lat + radius_m * math.cos(a) / M_PER_DEG)
        lons.append(lon + radius_m * math.sin(a) / (M_PER_DEG * math.cos(math.radians(lat))))
    return lons, lats


def ticks(vehicles: Iterable[VehicleSnap], length_m: float = TICK_M) -> tuple[list, list]:
    """Heading ticks (SUMO angle: degrees clockwise from north), None-separated."""
    lons: list[float | None] = []
    lats: list[float | None] = []
    for v in vehicles:
        h = math.radians(v.heading_deg)
        lons += [
            v.lon,
            v.lon + length_m * math.sin(h) / (M_PER_DEG * math.cos(math.radians(v.lat))),
        ]
        lats += [v.lat, v.lat + length_m * math.cos(h) / M_PER_DEG]
        lons.append(None)
        lats.append(None)
    return lons, lats


def distance_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Great-circle distance in metres."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6_371_000 * math.asin(math.sqrt(a))


def _layout(
    fig: go.Figure, lon: float, lat: float, zoom: float, height: int, style: str, rev: str
) -> go.Figure:
    fig.update_layout(
        map={"style": style, "center": {"lon": lon, "lat": lat}, "zoom": zoom},
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        height=height,
        showlegend=False,
        paper_bgcolor=BG,
        uirevision=rev,  # keep the user's zoom and pan across reruns
        hoverlabel={
            "bgcolor": RAISED,
            "bordercolor": BORDER_STRONG,
            "font": {"color": TEXT, "family": FONT, "size": 12},
        },
    )
    return fig


def _zone(fig: go.Figure, lon: float, lat: float, radius_m: float, line: str, fill: str) -> None:
    lons, lats = ring(lon, lat, radius_m)
    fig.add_trace(
        go.Scattermap(
            lon=lons,
            lat=lats,
            mode="lines",
            fill="toself" if fill else "none",
            fillcolor=fill or None,
            line={"color": line, "width": 1.2},
            hoverinfo="skip",
        )
    )


def _glow(
    fig: go.Figure, lon: float, lat: float, colour: str, sizes: Sequence[int], name: str
) -> None:
    for size, alpha in zip(sizes, ("26", "55", "ff"), strict=False):
        fig.add_trace(
            go.Scattermap(
                lon=[lon],
                lat=[lat],
                mode="markers",
                marker={"size": size, "color": colour + alpha if alpha != "ff" else colour},
                hovertext=name,
                hoverinfo="text" if alpha == "ff" else "skip",
            )
        )


def _vehicle_hover(v: VehicleSnap, label: str, result: PolicyResult | None, hazard: tuple) -> str:
    text = (
        f"<b>{v.vehicle_id}</b> · {label}<br>{v.speed_mps * 3.6:.0f} km/h · heading "
        f"{v.heading_deg:.0f}°<br>edge {v.edge_id}<br>"
        f"{distance_m(v.lon, v.lat, *hazard):.0f} m from hazard (straight line)"
    )
    if result is not None and v.vehicle_id in result.latency_ms:
        text += (
            f"<br>delivered in {result.latency_ms[v.vehicle_id]:.1f} ms, "
            f"{result.hops.get(v.vehicle_id, '?')} hop(s)"
        )
    return text


def live_map(
    trace: Trace,
    frame: ReportFrame,
    result: PolicyResult,
    outcome: dict[str, str],
    *,
    location: str,
    tracked: str | None,
    show_zones: bool,
    show_ticks: bool,
    style: str,
    height: int = 700,
) -> go.Figure:
    fig = go.Figure()
    hz = (trace.hazard_lon, trace.hazard_lat)
    if show_zones:
        centre = hz if location == "known" else (frame.fallback_lon, frame.fallback_lat)
        _zone(fig, *centre, trace.radius_m, LIME + "55", LIME + "0a")
        _zone(fig, *hz, SENSOR_RADIUS_M, RED + "aa", RED + "22")
    for label, colour in OUTCOME_COLOURS.items():
        vs = [v for v in frame.vehicles if outcome[v.vehicle_id] == label]
        if not vs:
            continue
        quiet = label == "not warned"
        if show_ticks:
            lons, lats = ticks(vs)
            fig.add_trace(
                go.Scattermap(
                    lon=lons,
                    lat=lats,
                    mode="lines",
                    line={"color": colour, "width": 1.6 if quiet else 2.4},
                    opacity=0.5 if quiet else 0.95,
                    hoverinfo="skip",
                )
            )
        fig.add_trace(
            go.Scattermap(
                lon=[v.lon for v in vs],
                lat=[v.lat for v in vs],
                mode="markers",
                marker={"size": 6 if quiet else 9, "color": colour, "opacity": 0.6 if quiet else 1},
                hovertext=[_vehicle_hover(v, label, result, hz) for v in vs],
                hoverinfo="text",
            )
        )
    by_id = {v.vehicle_id: v for v in (*frame.vehicles, frame.reporter)}
    if tracked in by_id:
        t = by_id[tracked]
        fig.add_trace(
            go.Scattermap(
                lon=[t.lon],
                lat=[t.lat],
                mode="markers",
                marker={"size": 26, "color": "rgba(255,255,255,0.18)"},
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scattermap(
                lon=[t.lon],
                lat=[t.lat],
                mode="markers",
                marker={"size": 12, "color": "#ffffff"},
                hovertext=f"tracking {t.vehicle_id}",
                hoverinfo="text",
            )
        )
    _glow(fig, *hz, RED, (54, 30, 15), "hazard (true position)")
    r = frame.reporter
    _glow(fig, r.lon, r.lat, PURPLE, (36, 22, 12), f"reporter {r.vehicle_id}")
    if location == "fallback":
        _glow(
            fig, frame.fallback_lon, frame.fallback_lat, ORANGE, (30, 18, 10), "fallback location"
        )
    return _layout(fig, *hz, 14.7, height, style, trace.spec.scenario_id)


def sites_map(traces: Sequence[Trace], style: str, height: int = 420) -> go.Figure:
    """Every recorded scenario's hazard site, coloured by condition."""
    fig = go.Figure()
    for condition, colour in CONDITION_COLOURS.items():
        ts = [t for t in traces if t.spec.condition == condition]
        if not ts:
            continue
        for size, alpha in ((34, "22"), (12, "")):
            fig.add_trace(
                go.Scattermap(
                    lon=[t.hazard_lon for t in ts],
                    lat=[t.hazard_lat for t in ts],
                    mode="markers",
                    marker={"size": size, "color": colour + alpha},
                    hovertext=[
                        f"<b>{t.spec.scenario_id}</b><br>{t.spec.hazard_class.value} · "
                        f"{t.spec.effect}<br>{', '.join(t.spec.landmark_names[:2])}<br>"
                        f"{len(t.frames)} report(s)"
                        for t in ts
                    ],
                    hoverinfo="text" if not alpha else "skip",
                )
            )
    lon = sum(t.hazard_lon for t in traces) / len(traces)
    lat = sum(t.hazard_lat for t in traces) / len(traces)
    return _layout(fig, lon, lat, 14.4, height, style, "sites")


def traffic_map(trace: Trace, style: str, show_zones: bool, height: int = 640) -> go.Figure:
    """All vehicle positions recorded in a run, coloured by speed: the road network in use."""
    spec = trace.spec
    approach, opposing = set(spec.approach_route), set(spec.opposing_route)
    snaps = [v for f in trace.frames for v in (*f.vehicles, f.reporter)]
    fig = go.Figure()
    if show_zones:
        _zone(fig, trace.hazard_lon, trace.hazard_lat, trace.radius_m, LIME + "55", LIME + "08")
        _zone(fig, trace.hazard_lon, trace.hazard_lat, trace.comm_range_m, BLUE + "aa", BLUE + "10")
        _zone(fig, trace.hazard_lon, trace.hazard_lat, SENSOR_RADIUS_M, RED + "aa", RED + "22")
    fig.add_trace(
        go.Scattermap(
            lon=[v.lon for v in snaps],
            lat=[v.lat for v in snaps],
            mode="markers",
            marker={
                "size": [8 if v.edge_id in approach or v.edge_id in opposing else 5 for v in snaps],
                "color": [v.speed_mps * 3.6 for v in snaps],
                "colorscale": [[0, RED], [0.35, ORANGE], [1, LIME]],
                "cmin": 0,
                "cmax": 60,
                "opacity": 0.8,
                "colorbar": {
                    "title": {"text": "km/h", "font": {"color": TEXT, "size": 11}},
                    "thickness": 8,
                    "len": 0.3,
                    "x": 0.985,
                    "y": 0.3,
                    "bgcolor": "rgba(26,26,26,0.85)",
                    "outlinewidth": 0,
                    "tickfont": {"color": TEXT, "size": 10},
                },
            },
            hovertext=[
                f"<b>{v.vehicle_id}</b><br>{v.speed_mps * 3.6:.0f} km/h<br>edge {v.edge_id}"
                + (
                    " · approach route"
                    if v.edge_id in approach
                    else " · opposing route"
                    if v.edge_id in opposing
                    else ""
                )
                for v in snaps
            ],
            hoverinfo="text",
        )
    )
    _glow(fig, trace.hazard_lon, trace.hazard_lat, RED, (54, 30, 15), "hazard (true position)")
    return _layout(fig, trace.hazard_lon, trace.hazard_lat, 14.3, height, style, spec.scenario_id)
