# ruff: noqa: E501 - long caption strings
"""Scroll-driven camera tour for the 3D simulation. Owner: P4.

Engine-free. The tour is a few hand-placed camera shots of the real corridor
(an establishing view, along the highway, the flyover, the hazard, the
geocast region, free exploring). Scrolling moves a progress value between
them; the camera follows a smooth Catmull-Rom path through the shots, so
scrolling tilts, turns and zooms the view instead of spinning the scene.

Shots are placed from the scene itself (its bounds, the main roads' axis,
the raised decks, the scenario's hazard), never from fixed coordinates.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace

Point3 = tuple[float, float, float]


@dataclass(frozen=True)
class Shot:
    """A camera pose: orbit ``dist`` metres from ``focus`` (Ursina coordinates),
    turned ``yaw`` degrees (clockwise from north) and tilted ``pitch`` down."""

    focus: Point3
    yaw: float
    pitch: float
    dist: float
    fov: float = 50.0


@dataclass(frozen=True)
class Chapter:
    key: str
    title: str
    caption: str
    shot: Shot


def catmull(p0: float, p1: float, p2: float, p3: float, t: float) -> float:
    """Catmull-Rom between p1 and p2 (t in 0..1)."""
    t2, t3 = t * t, t * t * t
    return 0.5 * (
        2 * p1
        + (p2 - p0) * t
        + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
        + (3 * p1 - p0 - 3 * p2 + p3) * t3
    )


def smootherstep(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * t * (t * (t * 6 - 15) + 10)


def unwrap(angles: Sequence[float]) -> list[float]:
    """Headings made continuous, so interpolation always turns the short way."""
    out: list[float] = []
    for a in angles:
        if out:
            a = out[-1] + (a - out[-1] + 180.0) % 360.0 - 180.0
        out.append(a)
    return out


def pose_at(shots: Sequence[Shot], progress: float) -> Shot:
    """The camera pose ``progress`` chapters into the tour (0 .. len - 1).

    Each chapter settles on its shot (the easing flattens near whole numbers),
    and the path between shots is a Catmull-Rom curve: position, heading, tilt,
    log-distance and field of view all move together."""
    n = len(shots)
    if n == 1:
        return shots[0]
    progress = max(0.0, min(float(n - 1), progress))
    i = min(int(progress), n - 2)
    t = smootherstep(progress - i)
    idx = [max(0, i - 1), i, i + 1, min(n - 1, i + 2)]
    yaws = unwrap([s.yaw for s in shots])

    def channel(values: Sequence[float]) -> float:
        return catmull(*(values[k] for k in idx), t)  # type: ignore[call-arg]

    focus = tuple(channel([s.focus[a] for s in shots]) for a in range(3))
    return Shot(
        focus=focus,  # type: ignore[arg-type]
        yaw=channel(yaws) % 360.0,
        pitch=channel([s.pitch for s in shots]),
        dist=math.exp(channel([math.log(s.dist) for s in shots])),
        fov=channel([s.fov for s in shots]),
    )


def blend(a: Shot, b: Shot, t: float) -> Shot:
    """Ease from pose ``a`` towards ``b`` (t in 0..1), heading the short way round."""
    t = max(0.0, min(1.0, t))
    dy = (b.yaw - a.yaw + 180.0) % 360.0 - 180.0
    return Shot(
        focus=tuple(p + (q - p) * t for p, q in zip(a.focus, b.focus, strict=True)),  # type: ignore[arg-type]
        yaw=(a.yaw + dy * t) % 360.0,
        pitch=a.pitch + (b.pitch - a.pitch) * t,
        dist=math.exp(math.log(a.dist) + (math.log(b.dist) - math.log(a.dist)) * t),
        fov=a.fov + (b.fov - a.fov) * t,
    )


def parallax(shot: Shot, mx: float, my: float, strength: float = 1.0) -> Shot:
    """A small look-around from the mouse position (-0.5 .. 0.5 on each axis)."""
    return replace(
        shot,
        yaw=(shot.yaw + mx * 7.0 * strength) % 360.0,
        pitch=max(4.0, min(88.0, shot.pitch - my * 4.0 * strength)),
    )


@dataclass
class Scroll:
    """Scroll position with inertia: the wheel moves ``target``; ``value``
    glides after it, like a smooth-scrolling web page."""

    count: int
    value: float = 0.0
    target: float = 0.0
    step: float = 0.25
    """Chapters per wheel notch (four notches from one shot to the next)."""
    rate: float = 2.6
    """How quickly ``value`` catches up (per second)."""

    def wheel(self, notches: float) -> None:
        self.target = max(0.0, min(float(self.count - 1), self.target + notches * self.step))

    def go(self, chapter: float) -> None:
        self.target = max(0.0, min(float(self.count - 1), chapter))

    def update(self, dt: float) -> float:
        k = 1.0 - math.exp(-self.rate * max(dt, 0.0))
        self.value += (self.target - self.value) * k
        if abs(self.target - self.value) < 1e-4:
            self.value = self.target
        return self.value

    @property
    def chapter(self) -> int:
        return round(self.value)


def principal_axis(points: Sequence[tuple[float, float]]) -> float:
    """Heading (degrees clockwise from north, 0..180) of the points' main direction."""
    if len(points) < 2:
        return 0.0
    mx = sum(p[0] for p in points) / len(points)
    my = sum(p[1] for p in points) / len(points)
    sxx = sum((p[0] - mx) ** 2 for p in points)
    syy = sum((p[1] - my) ** 2 for p in points)
    sxy = sum((p[0] - mx) * (p[1] - my) for p in points)
    angle = 0.5 * math.atan2(2 * sxy, sxx - syy)  # from +x (east), counter-clockwise
    return (90.0 - math.degrees(angle)) % 180.0


def chapters(
    extent: tuple[float, float],
    road_axis: float,
    road_focus: Point3,
    deck_focus: Point3 | None,
    hazard: Point3,
    *,
    hazard_name: str,
    place: str,
    lanes: int,
    buildings: int,
    decks: int,
    distribution: str,
    radius_m: float,
) -> list[Chapter]:
    """The tour for one scenario. Positions are Ursina (x, y-up, z) metres."""
    span = max(extent)
    deck = deck_focus or road_focus
    where = f" near {place}" if place else ""
    rule = {
        "directional": f"only vehicles still heading towards it, within {radius_m:.0f} m upstream",
        "circular": f"every vehicle within {radius_m:.0f} m",
        "broadcast": "every vehicle in radio reach, needed or not",
    }.get(distribution, distribution)
    return [
        Chapter(
            "intro",
            "A.C.T.I.O.N  V2V",
            "Informal driver reports, turned into vehicle-to-vehicle hazard warnings.\n"
            "Hebbal, Bengaluru: real OpenStreetMap roads, live SUMO traffic.",
            Shot(tuple(c * 0.5 for c in road_focus), road_axis + 205.0, 13.0, span * 0.6, 40.0),  # type: ignore[arg-type]
        ),
        Chapter(
            "corridor",
            "The corridor",
            f"{lanes:,} lanes and {buildings:,} buildings from the OpenStreetMap snapshot,\n"
            "with every vehicle driven by SUMO.",
            Shot(road_focus, road_axis + 8.0, 9.0, 520.0, 46.0),
        ),
        Chapter(
            "flyovers",
            "Flyovers",
            f"{decks} raised road segments. A warning must reach the right level,\n"
            "not just the right spot on the map.",
            Shot(deck, road_axis + 72.0, 16.0, 230.0, 50.0),
        ),
        Chapter(
            "hazard",
            "The hazard",
            f"A {hazard_name}{where}. Click it to see what the system knows,\n"
            "or press K to jump to the moment drivers report it.",
            Shot(hazard, road_axis - 38.0, 30.0, 300.0, 48.0),
        ),
        Chapter(
            "geocast",
            "Geocast",
            f"Warnings go to {rule}.\n"
            "Each glowing hop is a real relay in the simulated radio network.",
            Shot(hazard, road_axis + 14.0, 58.0, max(radius_m, 300.0) * 2.1, 44.0),
        ),
        Chapter(
            "explore",
            "Explore",
            "Drag to look around, click any vehicle or the hazard.\n"
            "Space plays the simulation; the panel at the bottom left has the rest.",
            Shot(hazard, road_axis + 128.0, 38.0, 560.0, 50.0),
        ),
    ]
