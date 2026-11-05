# ruff: noqa: E501 - long geometry and HUD expressions
"""Engine-free geometry for the 3D simulation: strips, polygons, sampling. Owner: P4.

SUMO coordinates are (x east, y north, z up) in metres. Ursina is y-up, so a
SUMO point (x, y, z) becomes (x - ox, z, y - oy), with the scene centred on the
hazard so float precision stays good. Headings are SUMO angles (degrees
clockwise from north), which is exactly Ursina's ``rotation_y``.
"""

import math
import random
from collections.abc import Sequence

Point2 = tuple[float, float]
Point3 = tuple[float, float, float]


def to_ursina(x: float, y: float, z: float, origin: Point2) -> Point3:
    return (x - origin[0], z, y - origin[1])


def normals(points: Sequence[Point3]) -> list[Point2]:
    """Unit left-hand normals (in the x/y plane) at each vertex of a polyline,
    averaged at corners so strips join without gaps."""
    out: list[Point2] = []
    n = len(points)
    for i in range(n):
        a = points[max(i - 1, 0)]
        b = points[min(i + 1, n - 1)]
        dx, dy = b[0] - a[0], b[1] - a[1]
        length = math.hypot(dx, dy) or 1.0
        out.append((-dy / length, dx / length))
    return out


def offset(points: Sequence[Point3], d: float) -> list[Point3]:
    """The polyline shifted ``d`` metres to its left (negative: right)."""
    return [
        (p[0] + nx * d, p[1] + ny * d, p[2])
        for p, (nx, ny) in zip(points, normals(points), strict=True)
    ]


def strip(points: Sequence[Point3], left: float, right: float) -> tuple[list[Point3], list[Point3]]:
    """Left and right edges of a band from ``right`` to ``left`` metres off the line."""
    return offset(points, left), offset(points, right)


def dashes(points: Sequence[Point3], on: float, off: float) -> list[list[Point3]]:
    """Cut a polyline into dash segments ``on`` metres long with ``off`` gaps."""
    out: list[list[Point3]] = []
    current: list[Point3] = []
    drawing, left = True, on
    for a, b in zip(points, points[1:], strict=False):
        seg = math.dist(a[:2], b[:2])
        pos = 0.0
        if drawing and not current:
            current = [a]
        while seg - pos > left:
            pos += left
            f = pos / seg
            p = (a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f)
            if drawing:
                current.append(p)
                out.append(current)
                current = []
            else:
                current = [p]
            drawing = not drawing
            left = on if drawing else off
        left -= seg - pos
        if drawing:
            current.append(b)
    if drawing and len(current) >= 2:
        out.append(current)
    return out


def along(points: Sequence[Point3], every: float, start: float = 0.0) -> list[tuple[Point3, float]]:
    """Points every ``every`` metres along a polyline, with the heading there."""
    out: list[tuple[Point3, float]] = []
    target, walked = start, 0.0
    for a, b in zip(points, points[1:], strict=False):
        seg = math.dist(a[:2], b[:2])
        if seg == 0:
            continue
        heading = math.degrees(math.atan2(b[0] - a[0], b[1] - a[1])) % 360
        while target <= walked + seg:
            f = (target - walked) / seg
            out.append(
                (
                    (a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f),
                    heading,
                )
            )
            target += every
        walked += seg
    return out


def signed_area(poly: Sequence[Point2]) -> float:
    return 0.5 * sum(
        a[0] * b[1] - b[0] * a[1] for a, b in zip(poly, [*poly[1:], poly[0]], strict=True)
    )


def ccw(poly: Sequence[Point2]) -> list[Point2]:
    """The polygon in counter-clockwise order."""
    return list(poly) if signed_area(poly) >= 0 else list(reversed(poly))


def inside(poly: Sequence[Point2], x: float, y: float) -> bool:
    hit = False
    for (x1, y1), (x2, y2) in zip(poly, [*poly[1:], poly[0]], strict=True):
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            hit = not hit
    return hit


def scatter(poly: Sequence[Point2], spacing: float, seed: str) -> list[Point2]:
    """Jittered grid points inside a polygon (decorative trees in a real park)."""
    rng = random.Random(seed)
    xs, ys = [p[0] for p in poly], [p[1] for p in poly]
    out: list[Point2] = []
    y = min(ys) + spacing / 2
    while y < max(ys):
        x = min(xs) + spacing / 2
        while x < max(xs):
            px = x + rng.uniform(-0.35, 0.35) * spacing
            py = y + rng.uniform(-0.35, 0.35) * spacing
            if inside(poly, px, py):
                out.append((px, py))
            x += spacing
        y += spacing
    return out


def slow_motion(latest_ms: float, show_s: float = 2.5) -> float:
    """Real seconds per message millisecond, so a whole V2V flood takes about
    ``show_s`` seconds on screen (radio latencies are only tens of ms)."""
    return show_s / max(latest_ms, 1.0)


def lerp_angle(a: float, b: float, t: float) -> float:
    """Shortest-way interpolation between two headings in degrees."""
    d = (b - a + 180.0) % 360.0 - 180.0
    return (a + d * t) % 360.0


TYPEABLE = ".,'-?!/"
MAX_TYPED = 160


def type_into(text: str, key: str) -> tuple[str, str | None]:
    """One key press for a report being typed: (the new text, then "send" or
    "cancel" when the key ends it, else None). Letters, digits and light
    punctuation are kept; every other key (shift, arrows, mouse) is ignored."""
    if key in ("enter", "return"):
        return text, "send"
    if key == "escape":
        return text, "cancel"
    if key == "backspace":
        return text[:-1], None
    if key == "space":
        return (text + " " if len(text) < MAX_TYPED else text), None
    if len(key) == 1 and (key.isalnum() or key in TYPEABLE) and len(text) < MAX_TYPED:
        return text + key, None
    return text, None
