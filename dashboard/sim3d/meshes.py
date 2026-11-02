# ruff: noqa: E501 - long geometry and HUD expressions
"""Turn the real corridor (``v2v.sim.world3d.Scene``) into a few big meshes. Owner: P4.

Everything static is merged into a handful of vertex-coloured meshes with the
lighting baked into the colours, drawn unlit with distance haze (atmosphere.fog_shader): cheap
enough for integrated graphics. Roads, flyovers, junctions, buildings, water
and parks are real (SUMO network + OpenStreetMap). Street lamps along the
main roads, trees inside the real parks and lit windows are decorative
placements, since the public OSM server cannot serve those nodes for the
snapshot date and OSM has no windows.
"""

import math
import random
from collections import defaultdict
from collections.abc import Iterable, Sequence

from atmosphere import fog_shader
from geom import Point2, Point3, along, ccw, dashes, scatter, strip, to_ursina
from panda3d.core import TransparencyAttrib, Triangulator
from ursina import Entity, Mesh, Vec3, color

from v2v.sim.world3d import Lane, Scene

MAX_VERTS = 60_000
MAIN_ROADS = {"motorway", "trunk", "primary", "trunk_link", "primary_link", "motorway_link"}
LIGHT = (-0.55, 0.8)
"""Direction the moonlight comes from (x, y), for baked wall shading."""

ASPHALT_MAIN = (0.13, 0.148, 0.185)
ASPHALT = (0.075, 0.085, 0.105)
JUNCTION = (0.085, 0.094, 0.115)
MARKING = (0.36, 0.4, 0.45)
EDGE_LINE = (0.2, 0.24, 0.3)
DECK_GLOW = (0.0, 0.85, 1.0)
ROAD_GLOW = (0.15, 0.55, 1.0)
STREET_GLOW = (0.1, 0.32, 0.6)
CONCRETE = (0.15, 0.165, 0.19)
WALL = (0.08, 0.098, 0.13)
ROOF = (0.12, 0.14, 0.18)
WATER = (0.016, 0.05, 0.092)
WATER_EDGE = (0.1, 0.35, 0.55)
PARK = (0.026, 0.055, 0.045)
GROUND = (0.026, 0.032, 0.044)
TREE = (0.04, 0.13, 0.09)
TRUNK = (0.1, 0.08, 0.065)
LAMP = (1.0, 0.62, 0.25)
WINDOW_WARM = (1.0, 0.66, 0.34)
WINDOW_COOL = (0.5, 0.74, 1.0)
FLOOR_M = 3.2


class Builder:
    """Accumulates coloured triangles and hands out meshes of at most MAX_VERTS."""

    def __init__(self) -> None:
        self.meshes: list[Mesh] = []
        self._v: list[Vec3] = []
        self._t: list[int] = []
        self._c: list[color.Color] = []

    def _room(self, n: int) -> None:
        if len(self._v) + n > MAX_VERTS:
            self.flush()

    def tri(
        self, a: Point3, b: Point3, c: Point3, rgb: Sequence[float], alpha: float = 1.0
    ) -> None:
        self._room(3)
        i = len(self._v)
        self._v += [Vec3(*a), Vec3(*b), Vec3(*c)]
        self._t += [i, i + 1, i + 2]
        self._c += [color.rgba(*rgb, alpha)] * 3

    def quad(
        self, a: Point3, b: Point3, c: Point3, d: Point3, rgb: Sequence[float], alpha: float = 1.0
    ) -> None:
        self._room(4)
        i = len(self._v)
        self._v += [Vec3(*a), Vec3(*b), Vec3(*c), Vec3(*d)]
        self._t += [i, i + 1, i + 2, i, i + 2, i + 3]
        self._c += [color.rgba(*rgb, alpha)] * 4

    def shaded(
        self,
        a: Point3,
        b: Point3,
        c: Point3,
        d: Point3,
        low: Sequence[float],
        high: Sequence[float],
    ) -> None:
        """A wall quad, ``a``/``d`` at the bottom in ``low`` and ``b``/``c`` at the top in ``high``."""
        self._room(4)
        i = len(self._v)
        self._v += [Vec3(*a), Vec3(*b), Vec3(*c), Vec3(*d)]
        self._t += [i, i + 1, i + 2, i, i + 2, i + 3]
        lo, hi = color.rgba(*low, 1), color.rgba(*high, 1)
        self._c += [lo, hi, hi, lo]

    def fade(
        self, inner: Sequence[Point3], outer: Sequence[Point3], rgb: Sequence[float], alpha: float
    ) -> None:
        """A strip lit at ``alpha`` along ``inner`` and fading to nothing at ``outer``."""
        on, off = color.rgba(*rgb, alpha), color.rgba(*rgb, 0.0)
        for a, b, c, d in zip(inner, inner[1:], outer[1:], outer, strict=False):
            self._room(4)
            i = len(self._v)
            self._v += [Vec3(*a), Vec3(*b), Vec3(*c), Vec3(*d)]
            self._t += [i, i + 1, i + 2, i, i + 2, i + 3]
            self._c += [on, on, off, off]

    def ribbon(
        self,
        line: Sequence[Point3],
        half: float,
        rgb: Sequence[float],
        alpha: float,
        origin: Point2,
        lift: float = 0.0,
    ) -> None:
        """Soft light along a polyline (SUMO coords): brightest on it, gone ``half`` metres either side.
        Wide and soft, so it never breaks into dots in the distance the way thin lines do."""
        centre = _u(line, origin, lift)
        left, right = strip(line, half, -half)
        self.fade(centre, _u(left, origin, lift), rgb, alpha)
        self.fade(centre, _u(right, origin, lift), rgb, alpha)

    def band(
        self,
        left: Sequence[Point3],
        right: Sequence[Point3],
        rgb: Sequence[float],
        alpha: float = 1.0,
    ) -> None:
        for a, b, c, d in zip(left, left[1:], right[1:], right, strict=False):
            self.quad(a, b, c, d, rgb, alpha)

    def box(self, centre: Point3, size: Point3, heading: float, rgb: Sequence[float]) -> None:
        """An upright box (Ursina coords), rotated about y by ``heading`` degrees."""
        cx, cy, cz = centre
        sx, sy, sz = (s / 2 for s in size)
        h = math.radians(heading)
        ch, sh = math.cos(h), math.sin(h)

        def p(x: float, y: float, z: float) -> Point3:
            return (cx + x * ch + z * sh, cy + y, cz - x * sh + z * ch)

        top = [p(-sx, sy, -sz), p(-sx, sy, sz), p(sx, sy, sz), p(sx, sy, -sz)]
        bot = [p(-sx, -sy, -sz), p(-sx, -sy, sz), p(sx, -sy, sz), p(sx, -sy, -sz)]
        self.quad(*top, rgb)
        for i in range(4):
            j = (i + 1) % 4
            shade = 0.75 + 0.25 * (i % 2)
            self.quad(bot[i], top[i], top[j], bot[j], [c * shade for c in rgb])

    def polygon(
        self,
        outline: Sequence[Point2],
        y: float,
        rgb: Sequence[float],
        origin: Point2,
        alpha: float = 1.0,
    ) -> None:
        """A flat filled polygon at height ``y`` (SUMO x/y outline)."""
        poly = ccw(outline)
        tri = Triangulator()
        for x, z in poly:
            tri.add_polygon_vertex(tri.add_vertex(x - origin[0], z - origin[1]))
        tri.triangulate()
        for k in range(tri.get_num_triangles()):
            a, b, c = (
                tri.get_vertex(i)
                for i in (tri.get_triangle_v0(k), tri.get_triangle_v1(k), tri.get_triangle_v2(k))
            )
            self.tri((a[0], y, a[1]), (c[0], y, c[1]), (b[0], y, b[1]), rgb, alpha)

    def flush(self) -> None:
        if self._v:
            self.meshes.append(
                Mesh(vertices=self._v, triangles=self._t, colors=self._c, static=True)
            )
        self._v, self._t, self._c = [], [], []

    def entities(self, see_through: bool = False, **kwargs: object) -> list[Entity]:
        self.flush()
        out = [
            Entity(model=m, shader=fog_shader, double_sided=True, **kwargs)  # type: ignore[arg-type]
            for m in self.meshes
        ]
        if see_through:
            for e in out:
                translucent(e)
        return out


def translucent(e: Entity) -> Entity:
    """Blend by vertex alpha and do not hide what is behind (zones, glows)."""
    e.setTransparency(TransparencyAttrib.M_alpha)
    e.setDepthWrite(False)
    e.setBin("transparent", 10)
    return e


def _u(points: Iterable[Point3], origin: Point2, lift: float = 0.0) -> list[Point3]:
    return [to_ursina(x, y, z + lift, origin) for x, y, z in points]


def lanes_by_edge(scene: Scene) -> dict[str, list[Lane]]:
    out: dict[str, list[Lane]] = defaultdict(list)
    for lane in scene.lanes:
        out[lane.edge_id].append(lane)
    for lanes in out.values():
        lanes.sort(key=lambda lane: int(lane.lane_id.rsplit("_", 1)[1]))
    return out


def build_roads(scene: Scene, origin: Point2) -> tuple[Builder, Builder]:
    """Asphalt, junctions, markings and flyover structure; and the light along decks and highways."""
    solid, glow = Builder(), Builder()
    for poly, z in zip(scene.junctions, scene.junction_z, strict=True):
        solid.polygon(poly, z + 0.06, JUNCTION, origin)
    for edge_id, lanes in lanes_by_edge(scene).items():
        main = lanes[0].road_class in MAIN_ROADS
        for k, lane in enumerate(lanes):
            half = lane.width_m / 2
            left, right = strip(lane.points, half, -half)
            solid.band(
                _u(left, origin, 0.08), _u(right, origin, 0.08), ASPHALT_MAIN if main else ASPHALT
            )
            if k > 0:  # dashed separator on this lane's right edge
                for dash in dashes(right, 3.0, 6.0):
                    a, b = strip(dash, 0.08, -0.08)
                    solid.band(_u(a, origin, 0.1), _u(b, origin, 0.1), MARKING)
        raised = max(p[2] for lane in lanes for p in lane.points) > 1.0
        # the network itself glows faintly, so it reads as the centrepiece from afar
        width = sum(lane.width_m for lane in lanes)
        mid = lanes[len(lanes) // 2].points
        glow.ribbon(
            mid,
            width / 2 + (7.0 if main else 3.5),
            ROAD_GLOW if main else STREET_GLOW,
            0.26 if main else 0.1,
            origin,
            0.14,
        )
        outer_right = strip(lanes[0].points, lanes[0].width_m / 2, -lanes[0].width_m / 2)[1]
        outer_left = strip(lanes[-1].points, lanes[-1].width_m / 2, -lanes[-1].width_m / 2)[0]
        for side in (outer_right, outer_left):
            a, b = strip(side, 0.07, -0.07)
            solid.band(_u(a, origin, 0.1), _u(b, origin, 0.1), EDGE_LINE)
            if main and not raised:  # light along the highway edges
                glow.ribbon(side, 1.1, ROAD_GLOW, 0.75, origin, 0.12)
            if raised:  # parapet with a neon strip along the flyover deck
                top = [(x, y, z + 1.0) for x, y, z in side]
                solid.band(_u(side, origin), _u(top, origin), CONCRETE)
                glow.ribbon(top, 0.6, DECK_GLOW, 0.9, origin, 0.02)
        if raised:
            for (x, y, z), heading in along(lanes[0].points, 28.0, 10.0):
                if z > 2.5:
                    w = sum(lane.width_m for lane in lanes) * 0.6
                    solid.box(
                        to_ursina(x, y, (z - 0.4) / 2, origin), (w, z - 0.4, 1.6), heading, CONCRETE
                    )
        del edge_id
    return solid, glow


def build_city(scene: Scene, origin: Point2) -> tuple[Builder, Builder]:
    """Ground, water, parks and extruded buildings with baked moonlight; and lit windows."""
    b, lights = Builder(), Builder()
    far = 7000.0  # ground out to the horizon; the haze hides its edge
    b.polygon([(-far, -far), (far, -far), (far, far), (-far, far)], -0.05, GROUND, (0.0, 0.0))
    for area in scene.areas:
        if area.kind == "water":
            b.polygon(area.outline, 0.02, WATER, origin)
            poly = [(x, y, 0.0) for x, y in ccw(area.outline)]
            ring_ = [*poly, poly[0]]
            left, right = strip(ring_, 0.6, -0.6)
            lights.band(_u(left, origin, 0.05), _u(right, origin, 0.05), WATER_EDGE, 0.35)
        elif area.kind == "park":
            b.polygon(area.outline, 0.01, PARK, origin)
    rng = random.Random(7)
    lx, ly = LIGHT
    for area in scene.areas:
        if area.kind != "building":
            continue
        h = area.height_m
        tint = rng.uniform(0.85, 1.15)
        floors = max(1, int(h // FLOOR_M))
        poly = ccw(area.outline)
        for (x1, y1), (x2, y2) in zip(poly, [*poly[1:], poly[0]], strict=True):
            nx, ny = (y2 - y1), -(x2 - x1)  # outward normal of a CCW outline
            n = math.hypot(nx, ny) or 1.0
            shade = 0.55 + 0.45 * max(0.0, (nx * lx + ny * ly) / n)
            low = [c * shade * tint * 0.45 for c in WALL]  # darker at street level
            high = [c * shade * tint * 1.5 for c in WALL]
            b.shaded(
                to_ursina(x1, y1, 0, origin),
                to_ursina(x1, y1, h, origin),
                to_ursina(x2, y2, h, origin),
                to_ursina(x2, y2, 0, origin),
                low,
                high,
            )
            if n < 6.0:
                continue
            ox, oy = nx / n * 0.06, ny / n * 0.06  # just proud of the wall
            for floor in range(floors):
                if rng.random() > 0.22:
                    continue
                z0 = floor * FLOOR_M + 1.1
                rgb = WINDOW_WARM if rng.random() < 0.7 else WINDOW_COOL
                k = rng.uniform(0.25, 0.55)
                count = max(1, int((n - 2.0) / 3.0))  # a window every 3 m, 1.5 m wide
                for w in range(count):
                    if rng.random() < 0.3:
                        continue  # some rooms are dark
                    f0 = (1.0 + w * 3.0 + 0.75) / n
                    f1 = f0 + 1.5 / n
                    pa = (x1 + (x2 - x1) * f0 + ox, y1 + (y2 - y1) * f0 + oy)
                    pb = (x1 + (x2 - x1) * f1 + ox, y1 + (y2 - y1) * f1 + oy)
                    lights.quad(
                        to_ursina(*pa, z0, origin),
                        to_ursina(*pa, z0 + 1.2, origin),
                        to_ursina(*pb, z0 + 1.2, origin),
                        to_ursina(*pb, z0, origin),
                        [c * k for c in rgb],
                    )
        b.polygon(area.outline, h, [c * tint for c in ROOF], origin)
    return b, lights


def build_decor(scene: Scene, origin: Point2) -> tuple[Builder, Builder]:
    """Decorative street lamps on the main roads and trees in the real parks; and lamp glows."""
    solid, glow = Builder(), Builder()
    for lanes in lanes_by_edge(scene).values():
        if lanes[0].road_class not in MAIN_ROADS:
            continue
        side = strip(lanes[0].points, 0, -lanes[0].width_m / 2 - 1.2)[1]
        for (x, y, z), heading in along(side, 45.0, 12.0):
            base = to_ursina(x, y, z, origin)
            solid.box((base[0], base[1] + 5.0, base[2]), (0.25, 10.0, 0.25), heading, CONCRETE)
            solid.box((base[0], base[1] + 9.9, base[2]), (0.7, 0.25, 0.7), heading, LAMP)
            r = 4.5
            ring = [
                (base[0] + r * math.cos(a), base[1] + 0.14, base[2] + r * math.sin(a))
                for a in (i * math.pi / 6 for i in range(13))
            ]
            for p, q in zip(ring, ring[1:], strict=False):
                glow.tri((base[0], base[1] + 0.14, base[2]), p, q, LAMP, 0.05)
    for area in scene.areas:
        if area.kind != "park":
            continue
        for x, y in scatter(area.outline, 16.0, f"{area.outline[0]}"):
            p = to_ursina(x, y, 0, origin)
            solid.box((p[0], 1.5, p[2]), (0.5, 3.0, 0.5), 0, TRUNK)
            solid.box((p[0], 5.0, p[2]), (3.6, 4.2, 3.6), 45, TREE)
    return solid, glow


def zone_overlay(
    scene_lanes: dict[str, list[Lane]], edges: Iterable[str], origin: Point2
) -> Builder:
    """Translucent cyan over the road edges a directional geocast covers."""
    b = Builder()
    for edge_id in edges:
        for lane in scene_lanes.get(edge_id, []):
            left, right = strip(lane.points, lane.width_m / 2, -lane.width_m / 2)
            b.band(_u(left, origin, 0.2), _u(right, origin, 0.2), DECK_GLOW, 0.32)
    return b


def disc(radius: float, rgb: Sequence[float], alpha: float, segments: int = 72) -> Mesh:
    """A flat disc (unit-centred) for geocast and sensor zones."""
    verts = [Vec3(0, 0, 0)] + [
        Vec3(
            radius * math.cos(2 * math.pi * i / segments),
            0,
            radius * math.sin(2 * math.pi * i / segments),
        )
        for i in range(segments + 1)
    ]
    tris = [k for i in range(1, segments + 1) for k in (0, i + 1, i)]
    return Mesh(vertices=verts, triangles=tris, colors=[color.rgba(*rgb, alpha)] * len(verts))


def ring(
    radius: float, width: float, rgb: Sequence[float], alpha: float, segments: int = 96
) -> Mesh:
    verts: list[Vec3] = []
    tris: list[int] = []
    for i in range(segments + 1):
        a = 2 * math.pi * i / segments
        verts += [
            Vec3((radius - width) * math.cos(a), 0, (radius - width) * math.sin(a)),
            Vec3(radius * math.cos(a), 0, radius * math.sin(a)),
        ]
    for i in range(segments):
        j = 2 * i
        tris += [j, j + 1, j + 3, j, j + 3, j + 2]
    return Mesh(vertices=verts, triangles=tris, colors=[color.rgba(*rgb, alpha)] * len(verts))


def car_mesh(length: float = 4.5, width: float = 1.9) -> Mesh:
    """A low-poly car, nose towards +z: white body (tinted per vehicle), dark glass, lights."""
    b = Builder()
    b.box((0, 0.55, 0), (width, 0.8, length), 0, (0.92, 0.93, 0.95))
    b.box((0, 1.25, -0.25), (width * 0.86, 0.6, length * 0.5), 0, (0.12, 0.14, 0.18))
    for sx in (-0.6, 0.6):
        b.box((sx, 0.6, length / 2), (0.45, 0.18, 0.06), 0, (1.0, 0.97, 0.85))
        b.box((sx, 0.65, -length / 2), (0.45, 0.16, 0.06), 0, (1.0, 0.12, 0.1))
    b.flush()
    if len(b.meshes) != 1:
        raise AssertionError("car mesh should fit one chunk")
    return b.meshes[0]


def beam(height: float = 1.0, width: float = 1.0, segments: int = 6) -> Mesh:
    """A vertical light shaft (crossed quads), bright at the base and fading upwards.
    White, so it can be tinted; draw it additively."""
    verts: list[Vec3] = []
    cols: list[color.Color] = []
    tris: list[int] = []
    for k in range(2):
        dx, dz = (width / 2, 0.0) if k == 0 else (0.0, width / 2)
        for s in range(segments):
            y0, y1 = height * s / segments, height * (s + 1) / segments
            a0, a1 = (1 - s / segments) ** 2, (1 - (s + 1) / segments) ** 2
            i = len(verts)
            verts += [Vec3(-dx, y0, -dz), Vec3(-dx, y1, -dz), Vec3(dx, y1, dz), Vec3(dx, y0, dz)]
            cols += [
                color.rgba(1, 1, 1, a0),
                color.rgba(1, 1, 1, a1),
                color.rgba(1, 1, 1, a1),
                color.rgba(1, 1, 1, a0),
            ]
            tris += [i, i + 1, i + 2, i, i + 2, i + 3]
    return Mesh(vertices=verts, triangles=tris, colors=cols)
