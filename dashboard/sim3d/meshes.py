# ruff: noqa: E501 - long geometry and HUD expressions
"""Turn the real corridor (``v2v.sim.world3d.Scene``) into a few big meshes. Owner: P4.

Everything static is merged into a handful of vertex-coloured meshes with the
lighting baked into the colours, drawn with Ursina's unlit shader: cheap
enough for integrated graphics. Roads, flyovers, junctions, buildings, water
and parks are real (SUMO network + OpenStreetMap). Street lamps along the
main roads and trees inside the real parks are decorative placements, since
the public OSM server cannot serve those nodes for the snapshot date.
"""

import math
import random
from collections import defaultdict
from collections.abc import Iterable, Sequence

from geom import Point2, Point3, along, ccw, dashes, scatter, strip, to_ursina
from panda3d.core import TransparencyAttrib, Triangulator
from ursina import Entity, Mesh, Vec3, color
from ursina.shaders import unlit_shader

from v2v.sim.world3d import Lane, Scene

MAX_VERTS = 60_000
MAIN_ROADS = {"motorway", "trunk", "primary", "trunk_link", "primary_link", "motorway_link"}
LIGHT = (-0.55, 0.8)
"""Direction the moonlight comes from (x, y), for baked wall shading."""

ASPHALT_MAIN = (0.105, 0.115, 0.14)
ASPHALT = (0.085, 0.09, 0.105)
JUNCTION = (0.095, 0.1, 0.12)
MARKING = (0.3, 0.33, 0.37)
EDGE_LINE = (0.22, 0.25, 0.3)
DECK_GLOW = (0.0, 0.85, 1.0)
CONCRETE = (0.16, 0.17, 0.19)
WALL = (0.085, 0.1, 0.13)
ROOF = (0.12, 0.14, 0.18)
WATER = (0.02, 0.07, 0.13)
PARK = (0.035, 0.07, 0.05)
GROUND = (0.03, 0.034, 0.042)
GRID = (0.0, 0.32, 0.42)
TREE = (0.05, 0.16, 0.1)
TRUNK = (0.12, 0.09, 0.07)
LAMP = (1.0, 0.62, 0.25)


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
            Entity(model=m, shader=unlit_shader, double_sided=True, **kwargs)  # type: ignore[arg-type]
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
    """Asphalt, junctions, markings and flyover structure; and the neon deck edges."""
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
        outer_right = strip(lanes[0].points, lanes[0].width_m / 2, -lanes[0].width_m / 2)[1]
        outer_left = strip(lanes[-1].points, lanes[-1].width_m / 2, -lanes[-1].width_m / 2)[0]
        for side in (outer_right, outer_left):
            a, b = strip(side, 0.07, -0.07)
            solid.band(_u(a, origin, 0.1), _u(b, origin, 0.1), EDGE_LINE)
            if raised:  # parapet with a neon strip along the flyover deck
                top = [(x, y, z + 1.0) for x, y, z in side]
                solid.band(_u(side, origin), _u(top, origin), CONCRETE)
                ga, gb = strip(top, 0.12, -0.12)
                glow.band(_u(ga, origin, 0.02), _u(gb, origin, 0.02), DECK_GLOW)
        if raised:
            for (x, y, z), heading in along(lanes[0].points, 28.0, 10.0):
                if z > 2.5:
                    w = sum(lane.width_m for lane in lanes) * 0.6
                    solid.box(
                        to_ursina(x, y, (z - 0.4) / 2, origin), (w, z - 0.4, 1.6), heading, CONCRETE
                    )
        del edge_id
    return solid, glow


def build_city(scene: Scene, origin: Point2) -> Builder:
    """Ground, water, parks and extruded buildings with baked moonlight."""
    b = Builder()
    xmin, ymin, xmax, ymax = scene.bounds
    pad = 400.0
    g = [
        (xmin - pad, ymin - pad),
        (xmax + pad, ymin - pad),
        (xmax + pad, ymax + pad),
        (xmin - pad, ymax + pad),
    ]
    b.polygon(g, -0.05, GROUND, origin)
    step = 100.0
    x = math.floor((xmin - pad) / step) * step
    while x <= xmax + pad:
        a = to_ursina(x, ymin - pad, 0.0, origin)
        c = to_ursina(x, ymax + pad, 0.0, origin)
        b.quad(
            (a[0] - 0.25, 0.0, a[2]),
            (c[0] - 0.25, 0.0, c[2]),
            (c[0] + 0.25, 0.0, c[2]),
            (a[0] + 0.25, 0.0, a[2]),
            GRID,
            0.35,
        )
        x += step
    y = math.floor((ymin - pad) / step) * step
    while y <= ymax + pad:
        a = to_ursina(xmin - pad, y, 0.0, origin)
        c = to_ursina(xmax + pad, y, 0.0, origin)
        b.quad(
            (a[0], 0.0, a[2] - 0.25),
            (c[0], 0.0, c[2] - 0.25),
            (c[0], 0.0, c[2] + 0.25),
            (a[0], 0.0, a[2] + 0.25),
            GRID,
            0.35,
        )
        y += step
    for area in scene.areas:
        if area.kind == "water":
            b.polygon(area.outline, 0.02, WATER, origin)
        elif area.kind == "park":
            b.polygon(area.outline, 0.01, PARK, origin)
    rng = random.Random(7)
    lx, ly = LIGHT
    for area in scene.areas:
        if area.kind != "building":
            continue
        h = area.height_m
        tint = rng.uniform(0.85, 1.15)
        poly = ccw(area.outline)
        for (x1, y1), (x2, y2) in zip(poly, [*poly[1:], poly[0]], strict=True):
            nx, ny = (y2 - y1), -(x2 - x1)  # outward normal of a CCW outline
            n = math.hypot(nx, ny) or 1.0
            shade = 0.55 + 0.45 * max(0.0, (nx * lx + ny * ly) / n)
            rgb = [c * shade * tint for c in WALL]
            b.quad(
                to_ursina(x1, y1, 0, origin),
                to_ursina(x1, y1, h, origin),
                to_ursina(x2, y2, h, origin),
                to_ursina(x2, y2, 0, origin),
                rgb,
            )
        b.polygon(area.outline, h, [c * tint for c in ROOF], origin)
    return b


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
