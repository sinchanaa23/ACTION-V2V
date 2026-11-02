# ruff: noqa: E501 - GLSL source and long colour tuples
"""Sky, distance haze and glow for the 3D simulation. Owner: P4.

A fog shader (unlit vertex colour, faded into the horizon colour with
distance), a sky dome with a gradient, stars and the warm glow of central
Bengaluru to the south, a ring of low hills on the horizon, and additive
blending for light. Hills and stars are scenery beyond the mapped area; the
corridor itself is unchanged real OpenStreetMap data.
"""

import math
import random

from panda3d.core import ColorBlendAttrib, TransparencyAttrib
from ursina import Entity, Mesh, Shader, Vec3, color

HORIZON = (0.065, 0.1, 0.145)
ZENITH = (0.006, 0.01, 0.024)
MID_SKY = (0.018, 0.03, 0.058)
CITY_GLOW = (0.32, 0.17, 0.08)
HILL_BASE = (0.03, 0.042, 0.062)
HILL_TOP = (0.045, 0.062, 0.088)
FOG_RANGE = (1400.0, 7600.0)

fog_shader = Shader(
    name="v2v_fog",
    language=Shader.GLSL,
    vertex="""
#version 140
uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat4 p3d_ModelViewMatrix;
in vec4 p3d_Vertex;
in vec4 p3d_Color;
in vec2 p3d_MultiTexCoord0;
out vec4 vcolor;
out vec2 uv;
out float depth;
void main() {
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
    vcolor = p3d_Color;
    uv = p3d_MultiTexCoord0;
    depth = length((p3d_ModelViewMatrix * p3d_Vertex).xyz);
}
""",
    fragment="""
#version 140
uniform sampler2D p3d_Texture0;
uniform vec4 p3d_ColorScale;
uniform vec3 fog_color;
uniform vec2 fog_range;
in vec4 vcolor;
in vec2 uv;
in float depth;
out vec4 fragColor;
void main() {
    vec4 c = texture(p3d_Texture0, uv) * vcolor * p3d_ColorScale;
    float f = clamp((depth - fog_range.x) / (fog_range.y - fog_range.x), 0.0, 1.0);
    f = f * f * (3.0 - 2.0 * f);
    c.rgb = mix(c.rgb, fog_color, f * 0.92);
    fragColor = c;
}
""",
    default_input={"fog_color": Vec3(*HORIZON), "fog_range": (FOG_RANGE[0], FOG_RANGE[1])},
)


def additive(e: Entity) -> Entity:
    """Draw as light: brightens what is behind it, never hides it."""
    e.setTransparency(TransparencyAttrib.M_alpha)
    e.setAttrib(
        ColorBlendAttrib.make(
            ColorBlendAttrib.M_add, ColorBlendAttrib.O_incoming_alpha, ColorBlendAttrib.O_one
        )
    )
    e.setDepthWrite(False)
    e.setDepthOffset(2)  # sits on the surface it lights without flickering through it
    e.setBin("transparent", 20)
    return e


def _mix(a: tuple[float, ...], b: tuple[float, ...], t: float) -> tuple[float, ...]:
    return tuple(x + (y - x) * t for x, y in zip(a, b, strict=True))


def sky_colour(elevation: float, azimuth: float) -> tuple[float, ...]:
    """Sky at an elevation (0 horizon .. 90 zenith) and azimuth (degrees from north)."""
    e = max(0.0, min(1.0, elevation / 90.0))
    base = (
        _mix(HORIZON, MID_SKY, min(1.0, e / 0.22))
        if e < 0.22
        else _mix(MID_SKY, ZENITH, (e - 0.22) / 0.78)
    )
    south = max(0.0, math.cos(math.radians(azimuth - 180.0))) ** 3  # the city lies south of Hebbal
    glow = south * max(0.0, 1.0 - e / 0.16) ** 2 * 0.55
    return _mix(base, CITY_GLOW, glow)


def sky_dome(radius: float = 9000.0, rings: int = 14, segments: int = 64, stars: int = 420) -> Mesh:
    verts: list[Vec3] = []
    cols: list[color.Color] = []
    tris: list[int] = []
    elevations = [-8.0] + [90.0 * (i / (rings - 1)) ** 1.6 for i in range(rings)]
    for el in elevations:
        for s in range(segments + 1):
            az = 360.0 * s / segments
            r = radius * math.cos(math.radians(el))
            verts.append(
                Vec3(
                    r * math.sin(math.radians(az)),
                    radius * math.sin(math.radians(el)),
                    r * math.cos(math.radians(az)),
                )
            )
            cols.append(color.rgba(*sky_colour(max(el, 0.0), az), 1))
    row = segments + 1
    for i in range(len(elevations) - 1):
        for s in range(segments):
            a, b = i * row + s, (i + 1) * row + s
            tris += [a, a + 1, b + 1, a, b + 1, b]
    rng = random.Random(11)
    for _ in range(stars):
        el = math.degrees(math.asin(rng.uniform(0.18, 0.995)))
        az = rng.uniform(0, 360)
        size = rng.uniform(5.0, 13.0)
        bright = rng.uniform(0.25, 0.85) * min(1.0, (el - 10) / 30)
        r = radius * 0.98
        c = Vec3(
            r * math.cos(math.radians(el)) * math.sin(math.radians(az)),
            r * math.sin(math.radians(el)),
            r * math.cos(math.radians(el)) * math.cos(math.radians(az)),
        )
        u = Vec3(math.cos(math.radians(az)), 0, -math.sin(math.radians(az))) * size
        v = c.normalized().cross(u).normalized() * size  # the quad faces the centre
        i = len(verts)
        verts += [c - u - v, c - u + v, c + u + v, c + u - v]
        cols += [color.rgba(0.75, 0.85, 1.0, bright)] * 4
        tris += [i, i + 1, i + 2, i, i + 2, i + 3]
    return Mesh(vertices=verts, triangles=tris, colors=cols, static=True)


def hills(inner: float, outer: float, seed: int = 3, segments: int = 160) -> Mesh:
    """A ring of low, layered hill silhouettes on the horizon (scenery, not map data)."""
    rng = random.Random(seed)
    waves = [
        (rng.uniform(2, 9), rng.uniform(0, 2 * math.pi), rng.uniform(25, 70)) for _ in range(6)
    ]
    verts: list[Vec3] = []
    cols: list[color.Color] = []
    tris: list[int] = []
    for band, (r0, r1, lift) in enumerate(((inner, inner + 500, 1.0), (inner + 700, outer, 1.7))):
        start = len(verts)
        for s in range(segments + 1):
            a = 2 * math.pi * s / segments
            h = lift * (60 + sum(amp * (1 + math.sin(f * a + p)) for f, p, amp in waves))
            sa, ca = math.sin(a), math.cos(a)
            verts += [Vec3(r0 * sa, -2, r0 * ca), Vec3(r1 * sa, h, r1 * ca)]
            shade = 1.0 - 0.25 * band
            cols += [
                color.rgba(*(c * shade for c in HILL_BASE), 1),
                color.rgba(*(c * shade for c in HILL_TOP), 1),
            ]
        for s in range(segments):
            j = start + 2 * s
            tris += [j, j + 1, j + 3, j, j + 3, j + 2]
    return Mesh(vertices=verts, triangles=tris, colors=cols, static=True)
