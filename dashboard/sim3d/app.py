# ruff: noqa: E501, E402 - long HUD strings; imports follow the sys.path setup
"""A.C.T.I.O.N V2V: interactive 3D hazard-warning simulation. Owner: P4.

    python -m v2v.sim.world3d --corridor hebbal         # once: buildings, water, parks
    python dashboard/sim3d/app.py

The real Hebbal corridor in 3D (SUMO roads and flyovers, OSM buildings, Hebbal
lake) with live SUMO traffic. Reports fire when the scenario says, run through
the real pipeline (P2's reading, P3's confidence, P4's geocast and Nakagami
delivery), and the view shows what happened: the geocast zone, every V2V hop
in slow motion along the actual relay tree, and who was warned, missed or
warned needlessly against SUMO's ground truth. Report text is fixture text
until P1's real test split exists, and the card says so.

It opens as a scroll-driven tour: the wheel (or the arrow keys) glides the
camera through a few chapters of the corridor. Dragging switches to free
exploring; T returns to the tour. Click a vehicle or the hazard for details.
Keys: space play/pause, N step, R reset, K skip to the event, F follow,
G ground truth, C controls, Tab hide the interface, 1-7 speed, [ ] scenario,
Esc deselect, F11 full screen.
"""

import argparse
import math
import os
import sys
import time
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling modules

REPO = Path(__file__).resolve().parents[2]
os.chdir(REPO)  # scenario files use repo-relative paths

from atmosphere import FOG_RANGE, HORIZON, additive, fog_shader, hills, sky_dome
from geom import lerp_angle, slow_motion, to_ursina, type_into
from hud import (
    AMBER,
    BLUE,
    CYAN,
    GREEN,
    MUTED,
    ORANGE,
    RED,
    WHITE,
    Hud,
    faint,
    muted,
    spaced,
    tint,
    wrap,
)
from meshes import (
    beam,
    build_city,
    build_decor,
    build_roads,
    car_mesh,
    disc,
    lanes_by_edge,
    ring,
    translucent,
    zone_overlay,
)
from tour import Chapter, Scroll, Shot, blend, chapters, parallax, pose_at, principal_axis
from ursina import (
    BoxCollider,
    Entity,
    Mesh,
    Text,
    Ursina,
    Vec3,
    camera,
    color,
    destroy,
    held_keys,
    mouse,
    scene,
    window,
)
from ursina import time as utime
from ursina.shaders import unlit_shader
from voice_proc import VoiceHandle
from worker import FiredEvent, SimWorker, Snapshot

from v2v.experiments import CONFIGS
from v2v.sim.delivery import DEFAULT_MODEL
from v2v.sim.driver import verdict
from v2v.sim.live import ORIGIN, WarningEvent, scenario_specs
from v2v.sim.osm_network import CORRIDORS
from v2v.sim.world3d import Scene, build_scene, context_paths
from v2v.voice.listener import models_ready

CORRIDOR = CORRIDORS["hebbal"]
NET_DIR = REPO / "sim_out" / "networks" / CORRIDOR.name
SCENARIO_DIR = REPO / "sim_out" / "scenarios" / CORRIDOR.name
SPEEDS = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
CONFIG_NAMES = tuple(CONFIGS)
PAINTS = (
    (0.86, 0.87, 0.89),
    (0.86, 0.87, 0.89),
    (0.55, 0.58, 0.62),
    (0.55, 0.58, 0.62),
    (0.2, 0.21, 0.24),
    (0.1, 0.1, 0.11),
    (0.16, 0.3, 0.3),
    (0.14, 0.24, 0.44),
    (0.62, 0.57, 0.47),
)
"""Ordinary car colours, picked per vehicle id so each car keeps its colour."""
SKY = color.rgba(*HORIZON, 1)


def rgba(rgb: tuple[float, float, float], a: float = 1.0) -> color.Color:
    return color.rgba(*rgb, a)


def instance(
    mesh: Mesh, tint_to: color.Color | None = None, shader: object = fog_shader, **kwargs: object
) -> Entity:
    """An entity drawing a shared mesh. Assigning one Mesh as ``model`` to many
    entities would reparent it each time; instancing shares it instead."""
    e = Entity(shader=shader, **kwargs)  # type: ignore[arg-type]
    mesh.instanceTo(e)
    if tint_to is not None:
        paint(e, tint_to)
    return e


def paint(e: Entity, c: color.Color) -> None:
    """Tint an instanced entity (Ursina's ``color`` only tints a ``model``)."""
    e.setColorScale(c)


def cardinal(h: float) -> str:
    return ("N", "NE", "E", "SE", "S", "SW", "W", "NW")[round(h / 45) % 8]


def paint_for(vid: str) -> tuple[float, float, float]:
    return PAINTS[zlib.crc32(vid.encode()) % len(PAINTS)]


# --- camera ------------------------------------------------------------------------------


class CameraRig(Entity):
    """The scroll-driven tour, or free orbit / pan / zoom / chase when exploring.

    Tour: the wheel moves ``scroll``; the pose glides along the chapter path,
    with a little mouse parallax. Explore: right-drag turns, middle-drag or
    WASD pans, the wheel zooms, F chases a vehicle."""

    def __init__(self, shots: list[Shot]) -> None:
        super().__init__()
        camera.parent = self
        camera.position = (0, 0, -600)
        camera.rotation = (0, 0, 0)
        camera.clip_plane_near = 1.5
        camera.clip_plane_far = 12000
        self.scroll = Scroll(len(shots))
        self.shots = shots
        self.mode = "tour"
        self.pose = shots[0]
        self.free = shots[0]
        self.follow: Entity | None = None
        self.mx = self.my = 0.0
        self.scrolled_at = -1e9
        self.locked = False
        """Typing a report: the camera keys must not move the view."""
        self.on_mode: object = None
        self.apply(self.pose)

    @property
    def shots(self) -> list[Shot]:
        return self._shots

    @shots.setter
    def shots(self, shots: list[Shot]) -> None:
        self._shots = shots
        self.scroll.count = len(shots)
        self.scroll.go(self.scroll.target)

    @property
    def dist(self) -> float:
        return self.pose.dist

    def explore(self) -> None:
        if self.mode != "explore":
            self.mode = "explore"
            self.free = self.pose
            if callable(self.on_mode):
                self.on_mode()

    def tour(self) -> None:
        self.mode = "tour"
        self.follow = None
        if callable(self.on_mode):
            self.on_mode()

    def fly_to(self, focus: Vec3, dist: float, pitch: float | None = None) -> None:
        self.explore()
        self.free = Shot(
            (focus.x, focus.y, focus.z),
            self.free.yaw,
            pitch if pitch is not None else self.free.pitch,
            dist,
            self.free.fov,
        )

    def apply(self, p: Shot) -> None:
        self.position = Vec3(*p.focus)
        self.rotation = (p.pitch, p.yaw, 0)
        camera.z = -p.dist
        camera.fov = p.fov

    def update(self) -> None:
        dt = utime.dt
        a = window.aspect_ratio
        k = 1.0 - math.exp(-dt * 2.5)
        self.mx += (max(-0.5, min(0.5, mouse.x / a)) - self.mx) * k
        self.my += (max(-0.5, min(0.5, mouse.y)) - self.my) * k
        progress = self.scroll.update(dt)
        dragging = mouse.right or mouse.middle
        keys = {k: 0 if self.locked else held_keys[k] for k in "wasdqe"}
        if (dragging or any(keys.values())) and self.mode == "tour":
            self.explore()
        if self.mode == "tour":
            breathe = math.sin(time.perf_counter() * 0.25) * 0.6  # the camera is held, not fixed
            target = parallax(pose_at(self.shots, progress), self.mx, self.my)
            target = Shot(target.focus, target.yaw + breathe, target.pitch, target.dist, target.fov)
            self.pose = blend(self.pose, target, 1.0 - math.exp(-dt * 5.0))
        else:
            f = self.free
            yaw, pitch, dist = f.yaw, f.pitch, f.dist
            focus = Vec3(*f.focus)
            if mouse.right:
                yaw += mouse.velocity[0] * 220
                pitch = min(86.0, max(4.0, pitch - mouse.velocity[1] * 140))
            yaw += (keys["e"] - keys["q"]) * 70 * dt
            fwd = Vec3(math.sin(math.radians(yaw)), 0, math.cos(math.radians(yaw)))
            right = Vec3(fwd.z, 0, -fwd.x)
            move = fwd * (keys["w"] - keys["s"]) + right * (keys["d"] - keys["a"])
            if mouse.middle:
                move -= right * mouse.velocity[0] * 60 + fwd * mouse.velocity[1] * 60
            if self.follow is not None:
                t = self.follow.world_position
                focus = Vec3(t.x, t.y + 2, t.z)
                if not mouse.right:  # swing in behind the vehicle
                    yaw = lerp_angle(yaw, self.follow.rotation_y, min(1.0, dt * 1.5))
            elif move.length() > 0:
                focus += move * dist * 0.9 * dt
            self.free = Shot((focus.x, focus.y, focus.z), yaw % 360, pitch, dist, 50.0)
            self.pose = blend(
                self.pose,
                self.free,
                1.0 - math.exp(-dt * (6.0 if self.follow is not None else 9.0)),
            )
        self.apply(self.pose)

    def input(self, key: str) -> None:
        notches = {"scroll down": 1, "scroll up": -1}.get(key)
        if notches is None:
            return
        if self.mode == "tour":
            self.scroll.wheel(notches)
            self.scrolled_at = time.perf_counter()
        else:
            f = self.free
            dist = max(12.0, min(4000.0, f.dist * (1.16 if notches > 0 else 0.86)))
            self.free = Shot(f.focus, f.yaw, f.pitch, dist, f.fov)


# --- vehicles and signals ----------------------------------------------------------------


class Car:
    def __init__(self, view: "SimView", vid: str, model: Mesh, blocking: bool = False) -> None:
        self.vid = vid
        self.blocking = blocking
        self.paint = RED if blocking else paint_for(vid)
        self.body = instance(model, tint_to=rgba(self.paint), double_sided=True)
        self.body.collider = BoxCollider(
            self.body, center=Vec3(0, 0.8, 0), size=Vec3(2.6, 2.2, 5.4)
        )
        self.body.on_click = lambda: view.select(("vehicle", vid))
        # not children of the body: a child's tint would be multiplied by the car's paint
        self.marker = additive(instance(view.beam_mesh, shader=unlit_shader))
        self.marker.enabled = False
        self.halo = translucent(instance(view.halo_mesh, shader=unlit_shader))
        self.halo.enabled = False
        self.status: tuple[float, float, float] | None = None
        self.target = Vec3(0, 0, 0)
        self.heading = 0.0
        self.fresh = True

    def place(self, p: Vec3, heading: float) -> None:
        self.target = p
        self.heading = heading
        if self.fresh:
            self.body.position = p
            self.body.rotation_y = heading
            self.fresh = False

    def mark(self, rgb: tuple[float, float, float] | None, selected: bool) -> None:
        """Ordinary cars keep their paint; a car the story is about glows."""
        if rgb == self.status and not selected and not self.marker.enabled:
            return
        self.status = rgb
        paint(self.body, rgba(rgb if rgb else self.paint))
        lit = rgb is not None or selected
        self.marker.enabled = self.halo.enabled = lit
        if lit:
            c = rgb or WHITE
            paint(self.marker, rgba(c, 0.55 if selected else 0.4))
            paint(self.halo, rgba(c, 0.5))

    def update(self, dt: float, size: float, beam_h: float) -> None:
        k = min(1.0, dt * 7)
        self.body.position = self.body.position + (self.target - self.body.position) * k
        self.body.rotation_y = lerp_angle(self.body.rotation_y, self.heading, k)
        self.body.scale = size
        if self.halo.enabled:
            self.halo.position = self.body.position + Vec3(0, 0.15, 0)
            self.halo.scale = size
        if self.marker.enabled:
            self.marker.position = self.body.position
            self.marker.scale = (max(0.6, beam_h * 0.05), beam_h, max(0.6, beam_h * 0.05))

    def destroy(self) -> None:
        destroy(self.body)
        destroy(self.marker)
        destroy(self.halo)


class PulseFx:
    """One V2V hop in slow motion: a small light travelling a soft arc, leaving a fading trace."""

    SEGMENTS = 14

    def __init__(
        self, a: Vec3, b: Vec3, t0: float, t1: float, rgb: tuple[float, float, float], on_arrive
    ) -> None:  # type: ignore[no-untyped-def]
        self.a, self.b, self.t0, self.t1, self.rgb = a, b, t0, t1, rgb
        self.on_arrive = on_arrive
        self.lift = min(18.0, 2.0 + (b - a).length() * 0.08)
        self.dot = additive(
            Entity(
                model="sphere",
                shader=unlit_shader,
                color=rgba(rgb, 1),
                scale=1.8,
                position=a,
                enabled=False,
            )
        )
        self.trail = additive(
            Entity(
                model=Mesh(
                    vertices=[a, a], colors=[rgba(rgb, 0), rgba(rgb, 0)], mode="line", thickness=2
                ),
                shader=unlit_shader,
                enabled=False,
            )
        )
        self.done = False

    def at(self, f: float) -> Vec3:
        return self.a + (self.b - self.a) * f + Vec3(0, math.sin(f * math.pi) * self.lift, 0)

    def update(self, now: float) -> None:
        if now < self.t0:
            return
        f = min(1.0, (now - self.t0) / max(self.t1 - self.t0, 1e-3))
        self.dot.enabled = self.trail.enabled = not self.done
        self.dot.position = self.at(f)
        n = self.SEGMENTS
        pts = [self.at(f * i / n) for i in range(n + 1)]
        self.trail.model.vertices = pts
        self.trail.model.colors = [rgba(self.rgb, 0.55 * (i / n) ** 1.5) for i in range(n + 1)]
        self.trail.model.generate()
        if f >= 1.0 and not self.done:
            self.done = True
            self.on_arrive()

    def fade(self, now: float) -> bool:
        """True once it has faded out and can be removed."""
        if not self.done:
            return False
        age = now - self.t1
        self.dot.enabled = False
        k = max(0.0, 1.0 - age / 1.2)
        self.trail.enabled = k > 0
        paint(self.trail, color.rgba(1, 1, 1, k))
        return age > 1.2

    def destroy(self) -> None:
        destroy(self.dot)
        destroy(self.trail)


class Wave:
    """An expanding ring of light (radio range, arrival flash)."""

    def __init__(
        self,
        view: "SimView",
        p: Vec3,
        radius: float,
        rgb: tuple[float, float, float],
        duration: float,
        strength: float = 0.7,
    ) -> None:
        self.e = additive(
            instance(
                view.unit_ring,
                tint_to=rgba(rgb, strength),
                shader=unlit_shader,
                position=p + Vec3(0, 0.6, 0),
                scale=0.01,
            )
        )
        self.radius, self.rgb, self.duration, self.strength = radius, rgb, duration, strength
        self.start = time.perf_counter()

    def update(self, now: float) -> bool:
        f = (now - self.start) / self.duration
        if f >= 1:
            destroy(self.e)
            return True
        ease = 1 - (1 - f) ** 3
        self.e.scale = max(0.01, self.radius * ease)
        paint(self.e, rgba(self.rgb, self.strength * (1 - f) ** 1.5))
        return False


# --- the simulation view -----------------------------------------------------------------


class SimView(Entity):
    def __init__(
        self, worker: SimWorker, world: Scene, origin: tuple[float, float], hud: Hud, rig: CameraRig
    ) -> None:
        super().__init__()
        self.worker, self.world, self.origin, self.hud, self.rig = worker, world, origin, hud, rig
        self.lanes = lanes_by_edge(world)
        self.car_model = car_mesh()
        self.beam_mesh = beam()
        self.halo_mesh = ring(3.0, 0.5, (1, 1, 1), 1.0, 40)
        self.unit_ring = ring(1.0, 0.03, (1, 1, 1), 1.0, 128)
        self.cars: dict[str, Car] = {}
        self.version = -1
        self.snap: Snapshot | None = None
        self.selected: tuple[str, ...] | None = None
        self.hazard_tab = "details"
        self.voice: VoiceHandle | None = None
        self.typing = False
        self.typed = ""
        self.voice_override: tuple[str, str, float] | None = None
        """Screenshots only: show the voice strip in a given state."""
        self.ground_truth = False
        self.warned_needed: set[str] = set()
        self.warned_unneeded: set[str] = set()
        self.missed: set[str] = set()
        self.reporters: set[str] = set()
        self.pulses: list[PulseFx] = []
        self.waves: list[Wave] = []
        self.zone_entities: list[Entity] = []
        self.bubble = Text(
            " ",
            parent=scene,
            billboard=True,
            scale=60,
            origin=(0, 0),
            color=rgba(ORANGE),
            enabled=False,
        )
        self.bubble_until = 0.0
        self.tag = Text(
            " ", parent=scene, billboard=True, scale=40, origin=(0, -0.5), enabled=False
        )
        self._hud_at = 0.0
        self._fps = 60.0
        self._slow_since: float | None = None
        self._build_hazard()
        self.load_scenario()

    # --- coordinates -----------------------------------------------------------------

    def u(self, x: float, y: float, z: float = 0.0) -> Vec3:
        return Vec3(*to_ursina(x, y, z, self.origin))

    def at_vehicle(self, v) -> Vec3:  # type: ignore[no-untyped-def]
        return self.u(v.x, v.y, self.world.elevation(v.edge_id, v.lane_pos_m))

    # --- the tour ------------------------------------------------------------------------

    def tour_chapters(self) -> list[Chapter]:
        spec, world = self.worker.spec, self.world
        h = spec.hazard
        hazard = self.u(h.x, h.y, world.elevation(h.edge_id, h.pos_m))
        main = [
            p
            for lanes in self.lanes.values()
            if lanes[0].road_class in ("trunk", "motorway", "primary")
            for p in lanes[0].points
        ]
        axis = principal_axis([(x, y) for x, y, _ in main]) if main else 0.0
        if main:  # along the highway, a little way from the hazard
            ax = math.sin(math.radians(axis)), math.cos(math.radians(axis))
            want = (h.x - ax[0] * 420, h.y - ax[1] * 420)
            x, y, z = min(main, key=lambda p: math.dist(p[:2], want))
            road_focus = self.u(x, y, z)
        else:
            road_focus = hazard
        raised = [
            lanes for lanes in self.lanes.values() if max(p[2] for p in lanes[0].points) > 1.0
        ]
        decks = [p for lanes in raised for p in lanes[0].points if p[2] > 3.0]
        deck = self.u(*min(decks, key=lambda p: math.dist(p[:2], (h.x, h.y)))) if decks else None
        xmin, ymin, xmax, ymax = world.bounds
        cfg = CONFIGS[self.worker.config_name]
        return chapters(
            (xmax - xmin, ymax - ymin),
            axis,
            (road_focus.x, road_focus.y, road_focus.z),
            (deck.x, deck.y, deck.z) if deck is not None else None,
            (hazard.x, hazard.y, hazard.z),
            hazard_name=spec.hazard_class.value.replace("_", " "),
            place=spec.landmark_names[0] if spec.landmark_names else "",
            lanes=len(world.lanes),
            buildings=sum(1 for a in world.areas if a.kind == "building"),
            decks=len(raised),
            distribution=cfg.distribution,
            radius_m=cfg.radius_m,
        )

    # --- hazard and zones --------------------------------------------------------------

    def _build_hazard(self) -> None:
        self.hazard = Entity(
            collider="box", model="cube", scale=(16, 16, 16), color=color.clear, shader=unlit_shader
        )
        self.hazard.on_click = lambda: self.select(("hazard",))
        self.core = Entity(
            model="cube", shader=unlit_shader, scale=2.4, color=rgba(RED), rotation=(45, 0, 45)
        )
        self.column = additive(
            instance(self.beam_mesh, tint_to=rgba(RED, 0.6), shader=unlit_shader, scale=(5, 140, 5))
        )
        self.beacon_ring = additive(Entity(model=ring(12, 0.7, RED, 1.0), shader=unlit_shader))
        self.sensor = translucent(Entity(model=disc(200, RED, 0.05), shader=unlit_shader))
        self.sensor_rim = additive(Entity(model=ring(200, 1.0, RED, 0.5), shader=unlit_shader))
        self.hazard_label = Text(
            " ",
            parent=scene,
            billboard=True,
            scale=70,
            origin=(0, -0.5),
            color=rgba(RED),
            enabled=False,
        )

    def load_scenario(self) -> None:
        spec = self.worker.spec
        h = spec.hazard
        p = self.u(h.x, h.y, self.world.elevation(h.edge_id, h.pos_m))
        self.hazard_pos = p
        for e in (self.hazard, self.beacon_ring, self.sensor, self.sensor_rim, self.column):
            e.position = p + Vec3(0, 0.35, 0)
        self.hazard.y = p.y + 8
        self.core.position = p + Vec3(0, 4, 0)
        for car in self.cars.values():
            car.destroy()
        self.cars.clear()
        for fx in self.pulses:
            fx.destroy()
        self.pulses.clear()
        self.clear_zone()
        self.warned_needed.clear()
        self.warned_unneeded.clear()
        self.missed.clear()
        self.reporters.clear()
        self.rig.follow = None
        self.chapters = self.tour_chapters()
        self.rig.shots = [c.shot for c in self.chapters]
        self.select(None)
        self.version = -1

    def clear_zone(self) -> None:
        for e in self.zone_entities:
            destroy(e)
        self.zone_entities.clear()

    def show_zone(self, event: WarningEvent) -> None:
        self.clear_zone()
        cx, cy = event.centre_xy
        approx = event.result.assessed.location is None or event.result.assessed.location.approx
        rgb = AMBER if approx else CYAN
        centre = self.u(cx, cy, 0.4)
        self.zone_entities += [
            translucent(
                Entity(model=disc(event.radius_m, rgb, 0.03), shader=unlit_shader, position=centre)
            ),
            additive(
                Entity(
                    model=ring(event.radius_m, 1.6, rgb, 0.5),
                    shader=unlit_shader,
                    position=centre + Vec3(0, 0.1, 0),
                )
            ),
        ]
        if event.zone_edges:
            for e in zone_overlay(self.lanes, event.zone_edges, self.origin).entities():
                self.zone_entities.append(additive(e))
        if event.spoken:  # the hazard is wherever the driver said it, so mark it
            column = instance(
                self.beam_mesh,
                tint_to=rgba(RED, 0.6),
                shader=unlit_shader,
                position=centre,
                scale=(5, 110, 5),
            )
            self.zone_entities.append(additive(column))
        self.waves.append(Wave(self, centre, event.radius_m, rgb, 2.6, 0.6))

    # --- events ------------------------------------------------------------------------

    def fire(self, fired: FiredEvent) -> None:
        e = fired.event
        r = e.result
        self.reporters.add(e.slot.reporter_id)
        sender = self.u(*e.sender_xy, 1.5)
        self.bubble.text = spaced("report")
        self.bubble.position = sender + Vec3(0, 14, 0)
        self.bubble.enabled = True
        self.bubble_until = time.perf_counter() + 3.5
        if not e.sent:
            self.toast(
                self.spoken_summary(e)
                if e.spoken
                else tint(AMBER, f"{e.slot.role.capitalize()} report at {e.t_s:.0f} s held back")
                + muted(f"  ·  {r.assessed.status}, no warning sent"),
                8.0 if e.spoken else 6.0,
            )
            return
        self.show_zone(e)
        self.waves.append(Wave(self, sender, DEFAULT_MODEL.comm_range_m, CYAN, 1.8, 0.5))
        now = time.perf_counter() + 0.4
        latest = max((p.end_ms for p in e.pulses), default=1.0)
        k = slow_motion(latest)
        relevant = e.relevant
        for p in e.pulses:
            src = fired.positions.get(p.src) if p.src != ORIGIN else e.centre_xy
            dst = fired.positions.get(p.dst)
            if src is None or dst is None:
                continue
            a, b = self.u(*src, 2.0), self.u(*dst, 2.0)
            car = self.cars.get(p.dst)
            if car is not None:
                b = car.target + Vec3(0, 2.0, 0)
            needed = p.dst in relevant

            def arrive(vid: str = p.dst, needed: bool = needed) -> None:
                (self.warned_needed if needed else self.warned_unneeded).add(vid)
                c = self.cars.get(vid)
                if c is not None:
                    self.waves.append(
                        Wave(self, c.body.position, 8, CYAN if needed else BLUE, 0.8, 0.8)
                    )

            self.pulses.append(
                PulseFx(
                    a, b, now + p.start_ms * k, now + p.end_ms * k, CYAN if needed else BLUE, arrive
                )
            )
        self.missed = set(e.missed)
        if e.spoken:
            self.toast(
                self.spoken_summary(e) + faint(f"  ·  {len(e.delivered)} drivers warned"), 9.0
            )
            # the hazard is where the driver is: fly there, not to the scenario's
            self.rig.fly_to(self.u(*e.centre_xy, 0.0), max(e.radius_m * 0.85, 420.0), 50.0)
            return
        self.toast(
            (tint(RED, "False alarm  ·  ") if e.false_alarm else "")
            + tint(CYAN, f"{r.assessed.status.capitalize()} warning sent")
            + muted(f"  ·  {len(e.delivered)} warned, ")
            + tint(RED, f"{len(e.missed)} missed")
            + faint(f"  ·  radio shown {k * 1000:.0f}× slower")
        )
        if self.rig.mode == "tour" and time.perf_counter() - self.rig.scrolled_at > 3.0:
            self.rig.scroll.go(next(i for i, c in enumerate(self.chapters) if c.key == "geocast"))

    def spoken_summary(self, e: WarningEvent) -> str:
        """What the driver said, and whether it was safe to send."""
        word, why = verdict(e)
        colour = {"Confirmed": GREEN, "Sent as unverified": AMBER, "Blocked": RED}[word]
        return muted(f"“{e.text}”   ") + tint(colour, word) + muted(f"  ·  {why}")

    def voice_vehicle(self) -> str | None:
        """Who speaks: the selected vehicle, else the one nearest the hazard (touring) or
        nearest where the camera looks (exploring)."""
        sel = self.selected
        if sel and sel[0] == "vehicle" and sel[1] in self.cars and not self.cars[sel[1]].blocking:
            return sel[1]
        # touring: where the action is (the hazard); exploring: wherever you are looking
        focus = Vec3(*self.rig.pose.focus) if self.rig.mode == "explore" else self.hazard_pos
        near = min(
            (c for c in self.cars.values() if not c.blocking),
            key=lambda c: (c.body.position - focus).length(),
            default=None,
        )
        return near.vid if near is not None else None

    def submit_voice(self, roman: str) -> None:
        """A spoken report from the speaking vehicle, through the real pipeline."""
        vid = self.voice_vehicle()
        if vid is None:
            self.toast(tint(AMBER, "No vehicle on the road yet to speak as. Press space to play."))
            return
        self.worker.send("voice", (vid, roman))
        self.toast(muted(f"{vid}:  ") + tint(WHITE, f"“{roman}”") + muted("   checking…"), 5.0)

    def refresh_voice(self) -> None:
        if self.voice_override is not None:
            phase, words, level = self.voice_override
            error = None
        elif self.voice is not None:
            v = self.voice
            phase, words, level, error = v.phase, v.words, v.level, v.error
        else:
            phase, words, level, error = "off", "", 0.0, None
        gated = self.voice is not None and self.voice.gated and self.voice_override is None
        vehicle = self.voice_vehicle() if phase != "off" else None
        self.hud.voice(phase, words, level, vehicle, error, gated)
        self.hud.report_box(self.typing, self.typed)

    def begin_typing(self) -> None:
        """Open the box for typing a report (Enter): works with or without a microphone."""
        self.typing, self.typed = True, ""
        self.rig.locked = True

    def end_typing(self, send: bool) -> None:
        text, self.typing, self.typed = self.typed.strip(), False, ""
        self.rig.locked = False
        if send and text:
            self.submit_voice(text)

    def type_key(self, key: str) -> None:
        """One key press while the report box is open."""
        self.typed, end = type_into(self.typed, key)
        if end is not None:
            self.end_typing(end == "send")

    def toast(self, text: str, seconds: float = 6.0) -> None:
        self.hud.toast.text = text
        self.hud.toast_until = time.perf_counter() + seconds

    # --- selection -----------------------------------------------------------------------

    def select(self, what: tuple[str, ...] | None) -> None:
        if what != self.selected and what == ("hazard",):
            self.hazard_tab = "details"
        self.selected = what
        if what is None:
            self.rig.follow = None
        self._hud_at = 0

    def toggle_follow(self) -> None:
        if self.rig.follow is not None:
            self.rig.follow = None
            self.rig.free = Shot(self.rig.free.focus, self.rig.free.yaw, 30.0, 220.0)
            return
        if self.selected and self.selected[0] == "vehicle" and self.selected[1] in self.cars:
            car = self.cars[self.selected[1]]
        else:
            near = self.nearest_vehicles(1)
            if not near:
                return
            car = self.cars[near[0].vehicle_id]
            self.select(("vehicle", car.vid))
        self.rig.explore()
        self.rig.follow = car.body
        f = self.rig.free
        self.rig.free = Shot(f.focus, f.yaw, 14.0, 34.0, f.fov)

    def nearest_vehicles(self, n: int):  # type: ignore[no-untyped-def]
        if self.snap is None:
            return []
        h = self.worker.spec.hazard
        vs = sorted(self.snap.vehicles.values(), key=lambda v: math.dist((v.x, v.y), (h.x, h.y)))
        return [v for v in vs if v.vehicle_id in self.cars][:n]

    # --- per frame -------------------------------------------------------------------------

    def status_colour(self, vid: str) -> tuple[tuple[float, float, float] | None, str]:
        snap = self.snap
        if vid in self.reporters:
            return ORANGE, "reporter"
        if vid in self.warned_needed:
            return CYAN, "warned"
        if vid in self.warned_unneeded:
            return BLUE, "warned, not needed"
        if vid in self.missed and snap is not None and vid in snap.relevant:
            return RED, "missed"
        if self.ground_truth and snap is not None and vid in snap.relevant:
            return RED, "will reach hazard"
        return None, ""

    def update(self) -> None:
        dt = utime.dt
        now = time.perf_counter()
        self._fps = self._fps * 0.95 + (1 / max(dt, 1e-4)) * 0.05
        self._guard_quality(now)
        version, snap = self.worker.latest()
        if version != self.version and snap is not None:
            self.version = version
            self.apply(snap)
        for fired in self.worker.drain():
            self.fire(fired)
        if self.voice is not None:
            for _heard, roman in self.voice.poll():
                self.submit_voice(roman)
        for note in self.worker.notes():
            self.toast(tint(AMBER, note))
        dist = self.rig.dist
        size = min(2.2, max(1.0, dist / 260))  # true size up close, a touch larger from afar
        beam_h = min(150.0, max(6.0, dist * 0.16))  # a hint up close, a beacon from afar
        for car in self.cars.values():
            car.update(dt, size, beam_h)
        words = max(14.0, dist * 0.16)
        self.tag.scale = words
        self.bubble.scale = words * 1.1
        self.hazard_label.scale = words * 2.2
        for fx in list(self.pulses):
            fx.update(now)
            if fx.fade(now):
                fx.destroy()
                self.pulses.remove(fx)
        self.waves = [w for w in self.waves if not w.update(now)]
        active = snap is not None and snap.event_active
        pulse = 0.5 + 0.5 * math.sin(now * 3)
        self.core.rotation_y += 50 * dt
        self.core.enabled = active
        paint(self.column, rgba(RED, 0.25 + 0.3 * pulse) if active else rgba(MUTED, 0.12))
        self.beacon_ring.scale = 1 + 0.25 * pulse if active else 1
        self.beacon_ring.color = rgba(RED, 0.5 + 0.4 * pulse) if active else rgba(MUTED, 0.3)
        hazard_on = self.selected == ("hazard",)
        self.sensor.enabled = self.sensor_rim.enabled = active and hazard_on
        for car in self.cars.values():
            if car.blocking:
                car.halo.enabled = True
                paint(car.halo, rgba(AMBER, 0.6) if int(now * 3) % 2 else rgba(RED, 0.3))
        if self.bubble.enabled and now > self.bubble_until:
            self.bubble.enabled = False
        if self.hud.toast.text.strip() and now > self.hud.toast_until:
            self.hud.toast.text = " "
        self.refresh_voice()
        self.refresh_tour()
        if now - self._hud_at > 0.2:
            self._hud_at = now
            self.refresh_hud()

    def _guard_quality(self, now: float) -> None:
        """If the frame rate stays low, drop the glow filter (the costliest effect)
        rather than let the view crawl, e.g. on integrated graphics while recording."""
        filters = getattr(base, "_v2v_filters", None)  # noqa: F821 - Panda3D's global
        if filters is None:
            return
        if self._fps >= 22.0:
            self._slow_since = None
        elif self._slow_since is None:
            self._slow_since = now
        elif now - self._slow_since > 3.0:
            filters.delBloom()
            base._v2v_filters = None  # noqa: F821
            self.toast(muted("Glow turned off to keep the view smooth on this GPU."), 5.0)

    def apply(self, snap: Snapshot) -> None:
        self.snap = snap
        seen = set()
        sel = self.selected[1] if self.selected and self.selected[0] == "vehicle" else None
        for group, blocking in ((snap.vehicles, False), (snap.blocking, True)):
            for vid, v in group.items():
                seen.add(vid)
                car = self.cars.get(vid)
                if car is None:
                    car = self.cars[vid] = Car(self, vid, self.car_model, blocking)
                car.place(self.at_vehicle(v), v.heading_deg)
                if not blocking:
                    car.mark(self.status_colour(vid)[0], vid == sel)
        for vid in [v for v in self.cars if v not in seen]:
            if self.rig.follow is self.cars[vid].body:
                self.rig.follow = None
            self.cars.pop(vid).destroy()

    # --- interface -------------------------------------------------------------------------

    def refresh_tour(self) -> None:
        rig = self.rig
        progress = rig.scroll.value
        c = self.chapters[round(progress)]
        exploring = rig.mode == "explore"
        self.hud.chapter(
            progress,
            [ch.key for ch in self.chapters],
            f"{round(progress) + 1:02d} / {len(self.chapters):02d}",
            c.title,
            c.caption,
            show_caption=not exploring
            and self.hud.shown
            and not self.hud.dock.open
            and not self.hud.card.visible
            and not self.typing,
        )

    def refresh_hud(self) -> None:
        hud, snap, spec, w = self.hud, self.snap, self.worker.spec, self.worker
        events = w.events()
        t = snap.t_s if snap else 0.0
        live = {
            "running": (GREEN, "live"),
            "paused": (AMBER, "paused"),
            "ready": (AMBER, "ready"),
            "finished": (MUTED, "finished"),
            "fast-forward": (CYAN, "skipping"),
        }.get(w.status, (AMBER, w.status))
        if w.error:
            live = (RED, "SUMO error")
        hud.live_dot.color = rgba(live[0])
        hud.status.text = (
            f"{faint(spaced(live[1]))}   {tint(WHITE, f'{t:.0f} s')}   {faint(f'{w.speed:g}×')}"
        )
        hud.mode_b.text = "Back to tour" if self.rig.mode == "explore" else "Explore freely"
        dock = hud.dock
        dock.play_icon.enabled = not w.playing
        dock.pause_icon.enabled = w.playing
        cfg = CONFIGS[w.config_name]
        dock.time_l.text = f"{tint(WHITE, f'{t:.0f} s')} {faint(f'/ {spec.end_s:.0f}')}   {muted(f'{len(snap.vehicles) if snap else 0} vehicles')}"
        if dock.open:
            dock.scenario_l.text = f"{spec.condition} · {spec.density} traffic · seed {spec.seed}"
            dock.config_l.text = f"{cfg.name}  {faint(cfg.distribution)}"
            dock.highlight(SPEEDS.index(w.speed) if w.speed in SPEEDS else None)
            dock.truth_b.color = (
                rgba(CYAN, 0.3) if self.ground_truth else color.rgba(1, 1, 1, 0.045)
            )
            dock.timeline(
                t,
                spec.end_s,
                (spec.event_start_s, spec.event_end_s) if spec.event_occurred else None,
                [e.t_s for e in events],
            )
            dock.timeline_l.text = faint(
                f"event {spec.event_start_s:.0f}–{spec.event_end_s:.0f} s"
                if spec.event_occurred
                else "no real event in this scenario"
            ) + faint(
                f"   ·   {len(events)} reports, {sum(e.sent for e in events)} warnings   ·   {self._fps:.0f} fps"
            )
        sel = self.selected
        if sel and sel[0] == "vehicle" and sel[1] in self.cars:
            car = self.cars[sel[1]]
            self.tag.enabled = True
            self.tag.text = sel[1]
            self.tag.position = car.body.position + Vec3(0, 6 * car.body.scale_y, 0)
        else:
            self.tag.enabled = False
        active = snap is not None and snap.event_active
        self.hazard_label.enabled = sel == ("hazard",)
        self.hazard_label.text = spec.hazard_class.value.replace("_", " ").upper()
        self.hazard_label.color = rgba(RED) if active else rgba(MUTED)
        self.hazard_label.position = self.hazard_pos + Vec3(0, 30, 0)
        if not hud.shown or sel is None:
            hud.card.hide()
        elif sel[0] == "vehicle":
            self.vehicle_card(sel[1], events)
        else:
            self.hazard_card(events)

    def vehicle_card(self, vid: str, events: list[WarningEvent]) -> None:
        snap, spec = self.snap, self.worker.spec
        v = (snap.vehicles.get(vid) or snap.blocking.get(vid)) if snap else None
        if v is None or snap is None:
            self.hud.card.show(
                ("vehicle", vid),
                faint(spaced("vehicle")),
                vid,
                muted("This vehicle has left the network."),
            )
            return
        rgb, what = self.status_colour(vid)
        role = (
            "hazard vehicle, blocking a lane"
            if vid in snap.blocking
            else "reporter"
            if vid in self.reporters
            else "vehicle"
        )
        height = self.world.elevation(v.edge_id, v.lane_pos_m)
        lines = [
            f"{muted('speed')}  {v.speed_mps * 3.6:.0f} km/h      {muted('heading')}  {v.heading_deg:.0f}° {cardinal(v.heading_deg)}",
            f"{muted('road')}  {v.edge_id}"
            + (muted(f"   ·   on a flyover, {height:.0f} m up") if height > 1 else ""),
            f"{muted('to hazard')}  {math.dist((v.x, v.y), (spec.hazard.x, spec.hazard.y)):.0f} m in a straight line",
            f"{muted('ground truth')}  "
            + (tint(RED, "will reach the hazard") if vid in snap.relevant else "will not reach it"),
        ]
        mine = [e for e in events if e.result.warning and vid in e.result.warning.recipients]
        if mine:
            e = mine[-1]
            got = e.delivered.get(vid)
            if got is not None and e.outcome is not None:
                relay = e.outcome.relays.get(vid, ORIGIN)
                lines.append(
                    tint(
                        rgb or CYAN,
                        f"warned in {got:.1f} ms, hop {e.outcome.hops.get(vid, '?')}, relayed by {relay}",
                    )
                )
            else:
                lines.append(tint(ORANGE, "addressed, but the radio never got it through"))
        elif what:
            lines.append(tint(rgb or MUTED, what))
        else:
            lines.append(faint("not addressed by any warning yet"))
        following = self.rig.follow is not None and self.rig.follow is self.cars[vid].body
        self.hud.card.show(
            ("vehicle", vid),
            tint(rgb or MUTED, spaced(role)),
            vid,
            "\n".join(lines),
            actions=[("Stop following" if following else "Follow", self.toggle_follow)],
        )

    def hazard_card(self, events: list[WarningEvent]) -> None:
        snap, spec = self.snap, self.worker.spec
        active = snap is not None and snap.event_active
        tab = self.hazard_tab
        if tab == "report" and not events:
            tab = "details"

        def go(t: str):  # type: ignore[no-untyped-def]
            def f() -> None:
                self.hazard_tab = t
                self._hud_at = 0

            return f

        tabs = [
            ("Details", go("details"), tab == "details"),
            ("Nearby", go("nearby"), tab == "nearby"),
        ]
        if events:
            tabs.insert(1, ("Report", go("report"), tab == "report"))
        items: list[tuple[str, object, bool]] = []
        if tab == "details":
            when = (
                (
                    f"{spec.event_start_s:.0f}–{spec.event_end_s:.0f} s   "
                    + (tint(RED, "on the road now") if active else muted("not on the road now"))
                )
                if spec.event_occurred
                else tint(BLUE, "never happened: the reports in this scenario are misleading")
            )
            body = "\n".join(
                [
                    f"{muted('effect')}  {spec.effect.replace('_', ' ')}",
                    f"{muted('at')}  {(', '.join(spec.landmark_names[:2]) or spec.landmark_kind)[:48]}",
                    f"{muted('OSM')}  {spec.landmark_id}   {muted('edge')}  {spec.hazard.edge_id} @ {spec.hazard.pos_m:.0f} m",
                    f"{muted('event')}  {when}",
                    f"{muted('sensors flagging within 200 m')}  {snap.sensor_flags if snap else 0}",
                    (
                        f"{muted('vehicles that will reach it')}  {len(snap.relevant) if snap else 0}"
                        if spec.event_occurred
                        else muted("nobody needs a warning")
                    ),
                ]
            )
        elif tab == "report":
            body = self.comm_text(events[-1])
        else:
            near = self.nearest_vehicles(6)
            h = spec.hazard
            body = (
                muted("Closest to the hazard now:")
                if near
                else muted("No vehicles yet. Press space to play.")
            )
            for v in near:
                rgb, what = self.status_colour(v.vehicle_id)
                text = f"{tint(rgb or WHITE, v.vehicle_id)}   {muted(f'{v.speed_mps * 3.6:.0f} km/h · {math.dist((v.x, v.y), (h.x, h.y)):.0f} m')}   {faint(what)}"
                items.append((text, lambda vid=v.vehicle_id: self.select(("vehicle", vid)), False))
        actions = []
        if (
            snap is not None
            and spec.event_occurred
            and snap.t_s < spec.event_start_s + 30
            and not events
        ):
            actions.append(("Jump to the event", self.skip))
        actions.append(("Fly here", lambda: self.rig.fly_to(self.hazard_pos, 300, 34)))
        name = spec.hazard_class.value.replace("_", " ")
        self.hud.card.show(
            ("hazard", tab),
            tint(RED if active else MUTED, spaced("hazard"))
            + faint("   ·   scenario ground truth"),
            name[:1].upper() + name[1:],
            body,
            tabs=tabs,
            actions=actions,
            items=items,  # type: ignore[arg-type]
        )

    def skip(self) -> None:
        spec = self.worker.spec
        self.worker.send("until", max(0.0, spec.event_start_s + 30))
        self.worker.send("play")

    def comm_text(self, e: WarningEvent) -> str:
        r = e.result
        i = r.interpretation
        a = r.assessed
        loc = a.location
        where = (
            "no location"
            if loc is None
            else (
                "approximate (the reporter's position)"
                if loc.approx
                else f"edge {loc.edge_id} @ {loc.pos_m:.0f} m"
            )
        )
        cfg = CONFIGS[r.config_name]
        status = {
            "corroborated": tint(GREEN, "corroborated"),
            "unverified": tint(AMBER, "unverified"),
            "suppressed": tint(MUTED, "suppressed"),
        }[a.status]
        lines = [
            faint(
                f"{'spoken' if e.spoken else e.slot.role} report at {e.t_s:.0f} s"
                f"  ·  {e.text_source} text"
            ),
            *(tint(WHITE, line) for line in wrap(f'"{e.text}"', 50)),
            f"{muted('read as')}  {i.hazard.value.replace('_', ' ')}  p={i.p_class:.2f}"
            + (tint(AMBER, "  negated") if i.negated else "")
            + (tint(AMBER, "  hedged") if i.hedged else "")
            + (f"   {muted('place')}  {i.location_text}" if i.location_text else ""),
            f"{muted('located')}  {where}",
            f"{muted('confidence')}  {a.interp_conf:.2f}   {muted('evidence')}  {a.event_evidence:.2f}   {muted('reports')}  {a.n_reports}   {status}",
        ]
        if r.warning is None:
            lines.append(muted("No warning sent."))
        else:
            o = e.outcome
            lines += [
                *(tint(CYAN, line) for line in wrap(f'"{r.warning.text}"', 50)),
                f"{muted('policy')}  {cfg.distribution}, {cfg.radius_m:.0f} m   {muted('addressed')}  {len(r.warning.recipients)}",
                f"{tint(CYAN, f'{len(e.delivered)} delivered')}   {tint(RED, f'{len(e.missed)} missed')}   {tint(BLUE, f'{len(e.unneeded)} not needed')}",
                faint(
                    f"{o.transmissions if o else 0} transmissions, {o.receptions if o else 0} receptions (Nakagami-m, multi-hop)"
                ),
            ]
        if e.false_alarm:
            lines.append(
                tint(
                    RED,
                    f"False alarm: no real event, so all {len(e.delivered)} deliveries were unneeded.",
                )
            )
        if e.spoken:
            lines.append(tint(AMBER, "Spoken live: a demo input, never data or results."))
        elif e.text_source == "fixture":
            lines.append(
                tint(AMBER, "Fixture text: a code-path demo until P1's\nreal reports exist.")
            )
        return "\n".join(lines)


# --- app ---------------------------------------------------------------------------------------


def bloom(app: Ursina) -> bool:
    """Soft glow around lights; skipped quietly where the GPU cannot do it."""
    try:
        from direct.filter.CommonFilters import CommonFilters

        filters = CommonFilters(app.win, app.cam)
        ok = filters.setBloom(
            blend=(0.3, 0.4, 0.3, 0.0),
            mintrigger=0.45,
            maxtrigger=1.0,
            desat=-0.3,
            intensity=1.6,
            size="medium",
        )
        app._v2v_filters = filters  # type: ignore[attr-defined]  # keep it alive
        return bool(ok)
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="A.C.T.I.O.N V2V 3D simulation")
    parser.add_argument("--scenario", help="scenario folder name, e.g. noisy-low-s0")
    parser.add_argument(
        "--config", choices=CONFIG_NAMES, default="proposed", help="experiment config to start with"
    )
    parser.add_argument(
        "--no-bloom", action="store_true", help="skip the glow filter (slower GPUs)"
    )
    parser.add_argument(
        "--speed", type=float, default=4.0, choices=SPEEDS, help="simulated seconds per second"
    )
    parser.add_argument(
        "--screenshot", type=Path, help="save one frame and quit (for docs and checks)"
    )
    parser.add_argument("--chapter", type=float, help="with --screenshot: tour position to hold")
    parser.add_argument(
        "--at", type=float, default=0.0, help="with --screenshot: simulated second to show"
    )
    parser.add_argument(
        "--wait", type=float, default=5.0, help="with --screenshot: real seconds of play first"
    )
    parser.add_argument(
        "--select", help="with --screenshot: 'hazard' or a vehicle id to open its card"
    )
    parser.add_argument(
        "--tab", default="details", help="with --screenshot --select hazard: card tab"
    )
    parser.add_argument("--dock", action="store_true", help="with --screenshot: open the controls")
    parser.add_argument(
        "--follow", action="store_true", help="with --screenshot: chase-camera view"
    )
    parser.add_argument("--say", help="with --screenshot: speak this report from a vehicle")
    parser.add_argument("--mic-preview", help="with --screenshot: show the strip hearing this")
    parser.add_argument("--type-preview", help="with --screenshot: show the report box with this")
    parser.add_argument(
        "--after-report",
        type=float,
        help="with --screenshot: shoot this many real seconds after the first warning",
    )
    args = parser.parse_args()
    specs = scenario_specs(SCENARIO_DIR)
    if not specs:
        sys.exit(
            f"No scenarios in {SCENARIO_DIR}. Generate them first:\n  python -m v2v.sim.scenarios --corridor hebbal"
        )
    from panda3d.core import AntialiasAttrib, loadPrcFileData

    loadPrcFileData("", "framebuffer-multisample 1")
    loadPrcFileData("", "multisamples 4")
    app = Ursina(title="A.C.T.I.O.N V2V", borderless=False, development_mode=False, vsync=True)
    window.color = SKY
    window.exit_button.enabled = False
    window.fps_counter.enabled = False
    window.entity_counter.enabled = False
    window.collider_counter.enabled = False
    scene.setShaderInput("fog_color", Vec3(*HORIZON))
    scene.setShaderInput("fog_range", FOG_RANGE)

    world = build_scene(
        NET_DIR / f"{CORRIDOR.name}.net.xml",
        NET_DIR / f"{CORRIDOR.name}.osm",
        context_paths(CORRIDOR),
    )
    xmin, ymin, xmax, ymax = world.bounds
    origin = ((xmin + xmax) / 2, (ymin + ymax) / 2)
    sky = Entity(model=sky_dome(), shader=unlit_shader)
    sky.setBin("background", 0)
    sky.setDepthWrite(False)
    translucent(sky).setBin("background", 0)
    Entity(
        model=hills(max(xmax - xmin, ymax - ymin) * 0.95, max(xmax - xmin, ymax - ymin) * 1.45),
        shader=fog_shader,
        double_sided=True,
    )
    city, windows = build_city(world, origin)
    roads, glow = build_roads(world, origin)
    decor, lamp_glow = build_decor(world, origin)
    city.entities()
    roads.entities()
    decor.entities()
    for e in [*windows.entities(), *glow.entities(), *lamp_glow.entities()]:
        additive(e)
    scene.setAntialias(AntialiasAttrib.M_multisample)
    if not args.no_bloom:
        bloom(app)

    names = [Path(s.sumocfg).parent.name for s in specs]
    worker = SimWorker(
        specs,
        index=names.index(args.scenario) if args.scenario in names else 0,
        config_name=args.config,
    )
    worker.start()
    voice = VoiceHandle()
    rig = CameraRig([Shot((0.0, 0.0, 0.0), 200.0, 24.0, 3000.0, 38.0)])

    class Sky(Entity):
        def update(self) -> None:  # the sky stays centred on the viewer
            p = camera.world_position
            sky.position = (p.x, 0, p.z)

    Sky()
    view: SimView

    def run(cmd: str, arg: object = None) -> None:
        worker.send(cmd, arg)

    def switch(cmd: str, arg: object) -> None:
        worker.send(cmd, arg)
        if cmd == "scenario":
            worker.index = int(arg) % len(specs)  # type: ignore[call-overload]
        elif cmd == "config":
            worker.config_name = str(arg)
        view.load_scenario()

    def cycle_config(step: int) -> None:
        switch(
            "config",
            CONFIG_NAMES[(CONFIG_NAMES.index(worker.config_name) + step) % len(CONFIG_NAMES)],
        )

    def toggle_truth() -> None:
        view.ground_truth = not view.ground_truth
        view._hud_at = 0

    def mic() -> None:
        if not voice.active and not models_ready():
            view.toast(
                tint(
                    AMBER, "Speech models are missing: run python -m v2v.voice.listener --download"
                ),
                8.0,
            )
            return
        voice.toggle()
        view._hud_at = 0

    actions = {
        "mic": mic,
        "type": lambda: view.end_typing(False) if view.typing else view.begin_typing(),
        "play": lambda: run("pause" if worker.playing else "play"),
        "step": lambda: run("step"),
        "reset": lambda: switch("reset", None),
        "skip": lambda: view.skip(),
        "truth": toggle_truth,
        "speed": lambda s: run("speed", s),
        "prev_scenario": lambda: switch("scenario", worker.index - 1),
        "next_scenario": lambda: switch("scenario", worker.index + 1),
        "prev_config": lambda: cycle_config(-1),
        "next_config": lambda: cycle_config(1),
        "dock": lambda: (hud.dock.set_open(not hud.dock.open), setattr(view, "_hud_at", 0)),
        "mode": lambda: rig.tour() if rig.mode == "explore" else rig.explore(),
        "chapter": lambda i: (
            rig.tour(),
            rig.scroll.go(i),
            setattr(rig, "scrolled_at", time.perf_counter()),
        ),
        "deselect": lambda: view.select(None),
    }
    hud = Hud(actions, SPEEDS, 6)
    view = SimView(worker, world, origin, hud, rig)
    view.voice = voice
    rig.pose = rig.free = rig.shots[0]
    rig.on_mode = lambda: setattr(view, "_hud_at", 0)
    run("speed", args.speed)

    class Keys(Entity):
        def input(self, key: str) -> None:
            if view.typing:  # every key goes into the report, none moves the camera
                view.type_key(key)
                return
            if key in ("enter", "return"):
                view.begin_typing()
            elif key == "space":
                actions["play"]()
            elif key == "n":
                actions["step"]()
            elif key == "r":
                actions["reset"]()
            elif key == "f":
                view.toggle_follow()
            elif key == "g":
                toggle_truth()
            elif key == "k":
                view.skip()
            elif key == "c":
                actions["dock"]()
            elif key == "m":
                actions["mic"]()
            elif key == "v":
                voice.tap()
            elif key == "t":
                rig.tour()
            elif key == "tab":
                hud.set_shown(not hud.shown)
            elif key in "1234567" and len(key) == 1:
                run("speed", SPEEDS[int(key) - 1])
            elif key == "[":
                actions["prev_scenario"]()
            elif key == "]":
                actions["next_scenario"]()
            elif key in ("down arrow", "page down", "up arrow", "page up"):
                step = 1 if key in ("down arrow", "page down") else -1
                actions["chapter"](round(rig.scroll.target) + step)
            elif key == "escape":
                view.select(None)
            elif key == "f11":
                window.fullscreen = not window.fullscreen

    Keys()
    if args.screenshot is not None:
        from panda3d.core import Filename
        from ursina import application

        if args.at:
            run("until", args.at)
        run("play")
        if args.chapter is not None:
            rig.scroll.value = rig.scroll.target = args.chapter
            rig.pose = pose_at(rig.shots, args.chapter)
            rig.scrolled_at = math.inf  # hold it there
        if args.select:
            view.select(("hazard",) if args.select == "hazard" else ("vehicle", args.select))
            view.hazard_tab = args.tab
        if args.dock:
            hud.dock.set_open(True)

        class Director(Entity):
            """Screenshot mode: wait (for a report, if asked), frame the shot, save, quit."""

            def __init__(self) -> None:
                super().__init__()
                self.started = time.perf_counter()
                self.shoot_at: float | None = (
                    None if args.after_report is not None else self.started + args.wait
                )
                self.saved = False
                self.said = False
                if args.mic_preview:
                    view.voice_override = ("listening", args.mic_preview, 0.5)
                if args.type_preview:
                    view.typing, view.typed = True, args.type_preview

            def update(self) -> None:
                now = time.perf_counter()
                if args.say and not self.said and now - self.started > 6.0 and view.cars:
                    self.said = True
                    view.submit_voice(args.say)
                if args.select and args.select != "hazard" and view.selected is None:
                    view.select(("vehicle", args.select))
                if args.select == "hazard":
                    view.hazard_tab = args.tab
                if self.shoot_at is None:
                    if any(e.sent for e in worker.events()):
                        run("speed", 1.0)
                        self.shoot_at = now + args.after_report
                        if args.follow:
                            view.toggle_follow()
                    return
                if args.follow and self.shoot_at - now > 1.5 and view.rig.follow is None:
                    view.toggle_follow()
                if now >= self.shoot_at and not self.saved:
                    self.saved = True
                    args.screenshot.parent.mkdir(parents=True, exist_ok=True)
                    app.win.saveScreenshot(Filename.fromOsSpecific(str(args.screenshot.resolve())))
                    application.quit()

        Director()
    app.run()
    worker.stop()
    voice.close()


if __name__ == "__main__":
    main()
