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
until P1's real test split exists, and the HUD says so.

Mouse: right-drag rotate, scroll zoom, middle-drag or WASD pan, left-click a
vehicle or the hazard. Keys: space play/pause, N step, R reset, F follow,
G ground truth, K skip to the event, Tab panels, 1-7 speed, [ ] scenario,
Esc deselect, F11 full screen.
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling modules

REPO = Path(__file__).resolve().parents[2]
os.chdir(REPO)  # scenario files use repo-relative paths

from geom import lerp_angle, slow_motion, to_ursina  # noqa: E402
from hud import AMBER, BLUE, CYAN, GREEN, MUTED, ORANGE, RED, WHITE, Hud, muted, tint, wrap  # noqa: E402
from meshes import (
    build_city,
    build_decor,
    build_roads,
    car_mesh,
    disc,
    lanes_by_edge,
    ring,
    translucent,
    zone_overlay,
)  # noqa: E402
from ursina import (  # noqa: E402
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
from ursina import (
    time as utime,
)
from ursina.shaders import unlit_shader  # noqa: E402
from worker import FiredEvent, SimWorker, Snapshot  # noqa: E402

from v2v.experiments import CONFIGS  # noqa: E402
from v2v.sim.delivery import DEFAULT_MODEL  # noqa: E402
from v2v.sim.live import ORIGIN, WarningEvent, scenario_specs  # noqa: E402
from v2v.sim.osm_network import CORRIDORS  # noqa: E402
from v2v.sim.world3d import build_scene, context_paths  # noqa: E402

CORRIDOR = CORRIDORS["hebbal"]
NET_DIR = REPO / "sim_out" / "networks" / CORRIDOR.name
SCENARIO_DIR = REPO / "sim_out" / "scenarios" / CORRIDOR.name
SPEEDS = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
CONFIG_NAMES = tuple(CONFIGS)
NEUTRAL = color.rgba(0.74, 0.79, 0.88, 1)
HALO = color.rgba(0.2, 0.24, 0.32, 1)
"""Dim ring under an ordinary vehicle: entity alpha is ignored by the unlit shader."""
SKY = color.rgba(0.018, 0.026, 0.05, 1)


def rgba(rgb: tuple[float, float, float], a: float = 1.0) -> color.Color:
    return color.rgba(*rgb, a)


def instance(mesh: Mesh, tint_to: color.Color | None = None, **kwargs: object) -> Entity:
    """An entity drawing a shared mesh. Assigning one Mesh as ``model`` to many
    entities would reparent it each time; instancing shares it instead."""
    e = Entity(shader=unlit_shader, **kwargs)  # type: ignore[arg-type]
    mesh.instanceTo(e)
    if tint_to is not None:
        paint(e, tint_to)
    return e


def paint(e: Entity, c: color.Color) -> None:
    """Tint an instanced entity (Ursina's ``color`` only tints a ``model``)."""
    e.setColorScale(c)


def cardinal(h: float) -> str:
    return ("N", "NE", "E", "SE", "S", "SW", "W", "NW")[round(h / 45) % 8]


# --- camera ------------------------------------------------------------------------------


class CameraRig(Entity):
    """Orbit / pan / zoom around a pivot, or chase a vehicle in follow mode."""

    def __init__(self) -> None:
        super().__init__()
        camera.parent = self
        camera.position = (0, 0, -600)
        camera.rotation = (0, 0, 0)
        camera.fov = 55
        camera.clip_plane_far = 7000
        self.yaw, self.pitch, self.dist = 20.0, 38.0, 650.0
        self.goal = Vec3(0, 0, 0)
        self.follow: Entity | None = None

    def look_at_point(self, p: Vec3, dist: float | None = None) -> None:
        self.goal = Vec3(p)
        if dist is not None:
            self.dist = dist

    def update(self) -> None:
        dt = utime.dt
        if mouse.right:
            self.yaw += mouse.velocity[0] * 220
            self.pitch = min(86.0, max(6.0, self.pitch - mouse.velocity[1] * 140))
        self.yaw += (held_keys["e"] - held_keys["q"]) * 70 * dt
        pan = self.dist * 0.9 * dt
        fwd = Vec3(math.sin(math.radians(self.yaw)), 0, math.cos(math.radians(self.yaw)))
        right = Vec3(fwd.z, 0, -fwd.x)
        move = fwd * (held_keys["w"] - held_keys["s"]) + right * (held_keys["d"] - held_keys["a"])
        if mouse.middle:
            move -= right * mouse.velocity[0] * 60 + fwd * mouse.velocity[1] * 60
        if move.length() > 0 and self.follow is None:
            self.goal += move * pan
        if self.follow is not None:
            target = self.follow.world_position
            self.goal = Vec3(target.x, target.y + 2, target.z)
            if not mouse.right:  # swing in behind the vehicle
                self.yaw = lerp_angle(self.yaw, self.follow.rotation_y, min(1.0, dt * 1.5))
        k = min(1.0, dt * (6 if self.follow is not None else 9))
        self.position = self.position + (self.goal - self.position) * k
        self.rotation_y += ((self.yaw - self.rotation_y + 180) % 360 - 180) * k
        self.rotation_x += (self.pitch - self.rotation_x) * k
        camera.z += (-self.dist - camera.z) * k

    def input(self, key: str) -> None:
        if key == "scroll up":
            self.dist = max(12.0, self.dist * 0.86)
        elif key == "scroll down":
            self.dist = min(3200.0, self.dist * 1.16)


# --- the simulation view -----------------------------------------------------------------


class Car:
    def __init__(self, view: "SimView", vid: str, model: Mesh, blocking: bool = False) -> None:
        self.vid = vid
        self.blocking = blocking
        self.body = instance(model, tint_to=NEUTRAL, double_sided=True)
        self.body.collider = BoxCollider(
            self.body, center=Vec3(0, 0.8, 0), size=Vec3(2.2, 1.8, 4.8)
        )
        self.body.on_click = lambda: view.select(("vehicle", vid))
        self.halo = instance(view.halo_mesh, tint_to=HALO, parent=self.body, y=0.12)
        self.halo.enabled = False
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

    def update(self, dt: float) -> None:
        k = min(1.0, dt * 7)
        self.body.position = self.body.position + (self.target - self.body.position) * k
        self.body.rotation_y = lerp_angle(self.body.rotation_y, self.heading, k)

    def destroy(self) -> None:
        destroy(self.body)


class PulseFx:
    """One V2V hop drawn in slow motion: a glowing packet with a beam behind it."""

    def __init__(
        self, a: Vec3, b: Vec3, t0: float, t1: float, rgb: tuple[float, float, float], on_arrive
    ) -> None:  # type: ignore[no-untyped-def]
        self.a, self.b, self.t0, self.t1 = a, b, t0, t1
        self.on_arrive = on_arrive
        self.dot = Entity(
            model="sphere",
            shader=unlit_shader,
            color=rgba(rgb),
            scale=3.2,
            position=a,
            enabled=False,
        )
        self.glow = Entity(
            parent=self.dot, model="sphere", shader=unlit_shader, color=rgba(rgb, 0.25), scale=2.4
        )
        self.beam = Entity(
            model=Mesh(vertices=[a, a], mode="line", thickness=3),
            shader=unlit_shader,
            color=rgba(rgb, 0.8),
            enabled=False,
        )
        self.done = False

    def update(self, now: float) -> None:
        if now < self.t0:
            return
        f = min(1.0, (now - self.t0) / max(self.t1 - self.t0, 1e-3))
        p = self.a + (self.b - self.a) * f + Vec3(0, math.sin(f * math.pi) * 6, 0)
        self.dot.enabled = self.beam.enabled = True
        self.dot.position = p
        self.beam.model.vertices = [self.a, p]
        self.beam.model.generate()
        if f >= 1.0 and not self.done:
            self.done = True
            self.on_arrive()

    def fade(self, now: float) -> bool:
        """True once it has faded out and can be removed."""
        if not self.done:
            return False
        age = now - self.t1
        self.dot.enabled = False
        self.beam.color = rgba(CYAN, max(0.0, 0.8 - age * 0.6))
        return age > 1.5

    def destroy(self) -> None:
        destroy(self.dot)
        destroy(self.beam)


class Wave:
    """An expanding ring (radio range, arrival flash)."""

    def __init__(
        self,
        view: "SimView",
        p: Vec3,
        radius: float,
        rgb: tuple[float, float, float],
        duration: float,
    ) -> None:
        self.e = translucent(
            instance(view.unit_ring, tint_to=rgba(rgb), position=p + Vec3(0, 0.6, 0), scale=0.01)
        )
        self.radius, self.rgb, self.duration, self.start = (
            radius,
            rgb,
            duration,
            time.perf_counter(),
        )

    def update(self, now: float) -> bool:
        f = (now - self.start) / self.duration
        if f >= 1:
            destroy(self.e)
            return True
        self.e.scale = max(0.01, self.radius * f)
        paint(self.e, rgba(tuple(c * (1 - f) for c in self.rgb)))  # type: ignore[arg-type]
        return False


class SimView(Entity):
    def __init__(
        self, worker: SimWorker, world, origin: tuple[float, float], hud: Hud, rig: CameraRig
    ) -> None:  # type: ignore[no-untyped-def]
        super().__init__()
        self.worker, self.world, self.origin, self.hud, self.rig = worker, world, origin, hud, rig
        self.lanes = lanes_by_edge(world)
        self.car_model = car_mesh()
        self.halo_mesh = ring(3.4, 0.9, (1, 1, 1), 1.0, 40)
        self.unit_ring = ring(1.0, 0.04, (1, 1, 1), 1.0, 96)
        self.cars: dict[str, Car] = {}
        self.version = -1
        self.snap: Snapshot | None = None
        self.selected: tuple[str, ...] | None = None
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
            scale=150,
            origin=(0, 0),
            color=rgba(ORANGE),
            enabled=False,
        )
        self.bubble_until = 0.0
        self.tag = Text(
            " ", parent=scene, billboard=True, scale=110, origin=(0, -0.5), enabled=False
        )
        self._hud_at = 0.0
        self._fps = 60.0
        self._build_hazard()
        self.load_scenario()

    # --- coordinates -----------------------------------------------------------------

    def u(self, x: float, y: float, z: float = 0.0) -> Vec3:
        return Vec3(*to_ursina(x, y, z, self.origin))

    def at_vehicle(self, v) -> Vec3:  # type: ignore[no-untyped-def]
        return self.u(v.x, v.y, self.world.elevation(v.edge_id, v.lane_pos_m))

    # --- hazard and zones --------------------------------------------------------------

    def _build_hazard(self) -> None:
        self.hazard = Entity(
            collider="box", model="cube", scale=(14, 14, 14), color=color.clear, shader=unlit_shader
        )
        self.hazard.on_click = lambda: self.select(("hazard",))
        self.beacon = Entity(
            parent=self.hazard,
            model="cube",
            shader=unlit_shader,
            scale=(0.08, 3.4, 0.08),
            y=1.7,
            color=rgba(RED, 0.75),
        )
        self.beacon_ring = Entity(model=ring(14, 2.2, RED, 0.9), shader=unlit_shader)
        self.sensor = translucent(Entity(model=disc(200, RED, 0.07), shader=unlit_shader))
        self.sensor_rim = translucent(Entity(model=ring(200, 1.4, RED, 0.45), shader=unlit_shader))
        self.hazard_label = Text(
            " ", parent=scene, billboard=True, scale=230, origin=(0, -0.5), color=rgba(RED)
        )

    def load_scenario(self) -> None:
        spec = self.worker.spec
        h = spec.hazard
        p = self.u(h.x, h.y, self.world.elevation(h.edge_id, h.pos_m))
        self.hazard_pos = p
        for e in (self.hazard, self.beacon_ring, self.sensor, self.sensor_rim):
            e.position = p + Vec3(0, 0.35, 0)
        self.hazard.y = p.y + 7
        self.hazard_label.position = p + Vec3(0, 52, 0)
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
        self.rig.look_at_point(p, 260)
        self.rig.pitch = 34.0
        self.selected = ("hazard",)
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
                Entity(model=disc(event.radius_m, rgb, 0.045), shader=unlit_shader, position=centre)
            ),
            translucent(
                Entity(
                    model=ring(event.radius_m, 2.5, rgb, 0.55),
                    shader=unlit_shader,
                    position=centre + Vec3(0, 0.1, 0),
                )
            ),
        ]
        if event.zone_edges:
            self.zone_entities += zone_overlay(self.lanes, event.zone_edges, self.origin).entities(
                see_through=True
            )
        self.waves.append(Wave(self, centre, event.radius_m, rgb, 2.2))

    # --- events ------------------------------------------------------------------------

    def fire(self, fired: FiredEvent) -> None:
        e = fired.event
        r = e.result
        self.reporters.add(e.slot.reporter_id)
        sender = self.u(*e.sender_xy, 1.5)
        self.bubble.text = f"REPORT  {r.interpretation.hazard.value.replace('_', ' ')}"
        self.bubble.position = sender + Vec3(0, 20, 0)
        self.bubble.enabled = True
        self.bubble_until = time.perf_counter() + 5
        if not e.sent:
            self.hud.toast.text = tint(
                AMBER, f"t={e.t_s:.0f}s  {e.slot.role} report suppressed"
            ) + muted(f"  ({r.assessed.status}: no warning sent)")
            self.hud.toast_until = time.perf_counter() + 5
            return
        self.show_zone(e)
        self.waves.append(Wave(self, sender, DEFAULT_MODEL.comm_range_m, CYAN, 1.6))
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
                    self.waves.append(Wave(self, c.body.position, 9, CYAN if needed else BLUE, 0.7))

            self.pulses.append(
                PulseFx(
                    a, b, now + p.start_ms * k, now + p.end_ms * k, CYAN if needed else BLUE, arrive
                )
            )
        self.missed = set(e.missed)
        self.hud.toast.text = (
            (tint(RED, "FALSE ALARM  ") if e.false_alarm else "")
            + tint(CYAN, f"{e.slot.role} report -> {r.assessed.status} warning")
            + muted(
                f"   {len(r.warning.recipients) if r.warning else 0} addressed · {len(e.delivered)} delivered · "
            )
            + tint(RED, f"{len(e.missed)} missed")
            + muted(f"   V2V shown {k * 1000:.0f}x slower")
        )
        self.hud.toast_until = time.perf_counter() + 6

    # --- selection -----------------------------------------------------------------------

    def select(self, what: tuple[str, ...] | None) -> None:
        self.selected = what
        if what is None:
            self.rig.follow = None
        self._hud_at = 0

    def toggle_follow(self) -> None:
        if self.rig.follow is not None:
            self.rig.follow = None
            self.rig.dist = 220
            return
        if self.selected and self.selected[0] == "vehicle" and self.selected[1] in self.cars:
            car = self.cars[self.selected[1]]
        else:
            near = self.nearest_vehicles(1)
            if not near:
                return
            car = self.cars[near[0].vehicle_id]
            self.select(("vehicle", car.vid))
        self.rig.follow = car.body
        self.rig.dist, self.rig.pitch = 42.0, 16.0

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
        version, snap = self.worker.latest()
        if version != self.version and snap is not None:
            self.version = version
            self.apply(snap)
        for fired in self.worker.drain():
            self.fire(fired)
        size = min(7.0, max(1.0, self.rig.dist / 110))  # true size up close, legible from afar
        words = max(18.0, self.rig.dist * 0.42)  # world labels stay readable at any zoom
        self.tag.scale = words * 0.8
        self.bubble.scale = words
        self.hazard_label.scale = max(40.0, self.rig.dist * 0.75)
        for car in self.cars.values():
            car.update(dt)
            car.body.scale = size
        for fx in list(self.pulses):
            fx.update(now)
            if fx.fade(now):
                fx.destroy()
                self.pulses.remove(fx)
        self.waves = [w for w in self.waves if not w.update(now)]
        self.beacon_ring.rotation_y += 40 * dt
        pulse = 0.5 + 0.5 * math.sin(now * 4)
        active = snap is not None and snap.event_active
        self.beacon.color = rgba(RED, 0.4 + 0.5 * pulse) if active else rgba(MUTED, 0.35)
        self.beacon_ring.color = rgba(RED, 0.5 + 0.4 * pulse) if active else rgba(MUTED, 0.3)
        self.sensor.enabled = self.sensor_rim.enabled = active
        for car in self.cars.values():
            if car.blocking:
                car.halo.enabled = True
                paint(car.halo, rgba(AMBER) if int(now * 3) % 2 else HALO)
        if self.bubble.enabled and now > self.bubble_until:
            self.bubble.enabled = False
        if self.hud.toast.text and now > self.hud.toast_until:
            self.hud.toast.text = ""
        if now - self._hud_at > 0.2:
            self._hud_at = now
            self.refresh_hud()

    def apply(self, snap: Snapshot) -> None:
        self.snap = snap
        seen = set()
        for group, blocking in ((snap.vehicles, False), (snap.blocking, True)):
            for vid, v in group.items():
                seen.add(vid)
                car = self.cars.get(vid)
                if car is None:
                    car = self.cars[vid] = Car(self, vid, self.car_model, blocking)
                    if blocking:
                        paint(car.body, rgba(RED))
                car.place(self.at_vehicle(v), v.heading_deg)
                if not blocking:
                    rgb, _ = self.status_colour(vid)
                    paint(car.body, rgba(rgb) if rgb else NEUTRAL)
                    car.halo.enabled = True
                    paint(car.halo, rgba(rgb) if rgb else HALO)
        for vid in [v for v in self.cars if v not in seen]:
            if self.rig.follow is self.cars[vid].body:
                self.rig.follow = None
            self.cars.pop(vid).destroy()

    # --- HUD ----------------------------------------------------------------------------------

    def refresh_hud(self) -> None:
        hud, snap, spec = self.hud, self.snap, self.worker.spec
        w = self.worker
        events = w.events()
        state = {
            "running": tint(GREEN, "SUMO LIVE"),
            "paused": tint(AMBER, "PAUSED"),
            "ready": tint(AMBER, "READY"),
            "finished": tint(MUTED, "FINISHED"),
            "fast-forward": tint(CYAN, "FAST-FORWARD"),
        }.get(w.status, tint(AMBER, w.status.upper()))
        if w.error:
            state = tint(RED, "SUMO ERROR")
        sent = [e for e in events if e.sent]
        cfg = CONFIGS[w.config_name]
        hud.status.text = (
            f"{state}   {muted('t')} {snap.t_s if snap else 0:.0f}s   "
            f"{muted('vehicles')} {len(snap.vehicles) if snap else 0}   "
            f"{muted('reports')} {len(events)}   {muted('warnings')} {len(sent)}   "
            f"{muted('config')} {cfg.name} ({cfg.distribution})   "
            f"{muted(f'{self._fps:.0f} fps')}"
        )
        hud.scenario_l.text = f"{spec.condition} · {spec.density} · seed {spec.seed}"
        hud.config_l.text = cfg.name
        hud.play_b.text = "Pause" if w.playing else "Play"
        hud.follow_b.text = "Orbit" if self.rig.follow is not None else "Follow"
        hud.highlight(hud.speed_bs, SPEEDS.index(w.speed) if w.speed in SPEEDS else None)
        hud.timeline(
            snap.t_s if snap else 0.0,
            spec.end_s,
            (spec.event_start_s, spec.event_end_s) if spec.event_occurred else None,
            [e.t_s for e in events],
        )
        hud.time_l.text = f"{snap.t_s if snap else 0:.0f} / {spec.end_s:.0f} s" + muted(
            f"   event {spec.event_start_s:.0f}-{spec.event_end_s:.0f} s"
            if spec.event_occurred
            else "   no real event"
        )
        active = snap is not None and snap.event_active
        name = spec.hazard_class.value.replace("_", " ")
        self.hazard_label.text = (
            name.upper()
            if active
            else (
                f"{name} (not on the road)"
                if not spec.event_occurred
                else f"{name} (not yet)"
                if snap and snap.t_s < spec.event_start_s
                else f"{name} (cleared)"
            )
        )
        self.hazard_label.color = rgba(RED) if active else rgba(MUTED)
        if hud.left_open:
            self.refresh_left(events)
        if hud.right_open:
            self.refresh_right(events)
        sel = self.selected
        if sel and sel[0] == "vehicle" and sel[1] in self.cars:
            car = self.cars[sel[1]]
            self.tag.enabled = True
            self.tag.text = sel[1]
            self.tag.position = car.body.position + Vec3(0, 7, 0)
        else:
            self.tag.enabled = False

    def refresh_left(self, events: list[WarningEvent]) -> None:
        hud, spec = self.hud, self.worker.spec
        h = spec.hazard
        near = self.nearest_vehicles(len(hud.vehicle_buttons))
        for i, b in enumerate(hud.vehicle_buttons):
            if i >= len(near):
                b.enabled = False
                continue
            v = near[i]
            rgb, what = self.status_colour(v.vehicle_id)
            b.enabled = True
            b.text = f"{tint(rgb or WHITE, f'{v.vehicle_id:<14}')} {v.speed_mps * 3.6:4.0f} km/h {math.dist((v.x, v.y), (h.x, h.y)):5.0f} m  {muted(what)}"
            b.on_click = lambda vid=v.vehicle_id: self.select(("vehicle", vid))
            b.color = (
                color.rgba(0, 0.45, 0.55, 0.9)
                if self.selected == ("vehicle", v.vehicle_id)
                else color.rgba(0.08, 0.11, 0.16, 0.92)
            )
        active = self.snap is not None and self.snap.event_active
        hud.hazard_button.text = (
            f"{tint(RED if active else MUTED, spec.hazard_class.value.replace('_', ' '))}  {muted(spec.effect.replace('_', ' '))}  "
            + (tint(RED, "ACTIVE") if active else muted("inactive"))
        )
        lines = []
        for e in events[-6:]:
            r = e.result
            status = {
                "corroborated": tint(GREEN, "corroborated"),
                "unverified": tint(AMBER, "unverified"),
                "suppressed": tint(MUTED, "suppressed"),
            }[r.assessed.status]
            lines.append(
                f"{e.t_s:4.0f}s {e.slot.role:<13} {status}  {muted(f'{len(e.delivered)} warned')}"
            )
        hud.log.text = "\n".join(lines) or muted(
            "No report yet. The first one fires after\nits reporter passes the hazard. Press K to skip."
        )

    def refresh_right(self, events: list[WarningEvent]) -> None:
        hud, snap, spec = self.hud, self.snap, self.worker.spec
        sel = self.selected
        out: list[str] = []
        focus: WarningEvent | None = events[-1] if events else None
        if sel and sel[0] == "vehicle" and snap is not None:
            vid = sel[1]
            v = snap.vehicles.get(vid) or snap.blocking.get(vid)
            if v is None:
                out.append(muted(f"{vid} has left the network."))
            else:
                rgb, what = self.status_colour(vid)
                role = (
                    "hazard vehicle (blocking a lane)"
                    if vid in snap.blocking
                    else "reporter"
                    if vid in self.reporters
                    else "vehicle"
                )
                height = self.world.elevation(v.edge_id, v.lane_pos_m)
                out += [
                    tint(CYAN, "SELECTED VEHICLE"),
                    f"{tint(WHITE, vid)}   {muted(role)}",
                    f"{muted('speed')} {v.speed_mps * 3.6:.0f} km/h   {muted('heading')} {v.heading_deg:.0f}° {cardinal(v.heading_deg)}",
                    f"{muted('road')} {v.edge_id}"
                    + (muted(f"   on flyover, {height:.0f} m up") if height > 1 else ""),
                    f"{muted('to hazard')} {math.dist((v.x, v.y), (spec.hazard.x, spec.hazard.y)):.0f} m straight line",
                    f"{muted('ground truth')} "
                    + (
                        tint(RED, "will reach the hazard")
                        if vid in snap.relevant
                        else "will not reach it"
                    ),
                ]
                mine = [
                    e for e in events if e.result.warning and vid in e.result.warning.recipients
                ]
                if mine:
                    e = mine[-1]
                    focus = e
                    got = e.delivered.get(vid)
                    if got is not None and e.outcome is not None:
                        relay = e.outcome.relays.get(vid, ORIGIN)
                        out.append(
                            tint(
                                rgb or CYAN,
                                f"warned in {got:.1f} ms, hop {e.outcome.hops.get(vid, '?')}, relayed by {relay}",
                            )
                        )
                    else:
                        out.append(tint(ORANGE, "addressed, but the radio never got it through"))
                elif what:
                    out.append(tint(rgb or MUTED, what))
                else:
                    out.append(muted("not addressed by any warning"))
        else:
            active = snap is not None and snap.event_active
            out += [
                tint(RED, "HAZARD") + muted("   (scenario ground truth)"),
                f"{tint(WHITE, spec.hazard_class.value.replace('_', ' '))}   {muted(spec.effect.replace('_', ' '))}",
                f"{muted('at')} {(', '.join(spec.landmark_names[:2]) or spec.landmark_kind)[:52]}",
                f"{muted('OSM')} {spec.landmark_id}   {muted('edge')} {spec.hazard.edge_id} @ {spec.hazard.pos_m:.0f} m",
                f"{muted('event')} "
                + (
                    f"{spec.event_start_s:.0f}-{spec.event_end_s:.0f} s  "
                    + (tint(RED, "on the road now") if active else muted("not on the road now"))
                    if spec.event_occurred
                    else tint(BLUE, "never happened: these reports are misleading")
                ),
                f"{muted('sensor flags within 200 m')} {snap.sensor_flags if snap else 0}   "
                + (
                    f"{muted('will reach it')} {len(snap.relevant) if snap else 0}"
                    if spec.event_occurred
                    else muted("nobody needs a warning")
                ),
            ]
        hud.detail.text = "\n".join(out)
        hud.comm.text = self.comm_text(focus)

    def comm_text(self, e: WarningEvent | None) -> str:
        if e is None:
            return tint(CYAN, "COMMUNICATION") + "\n" + muted("Waiting for the first report.")
        r = e.result
        i = r.interpretation
        a = r.assessed
        loc = a.location
        where = (
            "no location"
            if loc is None
            else (
                "approximate (reporter's position)"
                if loc.approx
                else f"edge {loc.edge_id} @ {loc.pos_m:.0f} m"
            )
        )
        cfg = CONFIGS[r.config_name]
        lines = [
            tint(CYAN, "COMMUNICATION")
            + muted(f"   report at {e.t_s:.0f} s · {e.slot.role} · {e.text_source} text"),
            *(f'"{line}"' if i == 0 else line for i, line in enumerate(wrap(e.text, 58))),
            f"{muted('NLP')} {i.hazard.value} p={i.p_class:.2f}"
            + (tint(AMBER, "  negated") if i.negated else "")
            + (tint(AMBER, "  hedged") if i.hedged else "")
            + (f"  {muted('place')} {i.location_text}" if i.location_text else ""),
            f"{muted('located')} {where}",
            f"{muted('confidence')} {a.interp_conf:.2f}  {muted('evidence')} {a.event_evidence:.2f}  {muted('reports')} {a.n_reports}  -> "
            + {
                "corroborated": tint(GREEN, "corroborated"),
                "unverified": tint(AMBER, "unverified"),
                "suppressed": tint(MUTED, "suppressed"),
            }[a.status],
        ]
        if r.warning is None:
            lines.append(muted("No warning sent."))
        else:
            o = e.outcome
            lines += [
                *(tint(WHITE, line) for line in wrap(f'"{r.warning.text}"', 58)),
                f"{muted('policy')} {cfg.distribution} {cfg.radius_m:.0f} m   {muted('addressed')} {len(r.warning.recipients)}",
                f"{tint(CYAN, f'delivered {len(e.delivered)}')}   {tint(RED, f'missed {len(e.missed)}')}   {tint(BLUE, f'unneeded {len(e.unneeded)}')}",
                muted(
                    f"{o.transmissions if o else 0} transmissions, {o.receptions if o else 0} receptions (Nakagami-m, multi-hop)"
                ),
            ]
        if e.false_alarm:
            lines.append(
                tint(
                    RED, f"False alarm: no real event, all {len(e.delivered)} deliveries unneeded."
                )
            )
        if e.text_source == "fixture":
            lines.append(
                tint(AMBER, "Fixture text: a code-path demo until P1's real reports exist.")
            )
        return "\n".join(lines)


# --- app ---------------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="A.C.T.I.O.N V2V 3D simulation")
    parser.add_argument("--scenario", help="scenario folder name, e.g. noisy-low-s0")
    parser.add_argument(
        "--config", choices=CONFIG_NAMES, default="proposed", help="experiment config to start with"
    )
    parser.add_argument(
        "--distance", type=float, help="with --screenshot: camera distance in metres"
    )
    parser.add_argument(
        "--screenshot", type=Path, help="save one frame and quit (for docs and checks)"
    )
    parser.add_argument(
        "--at", type=float, default=0.0, help="with --screenshot: simulated second to show"
    )
    parser.add_argument(
        "--wait", type=float, default=4.0, help="with --screenshot: real seconds of play first"
    )
    parser.add_argument(
        "--follow", action="store_true", help="with --screenshot: chase-camera view"
    )
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
    app = Ursina(
        title="A.C.T.I.O.N V2V · 3D simulation",
        borderless=False,
        development_mode=False,
        vsync=True,
    )
    window.color = SKY
    window.exit_button.enabled = False
    window.fps_counter.enabled = False
    window.entity_counter.enabled = False
    window.collider_counter.enabled = False
    scene.fog_color = SKY
    scene.fog_density = (900, 3200)

    world = build_scene(
        NET_DIR / f"{CORRIDOR.name}.net.xml",
        NET_DIR / f"{CORRIDOR.name}.osm",
        context_paths(CORRIDOR),
    )
    xmin, ymin, xmax, ymax = world.bounds
    origin = ((xmin + xmax) / 2, (ymin + ymax) / 2)
    city = build_city(world, origin)
    roads, glow = build_roads(world, origin)
    decor, lamp_glow = build_decor(world, origin)
    city.entities()
    roads.entities()
    decor.entities()
    glow.entities()
    lamp_glow.entities(see_through=True)

    names = [Path(s.sumocfg).parent.name for s in specs]
    worker = SimWorker(
        specs,
        index=names.index(args.scenario) if args.scenario in names else 0,
        config_name=args.config,
    )
    worker.start()
    rig = CameraRig()

    view: SimView

    def run(cmd: str, arg: object = None) -> None:
        worker.send(cmd, arg)

    def switch(cmd: str, arg: object) -> None:
        worker.send(cmd, arg)
        if cmd == "scenario":
            worker.index = int(arg) % len(specs)  # type: ignore[call-overload]
        view.load_scenario()

    actions = {
        "play": lambda: run("pause" if worker.playing else "play"),
        "step": lambda: run("step"),
        "reset": lambda: switch("reset", None),
        "speed": lambda s: run("speed", s),
        "prev_scenario": lambda: switch("scenario", worker.index - 1),
        "next_scenario": lambda: switch("scenario", worker.index + 1),
        "prev_config": lambda: switch(
            "config", CONFIG_NAMES[(CONFIG_NAMES.index(worker.config_name) - 1) % len(CONFIG_NAMES)]
        ),
        "next_config": lambda: switch(
            "config", CONFIG_NAMES[(CONFIG_NAMES.index(worker.config_name) + 1) % len(CONFIG_NAMES)]
        ),
        "follow": lambda: view.toggle_follow(),
        "panels": lambda: hud.set_open(not hud.left_open, not hud.left_open),
        "toggle_left": lambda: hud.set_open(left=not hud.left_open),
        "toggle_right": lambda: hud.set_open(right=not hud.right_open),
        "select_hazard": lambda: (
            view.select(("hazard",)),
            rig.look_at_point(view.hazard_pos, 420),
        ),
    }
    hud = Hud(actions, SPEEDS)  # type: ignore[arg-type]
    view = SimView(worker, world, origin, hud, rig)
    run("speed", 4.0)

    class Keys(Entity):
        def input(self, key: str) -> None:
            if key == "space":
                actions["play"]()
            elif key == "n":
                actions["step"]()
            elif key == "r":
                actions["reset"]()
            elif key == "f":
                actions["follow"]()
            elif key == "g":
                view.ground_truth = not view.ground_truth
            elif key == "k":
                spec = worker.spec
                run("until", max(0.0, spec.event_start_s + 30))
            elif key == "tab":
                actions["panels"]()
            elif key in "1234567" and len(key) == 1:
                run("speed", SPEEDS[int(key) - 1])
            elif key == "[":
                actions["prev_scenario"]()
            elif key == "]":
                actions["next_scenario"]()
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

        class Director(Entity):
            """Screenshot mode: wait (for a report, if asked), frame the shot, save, quit."""

            def __init__(self) -> None:
                super().__init__()
                self.started = time.perf_counter()
                self.shoot_at: float | None = (
                    None if args.after_report is not None else self.started + args.wait
                )
                self.saved = False

            def update(self) -> None:
                now = time.perf_counter()
                if self.shoot_at is None:
                    if any(e.sent for e in worker.events()):
                        run("speed", 1.0)
                        self.shoot_at = now + args.after_report
                        if args.follow:
                            view.toggle_follow()
                    return
                if args.follow and self.shoot_at - now > 1.5 and view.rig.follow is None:
                    view.toggle_follow()
                if args.distance:
                    view.rig.dist = args.distance
                if now >= self.shoot_at and not self.saved:
                    self.saved = True
                    args.screenshot.parent.mkdir(parents=True, exist_ok=True)
                    app.win.saveScreenshot(Filename.fromOsSpecific(str(args.screenshot.resolve())))
                    application.quit()

        Director()
    app.run()
    worker.stop()


if __name__ == "__main__":
    main()
