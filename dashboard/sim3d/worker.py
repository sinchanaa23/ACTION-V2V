# ruff: noqa: E501 - long geometry and HUD expressions
"""Runs the live SUMO session in its own process. Owner: P4.

TraCI and the pipeline run in a child process; the render loop sends commands
(play, pause, step, reset, scenario, config, speed, until) and reads the
latest snapshot. A separate process rather than a thread: a simulation thread
that falls behind (high speed, a busy CPU) steps back to back in Python and
takes the interpreter lock every time the renderer lets go of it, which
starved the window down to a frame every few seconds.
"""

import multiprocessing as mp
import time
import traceback
from dataclasses import dataclass, field
from multiprocessing.connection import Connection
from typing import Any

from v2v.contracts import VehicleState
from v2v.data.scrub import scrub
from v2v.sim.live import LiveSession, WarningEvent
from v2v.sim.scenarios import ScenarioSpec

MAX_BEHIND_S = 0.25
"""Never try to catch up more than this: fall back to real time instead."""


@dataclass(frozen=True)
class Snapshot:
    t_s: float
    vehicles: dict[str, VehicleState]
    blocking: dict[str, VehicleState]
    relevant: frozenset[str]
    """Ground truth now: vehicles that will reach the hazard."""
    sensor_flags: int
    event_active: bool
    finished: bool


@dataclass(frozen=True)
class FiredEvent:
    event: WarningEvent
    positions: dict[str, tuple[float, float]]
    """SUMO x/y of every vehicle at send time, for drawing the V2V hops."""


# --- the child process ---------------------------------------------------------------


@dataclass
class _Sim:
    conn: Connection
    specs: list[ScenarioSpec]
    index: int
    config_name: str
    speed: float
    radius_m: float | None = None
    scripted: bool = True
    playing: bool = False
    status: str = "starting SUMO"
    session: LiveSession | None = None
    pending: list[tuple[str, Any]] = field(default_factory=list)
    """Commands that arrived during a fast-forward, applied once it is done."""

    def state(self, error: str | None = None) -> None:
        self.conn.send(
            (
                "state",
                {
                    "index": self.index,
                    "config_name": self.config_name,
                    "speed": self.speed,
                    "radius_m": self.radius_m,
                    "playing": self.playing,
                    "status": self.status,
                    "error": error,
                },
            )
        )

    def load(self) -> None:
        self.status = "starting SUMO"
        self.state()
        if self.session is not None:
            self.session.close()
        self.session = LiveSession(
            self.specs[self.index],
            config_name=self.config_name,
            radius_m=self.radius_m,
            scripted=self.scripted,
        )
        self.session.open()
        self.playing = False
        self.conn.send(("reset", None))
        self.publish()
        self.status = "ready"
        self.state()

    def step(self) -> None:
        session = self.session
        assert session is not None
        new = session.advance()
        if new:
            positions = {v.vehicle_id: (v.x, v.y) for v in session.vehicles()}
            for e in new:
                positions.setdefault(e.slot.reporter_id, e.sender_xy)
            self.conn.send(("fired", [FiredEvent(e, positions) for e in new]))
        self.publish()

    def voice(self, vehicle_id: str, roman: str) -> None:
        """A driver's spoken report, now, through the pipeline the scenario's use.
        Personal details are scrubbed first."""
        session = self.session
        assert session is not None
        try:
            event = session.report_from(vehicle_id, scrub(roman))
        except KeyError:
            self.conn.send(("note", f"{vehicle_id} is not on the road any more"))
            return
        positions = {v.vehicle_id: (v.x, v.y) for v in session.vehicles()}
        positions.setdefault(vehicle_id, event.sender_xy)
        self.conn.send(("fired", [FiredEvent(event, positions)]))
        self.publish()

    def publish(self) -> None:
        session = self.session
        assert session is not None
        vehicles = {v.vehicle_id: v for v in session.vehicles()}
        self.conn.send(
            (
                "snap",
                Snapshot(
                    t_s=session.time_s,
                    vehicles=vehicles,
                    blocking={v.vehicle_id: v for v in session.blocking()},
                    # Nobody needs a warning about an event that never happened.
                    relevant=frozenset(session.will_reach(vehicles))
                    if session.spec.event_occurred
                    else frozenset(),
                    sensor_flags=session.world.sensor_flags(session.spec.hazard, 200.0),
                    event_active=session.event_active,
                    finished=session.finished,
                ),
            )
        )

    def command(self, cmd: str, arg: Any) -> bool:
        """Apply one command; False to quit."""
        session = self.session
        assert session is not None
        if cmd == "quit":
            return False
        if cmd == "play":
            self.playing, self.status = True, "running"
        elif cmd == "pause":
            self.playing, self.status = False, "paused"
        elif cmd == "step":
            self.playing, self.status = False, "paused"
            self.step()
        elif cmd == "until":  # fast-forward to a simulated time, showing progress
            self.status = "fast-forward"
            self.state()
            last = time.perf_counter()
            while session.time_s < float(arg) and not session.finished:
                new = session.advance()
                if new:
                    positions = {v.vehicle_id: (v.x, v.y) for v in session.vehicles()}
                    for e in new:
                        positions.setdefault(e.slot.reporter_id, e.sender_xy)
                    self.conn.send(("fired", [FiredEvent(e, positions) for e in new]))
                if time.perf_counter() - last > 0.25:
                    last = time.perf_counter()
                    self.publish()
                while self.conn.poll():
                    later = self.conn.recv()
                    if later[0] == "quit":
                        return False
                    self.pending.append(later)  # play, speed... after the skip
            self.publish()
            self.status = "running" if self.playing else "paused"
        elif cmd == "voice":
            self.voice(*arg)
        elif cmd == "radius":  # applies to the next report; nothing restarts
            self.radius_m = float(arg)
            session.radius_m = self.radius_m
        elif cmd == "speed":
            self.speed = float(arg)
        elif cmd == "reset":
            self.load()
        elif cmd == "scenario":
            self.index = int(arg) % len(self.specs)
            self.load()
        elif cmd == "config":
            self.config_name = str(arg)
            self.load()
        self.state()
        return True

    def run(self) -> None:
        self.load()
        next_step = time.perf_counter()
        while True:
            while self.pending or self.conn.poll():
                cmd, arg = self.pending.pop(0) if self.pending else self.conn.recv()
                if not self.command(cmd, arg):
                    return
                next_step = time.perf_counter()
            session = self.session
            assert session is not None
            if self.playing and not session.finished:
                now = time.perf_counter()
                if now >= next_step:
                    self.step()
                    next_step = max(next_step + 1.0 / self.speed, now - MAX_BEHIND_S)
                else:
                    self.conn.poll(min(0.01, next_step - now))
            else:
                if session.finished and self.playing:
                    self.playing, self.status = False, "finished"
                    self.state()
                self.conn.poll(0.05)


def _serve(
    conn: Connection,
    specs: list[ScenarioSpec],
    index: int,
    config_name: str,
    speed: float,
    radius_m: float | None,
    scripted: bool,
) -> None:
    sim = _Sim(conn, specs, index, config_name, speed, radius_m, scripted)
    try:
        sim.run()
    except (EOFError, BrokenPipeError, ConnectionError):
        pass  # the window closed
    except Exception:  # surface SUMO failures in the HUD instead of dying silently
        try:
            sim.status = "error"
            sim.state(traceback.format_exc(limit=3))
        except OSError:
            pass
    finally:
        if sim.session is not None:
            sim.session.close()


# --- the render side -------------------------------------------------------------------


@dataclass(eq=False)
class SimWorker:
    """The render side's handle on the simulation process."""

    specs: list[ScenarioSpec]
    index: int = 0
    config_name: str = "proposed"
    speed: float = 1.0
    """Simulated seconds per real second."""
    radius_m: float | None = None
    """Geocast radius in metres (None: the config's own)."""
    scripted: bool = True
    """Fire the scenario's own reports; off for the sandbox."""
    playing: bool = False
    status: str = "starting SUMO"
    error: str | None = None
    _conn: Connection | None = None
    _proc: Any = None
    _snapshot: Snapshot | None = None
    _version: int = 0
    _fired: list[FiredEvent] = field(default_factory=list)
    _events: list[WarningEvent] = field(default_factory=list)
    _notes: list[str] = field(default_factory=list)

    @property
    def spec(self) -> ScenarioSpec:
        return self.specs[self.index]

    def start(self) -> None:
        ctx = mp.get_context("spawn")
        here, there = ctx.Pipe()
        self._conn = here
        self._proc = ctx.Process(
            target=_serve,
            args=(
                there,
                self.specs,
                self.index,
                self.config_name,
                self.speed,
                self.radius_m,
                self.scripted,
            ),
            daemon=True,
            name="sumo",
        )
        self._proc.start()
        there.close()

    def stop(self) -> None:
        if self._conn is not None:
            try:
                self._conn.send(("quit", None))
            except OSError:
                pass
        if self._proc is not None:
            self._proc.join(timeout=3)
            if self._proc.is_alive():
                self._proc.terminate()

    def send(self, cmd: str, arg: object = None) -> None:
        if self._conn is not None:
            try:
                self._conn.send((cmd, arg))
            except OSError:
                self.error = self.error or "the simulation process has stopped"

    def _pump(self) -> None:
        """Take in everything the simulation has sent; keep only the latest snapshot."""
        conn = self._conn
        if conn is None:
            return
        try:
            while conn.poll():
                kind, data = conn.recv()
                if kind == "snap":
                    self._snapshot = data
                    self._version += 1
                elif kind == "fired":
                    self._fired += data
                    self._events += [f.event for f in data]
                elif kind == "note":
                    self._notes.append(data)
                elif kind == "reset":
                    self._fired, self._events = [], []
                elif kind == "state":
                    for key, value in data.items():
                        if key != "error" or value is not None:
                            setattr(self, key, value)
        except (EOFError, OSError):
            self.status = "error"
            self.error = self.error or "the simulation process has stopped"
            self._conn = None

    def latest(self) -> tuple[int, Snapshot | None]:
        self._pump()
        return self._version, self._snapshot

    def drain(self) -> list[FiredEvent]:
        self._pump()
        out, self._fired = self._fired, []
        return out

    def notes(self) -> list[str]:
        """Messages from the simulation process for the player, once each."""
        self._pump()
        out, self._notes = self._notes, []
        return out

    def events(self) -> list[WarningEvent]:
        self._pump()
        return list(self._events)
