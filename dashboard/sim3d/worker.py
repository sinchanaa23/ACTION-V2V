# ruff: noqa: E501 - long geometry and HUD expressions
"""Runs the live SUMO session on its own thread. Owner: P4.

TraCI is only ever touched from this thread. The render loop sends commands
(play, pause, step, reset, scenario, config, speed) and reads the latest
snapshot, so a slow SUMO step never stalls a frame.
"""

import queue
import threading
import time
import traceback
from dataclasses import dataclass, field

from v2v.contracts import VehicleState
from v2v.sim.live import LiveSession, WarningEvent
from v2v.sim.scenarios import ScenarioSpec


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


@dataclass(eq=False)  # a Thread: keep identity hashing, avoid its private names
class SimWorker(threading.Thread):
    specs: list[ScenarioSpec]
    index: int = 0
    config_name: str = "proposed"
    speed: float = 1.0
    """Simulated seconds per real second."""
    playing: bool = False
    status: str = "starting SUMO"
    error: str | None = None
    _cmds: "queue.Queue[tuple[str, object]]" = field(default_factory=queue.Queue)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _snapshot: Snapshot | None = None
    _version: int = 0
    _fired: list[FiredEvent] = field(default_factory=list)
    _session: LiveSession | None = None
    _halt: threading.Event = field(default_factory=threading.Event)

    def __post_init__(self) -> None:
        threading.Thread.__init__(self, daemon=True, name="sumo")

    # --- render-thread API -----------------------------------------------------

    @property
    def spec(self) -> ScenarioSpec:
        return self.specs[self.index]

    def send(self, cmd: str, arg: object = None) -> None:
        self._cmds.put((cmd, arg))

    def latest(self) -> tuple[int, Snapshot | None]:
        with self._lock:
            return self._version, self._snapshot

    def drain(self) -> list[FiredEvent]:
        with self._lock:
            out, self._fired = self._fired, []
            return out

    def events(self) -> list[WarningEvent]:
        with self._lock:
            return list(self._session.events) if self._session else []

    def stop(self) -> None:
        self._halt.set()

    # --- thread ------------------------------------------------------------------

    def run(self) -> None:
        try:
            self._load()
            next_step = time.perf_counter()
            while not self._halt.is_set():
                stepped = self._commands()
                session = self._session
                assert session is not None
                if self.playing and not session.finished:
                    now = time.perf_counter()
                    if now >= next_step:
                        self._step()
                        next_step = max(next_step + 1.0 / self.speed, now - 0.25)
                    else:
                        time.sleep(min(0.004, next_step - now))
                else:
                    if session.finished:
                        self.playing = False
                        self.status = "finished"
                    next_step = time.perf_counter()
                    if not stepped:
                        time.sleep(0.01)
        except Exception:  # surface SUMO failures in the HUD instead of dying silently
            self.error = traceback.format_exc(limit=3)
            self.status = "error"
        finally:
            if self._session is not None:
                self._session.close()

    def _commands(self) -> bool:
        stepped = False
        while True:
            try:
                cmd, arg = self._cmds.get_nowait()
            except queue.Empty:
                return stepped
            if cmd == "play":
                self.playing = True
                self.status = "running"
            elif cmd == "pause":
                self.playing = False
                self.status = "paused"
            elif cmd == "step":
                self.playing = False
                self.status = "paused"
                self._step()
                stepped = True
            elif cmd == "until":  # fast-forward (no waiting) to a simulated time
                session = self._session
                assert session is not None
                self.status = "fast-forward"
                while session.time_s < float(arg) and not session.finished:  # type: ignore[arg-type]
                    self._step()
                self.status = "running" if self.playing else "paused"
                stepped = True
            elif cmd == "speed":
                self.speed = float(arg)  # type: ignore[arg-type]
            elif cmd == "reset":
                self._load()
            elif cmd == "scenario":
                self.index = int(arg) % len(self.specs)  # type: ignore[call-overload]
                self._load()
            elif cmd == "config":
                self.config_name = str(arg)
                self._load()

    def _load(self) -> None:
        self.status = "starting SUMO"
        if self._session is not None:
            self._session.close()
        session = LiveSession(self.spec, config_name=self.config_name)
        session.open()
        with self._lock:
            self._session = session
            self._fired = []
        self.playing = False
        self._publish()
        self.status = "ready"

    def _step(self) -> None:
        session = self._session
        assert session is not None
        new = session.advance()
        if new:
            positions = {v.vehicle_id: (v.x, v.y) for v in session.vehicles()}
            for e in new:
                positions.setdefault(e.slot.reporter_id, e.sender_xy)
            with self._lock:
                self._fired += [FiredEvent(e, positions) for e in new]
        self._publish()

    def _publish(self) -> None:
        session = self._session
        assert session is not None
        vehicles = {v.vehicle_id: v for v in session.vehicles()}
        snap = Snapshot(
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
        )
        with self._lock:
            self._snapshot = snap
            self._version += 1
