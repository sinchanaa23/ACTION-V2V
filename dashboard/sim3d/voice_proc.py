"""The microphone for the 3D view, in its own process. Owner: P4.

Speech recognition is heavy, and a thread inside the render process would take
frames from the window (the SUMO thread did exactly that, see worker.py). So the
microphone, the wake phrase and the speech models run in a child process; the
render side keeps a handle with the latest state and the finished reports.
"""

import multiprocessing as mp
import time
from dataclasses import dataclass, field
from multiprocessing.connection import Connection
from typing import Any

STATE_EVERY_S = 0.1


def _serve(conn: Connection) -> None:
    from v2v.voice.listener import Listener
    from v2v.voice.roman import romanise

    mic = Listener()
    last_sent = 0.0
    try:
        while True:
            while conn.poll():
                cmd = conn.recv()
                if cmd == "quit":
                    return
                if cmd == "start":
                    mic.start()
                elif cmd == "stop":
                    mic.stop()
                elif cmd == "tap":
                    mic.tap()
            now = time.monotonic()
            if now - last_sent >= STATE_EVERY_S:
                last_sent = now
                conn.send(("state", mic.phase, romanise(mic.partial), mic.level, mic.error))
            for heard in mic.take():
                conn.send(("report", heard.devanagari, heard.roman))
            conn.poll(0.03)
    except (EOFError, BrokenPipeError, ConnectionError):
        pass  # the window closed
    finally:
        mic.stop()


@dataclass(eq=False)
class VoiceHandle:
    """The render side's handle on the microphone process."""

    phase: str = "off"
    """off, loading, waiting (for the wake phrase) or listening (to a report)."""
    words: str = ""
    """What is being said, live, in Hinglish."""
    level: float = 0.0
    error: str | None = None
    _conn: Connection | None = None
    _proc: Any = None
    _reports: list[tuple[str, str]] = field(default_factory=list)

    @property
    def active(self) -> bool:
        return self.phase != "off"

    def start(self) -> None:
        """Turn the microphone on (starts its process the first time)."""
        if self._proc is None or not self._proc.is_alive():
            ctx = mp.get_context("spawn")
            here, there = ctx.Pipe()
            self._conn = here
            self._proc = ctx.Process(target=_serve, args=(there,), daemon=True, name="voice")
            self._proc.start()
            there.close()
        self.phase, self.error = "loading", None
        self._send("start")

    def stop(self) -> None:
        self._send("stop")
        self.phase, self.words, self.level = "off", "", 0.0

    def toggle(self) -> None:
        self.stop() if self.active else self.start()

    def tap(self) -> None:
        """Start a report without the wake phrase."""
        if self.active:
            self._send("tap")

    def close(self) -> None:
        self._send("quit")
        if self._proc is not None:
            self._proc.join(timeout=3)
            if self._proc.is_alive():
                self._proc.terminate()

    def _send(self, cmd: str) -> None:
        if self._conn is not None:
            try:
                self._conn.send(cmd)
            except OSError:
                self.error = "the microphone process has stopped"
                self.phase = "off"

    def poll(self) -> list[tuple[str, str]]:
        """Take in what the microphone process sent; returns finished reports as
        (heard in Devanagari, Romanised Hinglish)."""
        conn = self._conn
        if conn is not None:
            try:
                while conn.poll():
                    kind, *data = conn.recv()
                    if kind == "state" and self.phase != "off":
                        self.phase, self.words, self.level, error = data
                        self.error = error or self.error
                    elif kind == "report":
                        self._reports.append((data[0], data[1]))
            except (EOFError, OSError):
                self.phase, self.error = "off", "the microphone process has stopped"
                self._conn = None
        out, self._reports = self._reports, []
        return out
