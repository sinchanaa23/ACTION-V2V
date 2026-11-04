"""Always-listening microphone: "hey ACTION", then a live transcript. Owner: P4.

    python -m v2v.voice.listener --download     # once: the two offline speech models
    python -m v2v.voice.listener                # try it: prints what it hears
    python -m v2v.voice.listener --wav test.wav # the same from a 16 kHz mono file

Runs entirely on this machine with Vosk: nothing is sent anywhere, and audio is
not stored. While waiting, only a tiny recogniser limited to the wake phrase
runs, which is cheap. After "hey ACTION" the Hindi model transcribes live until
the speaker pauses. ``VoiceFlow`` holds that logic and takes any recognisers,
so it is tested without a microphone.
"""

import argparse
import json
import math
import queue
import threading
import time
import urllib.request
import wave
import zipfile
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from v2v.voice.roman import romanise, strip_wake_words

MODEL_DIR = Path("sim_out/voice_models")
"""Git-ignored, and not under models/: the experiment runner hashes that folder."""
WAKE_MODEL = "vosk-model-small-en-in-0.4"
SPEECH_MODEL = "vosk-model-small-hi-0.22"
MODEL_URL = "https://alphacephei.com/vosk/models/{}.zip"
RATE = 16_000
BLOCK_SAMPLES = 1_600  # 0.1 s
WAKE_GRAMMAR = ["hey action", "[unk]"]
MAX_WORDS_BEFORE_WAKE = 1
"""The phrase must start an utterance. A limited grammar forces odd matches in
the middle of ordinary speech ("[unk] [unk] [unk] hey action"), so talking to a
passenger does not open the microphone."""
LOOKBACK_S = 0.6
"""Audio replayed from before the wake phrase was recognised, so the first words
of "hey ACTION, mujhe..." said in one breath are not cut off."""
CONFIRM_S = 2.0
"""Audio re-read by the full English model to confirm a wake phrase."""
CONFIRM_WORDS = frozenset({"action", "actions", "auction", "axon", "ashen"})
"""What the full model may write for "action" in an Indian accent."""
END_SILENCE_S = 1.2
"""A pause this long ends the report."""
NO_SPEECH_S = 7.0
"""Give up if nothing is said this long after the wake phrase."""
MAX_REPORT_S = 25.0

Phase = Literal["waiting", "listening"]


class Recogniser(Protocol):
    """The part of Vosk's KaldiRecognizer used here."""

    def AcceptWaveform(self, data: bytes) -> bool: ...  # noqa: N802
    def PartialResult(self) -> str: ...  # noqa: N802
    def Result(self) -> str: ...  # noqa: N802
    def FinalResult(self) -> str: ...  # noqa: N802
    def Reset(self) -> None: ...  # noqa: N802


@dataclass(frozen=True)
class Heard:
    """A finished spoken report."""

    devanagari: str
    roman: str


def _text(result: str, key: str = "text") -> str:
    return str(json.loads(result).get(key, "")).strip()


def is_wake_phrase(heard: str) -> bool:
    """True when "hey action" starts what the wake recogniser heard."""
    words = heard.split()
    for i in range(len(words) - 1):
        if words[i] == "hey" and words[i + 1] == "action":
            return i <= MAX_WORDS_BEFORE_WAKE and all(w == "[unk]" for w in words[:i])
    return False


def _flush(recogniser: Recogniser) -> None:
    """Clear a recogniser completely: Reset alone can leave its last partial."""
    recogniser.FinalResult()
    recogniser.Reset()


class VoiceFlow:
    """Wake word, then transcript, driven one block of audio at a time."""

    def __init__(
        self,
        wake: Recogniser,
        speech: Recogniser,
        rate: int = RATE,
        confirm: Recogniser | None = None,
    ) -> None:
        self.wake, self.speech, self.rate, self.confirm = wake, speech, rate, confirm
        self.phase: Phase = "waiting"
        self.partial = ""
        """Live text so far, in Devanagari."""
        self._done: list[str] = []
        self._recent: deque[bytes] = deque()
        self._recent_bytes = 0
        self._elapsed = 0.0
        self._quiet = 0.0

    def begin(self) -> None:
        """Start a report now, without the wake phrase (tap to talk)."""
        _flush(self.wake)
        _flush(self.speech)
        self.phase = "listening"
        self.partial = ""
        self._done = []
        self._elapsed = self._quiet = 0.0

    def feed(self, pcm: bytes) -> Heard | None:
        """Take one block of 16-bit mono audio; returns a report when one ends."""
        seconds = len(pcm) / 2 / self.rate
        if self.phase == "waiting":
            self._remember(pcm)
            done = self.wake.AcceptWaveform(pcm)
            heard = (
                _text(self.wake.Result()) if done else _text(self.wake.PartialResult(), "partial")
            )
            if is_wake_phrase(heard):
                recent = b"".join(self._recent)
                if self._confirmed(recent):
                    self.begin()
                    self._listen(recent[-int(LOOKBACK_S * self.rate) * 2 :])
                else:
                    _flush(self.wake)
                    self._recent.clear()
                    self._recent_bytes = 0
            return None
        self._elapsed += seconds
        self._quiet += seconds
        self._listen(pcm)
        limit = NO_SPEECH_S if not self.transcript() else END_SILENCE_S
        if self._quiet >= limit or self._elapsed >= MAX_REPORT_S:
            return self._finish()
        return None

    def transcript(self) -> str:
        """Everything said so far, in Devanagari."""
        return " ".join([*self._done, self.partial]).strip()

    def _confirmed(self, recent: bytes) -> bool:
        """A second opinion from a model free to hear any word, not just the phrase."""
        if self.confirm is None:
            return True
        self.confirm.AcceptWaveform(recent)
        words = _text(self.confirm.FinalResult()).split()
        self.confirm.Reset()
        return any(w in CONFIRM_WORDS for w in words)

    def _remember(self, pcm: bytes) -> None:
        self._recent.append(pcm)
        self._recent_bytes += len(pcm)
        while self._recent_bytes > CONFIRM_S * self.rate * 2:
            self._recent_bytes -= len(self._recent.popleft())

    def _listen(self, pcm: bytes) -> None:
        if self.speech.AcceptWaveform(pcm):
            text = _text(self.speech.Result())
            if text:
                self._done.append(text)
            self.partial = ""
            return
        partial = _text(self.speech.PartialResult(), "partial")
        if partial != self.partial:
            self._quiet = 0.0  # words are still arriving
            self.partial = partial

    def _finish(self) -> Heard | None:
        last = _text(self.speech.FinalResult())
        if last:
            self._done.append(last)
        spoken = " ".join(self._done).strip()
        self.phase = "waiting"
        self.partial = ""
        self._done = []
        self._recent.clear()
        self._recent_bytes = 0
        _flush(self.wake)
        roman = strip_wake_words(romanise(spoken))
        return Heard(spoken, roman) if roman else None


def models_ready(root: Path = MODEL_DIR) -> bool:
    return all((root / name).is_dir() for name in (WAKE_MODEL, SPEECH_MODEL))


def download_models(root: Path = MODEL_DIR) -> None:
    """Fetch and unpack the speech models (about 80 MB in all)."""
    root.mkdir(parents=True, exist_ok=True)
    for name in (WAKE_MODEL, SPEECH_MODEL):
        if (root / name).is_dir():
            continue
        archive = root / f"{name}.zip"
        print(f"downloading {name}")
        urllib.request.urlretrieve(MODEL_URL.format(name), archive)  # noqa: S310
        with zipfile.ZipFile(archive) as z:
            z.extractall(root)
        archive.unlink()


def build_flow(root: Path = MODEL_DIR) -> VoiceFlow:
    """A flow on the real Vosk models."""
    import vosk

    vosk.SetLogLevel(-1)
    english = vosk.Model(str(root / WAKE_MODEL))
    wake = vosk.KaldiRecognizer(english, RATE, json.dumps(WAKE_GRAMMAR))
    speech = vosk.KaldiRecognizer(vosk.Model(str(root / SPEECH_MODEL)), RATE)
    return VoiceFlow(wake, speech, confirm=vosk.KaldiRecognizer(english, RATE))


class Listener:
    """The microphone on its own thread. The dashboard reads ``phase``, ``partial``
    and ``level``, and takes finished reports with ``take``."""

    def __init__(self, flow_factory: Callable[[], VoiceFlow] = build_flow) -> None:
        self._factory = flow_factory
        self._flow: VoiceFlow | None = None
        self._thread: threading.Thread | None = None
        self._halt = threading.Event()
        self._tap = threading.Event()
        self._reports: queue.Queue[Heard] = queue.Queue()
        self.phase: Literal["off", "loading", "waiting", "listening"] = "off"
        self.partial = ""
        self.level = 0.0
        self.error: str | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.running:
            return
        self._halt.clear()
        self.error = None
        self.phase = "loading"
        self._thread = threading.Thread(target=self._run, daemon=True, name="voice")
        self._thread.start()

    def stop(self) -> None:
        self._halt.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        self._thread = None
        self.phase, self.partial, self.level = "off", "", 0.0

    def tap(self) -> None:
        """Start a report without the wake phrase."""
        self._tap.set()

    def take(self) -> list[Heard]:
        out = []
        while True:
            try:
                out.append(self._reports.get_nowait())
            except queue.Empty:
                return out

    def _run(self) -> None:
        try:
            import sounddevice

            flow = self._flow = self._factory()
            blocks: queue.Queue[bytes] = queue.Queue()

            def capture(data: Any, frames: int, when: Any, status: Any) -> None:
                blocks.put(bytes(data))

            with sounddevice.RawInputStream(
                samplerate=RATE,
                blocksize=BLOCK_SAMPLES,
                channels=1,
                dtype="int16",
                callback=capture,
            ):
                self.phase = "waiting"
                while not self._halt.is_set():
                    if self._tap.is_set():
                        self._tap.clear()
                        flow.begin()
                    try:
                        pcm = blocks.get(timeout=0.2)
                    except queue.Empty:
                        continue
                    self.level = rms(pcm)
                    report = flow.feed(pcm)
                    self.phase, self.partial = flow.phase, flow.transcript()
                    if report is not None:
                        self._reports.put(report)
        except Exception as e:  # no microphone, no models: say so in the interface
            self.error = f"{type(e).__name__}: {e}"
            self.phase = "off"


def rms(pcm: bytes) -> float:
    """Loudness of a block of 16-bit audio, 0..1."""
    samples = memoryview(pcm).cast("h")
    if not len(samples):
        return 0.0
    return min(1.0, math.sqrt(sum(s * s for s in samples) / len(samples)) / 8_000)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="fetch the speech models")
    parser.add_argument("--wav", type=Path, help="read a 16 kHz mono WAV instead of the mic")
    args = parser.parse_args(argv)
    if args.download:
        download_models()
        return
    if args.wav:
        flow = build_flow()
        with wave.open(str(args.wav)) as w:
            if w.getframerate() != RATE or w.getnchannels() != 1:
                parser.error("needs 16 kHz mono audio")
            frames = w.readframes(w.getnframes()) + bytes(int(RATE * 2 * (END_SILENCE_S + 1)))
        was = flow.phase
        for i in range(0, len(frames), BLOCK_SAMPLES * 2):
            report = flow.feed(frames[i : i + BLOCK_SAMPLES * 2])
            if flow.phase != was:
                print(f"[{i / 2 / RATE:5.1f} s] {flow.phase}")
                was = flow.phase
            if report is not None:
                print(f"heard: {report.devanagari}\nroman: {report.roman}")
        return
    listener = Listener()
    listener.start()
    print('say "hey ACTION", Ctrl+C to stop')
    last = ""
    try:
        while True:
            time.sleep(0.2)
            if listener.error:
                raise SystemExit(listener.error)
            line = f"{listener.phase}: {listener.partial}"
            if line != last:
                print(line)
                last = line
            for report in listener.take():
                print(f"REPORT: {report.roman}")
    except KeyboardInterrupt:
        listener.stop()


if __name__ == "__main__":
    main()
