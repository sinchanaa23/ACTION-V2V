"""Always-listening microphone: "hey ACTION", then a live transcript. Owner: P4.

    python -m v2v.voice.listener --download     # once: the two offline speech models
    python -m v2v.voice.listener --check        # nothing heard? test the microphone
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
import os
import queue
import threading
import time
import urllib.request
import wave
import zipfile
from collections import deque
from collections.abc import Callable, Sequence
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
GATE_BLOCKS = 60
"""Blocks (6 s) watched for a microphone signal that drops to digital silence."""
GATE_SHARE = 0.15
"""A real microphone in a room never reads exactly zero. If this share of recent
blocks is pure digital silence, something between the microphone and this program
(typically Windows' or Realtek's noise suppression, "audio enhancements") is cutting
the signal between and inside words, and no speech model can read what is left."""

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


def is_silent(pcm: bytes) -> bool:
    """A block with every sample exactly zero."""
    return pcm.count(0) == len(pcm)


def is_gated(recent: "Sequence[bool]") -> bool:
    """Whether the recent blocks show a cut-off signal (see GATE_SHARE)."""
    return len(recent) >= GATE_BLOCKS // 2 and sum(recent) / len(recent) >= GATE_SHARE


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
        limit = END_SILENCE_S if self.said_something() else NO_SPEECH_S
        if self._quiet >= limit or self._elapsed >= MAX_REPORT_S:
            return self._finish()
        return None

    def transcript(self) -> str:
        """Everything said so far, in Devanagari."""
        return " ".join([*self._done, self.partial]).strip()

    def said_something(self) -> bool:
        """Whether any real words came after the wake phrase. The tail of "action"
        replayed from the lookback does not count, or a pause before the report
        would end it after 1.2 s with nothing in it."""
        return bool(strip_wake_words(romanise(self.transcript())))

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
                self._quiet = 0.0  # words just arrived
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
        self.gated = False
        """The signal keeps dropping to digital silence: see GATE_SHARE."""
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
        self.phase, self.partial, self.level, self.gated = "off", "", 0.0, False

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
                device=input_device(),
            ):
                self.phase = "waiting"
                silent: deque[bool] = deque(maxlen=GATE_BLOCKS)
                while not self._halt.is_set():
                    if self._tap.is_set():
                        self._tap.clear()
                        flow.begin()
                    try:
                        pcm = blocks.get(timeout=0.2)
                    except queue.Empty:
                        continue
                    self.level = rms(pcm)
                    silent.append(is_silent(pcm))
                    self.gated = is_gated(silent)
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


GATE_ADVICE = (
    "Windows (or the sound driver) is cutting your microphone signal to digital silence,\n"
    "which no speech model can read. Turn off the microphone's noise suppression:\n"
    "  Windows: Settings > System > Sound > Input > pick your microphone > turn\n"
    "           'Audio enhancements' / 'Enhance audio' OFF.\n"
    "  Realtek: open 'Realtek Audio Console' > Microphone > turn Noise Suppression and\n"
    "           Beam Forming off.\n"
    "Then run this check again. Until then you can type reports instead."
)


def input_device() -> int | str | None:
    """The microphone to use: ``V2V_MIC`` (a device number or part of its name), else the
    system default. Windows often lists several; the default is not always the one that works."""
    choice = os.environ.get("V2V_MIC", "").strip()
    if not choice:
        return None
    return int(choice) if choice.isdigit() else choice


def list_inputs() -> list[str]:
    import sounddevice

    return [
        f"{i:3d}  {d['name']}  ({sounddevice.query_hostapis(d['hostapi'])['name']})"
        for i, d in enumerate(sounddevice.query_devices())
        if d["max_input_channels"] > 0
    ]


def check(seconds: float = 8.0) -> int:
    """Record once from the default microphone and say what is wrong, if anything."""
    import sounddevice

    print(
        f"Recording {seconds:.0f} s. Say: hey ACTION, then: mujhe ek bandar road pe dikh raha hai"
    )
    chunks: list[bytes] = []

    def capture(data: Any, frames: int, when: Any, status: Any) -> None:
        chunks.append(bytes(data))

    with sounddevice.RawInputStream(
        samplerate=RATE,
        blocksize=BLOCK_SAMPLES,
        channels=1,
        dtype="int16",
        callback=capture,
        device=input_device(),
    ):
        t0 = time.monotonic()
        while time.monotonic() - t0 < seconds:
            time.sleep(0.5)
            level = rms(chunks[-1]) if chunks else 0.0
            print(f"  level {level:.2f} " + "#" * int(level * 40))
    silent = [is_silent(c) for c in chunks]
    share = sum(silent) / max(len(silent), 1)
    peak = max((rms(c) for c in chunks), default=0.0)
    print(f"\nsilent blocks: {share:.0%}   loudest: {peak:.2f}")
    if share >= GATE_SHARE:
        print("PROBLEM:", GATE_ADVICE)
        return 1
    if peak < 0.03:
        print("PROBLEM: almost nothing was heard. Check the microphone is not muted and raise its")
        print("input volume (Settings > System > Sound > Input), or choose another microphone.")
        return 1
    if not models_ready():
        print("Signal looks healthy. Download the speech models next: --download")
        return 0
    flow = build_flow()
    audio = b"".join(chunks) + bytes(int(RATE * 2 * (END_SILENCE_S + 1)))
    for i in range(0, len(audio), BLOCK_SAMPLES * 2):
        heard = flow.feed(audio[i : i + BLOCK_SAMPLES * 2])
        if heard is not None:
            print(f"Signal looks healthy. It understood: {heard.roman}")
            return 0
    print("Signal looks healthy but no report was recognised. Say 'hey ACTION', pause, then the")
    print("report, a little slower and closer to the microphone, and try again.")
    return 0


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="test the microphone and explain")
    parser.add_argument("--devices", action="store_true", help="list the microphones")
    parser.add_argument("--mic", help="microphone number or name for --check (or set V2V_MIC)")
    parser.add_argument("--download", action="store_true", help="fetch the speech models")
    parser.add_argument("--wav", type=Path, help="read a 16 kHz mono WAV instead of the mic")
    args = parser.parse_args(argv)
    if args.mic:
        os.environ["V2V_MIC"] = args.mic
    if args.devices:
        print("\n".join(list_inputs()))
        print("\nTry one:  python -m v2v.voice.listener --check --mic <number>")
        return
    if args.check:
        raise SystemExit(check())
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
    print('say "hey ACTION" and then your report, Ctrl+C to stop')
    last = ""
    beat = 0.0
    try:
        while True:
            time.sleep(0.2)
            if time.monotonic() - beat > 3.0:
                beat = time.monotonic()
                print(f"  (mic level {listener.level:.2f}, {listener.phase})")
            if listener.error:
                raise SystemExit(listener.error)
            line = f"{listener.phase}: {romanise(listener.partial)}"
            if line != last:
                print(line)
                last = line
            for report in listener.take():
                print(f"REPORT: {report.roman}")
    except KeyboardInterrupt:
        listener.stop()


if __name__ == "__main__":
    main()
