"""Spoken reports. Hand-written code-path strings only; never data."""

import importlib.util
import json
import os
from pathlib import Path

import pytest

from v2v.contracts import HazardType
from v2v.experiments import CONFIGS
from v2v.nlp import extract
from v2v.nlp.keyword_baseline import classify
from v2v.pipeline import run
from v2v.voice.listener import (
    BLOCK_SAMPLES,
    END_SILENCE_S,
    GATE_BLOCKS,
    GATE_SHARE,
    NO_SPEECH_S,
    RATE,
    VoiceFlow,
    is_gated,
    is_silent,
    is_wake_phrase,
    rms,
)
from v2v.voice.roman import romanise, strip_wake_words

BLOCK = bytes(BLOCK_SAMPLES * 2)
SPOKEN = "मुझे एक बंदर रोड पे नाचा हुआ दिख रहा है"
ROMAN = "mujhe ek bandar road pe naacha hua dikh raha hai"


@pytest.mark.parametrize(
    ("devanagari", "roman"),
    [
        (SPOKEN, ROMAN),
        ("फ्लाईओवर के पास एक्सीडेंट हुआ है", "flyover ke paas accident hua hai"),
        ("शायद सड़क पर गड्ढा है", "shayad sadak par gaddha hai"),
        ("करना देखना जल्दी", "karna dekhna jaldi"),  # silent "a" goes
        ("कमल", "kamal"),
        ("already roman", "already roman"),
        ("रोड है।", "road hai"),
    ],
)
def test_devanagari_becomes_the_hinglish_the_nlp_reads(devanagari: str, roman: str) -> None:
    assert romanise(devanagari) == roman


def test_wake_words_are_dropped_from_the_start_only() -> None:
    assert strip_wake_words("hey action mujhe gaddha dikha") == "mujhe gaddha dikha"
    assert strip_wake_words("action, road pe tel hai") == "road pe tel hai"
    assert strip_wake_words("koi action nahi hua") == "koi action nahi hua"


@pytest.mark.parametrize(
    ("heard", "wake"),
    [
        ("hey action", True),
        ("[unk] hey action", True),  # a breath or a click before it
        ("[unk] hey action [unk]", True),
        ("[unk] [unk] [unk] hey action", False),  # forced match in ordinary speech
        ("action", False),
        ("hey", False),
        ("", False),
    ],
)
def test_the_wake_phrase_must_start_an_utterance(heard: str, wake: bool) -> None:
    assert is_wake_phrase(heard) is wake


class FakeRecogniser:
    """Plays back a script, one entry per block: partial text, or ("final", text)."""

    def __init__(self, script: list[object] | None = None, final: str = "") -> None:
        self.script = list(script or [])
        self.final = final
        self.current: object = ""
        self.fed = 0
        self.flushed = 0

    def AcceptWaveform(self, data: bytes) -> bool:  # noqa: N802
        self.fed += 1
        self.current = self.script.pop(0) if self.script else ""
        return isinstance(self.current, tuple)

    def PartialResult(self) -> str:  # noqa: N802
        return json.dumps({"partial": self.current})

    def Result(self) -> str:  # noqa: N802
        assert isinstance(self.current, tuple)
        return json.dumps({"text": self.current[1]})

    def FinalResult(self) -> str:  # noqa: N802
        out, self.final = self.final, ""
        return json.dumps({"text": out})

    def Reset(self) -> None:  # noqa: N802
        self.flushed += 1
        self.current = ""


def _silence_until_done(flow: VoiceFlow, seconds: float) -> list[object]:
    out = []
    for _ in range(int(seconds / (BLOCK_SAMPLES / RATE)) + 2):
        report = flow.feed(BLOCK)
        if report is not None:
            out.append(report)
    return out


def test_wake_phrase_then_live_text_then_report() -> None:
    wake = FakeRecogniser(["", "hey", "hey action"])
    speech = FakeRecogniser(["", "मुझे", "मुझे एक", ("final", SPOKEN)])
    flow = VoiceFlow(wake, speech)
    flow.feed(BLOCK)
    flow.feed(BLOCK)
    assert flow.phase == "waiting" and speech.fed == 0  # the big model sleeps until woken
    flow.feed(BLOCK)
    assert flow.phase == "listening"
    assert speech.fed == 1  # replays the words said in the same breath
    flow.feed(BLOCK)
    assert flow.partial == "मुझे"
    flow.feed(BLOCK)
    assert flow.transcript() == "मुझे एक"
    flow.feed(BLOCK)
    assert flow.transcript() == SPOKEN and flow.phase == "listening"  # a pause may not be the end
    (report,) = _silence_until_done(flow, END_SILENCE_S)
    assert (report.devanagari, report.roman) == (SPOKEN, ROMAN)  # type: ignore[attr-defined]
    assert flow.phase == "waiting"


def test_a_second_opinion_can_reject_the_wake_phrase() -> None:
    no = FakeRecogniser(final="the traffic is very slow")
    flow = VoiceFlow(FakeRecogniser(["hey action"]), FakeRecogniser(), confirm=no)
    flow.feed(BLOCK)
    assert flow.phase == "waiting"
    yes = FakeRecogniser(final="hey action")
    flow = VoiceFlow(FakeRecogniser(["hey action"]), FakeRecogniser(), confirm=yes)
    flow.feed(BLOCK)
    assert flow.phase == "listening"


def test_silence_after_waking_gives_up_with_no_report() -> None:
    flow = VoiceFlow(FakeRecogniser(["hey action"]), FakeRecogniser())
    flow.feed(BLOCK)
    assert _silence_until_done(flow, NO_SPEECH_S) == []
    assert flow.phase == "waiting"


def test_tap_to_talk_skips_the_wake_phrase() -> None:
    flow = VoiceFlow(FakeRecogniser(), FakeRecogniser([("final", "गड्ढा है")]))
    flow.begin()
    flow.feed(BLOCK)
    (report,) = _silence_until_done(flow, END_SILENCE_S)
    assert report.roman == "gaddha hai"  # type: ignore[attr-defined]


def test_only_the_wake_words_said_is_not_a_report() -> None:
    flow = VoiceFlow(FakeRecogniser(["hey action"]), FakeRecogniser([("final", "हे एक्शन")]))
    flow.feed(BLOCK)
    assert _silence_until_done(flow, END_SILENCE_S) == []


def test_loudness() -> None:
    assert rms(bytes(100)) == 0.0
    assert rms(b"") == 0.0
    assert rms((8000).to_bytes(2, "little", signed=True) * 50) == pytest.approx(1.0)


@pytest.mark.parametrize("word", ["bandar", "monkey", "gaay", "kutta", "dog", "cattle"])
def test_animals_on_the_road_are_obstructions(word: str) -> None:
    # docs/annotation_guide.md lists animals under road_obstruction.
    assert classify("r", f"{word} road pe hai").hazard == HazardType.OBSTRUCTION


def test_filler_words_are_not_a_location() -> None:
    assert extract.extract_location(ROMAN) is None
    assert extract.extract_location("mujhe bandar dikha bellary road pe") == "bellary road"


def test_the_spoken_example_end_to_end() -> None:
    result = run(CONFIGS["proposed"], ROMAN)
    assert result.interpretation.hazard == HazardType.OBSTRUCTION
    assert result.assessed.status == "unverified"
    assert result.assessed.location is not None and result.assessed.location.approx
    assert result.warning is not None
    assert "bandar" not in result.warning.text  # template text, not the report
    assert "unverified" in result.warning.text and "approximate" in result.warning.text


@pytest.mark.parametrize("text", ["koi bandar nahi hai road pe", "traffic is fine today"])
def test_unsafe_reports_send_nothing(text: str) -> None:
    result = run(CONFIGS["proposed"], text)
    assert result.assessed.status == "suppressed" and result.warning is None


@pytest.mark.skipif(
    importlib.util.find_spec("traci") is None or not os.environ.get("V2V_NETWORK_TESTS"),
    reason="needs the [sim] extras and network access; set V2V_NETWORK_TESTS=1",
)
def test_voice_report_from_a_vehicle_on_the_real_network(tmp_path: Path) -> None:
    from v2v.sim.driver import DriverSession, verdict
    from v2v.sim.osm_network import CORRIDORS
    from v2v.sim.scenarios import generate

    (spec,) = generate(CORRIDORS["hebbal"], ["noisy"], ["medium"], [0], tmp_path / "networks")
    session = DriverSession(spec)
    session.open()
    try:
        while session.time_s < session.depart_s + 30:
            session.advance()
        sent = session.voice_report("demo_car", SPOKEN, f"{ROMAN} 9876543210")
        assert "[PHONE]" in sent.cleaned and "9876543210" not in sent.cleaned
        assert verdict(sent.event)[0] != "Blocked"
        assert "demo_car" not in sent.recipients and "demo_car" not in sent.delivered
        assert "demo_scooter" not in sent.recipients  # the other carriageway
        assert sent.event in session.live.events
        blocked = session.voice_report("demo_car", "", "traffic is fine today")
        assert verdict(blocked.event) == ("Blocked", "No road hazard in this report")
        assert blocked.recipients == ()
        while session.on_road("demo_car") and session.time_s < spec.end_s:
            session.advance()
        assert session.respawn("demo_car")
        session.advance()
        assert session.on_road("demo_car")
    finally:
        session.close()


def test_a_pause_after_the_wake_phrase_does_not_end_the_report() -> None:
    """The replayed tail of "action" reaches the speech model as a stray word;
    it must not shorten the wait for the real report."""
    wake = FakeRecogniser(["hey action"])
    speech = FakeRecogniser([("final", "क्शन")])  # what the replay produced
    flow = VoiceFlow(wake, speech)
    flow.feed(BLOCK)
    assert flow.phase == "listening" and not flow.said_something()
    assert _silence_until_done(flow, 3.0) == []  # a 3 s pause, longer than 1.2 s
    assert flow.phase == "listening"
    speech.script = [("final", "गड्ढा है")]
    flow.feed(BLOCK)
    assert flow.said_something()
    (report,) = _silence_until_done(flow, END_SILENCE_S)
    assert report.roman == "gaddha hai"  # type: ignore[attr-defined]


def test_wake_word_remnants_are_not_words() -> None:
    assert strip_wake_words("kshan mujhe gaddha dikha") == "mujhe gaddha dikha"
    assert strip_wake_words(romanise("क्शन")) == ""


def _event(slot_id: str, text: str = ROMAN):  # type: ignore[no-untyped-def]
    from v2v.sim.live import WarningEvent
    from v2v.sim.scenarios import ReportSlot

    slot = ReportSlot(
        slot_id=slot_id,
        role="original",
        needs="hazard",
        hazard_class=HazardType.AMBIGUOUS,
        reporter_id="rep",
        route=("a",),
        depart_s=0.0,
        delay_s=0.0,
        expected_report_s=0.0,
    )
    return WarningEvent(
        t_s=0.0,
        t_pass_s=0.0,
        slot=slot,
        text=text,
        text_source="fixture",
        result=run(CONFIGS["proposed"], text),
        outcome=None,
        sender_xy=(0.0, 0.0),
        centre_xy=(0.0, 0.0),
        radius_m=800.0,
    )


def test_spoken_events_are_recognisable() -> None:
    from v2v.sim.live import VOICE_SLOT

    assert _event(f"rep{VOICE_SLOT}1").spoken
    assert not _event("slot_0").spoken


@pytest.mark.parametrize(
    ("text", "word", "why"),
    [
        (ROMAN, "Sent as unverified", "The reading is confident, but nothing corroborates it yet"),
        ("koi bandar nahi hai road pe", "Blocked", "The report says the hazard is not there"),
        ("traffic is fine today", "Blocked", "No road hazard in this report"),
    ],
)
def test_the_verdict_says_whether_it_was_safe_to_send_and_why(
    text: str, word: str, why: str
) -> None:
    from v2v.sim.driver import verdict

    assert verdict(_event("rep-voice-1", text)) == (word, why)


def test_a_microphone_cut_to_digital_silence_is_detected() -> None:
    """A healthy microphone in a room never reads exactly zero; Windows or Realtek
    noise suppression that gates the signal does, and no speech model can read it."""
    assert is_silent(bytes(3200)) and not is_silent(bytes(3199) + b"")
    healthy = [False] * GATE_BLOCKS
    assert not is_gated(healthy)
    cut = [i % 3 == 0 for i in range(GATE_BLOCKS)]  # a third of the blocks, between words
    assert is_gated(cut)
    barely = [i < GATE_BLOCKS * GATE_SHARE * 0.5 for i in range(GATE_BLOCKS)]
    assert not is_gated(barely)  # a few quiet blocks are normal
    assert not is_gated([True] * 10)  # too little evidence yet
