"""Hand-written code-path strings only. Never training data."""

import pytest

from v2v.data.common import RawRecord, make_record
from v2v.data.filter import filter_candidates, in_net, summary
from v2v.nlp.preprocess import language_mix


def _record(text: str, source: str = "fixture", source_id: str = "1") -> RawRecord:
    rec = make_record(text, source, source_id)  # type: ignore[arg-type]
    assert rec is not None
    return rec


@pytest.mark.parametrize(
    "text",
    [
        "truck crashed near the bridge",
        "accident nahi hua yahan",
        "road pe bahut paani hai",
        "सड़क पर बड़ा गड्ढा है",
    ],
)
def test_net_keeps_hazards_and_negated_near_misses(text: str) -> None:
    assert in_net(text)


@pytest.mark.parametrize(
    "text",
    ["best biryani in town tonight", "accident", " ".join(["accident"] * 130)],
)
def test_net_drops_off_topic_and_bad_lengths(text: str) -> None:
    assert not in_net(text)


@pytest.mark.parametrize(
    ("text", "tag"),
    [
        ("Bhai aage accident hua hai", "hi-en"),
        ("सड़क पर गड्ढा है", "hi-en"),
        ("truck stopped on the left lane", "en"),
        ("accident", "unknown"),
        ("12345 !!!", "unknown"),
        ("சாலை விபத்து நடந்தது இங்கே", "unknown"),
    ],
)
def test_language_mix(text: str, tag: str) -> None:
    assert language_mix(text) == tag


def test_filter_removes_near_duplicates_and_prefers_survey() -> None:
    pytest.importorskip("rapidfuzz")
    records = [
        _record("big accident near the flyover today", "reddit", "a"),
        _record("big accident near the flyover today!", "survey", "b"),
        _record("oil spill on the left lane, go slow", "youtube", "c"),
        _record("nice weather for a walk today", "reddit", "d"),
    ]

    candidates = filter_candidates(records)

    assert {c.source_id for c in candidates} == {"b", "c"}
    assert {c.language for c in candidates} == {"en"}
    stats = summary(len(records), candidates)
    assert stats["raw"] == 4
    assert stats["candidates"] == 2
    assert stats["by_source"] == {"survey": 1, "youtube": 1}
