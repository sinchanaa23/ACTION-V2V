"""P2-02. Hand-written code-path strings only; never data."""

import pytest

from v2v.nlp import extract


@pytest.mark.parametrize(
    "text",
    [
        "accident nahi hua hai",
        "aage koi accident nahi hai bhai",
        "there is no accident ahead",
        "it wasn't a crash, just a slow truck",
        "pothole nhi hai ab",
    ],
)
def test_explicit_negation(text: str) -> None:
    result = extract.find_negation(text)
    assert result is not None and result[0] == "explicit"
    assert extract.detect_negation(text)


@pytest.mark.parametrize(
    "text",
    ["accident cleared now", "truck hata diya, road clear", "jam khul gaya"],
)
def test_cleared_is_negation_of_kind_cleared(text: str) -> None:
    result = extract.find_negation(text)
    assert result is not None and result[0] == "cleared"


@pytest.mark.parametrize(
    "text",
    [
        "bhai aage accident hua hai bridge ke paas",
        "no entry board ke paas accident hua",  # "no entry" is a sign, not negation
        "pata nahi kya hua, accident lag raha hai",  # "pata nahi" is a hedge
        "slow down nahi to accident hoga",  # "nahi to" means "otherwise"
        "traffic is not moving, big accident ahead",  # negation stops at the clause
    ],
)
def test_not_negated(text: str) -> None:
    assert not extract.detect_negation(text)


def test_negator_out_of_scope_is_ignored() -> None:
    # The Hindi negator scopes backwards only; the hazard word comes after it.
    assert not extract.detect_negation("ruka nahi koi bhi, gaddha bahut bada hai")


@pytest.mark.parametrize(
    ("text", "cue"),
    [
        ("shayad accident hua hai", "shayad"),
        ("i think there is oil on the road", "i think"),
        ("suna hai flyover pe truck palat gaya", "suna hai"),
        ("pata nahi kya hua, jam hai", "pata nahi"),
        ("looks like a breakdown near the signal", "looks like"),
    ],
)
def test_hedges(text: str, cue: str) -> None:
    assert extract.find_hedge(text) == cue
    assert extract.detect_hedging(text)


def test_plain_report_not_hedged() -> None:
    assert not extract.detect_hedging("bhai aage accident hua hai bridge ke paas")


@pytest.mark.parametrize(
    "text",
    [
        "the stalled truck has moved now",
        "water drained away, route is fine",
        "the fallen tree was towed away",
        "signal is working again",
        "gaadi chali gayi, road theek ho gaya",
    ],
)
def test_more_cleared_phrasings(text: str) -> None:
    result = extract.find_negation(text)
    assert result is not None and result[0] == "cleared"


@pytest.mark.parametrize(
    "text",
    [
        "i couldn't tell what it was, maybe debris",
        "wasn't sure if it was an accident",
        "hard to tell from far, something on the road",
        "kuch pada hai, samajh nahi aaya kya hai",
        "it was unclear whether the lane was blocked",
    ],
)
def test_more_hedges(text: str) -> None:
    assert extract.detect_hedging(text)


def test_uncertainty_phrases_are_not_negation() -> None:
    # "wasn't sure ... accident" is doubt about the hazard, not a denial of it.
    assert not extract.detect_negation("wasn't sure if it was an accident")
    assert not extract.detect_negation("couldn't tell if the accident was serious")


@pytest.mark.parametrize(
    "text", ["road is not clear now", "lane isn't cleared yet, truck still there"]
)
def test_not_clear_is_not_cleared_or_hedged(text: str) -> None:
    result = extract.find_negation(text)
    assert result is None or result[0] != "cleared"
    assert not extract.detect_hedging(text)
