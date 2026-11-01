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
