"""P2-03. Hand-written code-path strings only; never data."""

import pytest

from v2v.nlp.extract import extract_location, find_locations


@pytest.mark.parametrize(
    ("text", "location"),
    [
        ("bhai aage accident hua hai bridge ke paas slow down", "bridge"),
        ("truck broke down on hebbal flyover", "hebbal flyover"),
        ("huge pothole just after silk board junction", "silk board junction"),
        ("oil spill near the petrol pump", "petrol pump"),
        ("esteem ke paas gaadi palat gayi", "esteem"),
        ("accident near esteem mall", "esteem mall"),
        ("jam on outer ring road", "outer ring road"),
        ("tree fallen on bellary road", "bellary road"),
        ("signal pe accident", "signal"),
    ],
)
def test_location_span(text: str, location: str) -> None:
    assert extract_location(text) == location


@pytest.mark.parametrize(
    "text",
    [
        "bhai aage accident hua hai slow down",
        "oil on the road, careful",  # "road" alone is not a location
        "speeding car behind me",
        "gaadi ke peeche truck hai",  # "peeche" is direction, not proximity
        "near me there is a jam",
    ],
)
def test_no_location_stays_none(text: str) -> None:
    assert extract_location(text) is None


def test_span_is_copied_verbatim() -> None:
    text = "accident at 100 feet road and also near the metro"
    for span in find_locations(text):
        assert span in text
    assert find_locations(text) == ["100 feet road", "metro"]


def test_longest_span_wins() -> None:
    assert extract_location("signal ke baad, hebbal flyover pe jam") == "hebbal flyover"
