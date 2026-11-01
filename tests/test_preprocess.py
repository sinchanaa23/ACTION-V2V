"""Hand-written code-path strings only. Never training data."""

import json

import pytest

from v2v.nlp import preprocess
from v2v.nlp.preprocess import mine_variants, normalise


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("  Bhai   AAGEY  gadi\tkharab ", "bhai aage gaadi kharab"),
        ("accident nhi hua", "accident nahi hua"),
        ("bhaiiiii sloooow", "bhaii sloow"),
        ("speed 1000 km", "speed 1000 km"),
        ("my age is 30", "my age is 30"),
        ("call [PHONE] now", "call [phone] now"),
        ("सड़क पर  पानी", "सड़क पर पानी"),
    ],
)
def test_normalise(text: str, expected: str) -> None:
    assert normalise(text) == expected


def test_normalise_is_idempotent() -> None:
    once = normalise("Bhaiii aagey accidnt huwa")
    assert normalise(once) == once


def test_variants_map_only_to_final_forms() -> None:
    variants = json.loads(preprocess.VARIANTS_PATH.read_text(encoding="utf-8"))
    assert not set(variants) & set(variants.values())
    assert all(key == key.lower() for key in variants)


def test_mine_variants_groups_respellings_for_review() -> None:
    texts = ["khatra ahead", "khatra khatra here", "khatraa close", "bridge bridge bridge"]
    assert mine_variants(texts) == {"khatra": {"khatraa": 1}}


def test_mine_variants_skips_rare_and_known_forms() -> None:
    assert mine_variants(["khatra khatraa"]) == {}
    assert mine_variants(["trafic traffic traffic traffic"]) == {}


def test_skeleton_ignores_vowels_and_doubles() -> None:
    assert preprocess._skeleton("gaddha") == preprocess._skeleton("gadha")
