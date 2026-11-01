"""Stage 1: normalise informal English/Hinglish text. Owner: P1 (P1-05)."""

import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path

VARIANTS_PATH = Path(__file__).with_name("variants.json")
"""Reviewed ``{variant: canonical}`` pairs. English words stay out, so "age" is
not mapped to "aage"."""
MIN_VARIANT_LENGTH = 3

_SPACES = re.compile(r"\s+")
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_OTHER_SCRIPT = re.compile(r"[^\W\d_a-zA-ZÀ-ɏ]")
_LATIN_WORD = re.compile(r"[a-z]+")
_STRETCHED = re.compile(r"([a-z])\1{2,}")
_VOWELS = re.compile(r"[aeiouyh]")
_DOUBLED = re.compile(r"(.)\1+")
# Romanised Hindi that is not also an English word, so "to", "me" and "the" stay out.
_HINDI_MARKERS: frozenset[str] = frozenset(
    {
        "aage", "abhi", "achha", "accha", "aur", "bahut", "bhai", "bhaiya", "bohot", "chal",
        "dekho", "gaadi", "gadi", "gaya", "gayi", "gaye", "hai", "hain", "hoga", "hua", "hui",
        "idhar", "kal", "kuch", "kya", "kyun", "lagta", "lekin", "mein", "nahi", "nahin",
        "nhi", "paani", "paas", "pani", "pata", "raha", "rahe", "rahi", "sadak", "shayad",
        "tha", "thi", "thoda", "udhar", "wahan", "yaar", "yahan",
    }
)  # fmt: skip


def normalise(text: str) -> str:
    """Return a normalised form of ``text`` for the classifiers.

    Lowercases, unifies Unicode forms, collapses whitespace, shortens stretched
    letters ("bhaiiii") and applies the reviewed spelling-variant list. Words are
    only respelled, never translated or added.
    """
    lowered = unicodedata.normalize("NFKC", text).lower()
    squeezed = _STRETCHED.sub(r"\1\1", _SPACES.sub(" ", lowered).strip())
    variants = _variants()
    return _LATIN_WORD.sub(lambda m: variants.get(m.group(0), m.group(0)), squeezed)


@lru_cache(maxsize=1)
def _variants() -> dict[str, str]:
    loaded: dict[str, str] = json.loads(VARIANTS_PATH.read_text(encoding="utf-8"))
    return loaded


def _skeleton(word: str) -> str:
    """Consonant outline, the part that survives informal respelling."""
    outline = _VOWELS.sub("", word[1:])
    return word[0] + _DOUBLED.sub(r"\1", outline)


def mine_variants(
    texts: Iterable[str], min_count: int = 3, min_ratio: float = 0.7
) -> dict[str, dict[str, int]]:
    """Group likely respellings under their most frequent form.

    Output is ``{canonical: {variant: count}}`` for a person to review. Many
    groups are false friends, so nothing here is applied automatically: approved
    pairs are added to variants.json by hand.
    """
    counts = Counter(
        word
        for text in texts
        for word in _LATIN_WORD.findall(_STRETCHED.sub(r"\1\1", text.lower()))
        if len(word) >= MIN_VARIANT_LENGTH
    )
    groups: defaultdict[str, list[str]] = defaultdict(list)
    for word in counts:
        groups[_skeleton(word)].append(word)
    known = _variants()
    mined: dict[str, dict[str, int]] = {}
    for words in groups.values():
        canonical = min(words, key=lambda w: (-counts[w], w))
        if counts[canonical] < min_count:
            continue
        similar = {
            word: counts[word]
            for word in words
            if word != canonical
            and word not in known
            and SequenceMatcher(None, word, canonical).ratio() >= min_ratio
        }
        if similar:
            mined[canonical] = similar
    return mined


def language_mix(text: str) -> str:
    """Tag a report as ``en``, ``hi-en`` or ``unknown``.

    Devanagari or an unambiguous romanised Hindi word means ``hi-en``. Other
    scripts, and texts too short to judge, stay ``unknown``.
    """
    if _DEVANAGARI.search(text):
        return "hi-en"
    if _OTHER_SCRIPT.search(text):
        return "unknown"
    words = _LATIN_WORD.findall(text.lower())
    if not _HINDI_MARKERS.isdisjoint(words):
        return "hi-en"
    return "en" if len(words) >= 2 else "unknown"


def main() -> None:
    """Mine spelling-variant candidates from collected JSONL for review."""
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=Path("data/raw/variant_candidates.json"))
    args = parser.parse_args()
    texts = [
        json.loads(line)["text"]
        for path in args.inputs
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    mined = mine_variants(texts)
    args.out.write_text(json.dumps(mined, indent=2, ensure_ascii=False), encoding="utf-8")
    print(len(mined), "groups to review in", args.out)


if __name__ == "__main__":
    main()
