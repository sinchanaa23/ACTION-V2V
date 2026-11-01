"""Stage 1: normalise informal English/Hinglish text. Owner: P1 (P1-05)."""

import re

_SPACES = re.compile(r"\s+")
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_OTHER_SCRIPT = re.compile(r"[^\W\d_a-zA-ZÀ-ɏ]")
_LATIN_WORD = re.compile(r"[a-z]+")
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

    STUB(P1-05): lowercases and collapses whitespace only. The real version adds
    the spelling-variant list built from the collected data (aage/age, ...).
    """
    return _SPACES.sub(" ", text).strip().lower()


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
