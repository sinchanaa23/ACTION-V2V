"""Stage 1: normalise informal English/Hinglish text. Owner: P1 (P1-05)."""

import re

_SPACES = re.compile(r"\s+")


def normalise(text: str) -> str:
    """Return a normalised form of ``text`` for the classifiers.

    STUB(P1-05): lowercases and collapses whitespace only. The real version adds
    the spelling-variant list built from the collected data (aage/age, ...).
    """
    return _SPACES.sub(" ", text).strip().lower()


def language_mix(text: str) -> str:
    """Tag a report as ``en``, ``hi-en`` or ``unknown``.

    STUB(P1-04): always ``unknown``.
    """
    return "unknown"
