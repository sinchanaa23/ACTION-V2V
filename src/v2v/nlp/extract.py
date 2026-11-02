"""Location, negation and hedging extraction. Owner: P2 (P2-02, P2-03, P2-06)."""

import re


def extract_location(normalised_text: str) -> str | None:
    """Return the location phrase stated in the report, or None.

    STUB(P2-03): always None. Must never guess a location.
    """
    return None


# --- Negation and hedging (P2-02) -------------------------------------------
#
# Cue lists cover English and Romanised Hinglish. They are matched on the
# normalised (lowercased) text from nlp.preprocess. A negation cue only counts
# when a hazard word is inside its scope window: English negators scope
# forwards ("no accident"), Hindi negators follow the verb and scope backwards
# ("accident nahi hua").

_WORD = re.compile(r"[\w']+")
# Negation never scopes across a clause ("not moving, accident ahead").
_CLAUSE_BREAK = re.compile(r"[,.;:!?|]+|\b(?:but|lekin|magar|par|and|aur)\b")

NEGATION_SCOPE = 4
"""Tokens between a negation cue and the hazard word it negates."""

# Words that name a hazard. Only used to scope negation, never to classify.
HAZARD_WORDS: frozenset[str] = frozenset(
    {
        "accident", "accidents", "crash", "crashed", "collision", "collided", "takkar",
        "takar", "thoka", "thok", "palat", "palti", "overturned", "flipped", "pothole",
        "potholes", "gaddha", "gadda", "gaddhe", "khadda", "oil", "diesel", "slippery",
        "fisalan", "kichad", "mud", "waterlogging", "waterlogged", "flood", "flooded",
        "paani", "pani", "debris", "tree", "ped", "jam", "blocked", "block", "breakdown",
        "broken", "kharab", "stalled", "puncture", "stuck", "speeding", "rash",
        "hazard", "obstruction", "barricade", "construction", "digging", "khudai",
    }
)  # fmt: skip

_EN_NEGATORS: frozenset[str] = frozenset(
    {
        "no", "not", "never", "without", "nothing", "none", "isn't", "isnt", "wasn't",
        "wasnt", "aren't", "arent", "weren't", "werent", "don't", "dont", "didn't",
        "didnt", "hasn't", "hasnt", "haven't", "havent", "ain't",
    }
)  # fmt: skip
# "na" is left out: it is far more often a tag question ("accident hua na?").
_HI_NEGATORS: frozenset[str] = frozenset({"nahi", "nahin", "nahee", "nai", "nhi", "nehi"})

# Phrases that contain a negator but do not negate a hazard. They are blanked
# out before negation is checked; several are hedges instead ("pata nahi").
_PSEUDO_NEGATIONS = re.compile(
    r"\b(?:no doubt|not sure|not only|no idea|no entry|no parking|no u ?turn|"
    r"pata nahi|pata nahin|pata nhi|nahi to|nahin to|warna)\b"
)

# The hazard has gone. Reported as negated so no new alert is raised, with
# ``kind="cleared"`` so P3 can treat it as a contradiction of an earlier report.
_CLEARED = re.compile(
    r"\b(?:cleared|clear now|road is clear|road clear|all clear|been removed|removed|"
    r"cleaned up|hata diya|hata diye|hat gaya|hat gayi|saaf ho gaya|saaf ho gayi|"
    r"clear ho gaya|clear ho gayi|khul gaya|khul gayi|reopened|open now)\b"
)

_HEDGES = re.compile(
    r"\b(?:shayad|shaayad|lagta hai|lagta h|lag raha hai|lag rha hai|lagti hai|"
    r"ho sakta hai|ho sakti hai|pata nahi|pata nahin|pata nhi|suna hai|suna h|"
    r"sunne mein aaya|kisi ne bataya|keh rahe the|maybe|may be|perhaps|possibly|"
    r"probably|might|i think|i guess|not sure|seems|seem to|looks like|apparently|"
    r"reportedly|i heard|heard that|someone said|people are saying|could be|rumou?r)\b"
)


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text)


def find_negation(normalised_text: str) -> tuple[str, str] | None:
    """Return ``(kind, cue)`` if the report negates its hazard, else None.

    ``kind`` is ``"explicit"`` ("no accident", "accident nahi hua") or
    ``"cleared"`` ("accident cleared now").
    """
    cleared = _CLEARED.search(normalised_text)
    if cleared:
        return ("cleared", cleared.group(0))
    for clause in _CLAUSE_BREAK.split(_PSEUDO_NEGATIONS.sub(" ", normalised_text)):
        tokens = _tokens(clause)
        for i, token in enumerate(tokens):
            if token in _EN_NEGATORS:
                scope = tokens[i + 1 : i + 1 + NEGATION_SCOPE]
            elif token in _HI_NEGATORS:
                scope = tokens[max(0, i - NEGATION_SCOPE) : i]
            else:
                continue
            if any(t in HAZARD_WORDS for t in scope):
                return ("explicit", token)
    return None


def find_hedge(normalised_text: str) -> str | None:
    """Return the first hedging cue ("shayad", "i think", "suna hai"), or None."""
    match = _HEDGES.search(normalised_text)
    return match.group(0) if match else None


def detect_negation(normalised_text: str) -> bool:
    """True if the hazard is negated ("accident nahi hua") or reported cleared."""
    return find_negation(normalised_text) is not None


def detect_hedging(normalised_text: str) -> bool:
    """True if the report is hedged ("shayad", "lagta hai")."""
    return find_hedge(normalised_text) is not None
