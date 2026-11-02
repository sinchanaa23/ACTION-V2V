"""Location, negation and hedging extraction. Owner: P2 (P2-02, P2-03, P2-06)."""

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

# --- Rule-based location spans (P2-03) --------------------------------------
#
# A location is returned only as a span copied from the report, so it can never
# be invented. Turning the span into a point on the network is the landmark
# list's job (World.resolve_location); a span it can't resolve goes to the
# fallback policy.

_TOKEN = re.compile(r"[\w']+")

MAX_MODIFIERS = 3
"""Name tokens kept to the left of a place word ("silk board junction")."""

# Place words that locate a hazard on their own ("bridge ke paas").
STRONG_PLACE_WORDS: frozenset[str] = frozenset(
    {
        "bridge", "flyover", "pul", "underpass", "overbridge", "signal", "junction",
        "circle", "chowk", "chauraha", "mall", "lake", "talab", "talaab", "hospital",
        "school", "college", "temple", "mandir", "masjid", "church", "gurudwara", "metro",
        "station", "toll", "tollgate", "petrol pump", "petrol bunk", "bus stop",
        "bus stand", "traffic light", "traffic signal", "toll plaza", "outer ring road",
        "ring road", "orr", "service road", "airport", "stadium", "depot",
    }
)  # fmt: skip

# Place words too generic to locate anything without a name ("oil on the
# road" has no location; "bellary road" does).
WEAK_PLACE_WORDS: frozenset[str] = frozenset(
    {
        "road", "rd", "main", "cross", "highway", "expressway", "nagar", "layout", "gate",
        "market", "bazaar", "park", "campus", "colony", "street", "marg",
    }
)  # fmt: skip

# Never part of a location name: function words, pronouns, directions, verbs,
# vehicles and hazard words.
_NOT_A_NAME: frozenset[str] = frozenset(
    {
        "a", "an", "the", "this", "that", "is", "are", "was", "were", "be", "been", "it",
        "its", "there", "here", "near", "at", "on", "in", "of", "to", "from", "after",
        "before", "under", "over", "opposite", "beside", "behind", "front", "next", "by",
        "and", "or", "but", "just", "right", "left", "ahead", "back", "side", "lane",
        "me", "my", "you", "your", "we", "our", "he", "she", "they", "i", "bhai", "bro",
        "yaar", "boss", "guys", "please", "pls", "ke", "ki", "ka", "ko", "se", "pe",
        "par", "mein", "mai", "paas", "pass", "aage", "age", "peeche", "piche",
        "samne", "saamne", "neeche", "niche", "upar", "baad", "pehle", "wala", "wali",
        "wale", "hai", "hain", "h", "tha", "thi", "hua", "hui", "hue", "ho", "gaya",
        "gayi", "gaye", "raha", "rahi", "rahe", "kar", "karo", "koi", "kuch", "bahut",
        "bohot", "bada", "badi", "big", "huge", "small", "slow", "down", "careful",
        "accident", "crash", "pothole", "gaddha", "oil", "jam", "traffic", "water",
        "paani", "pani", "tree", "debris", "breakdown", "truck", "car", "bus", "bike",
        "auto", "gaadi", "gadi", "scooty", "lorry", "tempo", "cab", "vehicle", "someone",
        "very", "too", "so", "also", "now", "abhi", "today", "aaj", "not", "no", "nahi",
    }
)  # fmt: skip

# "<name> ke paas", "near <name>", "opposite <name>": a proximity phrase marks
# a location even without a place word.
_PROXIMITY_AFTER = re.compile(r"\b(?:ke|ki)\s+(?:paas|pass|saamne|samne|bagal)\b")
_PROXIMITY_BEFORE = re.compile(r"\b(?:near|opposite|beside|outside|next to|in front of)\s")


def _token_spans(text: str) -> list[tuple[str, int, int]]:
    return [(m.group(0), m.start(), m.end()) for m in _TOKEN.finditer(text)]


# Words describing the hazard or the road's state, never part of a place name
# ("slippery road" is a hazard, not a road called Slippery).
_ROAD_STATE_WORDS: frozenset[str] = frozenset(
    {
        "slippery", "wet", "oily", "muddy", "flooded", "waterlogged", "blocked", "jammed",
        "broken", "damaged", "bad", "narrow", "dark", "busy", "empty", "closed", "same",
        "kharab", "toota", "tooti", "gila", "geela",
        # hedges and fillers: "shayad flyover ke paas"
        "shayad", "shaayad", "maybe", "probably", "lagta", "dhyan", "please", "pls", "jaldi",
        # movement verbs: "reached home", "going to market"
        "reached", "reach", "going", "went", "came", "coming", "leaving", "crossed", "passing",
    }
)  # fmt: skip


def _is_name(token: str) -> bool:
    return (
        token not in _NOT_A_NAME
        and token not in STRONG_PLACE_WORDS
        and token not in _ROAD_STATE_WORDS
        and token not in HAZARD_WORDS
    )


def _place_word_spans(tokens: list[tuple[str, int, int]]) -> list[tuple[int, int, bool]]:
    """(first token, last token, strong) for each place word, longest first."""
    found: list[tuple[int, int, bool]] = []
    taken: set[int] = set()
    for size in (3, 2, 1):
        for i in range(len(tokens) - size + 1):
            if taken & set(range(i, i + size)):
                continue
            phrase = " ".join(t[0] for t in tokens[i : i + size])
            if phrase in STRONG_PLACE_WORDS or phrase in WEAK_PLACE_WORDS:
                found.append((i, i + size - 1, phrase in STRONG_PLACE_WORDS))
                taken.update(range(i, i + size))
    return sorted(found)


def find_locations(normalised_text: str) -> list[str]:
    """Every location span in the report, in text order. Spans are copied
    verbatim from ``normalised_text``."""
    tokens = _token_spans(normalised_text)
    spans: list[tuple[int, int]] = []  # token index ranges, inclusive

    places = _place_word_spans(tokens)
    in_place = {i for first, last, strong in places if strong for i in range(first, last + 1)}

    def joined(i: int) -> bool:
        """Token i-1 and token i are separated by spaces only (no comma etc.)."""
        return not normalised_text[tokens[i - 1][2] : tokens[i][1]].strip()

    for first, last, strong in places:
        start = first
        while (
            start > 0
            and first - start < MAX_MODIFIERS
            and joined(start)
            and start - 1 not in in_place
            and _is_name(tokens[start - 1][0])
        ):
            start -= 1
        # Weak place words ("road", "cross") need a name: "bellary road".
        if strong or start < first:
            spans.append((start, last))

    for match in _PROXIMITY_AFTER.finditer(normalised_text):
        before = [i for i, t in enumerate(tokens) if t[2] <= match.start()]
        names: list[int] = []
        for i in reversed(before[-2:]):
            if not _is_name(tokens[i][0]):
                break
            names.insert(0, i)
        if names:
            spans.append((names[0], names[-1]))

    for match in _PROXIMITY_BEFORE.finditer(normalised_text):
        after = [i for i, t in enumerate(tokens) if t[1] >= match.end()]
        names = []
        for i in after[:2]:
            if not _is_name(tokens[i][0]):
                break
            names.append(i)
        if names:
            spans.append((names[0], names[-1]))

    # Drop spans contained in a longer one, then keep text order.
    kept = [s for s in spans if not any(o != s and o[0] <= s[0] and s[1] <= o[1] for o in spans)]
    return [normalised_text[tokens[a][1] : tokens[b][2]] for a, b in sorted(set(kept))]


SPAN_TAGGER_ENV_VAR = "V2V_SPAN_TAGGER"
"""Path to a trained span tagger (P2-06). When set and the file exists,
extract_location uses it instead of the rules; choose on validation."""


@lru_cache(maxsize=2)
def _span_tagger(path: str) -> Any:
    from v2v.nlp.span_tagger import SpanTagger

    return SpanTagger.load(Path(path)) if Path(path).exists() else None


def extract_location(normalised_text: str) -> str | None:
    """Return the location phrase stated in the report, or None.

    The longest span wins (it names the place most precisely); ties go to the
    first one. Never guesses: no span in the text means None. Uses the
    trained span tagger instead of the rules when V2V_SPAN_TAGGER points to one.
    """
    tagger_path = os.environ.get(SPAN_TAGGER_ENV_VAR)
    tagger = _span_tagger(tagger_path) if tagger_path else None
    if tagger is not None:
        found: str | None = tagger.extract_location(normalised_text)
        return found
    spans = find_locations(normalised_text)
    if not spans:
        return None
    return max(spans, key=lambda s: len(s.split()))


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
    r"(?:was|is|were|are)n'?t sure|not certain|"
    r"(?:could|can)(?: not|n'?t) (?:tell|say|see|work out|make out)|"
    r"pata nahi|pata nahin|pata nhi|samajh nahi|dikha nahi|nahi to|nahin to|warna)\b"
)

# The hazard has gone. Reported as negated so no new alert is raised, with
# ``kind="cleared"`` so P3 can treat it as a contradiction of an earlier report.
_CLEARED = re.compile(
    # Not after "not"/"n't": "road is not clear now" means it is still blocked.
    r"\b(?<!not )(?<!n't )(?:cleared|clear now|road is clear|road clear|all clear|"
    r"been removed|removed|cleaned up|hata diya|hata diye|hat gaya|hat gayi|"
    r"saaf ho gaya|saaf ho gayi|"
    r"clear ho gaya|clear ho gayi|khul gaya|khul gayi|reopened|open now|"
    r"(?:has|had|have|been) (?:moved|gone|shifted|towed|fixed|repaired|drained)|"
    r"moved away|towed away|drained away|swept (?:aside|away)|no longer there|"
    r"working again|back to normal|normal now|chala gaya|chali gayi|theek ho gaya|"
    r"theek ho gayi|nikal gaya|nikal gayi)\b"
)

_HEDGES = re.compile(
    r"\b(?:shayad|shaayad|lagta hai|lagta h|lag raha hai|lag rha hai|lagti hai|"
    r"ho sakta hai|ho sakti hai|pata nahi|pata nahin|pata nhi|suna hai|suna h|"
    r"sunne mein aaya|kisi ne bataya|keh rahe the|maybe|may be|perhaps|possibly|"
    r"probably|might|i think|i guess|not sure|seems|seem to|looks like|apparently|"
    r"reportedly|i heard|heard that|someone said|people are saying|could be|rumou?r|"
    r"unsure|unclear|(?:was|is|were|are)n'?t sure|not certain|"
    r"(?:could|can)(?: not|n'?t) (?:tell|say|see|work out|make out)|hard to (?:tell|say)|"
    r"seemed|looked like|samajh nahi aaya|samajh nahi aa raha|dikha nahi|confirm nahi)\b"
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
