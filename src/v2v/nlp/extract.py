"""Location, negation and hedging extraction. Owner: P2 (P2-02, P2-03, P2-06)."""

import re

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
        "market", "bazaar", "park", "office", "campus", "colony", "street", "marg",
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


def _is_name(token: str) -> bool:
    return token not in _NOT_A_NAME and token not in STRONG_PLACE_WORDS


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

    for first, last, strong in _place_word_spans(tokens):
        start = first
        while start > 0 and first - start < MAX_MODIFIERS and _is_name(tokens[start - 1][0]):
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


def extract_location(normalised_text: str) -> str | None:
    """Return the location phrase stated in the report, or None.

    The longest span wins (it names the place most precisely); ties go to the
    first one. Never guesses: no span in the text means None.
    """
    spans = find_locations(normalised_text)
    if not spans:
        return None
    return max(spans, key=lambda s: len(s.split()))


def detect_negation(normalised_text: str) -> bool:
    """True if the hazard is negated ("accident nahi hua").

    STUB(P2-02): always False.
    """
    return False


def detect_hedging(normalised_text: str) -> bool:
    """True if the report is hedged ("shayad", "lagta hai").

    STUB(P2-02): always False.
    """
    return False
