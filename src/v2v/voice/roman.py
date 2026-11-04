"""Devanagari speech output to the Romanised Hinglish the NLP stage reads. Owner: P4.

The Hindi speech model writes Devanagari, while the normaliser, the classifier
and the negation and hedging cues work on Romanised Hinglish. This respells the
words and never translates them. Common words and English loanwords come from a
table, because spelling them out by rule gives forms the NLP does not know
("रोड" would become "rod"). Everything else goes through the letter rules, with
Hindi's silent final and medial "a" dropped ("बंदर" -> "bandar").
"""

import re

_INDEPENDENT = {
    "अ": "a", "आ": "aa", "इ": "i", "ई": "i", "उ": "u", "ऊ": "u", "ऋ": "ri",
    "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au", "ऑ": "o",
}  # fmt: skip
_MATRA = {
    "ा": "aa", "ि": "i", "ी": "i", "ु": "u", "ू": "u", "ृ": "ri",
    "े": "e", "ै": "ai", "ो": "o", "ौ": "au", "ॉ": "o",
}  # fmt: skip
_CONSONANT = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "n",
    "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "n",
    "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n",
    "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m",
    "य": "y", "र": "r", "ल": "l", "व": "v",
    "श": "sh", "ष": "sh", "स": "s", "ह": "h",
}  # fmt: skip
_NUKTA = {"ड": "d", "ढ": "dh", "क": "q", "ख": "kh", "ग": "g", "ज": "z", "फ": "f"}
_VIRAMA = "्"
_NUKTA_SIGN = "़"
_NASAL = {"ं", "ँ"}

WORDS: dict[str, str] = {
    # everyday Hindi the NLP cues expect, spelled the way they do
    "मुझे": "mujhe", "मैं": "main", "में": "mein", "है": "hai", "हैं": "hain", "हूं": "hoon",
    "हुआ": "hua", "हुई": "hui", "हुए": "hue", "था": "tha", "थी": "thi", "रहा": "raha",
    "रही": "rahi", "रहे": "rahe", "गया": "gaya", "गई": "gayi", "गयी": "gayi", "गए": "gaye",
    "नहीं": "nahi", "ना": "na", "एक": "ek", "पे": "pe", "पर": "par", "के": "ke", "की": "ki",
    "का": "ka", "को": "ko", "से": "se", "और": "aur", "पास": "paas", "आगे": "aage",
    "पीछे": "peeche", "सामने": "saamne", "बहुत": "bahut", "शायद": "shayad", "भाई": "bhai",
    "सड़क": "sadak", "गड्ढा": "gaddha", "गड्ढे": "gaddhe", "पानी": "paani", "गाड़ी": "gaadi",
    "दिख": "dikh", "रहा।": "raha", "बंदर": "bandar", "गाय": "gaay", "कुत्ता": "kutta",
    "कुत्ते": "kutte", "भैंस": "bhains", "सांड": "saand", "जानवर": "janwar",
    # English the speech model hears as Hindi
    "रोड": "road", "एक्सीडेंट": "accident", "एक्सिडेंट": "accident", "ट्रक": "truck",
    "कार": "car", "बस": "bus", "बाइक": "bike", "फ्लाईओवर": "flyover", "फ्लाइओवर": "flyover",
    "ब्रिज": "bridge", "सिग्नल": "signal", "जंक्शन": "junction", "ट्रैफिक": "traffic",
    "ब्रेकडाउन": "breakdown", "पंचर": "puncture", "आयल": "oil", "ऑयल": "oil", "मंकी": "monkey",
    "मेट्रो": "metro", "मॉल": "mall", "हॉस्पिटल": "hospital", "स्कूल": "school",
    "हे": "hey", "एक्शन": "action",
}  # fmt: skip

_DEVANAGARI_RUN = re.compile(r"[ऀ-ॿ]+")
_DANDA = re.compile(r"[।॥]")

# Letter classes while spelling a word: (consonant, vowel, explicit vowel?)
_Unit = tuple[str, str, bool]


def _units(word: str) -> list[_Unit]:
    units: list[_Unit] = []
    i = 0
    while i < len(word):
        ch = word[i]
        if ch in _CONSONANT:
            sound = _CONSONANT[ch]
            if i + 1 < len(word) and word[i + 1] == _NUKTA_SIGN:
                sound = _NUKTA.get(ch, sound)
                i += 1
            nxt = word[i + 1] if i + 1 < len(word) else ""
            if nxt == _VIRAMA:
                units.append((sound, "", True))
                i += 2
                continue
            if nxt in _MATRA:
                units.append((sound, _MATRA[nxt], True))
                i += 2
                continue
            units.append((sound, "a", False))  # the inherent, droppable "a"
        elif ch in _INDEPENDENT:
            units.append(("", _INDEPENDENT[ch], True))
        elif ch in _NASAL and units:
            cons, vowel, fixed = units[-1]
            units[-1] = (cons, vowel + "n", True)
        i += 1
    return units


def _spell(word: str) -> str:
    units = _units(word)
    # Hindi drops the inherent "a" at the end of a word, and between consonants
    # when a vowel sounds on both sides of it ("करना" -> "karna").
    if units and not units[-1][2] and len(units) > 1:
        units[-1] = (units[-1][0], "", True)
    for i in range(1, len(units) - 1):
        cons, vowel, fixed = units[i]
        if not fixed and vowel == "a" and units[i - 1][1] and units[i + 1][1]:
            units[i] = (cons, "", True)
    text = "".join(c + v for c, v, _ in units)
    return text[:-1] if text.endswith("aa") and len(text) > 3 else text


def romanise(text: str) -> str:
    """Spoken Hindi in Devanagari as Romanised Hinglish; Latin text stays as it is."""

    def word(match: re.Match[str]) -> str:
        w = match.group(0)
        return WORDS.get(w, _spell(w))

    return _DEVANAGARI_RUN.sub(word, _DANDA.sub(" ", text)).strip()


WAKE_WORDS = ("hey", "he", "hay", "action", "aekshan", "ekshan")


def strip_wake_words(text: str) -> str:
    """Drop the end of "hey ACTION" when it runs into the start of the report."""
    words = text.split()
    while words and words[0].lower().strip(",.!?") in WAKE_WORDS:
        words.pop(0)
    return " ".join(words)
