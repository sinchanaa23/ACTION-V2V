"""PII scrubbing. Runs before anything is stored.

Biased towards over-scrubbing: losing a few tokens costs little, while a stored
phone number or plate is a privacy failure.
"""

import re

_STATE_CODES = (
    "AN|AP|AR|AS|BR|CG|CH|DD|DL|GA|GJ|HP|HR|JH|JK|KA|KL|LA|LD|MH|ML|MN|MP|MZ|"
    "NL|OD|OR|PB|PY|RJ|SK|TN|TR|TS|UK|UP|WB"
)
_SEP = r"[ .-]?"

_URL = re.compile(r"\b(?:https?://|www\.)\S+", re.IGNORECASE)
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
_REDDIT_USER = re.compile(r"(?<![\w/])/?u/[\w-]{2,}", re.IGNORECASE)
_HANDLE = re.compile(r"(?<!\w)@[\w.-]{2,}")
# State series (KA 01 AB 1234, DL 3C AB 1234) and Bharat series (22 BH 1234 AA).
_PLATE = re.compile(
    rf"\b(?:{_STATE_CODES}){_SEP}\d{{1,2}}{_SEP}[A-Z]{{0,1}}{_SEP}[A-Z]{{1,3}}{_SEP}\d{{1,4}}\b"
    rf"|\b\d{{2}}{_SEP}BH{_SEP}\d{{4}}{_SEP}[A-Z]{{1,2}}\b",
    re.IGNORECASE,
)
_PHONE = re.compile(
    r"(?<![\d])(?:(?:\+?91|0)[ -]?)?(?:[6-9]\d{4}[ -]?\d{5}|[6-9]\d{2}[ -]\d{3}[ -]\d{4})(?!\d)"
    r"|(?<![\d])0\d{2,4}[ -]\d{6,8}(?!\d)"
)

# Order matters: emails before handles, or "@domain.com" is taken for a handle.
_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (_URL, "[URL]"),
    (_EMAIL, "[EMAIL]"),
    (_REDDIT_USER, "[USER]"),
    (_HANDLE, "[USER]"),
    (_PLATE, "[PLATE]"),
    (_PHONE, "[PHONE]"),
)


def scrub(text: str) -> str:
    """Replace usernames, phone numbers, plates, emails and URLs with tags."""
    for pattern, tag in _RULES:
        text = pattern.sub(tag, text)
    return text.strip()
