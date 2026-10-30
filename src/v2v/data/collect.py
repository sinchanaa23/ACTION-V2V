"""Collectors for the real data sources. Owner: P1 (P1-01, P1-02, P1-08).

Each collector returns DriverReport records with PII already scrubbed and
``source`` set to the real platform. No collector generates or rewrites text.
"""

from v2v.contracts import DriverReport


def survey(export_csv: str) -> list[DriverReport]:
    """STUB(P1-01): consent-based driver survey export."""
    raise NotImplementedError("P1-01")


def reddit(subreddits: list[str], limit: int) -> list[DriverReport]:
    """STUB(P1-02): official API via praw, rate-limited."""
    raise NotImplementedError("P1-02")


def youtube(video_ids: list[str]) -> list[DriverReport]:
    """STUB(P1-02): YouTube Data API v3 comments."""
    raise NotImplementedError("P1-02")


def scrub(text: str) -> str:
    """STUB(P1-02): remove usernames, phone numbers and plate numbers."""
    raise NotImplementedError("P1-02")
