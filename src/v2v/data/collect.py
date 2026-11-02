"""Collectors for the real data sources. Owner: P1 (P1-01, P1-02, P1-08).

Each collector returns DriverReport records with PII already scrubbed and
``source`` set to the real platform. No collector generates or rewrites text.
The per-source modules keep extra fields (time, city) in their JSONL output.
"""

from pathlib import Path

from v2v.contracts import DriverReport
from v2v.data import reddit as reddit_source
from v2v.data import survey as survey_source
from v2v.data import youtube as youtube_source
from v2v.data.scrub import scrub

__all__ = ["reddit", "scrub", "survey", "youtube"]

_PAGES_PER_VIDEO = 3


def survey(export_csv: str) -> list[DriverReport]:
    """Consent-based driver survey export."""
    return [rec.to_report() for rec in survey_source.read_responses(Path(export_csv))]


def reddit(subreddits: list[str], limit: int) -> list[DriverReport]:
    """Official API via praw, rate-limited."""
    cities = {name: reddit_source.SUBREDDIT_CITIES.get(name) for name in subreddits}
    records = reddit_source.collect(reddit_source.client_from_env(), cities, limit=limit)
    return [rec.to_report() for rec in records]


def youtube(video_ids: list[str]) -> list[DriverReport]:
    """YouTube Data API v3 comments."""
    client = youtube_source.client_from_env()
    return [
        rec.to_report()
        for video_id in video_ids
        for rec in youtube_source.comments_of(client, video_id, _PAGES_PER_VIDEO)
    ]
