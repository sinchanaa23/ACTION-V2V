"""Reddit collector (D2). Official API only, read-only credentials.

Run: python -m v2v.data.reddit --out data/raw/reddit.jsonl
Needs REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET and REDDIT_USER_AGENT.
"""

import argparse
import importlib
import os
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from v2v.data.common import RawRecord, append_new, make_record, with_retry

SUBREDDIT_CITIES = {
    "bangalore": "Bengaluru",
    "delhi": "Delhi",
    "mumbai": "Mumbai",
    "chennai": "Chennai",
    "hyderabad": "Hyderabad",
}
QUERIES = (
    "accident",
    "waterlogging",
    "pothole",
    "breakdown",
    "traffic jam",
    "road blocked",
    "flyover",
    "rash driving",
)
_GONE = {"[deleted]", "[removed]"}


def _text_of(*parts: str) -> str:
    return "\n".join(p for p in parts if p and p not in _GONE)


def records_from_submission(submission: Any, city: str | None) -> Iterator[RawRecord]:
    """Yield the post and its comments. Authors are never read."""
    post = make_record(
        _text_of(submission.title, submission.selftext),
        "reddit",
        submission.id,
        submission.created_utc,
        city,
    )
    if post:
        yield post
    with_retry(lambda: submission.comments.replace_more(limit=0))
    for comment in submission.comments.list():
        rec = make_record(_text_of(comment.body), "reddit", comment.id, comment.created_utc, city)
        if rec:
            yield rec


def _search(subreddit: Any, query: str, limit: int) -> list[Any]:
    return with_retry(lambda: list(subreddit.search(query, limit=limit)))


def collect(
    reddit: Any,
    subreddits: Mapping[str, str | None] = SUBREDDIT_CITIES,
    queries: Sequence[str] = QUERIES,
    limit: int = 100,
) -> Iterator[RawRecord]:
    """Search each subreddit for each query. PRAW paces requests itself."""
    visited: set[str] = set()
    for name, city in subreddits.items():
        subreddit = reddit.subreddit(name)
        for query in queries:
            for submission in _search(subreddit, query, limit):
                if submission.id in visited:
                    continue
                visited.add(submission.id)
                yield from records_from_submission(submission, city)


def client_from_env() -> Any:
    praw = importlib.import_module("praw")
    return praw.Reddit(
        client_id=os.environ["REDDIT_CLIENT_ID"],
        client_secret=os.environ["REDDIT_CLIENT_SECRET"],
        user_agent=os.environ["REDDIT_USER_AGENT"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data/raw/reddit.jsonl"))
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()
    print(append_new(collect(client_from_env(), limit=args.limit), args.out), "new records")


if __name__ == "__main__":
    main()
