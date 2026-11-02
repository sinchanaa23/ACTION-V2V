"""YouTube comment collector (D3), using the Data API v3.

Run: python -m v2v.data.youtube --out data/raw/youtube.jsonl
Needs YOUTUBE_API_KEY. A search costs 100 quota units and a comment page 1, out
of 10,000 a day, so keep the query list short and spend quota on comments.
"""

import argparse
import importlib
import os
from collections.abc import Iterator, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from v2v.data.common import RawRecord, append_new, make_record, with_retry

QUERIES = (
    "bangalore dashcam accident",
    "india dashcam road rage",
    "indian roads dashcam",
    "delhi traffic accident dashcam",
    "mumbai waterlogging traffic",
)


def _is_transient(exc: Exception) -> bool:
    status = getattr(getattr(exc, "resp", None), "status", None)
    return status == 429 or (isinstance(status, int) and status >= 500)


def _comments_closed(exc: Exception) -> bool:
    return "commentsDisabled" in str(getattr(exc, "content", b""))


def _timestamp(value: str) -> float:
    return datetime.fromisoformat(value).timestamp()


def search_videos(client: Any, query: str, max_results: int) -> list[str]:
    request = client.search().list(
        q=query,
        part="id",
        type="video",
        regionCode="IN",
        maxResults=max_results,
    )
    response = with_retry(request.execute, should_retry=_is_transient)
    return [item["id"]["videoId"] for item in response["items"]]


def comments_of(client: Any, video_id: str, max_pages: int) -> Iterator[RawRecord]:
    """Yield top-level comments. Only text, id and time are read."""
    token = None
    for _ in range(max_pages):
        request = client.commentThreads().list(
            videoId=video_id,
            part="snippet",
            textFormat="plainText",
            maxResults=100,
            pageToken=token,
        )
        try:
            response = with_retry(request.execute, should_retry=_is_transient)
        except Exception as exc:
            if _comments_closed(exc):
                return
            raise
        for item in response["items"]:
            top = item["snippet"]["topLevelComment"]
            snippet = top["snippet"]
            rec = make_record(
                snippet["textDisplay"],
                "youtube",
                top["id"],
                _timestamp(snippet["publishedAt"]),
            )
            if rec:
                yield rec
        token = response.get("nextPageToken")
        if not token:
            return


def collect(
    client: Any,
    queries: Sequence[str] = QUERIES,
    videos_per_query: int = 25,
    pages_per_video: int = 3,
) -> Iterator[RawRecord]:
    visited: set[str] = set()
    for query in queries:
        for video_id in search_videos(client, query, videos_per_query):
            if video_id in visited:
                continue
            visited.add(video_id)
            yield from comments_of(client, video_id, pages_per_video)


def client_from_env() -> Any:
    discovery = importlib.import_module("googleapiclient.discovery")
    return discovery.build("youtube", "v3", developerKey=os.environ["YOUTUBE_API_KEY"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data/raw/youtube.jsonl"))
    args = parser.parse_args()
    print(append_new(collect(client_from_env()), args.out), "new records")


if __name__ == "__main__":
    main()
