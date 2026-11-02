"""Hand-written code-path strings only. Never training data."""

from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from v2v.data import collect, reddit, youtube
from v2v.data.common import append_new, make_record, read_records, with_retry
from v2v.data.scrub import scrub


@pytest.mark.parametrize(
    ("text", "tag"),
    [
        ("call 9876543210 now", "[PHONE]"),
        ("call +91 98765 43210 now", "[PHONE]"),
        ("call 987-654-3210 now", "[PHONE]"),
        ("call 080-12345678 now", "[PHONE]"),
        ("car KA 01 AB 1234 hit it", "[PLATE]"),
        ("car KA01AB1234 hit it", "[PLATE]"),
        ("car ka-05-mh-7788 hit it", "[PLATE]"),
        ("car DL 3C AB 1234 hit it", "[PLATE]"),
        ("car 22 BH 1234 AA hit it", "[PLATE]"),
        ("thanks u/some_name", "[USER]"),
        ("thanks /u/some_name", "[USER]"),
        ("thanks @some.name", "[USER]"),
        ("mail me at a.b@example.co.in", "[EMAIL]"),
        ("see https://example.com/x?u=1", "[URL]"),
    ],
)
def test_scrub_removes_pii(text: str, tag: str) -> None:
    result = scrub(text)
    assert tag in result
    assert not any(ch.isdigit() for ch in result)


@pytest.mark.parametrize(
    "text",
    [
        "accident near NH 44 at km 120",
        "on 12 km stretch",
        "traffic since 2019 at 100 ft road",
        "bridge ke paas 2 trucks",
        "27th main, sector 5",
    ],
)
def test_scrub_keeps_road_text(text: str) -> None:
    assert scrub(text) == text


def test_make_record_scrubs_and_drops_empty() -> None:
    rec = make_record("call 9876543210", "fixture", "abc")
    assert rec is not None
    assert rec.report_id == "fixture:abc"
    assert "9876543210" not in rec.model_dump_json()
    assert make_record("   ", "fixture", "abc") is None


def test_record_has_no_author_field_and_converts() -> None:
    rec = make_record("x", "fixture", "1", city="Pune")
    assert rec is not None
    assert not {"author", "username", "name"} & set(rec.model_dump())
    report = rec.to_report()
    assert report.report_id == "fixture:1"
    assert report.reporter_id is None


def test_with_retry_backs_off_then_succeeds() -> None:
    calls: list[int] = []
    delays: list[float] = []

    def flaky() -> str:
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError
        return "ok"

    assert with_retry(flaky, sleep=delays.append) == "ok"
    assert delays == [2.0, 4.0]


def test_with_retry_stops_when_not_retryable() -> None:
    def fail() -> None:
        raise RuntimeError

    with pytest.raises(RuntimeError):
        with_retry(fail, should_retry=lambda _: False, sleep=lambda _: None)


def test_append_new_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "raw" / "x.jsonl"
    rec = make_record("x", "fixture", "1")
    assert rec is not None
    assert append_new([rec, rec], path) == 1
    assert append_new([rec], path) == 0
    assert len(read_records(path)) == 1


class FakeComments(list):
    def replace_more(self, limit: int) -> None:
        pass

    def list(self) -> "FakeComments":
        return self


def test_reddit_collect_skips_deleted_and_repeats() -> None:
    comments = FakeComments(
        [
            NS(id="c1", body="by u/someone", created_utc=2.0, author="SECRET"),
            NS(id="c2", body="[deleted]", created_utc=3.0, author=None),
        ]
    )
    post = NS(
        id="p1",
        title="title",
        selftext="[removed]",
        created_utc=1.0,
        author="SECRET",
        comments=comments,
    )
    client = NS(subreddit=lambda _: NS(search=lambda q, limit: [post]))

    records = list(reddit.collect(client, {"bangalore": "Bengaluru"}, ("a", "b")))

    assert [r.source_id for r in records] == ["p1", "c1"]
    assert records[0].text == "title"
    assert records[0].city == "Bengaluru"
    assert "[USER]" in records[1].text
    assert "SECRET" not in "".join(r.model_dump_json() for r in records)


class Request:
    def __init__(self, result: object) -> None:
        self.result = result

    def execute(self) -> object:
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def _comment(cid: str, text: str) -> dict:
    snippet = {
        "textDisplay": text,
        "publishedAt": "2026-01-02T03:04:05Z",
        "authorDisplayName": "SECRET",
    }
    return {"snippet": {"topLevelComment": {"id": cid, "snippet": snippet}}}


def _youtube_client(pages: dict[str, list[object]]) -> NS:
    return NS(
        search=lambda: NS(
            list=lambda **_: Request({"items": [{"id": {"videoId": v}} for v in pages]})
        ),
        commentThreads=lambda: NS(list=lambda videoId, **_: Request(pages[videoId].pop(0))),
    )


def test_youtube_pages_and_skips_closed_comments() -> None:
    closed = RuntimeError()
    closed.content = b'{"reason": "commentsDisabled"}'  # type: ignore[attr-defined]
    client = _youtube_client(
        {
            "v1": [
                {"items": [_comment("a", "one")], "nextPageToken": "t"},
                {"items": [_comment("b", "two")]},
            ],
            "v2": [closed],
        }
    )

    records = list(youtube.collect(client, ("q",), 2, 5))

    assert [r.source_id for r in records] == ["a", "b"]
    assert records[0].created_utc == pytest.approx(1767323045.0)
    assert "SECRET" not in records[0].model_dump_json()


def test_youtube_quota_error_is_raised() -> None:
    quota = RuntimeError()
    quota.content = b'{"reason": "quotaExceeded"}'  # type: ignore[attr-defined]
    client = _youtube_client({"v1": [quota]})
    with pytest.raises(RuntimeError):
        list(youtube.collect(client, ("q",), 1, 1))


def test_collect_reddit_uses_given_subreddits(monkeypatch: pytest.MonkeyPatch) -> None:
    post = NS(id="p", title="t", selftext="", created_utc=1.0, comments=FakeComments())
    names: list[str] = []

    def subreddit(name: str) -> NS:
        names.append(name)
        return NS(search=lambda q, limit: [post])

    monkeypatch.setattr(reddit, "client_from_env", lambda: NS(subreddit=subreddit))
    reports = collect.reddit(["delhi"], 5)
    assert [r.report_id for r in reports] == ["reddit:p"]
    assert set(names) == {"delhi"}


def test_collect_youtube_reads_given_videos(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _youtube_client({"v1": [{"items": [_comment("a", "one")]}]})
    monkeypatch.setattr(youtube, "client_from_env", lambda: client)
    assert [r.report_id for r in collect.youtube(["v1"])] == ["youtube:a"]


def test_collect_scrub_is_the_real_scrubber() -> None:
    assert collect.scrub("call 9876543210") == "call [PHONE]"
