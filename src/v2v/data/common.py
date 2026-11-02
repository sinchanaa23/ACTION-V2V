"""Record shape, retry and storage shared by the collectors."""

import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from v2v.contracts import DriverReport, Source
from v2v.data.scrub import scrub

T = TypeVar("T")


class RawRecord(BaseModel):
    """One scrubbed candidate text. Holds no author field by design."""

    report_id: str
    text: str
    source: Source
    source_id: str
    created_utc: float | None = None
    city: str | None = None

    def to_report(self) -> DriverReport:
        return DriverReport(
            report_id=self.report_id,
            text=self.text,
            source=self.source,
            reporter_id=None,
            t_sim=None,
        )


def make_record(
    text: str,
    source: Source,
    source_id: str,
    created_utc: float | None = None,
    city: str | None = None,
) -> RawRecord | None:
    """Build a record from scrubbed text, or None if nothing is left."""
    cleaned = scrub(text)
    if not cleaned:
        return None
    return RawRecord(
        report_id=f"{source}:{source_id}",
        text=cleaned,
        source=source,
        source_id=source_id,
        created_utc=created_utc,
        city=city,
    )


def with_retry(
    call: Callable[[], T],
    retries: int = 4,
    base_delay_s: float = 2.0,
    should_retry: Callable[[Exception], bool] = lambda _: True,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Run call, backing off exponentially on retryable errors."""
    for attempt in range(retries + 1):
        try:
            return call()
        except Exception as exc:
            if attempt == retries or not should_retry(exc):
                raise
            sleep(base_delay_s * 2**attempt)
    raise AssertionError("unreachable")


def append_new(records: Iterable[RawRecord], path: Path) -> int:
    """Append records whose id is not already in path, so reruns are safe."""
    path.parent.mkdir(parents=True, exist_ok=True)
    seen = {rec.report_id for rec in read_records(path)}
    written = 0
    with path.open("a", encoding="utf-8") as out:
        for rec in records:
            if rec.report_id in seen:
                continue
            seen.add(rec.report_id)
            out.write(rec.model_dump_json() + "\n")
            written += 1
    return written


def read_records(path: Path) -> list[RawRecord]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as src:
        return [RawRecord.model_validate_json(line) for line in src if line.strip()]
