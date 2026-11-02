"""Driver survey importer (D1).

Reads the CSV exported from the Google Form in docs/survey_form.md. A response
is kept only if the person consented and confirmed they are 18 or over. Only
the columns named here are read, so anything else in the export is dropped.

Run: python -m v2v.data.survey data/raw/responses.csv --out data/raw/survey.jsonl
"""

import argparse
import csv
import hashlib
from collections.abc import Iterator
from pathlib import Path

from v2v.data.common import RawRecord, append_new, make_record

CONSENT_COLUMN = "Consent"
ADULT_COLUMN = "Are you 18 or older?"
CITY_COLUMN = "Which city do you mostly drive in?"
REPORT_COLUMNS = (
    "Hazard report 1",
    "Hazard report 2",
    "Warning message",
    "Unsure report",
    "All clear report",
    "False alarm report",
)
CONSENT_YES = "I agree"
ADULT_YES = "Yes"
_REQUIRED = {CONSENT_COLUMN, ADULT_COLUMN, CITY_COLUMN, *REPORT_COLUMNS}


def _source_id(row_number: int, column: str, text: str) -> str:
    return hashlib.sha256(f"{row_number}|{column}|{text}".encode()).hexdigest()[:16]


def read_responses(path: Path) -> Iterator[RawRecord]:
    """Yield one record per answered report question of an eligible response."""
    with path.open(encoding="utf-8-sig", newline="") as src:
        reader = csv.DictReader(src)
        missing = _REQUIRED - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"survey export is missing columns: {sorted(missing)}")
        for row_number, row in enumerate(reader):
            if row[CONSENT_COLUMN].strip() != CONSENT_YES:
                continue
            if row[ADULT_COLUMN].strip() != ADULT_YES:
                continue
            city = row[CITY_COLUMN].strip() or None
            for column in REPORT_COLUMNS:
                text = row[column].strip()
                if not text:
                    continue
                rec = make_record(text, "survey", _source_id(row_number, column, text), city=city)
                if rec:
                    yield rec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--out", type=Path, default=Path("data/raw/survey.jsonl"))
    args = parser.parse_args()
    print(append_new(read_responses(args.csv_path), args.out), "new records")


if __name__ == "__main__":
    main()
