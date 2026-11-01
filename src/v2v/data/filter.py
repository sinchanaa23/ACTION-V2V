"""Candidate filter (P1-04): hazard recall net, language tag, near-duplicate removal.

Run: python -m v2v.data.filter data/raw/*.jsonl --out data/raw/candidates.jsonl

The net is deliberately wide. Near-misses and negated reports must survive it,
because the non_actionable class and the hard subsets come from the same text.
"""

import argparse
import importlib
import json
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path

from v2v.data.common import RawRecord, read_records
from v2v.nlp.preprocess import language_mix

MIN_WORDS = 3
MAX_WORDS = 120
DUPLICATE_SCORE = 90.0

# Word-start matches, so "crash" also catches "crashed".
_NET = re.compile(
    r"\b(?:accident|acident|crash|collid|collision|hit by|rammed|run over|overturn|flipped|"
    r"takkar|takar|thok|palat|palti|pothole|gaddh|gadd?a|khadd|waterlog|flood|paani|pani|"
    r"oil spill|diesel|slipp|skid|fisal|kichad|mud|gravel|breakdown|broke down|broken down|"
    r"kharab|stalled|stuck|puncture|jam|blocked|road block|barricade|debris|fallen tree|"
    r"tree fell|ped gir|construction|digging|khudai|diversion|wrong side|ulta|speeding|"
    r"overspeed|rash|racing|overtak|careful|bach ke|sambhal|hazard|danger|khatra|road clear)"
    r"|दुर्घटना|एक्सीडेंट|टक्कर|गड्ढ|पानी|जाम|खराब",
    re.IGNORECASE,
)
_SOURCE_PRIORITY = {"survey": 0}
"""Survey text wins a near-duplicate tie: it is the primary Hinglish source."""


class Candidate(RawRecord):
    language: str


def in_net(text: str) -> bool:
    """True if the text could be a hazard report, a near-miss or a negated one."""
    return MIN_WORDS <= len(text.split()) <= MAX_WORDS and _NET.search(text) is not None


def drop_near_duplicates(
    records: Sequence[RawRecord], score: float = DUPLICATE_SCORE
) -> list[RawRecord]:
    """Keep the first of each near-duplicate group, survey text first."""
    fuzz = importlib.import_module("rapidfuzz.fuzz")
    process = importlib.import_module("rapidfuzz.process")
    ordered = sorted(records, key=lambda r: _SOURCE_PRIORITY.get(r.source, 1))
    kept: list[RawRecord] = []
    kept_texts: list[str] = []
    for rec in ordered:
        text = rec.text.lower()
        if process.extractOne(text, kept_texts, scorer=fuzz.token_sort_ratio, score_cutoff=score):
            continue
        kept.append(rec)
        kept_texts.append(text)
    return kept


def filter_candidates(records: Iterable[RawRecord]) -> list[Candidate]:
    """Apply the net, remove near-duplicates and tag each record's language."""
    netted = [rec for rec in records if in_net(rec.text)]
    return [
        Candidate(**rec.model_dump(), language=language_mix(rec.text))
        for rec in drop_near_duplicates(netted)
    ]


def summary(raw_count: int, candidates: Sequence[Candidate]) -> dict[str, object]:
    return {
        "raw": raw_count,
        "candidates": len(candidates),
        "by_source": dict(Counter(c.source for c in candidates)),
        "by_language": dict(Counter(c.language for c in candidates)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=Path("data/raw/candidates.jsonl"))
    args = parser.parse_args()
    records = [rec for path in args.inputs for rec in read_records(path)]
    candidates = filter_candidates(records)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as out:
        for candidate in candidates:
            out.write(candidate.model_dump_json() + "\n")
    print(json.dumps(summary(len(records), candidates), indent=2))


if __name__ == "__main__":
    main()
