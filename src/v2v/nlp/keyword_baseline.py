"""Baseline A: keyword hazard classifier. Owner: P1 (P1-07).

    python -m v2v.nlp.keyword_baseline build --train data/splits/train
    python -m v2v.nlp.keyword_baseline score --split data/splits/val --out val.json

Deliberately naive: it has no negation, hedging or context handling, because it
is the reference the NLP classifier is compared against. The word list is the
seed list below plus keywords.json, which ``build`` mines from the training split.
"""

import argparse
import json
import math
import re
from collections import Counter
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

from v2v.contracts import NON_HAZARD_TYPES, HazardInterpretation, HazardType
from v2v.data.annotate import GoldRecord, read_gold
from v2v.data.scoring import classification_scores
from v2v.nlp.preprocess import normalise

KEYWORDS_PATH = Path(__file__).with_name("keywords.json")
MIN_KEYWORD_LENGTH = 3
MIN_LOG_ODDS = 1.0

SEED_KEYWORDS: dict[HazardType, tuple[str, ...]] = {
    HazardType.ACCIDENT: ("accident", "crash", "collision", "collided", "takkar", "overturned"),
    HazardType.OBSTRUCTION: (
        "pothole",
        "gaddha",
        "debris",
        "barricade",
        "fallen tree",
        "blocked",
        # animals on the road count as obstructions (docs/annotation_guide.md)
        "animal",
        "janwar",
        "bandar",
        "monkey",
        "gaay",
        "cow",
        "cattle",
        "bhains",
        "buffalo",
        "saand",
        "bull",
        "kutta",
        "kutte",
        "dog",
        "stray",
    ),  # fmt: skip
    HazardType.SLIPPERY: ("slippery", "waterlogging", "waterlogged", "flooded", "oil spill"),
    HazardType.STOPPED: ("breakdown", "broke down", "broken down", "stalled", "puncture"),
    HazardType.SPEEDING: ("speeding", "overspeeding", "wrong side", "rash driving", "racing"),
}
_WORD = re.compile(r"[^\W\d_]+")


def _load_mined(path: Path) -> dict[HazardType, list[str]]:
    if not path.exists():
        return {}
    mined: dict[str, list[str]] = json.loads(path.read_text(encoding="utf-8"))
    return {HazardType(name): words for name, words in mined.items()}


@lru_cache(maxsize=1)
def _rules() -> tuple[tuple[re.Pattern[str], HazardType], ...]:
    """Word-start patterns: seeds first, then mined words, in class order."""
    mined = _load_mined(KEYWORDS_PATH)
    return tuple(
        (re.compile(rf"\b{re.escape(word)}"), hazard)
        for source in (SEED_KEYWORDS, mined)
        for hazard, words in source.items()
        for word in words
    )


def classify(report_id: str, normalised_text: str) -> HazardInterpretation:
    """Classify by first keyword hit. Keyword rules have no real probability,
    so ``p_class`` is always 1.0."""
    for pattern, hazard in _rules():
        if pattern.search(normalised_text):
            return HazardInterpretation(report_id=report_id, hazard=hazard, p_class=1.0)
    return HazardInterpretation(report_id=report_id, hazard=HazardType.NON_ACTIONABLE, p_class=1.0)


def mine_keywords(
    records: Sequence[GoldRecord], min_count: int = 5, top_k: int = 25
) -> dict[str, list[str]]:
    """Pick each hazard class's most distinctive words by smoothed log-odds.

    Run on the training split only, so no validation or test text shapes the list.
    """
    tokens = [set(_WORD.findall(normalise(r.text))) for r in records]
    mined: dict[str, list[str]] = {}
    for hazard in HazardType:
        if hazard in NON_HAZARD_TYPES:
            continue
        inside = [t for t, r in zip(tokens, records, strict=True) if r.hazard == hazard]
        outside = [t for t, r in zip(tokens, records, strict=True) if r.hazard != hazard]
        in_counts = Counter(word for words in inside for word in words)
        out_counts = Counter(word for words in outside for word in words)
        scored = []
        for word, count in in_counts.items():
            if count < min_count or len(word) < MIN_KEYWORD_LENGTH:
                continue
            odds_in = (count + 0.5) / (len(inside) - count + 0.5)
            odds_out = (out_counts[word] + 0.5) / (len(outside) - out_counts[word] + 0.5)
            log_odds = math.log(odds_in / odds_out)
            if log_odds >= MIN_LOG_ODDS:
                scored.append((-log_odds, word))
        mined[hazard.value] = [word for _, word in sorted(scored)[:top_k]]
    return mined


def predict(records: Sequence[GoldRecord]) -> list[dict[str, str]]:
    """Predictions in the format the RQ1 error analysis reads."""
    return [
        {"report_id": r.report_id, "hazard": classify(r.report_id, normalise(r.text)).hazard.value}
        for r in records
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--train", type=Path, default=Path("data/splits/train"))
    score = commands.add_parser("score")
    score.add_argument("--split", type=Path, required=True)
    score.add_argument("--out", type=Path, required=True)
    score.add_argument("--predictions", type=Path)
    args = parser.parse_args()

    if args.command == "build":
        mined = mine_keywords(read_gold(args.train / "reports.jsonl"))
        KEYWORDS_PATH.write_text(json.dumps(mined, indent=2, ensure_ascii=False), encoding="utf-8")
        print("wrote", KEYWORDS_PATH)
        return
    records = read_gold(args.split / "reports.jsonl")
    predictions = predict(records)
    scores = classification_scores(
        [r.hazard for r in records], [HazardType(p["hazard"]) for p in predictions]
    )
    args.out.write_text(json.dumps(scores, indent=2), encoding="utf-8")
    if args.predictions:
        lines = [json.dumps(p) + "\n" for p in predictions]
        args.predictions.write_text("".join(lines), encoding="utf-8")
    print("macro-F1", scores["macro_f1"], "n", scores["n"])


if __name__ == "__main__":
    main()
