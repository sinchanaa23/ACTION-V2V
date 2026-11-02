"""RQ1 error analysis on the hard subsets (P1-09).

    python -m v2v.data.error_analysis --gold data/splits/val/reports.jsonl \
        --pred keyword=val_keyword.jsonl --pred tfidf=val_tfidf.jsonl --out rq1_val.md

Predictions are JSONL with ``report_id`` and ``hazard``. The report shows IDs
rather than text, so it can be shared. A subset's size is always shown beside its
score, and accuracy carries a 95% interval, so small subsets read as uncertain.
"""

import argparse
import json
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from v2v.contracts import HazardType
from v2v.data.annotate import GoldRecord, read_gold
from v2v.data.scoring import classification_scores, wilson_interval
from v2v.nlp.preprocess import language_mix

TOP_CONFUSIONS = 10
EXAMPLE_IDS = 3

SUBSETS: dict[str, Callable[[GoldRecord], bool]] = {
    "all": lambda r: True,
    "negated": lambda r: r.negated,
    "hedged": lambda r: r.hedged,
    "no location": lambda r: not r.locations,
    "code-mixed": lambda r: language_mix(r.text) == "hi-en",
    "english": lambda r: language_mix(r.text) == "en",
}


def read_predictions(path: Path) -> dict[str, HazardType]:
    lines = path.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    return {row["report_id"]: HazardType(row["hazard"]) for row in rows}


def subset_row(
    name: str, records: Sequence[GoldRecord], predictions: Mapping[str, HazardType]
) -> str:
    """One table row. Empty subsets are shown as empty, never scored."""
    if not records:
        return f"| {name} | 0 | - | - | - |"
    gold = [r.hazard for r in records]
    scores = classification_scores(gold, [predictions[r.report_id] for r in records])
    correct = sum(predictions[r.report_id] == r.hazard for r in records)
    low, high = wilson_interval(correct, len(records))
    return (
        f"| {name} | {len(records)} | {scores['macro_f1']:.3f} | "
        f"{scores['accuracy']:.3f} | {low:.3f}-{high:.3f} |"
    )


def top_confusions(
    records: Sequence[GoldRecord], predictions: Mapping[str, HazardType]
) -> list[str]:
    """Most common mistakes as table rows, with a few report IDs to inspect."""
    errors = [r for r in records if predictions[r.report_id] != r.hazard]
    counts = Counter((r.hazard, predictions[r.report_id]) for r in errors)
    rows = []
    for (gold, predicted), count in counts.most_common(TOP_CONFUSIONS):
        ids = [
            r.report_id for r in errors if (r.hazard, predictions[r.report_id]) == (gold, predicted)
        ]
        example = ", ".join(ids[:EXAMPLE_IDS])
        rows.append(f"| {gold.value} | {predicted.value} | {count} | {example} |")
    return rows


def render(records: Sequence[GoldRecord], models: Mapping[str, Mapping[str, HazardType]]) -> str:
    """Build the report. Every record must have a prediction from every model."""
    sections = ["# RQ1 error analysis", "", f"Records: {len(records)}"]
    for name, predictions in models.items():
        missing = [r.report_id for r in records if r.report_id not in predictions]
        if missing:
            raise ValueError(f"{name}: {len(missing)} predictions missing, e.g. {missing[:3]}")
        sections += [
            "",
            f"## {name}",
            "",
            "| Subset | n | Macro-F1 | Accuracy | Accuracy 95% CI |",
            "|---|---|---|---|---|",
        ]
        for subset, keep in SUBSETS.items():
            chosen = [r for r in records if keep(r)]
            sections.append(subset_row(subset, chosen, predictions))
        sections += [
            "",
            "| Gold | Predicted | Count | Example IDs |",
            "|---|---|---|---|",
            *top_confusions(records, predictions),
        ]
    return "\n".join(sections) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--pred", action="append", required=True, help="name=path")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    models = {
        name: read_predictions(Path(path))
        for name, path in (entry.split("=", 1) for entry in args.pred)
    }
    args.out.write_text(render(read_gold(args.gold), models), encoding="utf-8")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
