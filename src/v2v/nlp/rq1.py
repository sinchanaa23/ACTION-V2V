"""RQ1 results: classifiers, calibration and extractors on a frozen split. Owner: P2 (P2-08).

    python -m v2v.nlp.rq1 --split data/splits/val --model tfidf=models/tfidf_lr.pkl \
        --model transformer=models/transformer --out docs/report/rq1_val
    python -m v2v.nlp.rq1 --split data/splits/val --tune-factors tfidf=models/tfidf_lr.pkl

For each model (plus Baseline A, the keyword classifier, always included) it
writes predictions as JSONL in the format P1's error analysis reads, and one
report with:

- macro-F1 and per-class precision / recall / F1 (P1's shared scoring), with a
  95% Wilson interval on accuracy;
- calibration: ECE and a reliability diagram per model (Baseline A has no
  probabilities, so it has no ECE);
- negation and hedging detection (P2-02) against the gold flags;
- location extraction: the rules (P2-03) and, if given, the tagger (P2-06).

``--tune-factors`` picks the interpretation-confidence factors for hedged and
location-less reports on *validation*, by the grid value that minimises the
ECE of interpretation confidence against correctness. It prints them for a
person to copy into classifier.py; nothing is changed automatically.

The test split is read only with ``--allow-test``, once models and factors
are frozen.
"""

import argparse
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from v2v.contracts import HazardType
from v2v.data.annotate import GoldRecord, read_gold
from v2v.data.scoring import classification_scores, wilson_interval
from v2v.nlp import calibrate, classifier, extract, keyword_baseline, preprocess

SPLIT_FILE = "reports.jsonl"
FACTOR_GRID: tuple[float, ...] = tuple(round(0.3 + 0.05 * i, 2) for i in range(15))
"""0.30, 0.35, ..., 1.00."""

Prediction = tuple[HazardType, float | None]
"""(class, calibrated probability); None for Baseline A."""


def read_split(split_dir: Path, allow_test: bool = False) -> list[GoldRecord]:
    if not allow_test and "test" in split_dir.resolve().parts:
        raise ValueError(f"refusing to read the frozen test split: {split_dir}")
    return read_gold(split_dir / SPLIT_FILE)


def keyword_predictions(records: Sequence[GoldRecord]) -> list[Prediction]:
    return [
        (keyword_baseline.classify(r.report_id, preprocess.normalise(r.text)).hazard, None)
        for r in records
    ]


def model_predictions(model: Any, records: Sequence[GoldRecord]) -> list[Prediction]:
    """Any trained classifier with ``predict(texts) -> [(class, p)]``."""
    return [(h, p) for h, p in model.predict([r.text for r in records])]


def load_model(path: Path) -> Any:
    model = classifier._load_cached(path)
    if model is None:
        raise FileNotFoundError(f"no trained model at {path}")
    return model


def classification_report(
    records: Sequence[GoldRecord], preds: Sequence[Prediction]
) -> dict[str, Any]:
    gold = [r.hazard for r in records]
    predicted = [h for h, _ in preds]
    scores = classification_scores(gold, predicted)
    correct = [g == p for g, p in zip(gold, predicted, strict=True)]
    low, high = wilson_interval(sum(correct), len(correct))
    scores["accuracy_ci"] = (low, high)
    probs = [p for _, p in preds]
    if all(p is not None for p in probs):
        confidences = [float(p) for p in probs if p is not None]
        scores["ece"] = calibrate.expected_calibration_error(confidences, correct)
        scores["reliability"] = calibrate.reliability_bins(confidences, correct)
    else:
        scores["ece"] = None
    return scores


def flag_scores(gold: Sequence[bool], predicted: Sequence[bool]) -> dict[str, float | int]:
    tp = sum(g and p for g, p in zip(gold, predicted, strict=True))
    fp = sum(p and not g for g, p in zip(gold, predicted, strict=True))
    fn = sum(g and not p for g, p in zip(gold, predicted, strict=True))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "support": sum(gold)}


def extractor_report(records: Sequence[GoldRecord]) -> dict[str, Any]:
    texts = [preprocess.normalise(r.text) for r in records]
    return {
        "negation": flag_scores(
            [r.negated for r in records], [extract.detect_negation(t) for t in texts]
        ),
        "hedging": flag_scores(
            [r.hedged for r in records], [extract.detect_hedging(t) for t in texts]
        ),
    }


def location_report(records: Sequence[GoldRecord], tagger_path: Path | None) -> dict[str, Any]:
    from v2v.nlp import span_tagger

    tagger = span_tagger.SpanTagger.load(tagger_path) if tagger_path else None
    result: dict[str, Any] = span_tagger.compare(records, tagger)
    return result


def interpretation_ece(
    records: Sequence[GoldRecord],
    preds: Sequence[Prediction],
    hedged_factor: float,
    no_location_factor: float,
) -> float:
    """ECE of interpretation confidence (p_class lowered for hedged and
    location-less reports, as classifier.interpretation_confidence does)
    against whether the class was right."""
    confidences, correct = [], []
    for r, (h, p) in zip(records, preds, strict=True):
        if p is None:
            raise ValueError("factor tuning needs calibrated probabilities")
        text = preprocess.normalise(r.text)
        conf = p
        if extract.detect_hedging(text):
            conf *= hedged_factor
        if extract.extract_location(text) is None:
            conf *= no_location_factor
        confidences.append(conf)
        correct.append(h == r.hazard)
    return calibrate.expected_calibration_error(confidences, correct)


def tune_factors(
    records: Sequence[GoldRecord], preds: Sequence[Prediction], grid: Sequence[float] = FACTOR_GRID
) -> dict[str, float]:
    """Grid search on validation for the two factors."""
    best = min(
        ((h, n) for h in grid for n in grid),
        key=lambda hn: (interpretation_ece(records, preds, *hn), -hn[0], -hn[1]),
    )
    return {
        "hedged_factor": best[0],
        "no_location_factor": best[1],
        "ece_tuned": interpretation_ece(records, preds, *best),
        "ece_current": interpretation_ece(
            records, preds, classifier.HEDGED_FACTOR, classifier.NO_LOCATION_FACTOR
        ),
        "ece_no_factors": interpretation_ece(records, preds, 1.0, 1.0),
    }


def write_predictions(
    records: Sequence[GoldRecord], preds: Sequence[Prediction], out: Path
) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "".join(
            json.dumps({"report_id": r.report_id, "hazard": h.value, "p_class": p}) + "\n"
            for r, (h, p) in zip(records, preds, strict=True)
        ),
        encoding="utf-8",
    )


def _row(cells: Sequence[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _f(x: Any) -> str:
    return "-" if x is None else f"{x:.3f}" if isinstance(x, float) else str(x)


def render(
    split: str,
    n: int,
    models: Mapping[str, Mapping[str, Any]],
    extractors: Mapping[str, Any],
    locations: Mapping[str, Any],
) -> str:
    lines = [f"# RQ1 results on {split}", "", f"Reports: {n}", "", "## Classifiers", ""]
    lines += ["| Model | Macro-F1 | Accuracy | 95% CI | ECE |", "|---|---|---|---|---|"]
    for name, s in models.items():
        lo, hi = s["accuracy_ci"]
        cells = [name, _f(s["macro_f1"]), _f(s["accuracy"]), f"{lo:.3f}-{hi:.3f}", _f(s["ece"])]
        lines.append(_row(cells))
    for name, s in models.items():
        lines += [
            "",
            f"### {name}: per class",
            "",
            "| Class | P | R | F1 | Support |",
            "|---|---|---|---|---|",
        ]
        for cls, c in s["per_class"].items():
            lines.append(_row([cls, *(_f(c[k]) for k in ("precision", "recall", "f1", "support"))]))
    lines += [
        "",
        "## Negation and hedging (rule-based, P2-02)",
        "",
        "| Flag | P | R | F1 | Support |",
    ]
    lines.append("|---|---|---|---|---|")
    for flag, s in extractors.items():
        lines.append(_row([flag, *(_f(s[k]) for k in ("precision", "recall", "f1", "support"))]))
    lines += [
        "",
        "## Location extraction",
        "",
        "| Extractor | Exact F1 | Token-overlap F1 | No-location kept None |",
    ]
    lines.append("|---|---|---|---|")
    for name, s in locations.items():
        cells = [
            name,
            _f(s["exact"]["f1"]),
            _f(s["token_overlap"]["f1"]),
            _f(s["no_location_kept_none"]),
        ]
        lines.append(_row(cells))
    return "\n".join(lines) + "\n"


def run(
    split_dir: Path,
    model_paths: Mapping[str, Path],
    out_dir: Path,
    tagger_path: Path | None = None,
    allow_test: bool = False,
    loader: Callable[[Path], Any] = load_model,
) -> Path:
    records = read_split(split_dir, allow_test)
    split = split_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    all_preds: dict[str, list[Prediction]] = {"keyword": keyword_predictions(records)}
    for name, path in model_paths.items():
        all_preds[name] = model_predictions(loader(path), records)
    reports = {}
    for name, preds in all_preds.items():
        write_predictions(records, preds, out_dir / f"{split}_{name}.jsonl")
        reports[name] = classification_report(records, preds)
        if reports[name].get("reliability"):
            try:
                calibrate.plot_reliability(
                    reports[name]["reliability"], out_dir / f"reliability_{name}.png", name
                )
            except ImportError:
                pass
    path = out_dir / f"rq1_{split}.md"
    path.write_text(
        render(
            split,
            len(records),
            reports,
            extractor_report(records),
            location_report(records, tagger_path),
        ),
        encoding="utf-8",
    )
    return path


def _named(entries: Sequence[str]) -> dict[str, Path]:
    out = {}
    for entry in entries:
        name, _, path = entry.partition("=")
        if not path:
            raise ValueError(f"expected name=path, got {entry!r}")
        out[name] = Path(path)
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, default=Path("data/splits/val"))
    parser.add_argument("--model", action="append", default=[], help="name=path (repeatable)")
    parser.add_argument("--tagger", type=Path, help="trained span tagger to compare")
    parser.add_argument("--out", type=Path, default=Path("docs/report/rq1"))
    parser.add_argument("--tune-factors", metavar="NAME=PATH", help="tune on this model only")
    parser.add_argument("--allow-test", action="store_true", help="final RQ1 run only")
    args = parser.parse_args(argv)
    if args.tune_factors:
        if args.allow_test:
            parser.error("factors are tuned on validation, never on test")
        ((name, path),) = _named([args.tune_factors]).items()
        records = read_split(args.split)
        print(
            json.dumps(
                {
                    "model": name,
                    **tune_factors(records, model_predictions(load_model(path), records)),
                },
                indent=2,
            )
        )
        return
    print("wrote", run(args.split, _named(args.model), args.out, args.tagger, args.allow_test))


if __name__ == "__main__":
    main()
