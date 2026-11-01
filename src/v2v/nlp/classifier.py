"""Stage 2: NLP hazard interpretation. Owner: P2 (P2-01, P2-05).

    python -m v2v.nlp.classifier train --train data/splits/train --val data/splits/val

Baseline B is TF-IDF (word 1-2 grams + character 2-5 grams, which absorb
Hinglish spelling variants) with logistic regression. Its scores are
temperature-scaled on the validation split, so ``p_class`` is calibrated.
Until a trained model exists the pipeline falls back to the keyword baseline.
"""

import argparse
import json
import os
import pickle
from collections.abc import Collection, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from v2v.contracts import REAL_SOURCES, HazardInterpretation, HazardType
from v2v.nlp import calibrate, extract, keyword_baseline, preprocess

DEFAULT_MODEL_PATH = Path("models/tfidf_lr.pkl")
MODEL_ENV_VAR = "V2V_CLASSIFIER"
"""Set to a model path to override DEFAULT_MODEL_PATH."""

LABEL_FIELD = "hazard"
"""Gold label key in the annotated JSONL (a HazardType value)."""

# Interpretation confidence = calibrated p_class times these factors. Starting
# values: P2-08 tunes them on validation (never on test).
HEDGED_FACTOR = 0.7
NO_LOCATION_FACTOR = 0.85


class LabelledReport(BaseModel):
    """One annotated record from a frozen split."""

    model_config = ConfigDict(frozen=True)

    report_id: str
    text: str
    source: str
    hazard: HazardType


def load_split(
    split_dir: Path, allowed_sources: Collection[str] = REAL_SOURCES
) -> list[LabelledReport]:
    """Read every ``*.jsonl`` under ``split_dir``. Refuses the frozen test
    split and any record whose source is not real."""
    if "test" in split_dir.resolve().parts:
        raise ValueError(f"refusing to read the frozen test split: {split_dir}")
    records: list[LabelledReport] = []
    for path in sorted(split_dir.rglob("*.jsonl")):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if raw.get("source") not in allowed_sources:
                raise ValueError(f"{path}:{lineno}: source={raw.get('source')!r} not allowed")
            records.append(
                LabelledReport(
                    report_id=str(raw["report_id"]),
                    text=raw["text"],
                    source=raw["source"],
                    hazard=HazardType(raw[LABEL_FIELD]),
                )
            )
    if not records:
        raise ValueError(f"no records found under {split_dir}")
    return records


class TfidfHazardClassifier:
    """TF-IDF + logistic regression over normalised text. Needs ``.[nlp]``."""

    name = "tfidf_lr"

    def __init__(self, seed: int = 0) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore[import-untyped, import-not-found, unused-ignore]
        from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped, import-not-found, unused-ignore]
        from sklearn.pipeline import FeatureUnion, Pipeline  # type: ignore[import-untyped, import-not-found, unused-ignore]

        features = FeatureUnion(
            [
                ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True)),
                (
                    "char",
                    TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True),
                ),
            ]
        )
        self._model: Any = Pipeline(
            [
                ("features", features),
                (
                    "lr",
                    LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
                ),
            ]
        )
        self.classes: tuple[HazardType, ...] = ()
        self.temperature = 1.0

    def fit(self, texts: Sequence[str], labels: Sequence[HazardType]) -> None:
        if len(set(labels)) < 2:
            raise ValueError("need at least two classes to train")
        self._model.fit([preprocess.normalise(t) for t in texts], [h.value for h in labels])
        self.classes = tuple(HazardType(c) for c in self._model.classes_)
        self.temperature = 1.0

    def logits(self, texts: Sequence[str]) -> list[list[float]]:
        scores = self._model.decision_function([preprocess.normalise(t) for t in texts])
        if len(self.classes) == 2:  # sklearn returns one column for two classes
            return [[0.0, float(s)] for s in scores]
        return [[float(v) for v in row] for row in scores]

    def calibrate_on(self, texts: Sequence[str], labels: Sequence[HazardType]) -> None:
        """Fit the temperature on held-out (validation) data."""
        index = {c: i for i, c in enumerate(self.classes)}
        self.temperature = calibrate.fit_temperature(self.logits(texts), [index[h] for h in labels])

    def predict(self, texts: Sequence[str]) -> list[tuple[HazardType, float]]:
        """(class, calibrated probability) for each text."""
        out = []
        for row in self.logits(texts):
            probs = calibrate.softmax(row, self.temperature)
            best = max(range(len(probs)), key=probs.__getitem__)
            out.append((self.classes[best], probs[best]))
        return out

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(self))

    @staticmethod
    def load(path: Path) -> "TfidfHazardClassifier":
        model = pickle.loads(path.read_bytes())  # our own trained artefact
        if not isinstance(model, TfidfHazardClassifier):
            raise TypeError(f"{path} does not hold a TfidfHazardClassifier")
        return model


def model_path() -> Path:
    return Path(os.environ.get(MODEL_ENV_VAR, DEFAULT_MODEL_PATH))


@lru_cache(maxsize=4)
def _load_cached(path: Path) -> TfidfHazardClassifier | None:
    return TfidfHazardClassifier.load(path) if path.exists() else None


def interpret(report_id: str, normalised_text: str) -> HazardInterpretation:
    """Classify a report and attach extracted details and a calibrated probability."""
    model = _load_cached(model_path())
    if model is None:
        base = keyword_baseline.classify(report_id, normalised_text)
        hazard, p_class, used = base.hazard, base.p_class, "keyword_fallback"
    else:
        hazard, p_class = model.predict([normalised_text])[0]
        used = model.name
    return HazardInterpretation(
        report_id=report_id,
        hazard=hazard,
        p_class=p_class,
        location_text=extract.extract_location(normalised_text),
        negated=extract.detect_negation(normalised_text),
        hedged=extract.detect_hedging(normalised_text),
        details={"model": used},
    )


def interpretation_confidence(interp: HazardInterpretation) -> float:
    """How sure we are of the reading, for the warning policy's theta_i.

    ``p_class`` stays the calibrated class probability (that is what ECE
    measures); this lowers it for hedged reports and reports with no stated
    location. Negated reports are already suppressed by the policy.
    """
    conf = interp.p_class
    if interp.hedged:
        conf *= HEDGED_FACTOR
    if interp.location_text is None:
        conf *= NO_LOCATION_FACTOR
    return conf


def macro_f1(gold: Sequence[HazardType], pred: Sequence[HazardType]) -> float:
    """Unweighted mean F1 over the classes present in ``gold``."""
    scores = []
    for c in set(gold):
        tp = sum(g == c and p == c for g, p in zip(gold, pred, strict=True))
        fp = sum(g != c and p == c for g, p in zip(gold, pred, strict=True))
        fn = sum(g == c and p != c for g, p in zip(gold, pred, strict=True))
        scores.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return sum(scores) / len(scores)


def evaluate(model: TfidfHazardClassifier, records: Sequence[LabelledReport]) -> dict[str, float]:
    preds = model.predict([r.text for r in records])
    gold = [r.hazard for r in records]
    return {
        "n": float(len(records)),
        "macro_f1": macro_f1(gold, [p for p, _ in preds]),
        "ece": calibrate.expected_calibration_error(
            [conf for _, conf in preds], [p == g for (p, _), g in zip(preds, gold, strict=True)]
        ),
    }


def train(train_dir: Path, val_dir: Path, out: Path, seed: int = 0) -> dict[str, float]:
    """Fit on train, fit the temperature on val, save, and report val scores."""
    train_set, val_set = load_split(train_dir), load_split(val_dir)
    model = TfidfHazardClassifier(seed=seed)
    model.fit([r.text for r in train_set], [r.hazard for r in train_set])
    before = evaluate(model, val_set)
    model.calibrate_on([r.text for r in val_set], [r.hazard for r in val_set])
    after = evaluate(model, val_set)
    model.save(out)
    return {
        "val_n": after["n"],
        "val_macro_f1": after["macro_f1"],
        "val_ece_uncalibrated": before["ece"],
        "val_ece_calibrated": after["ece"],
        "temperature": model.temperature,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train", help="train Baseline B and calibrate it on val")
    t.add_argument("--train", type=Path, default=Path("data/splits/train"))
    t.add_argument("--val", type=Path, default=Path("data/splits/val"))
    t.add_argument("--out", type=Path, default=DEFAULT_MODEL_PATH)
    t.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    print(json.dumps(train(args.train, args.val, args.out, args.seed), indent=2))


if __name__ == "__main__":
    main()
