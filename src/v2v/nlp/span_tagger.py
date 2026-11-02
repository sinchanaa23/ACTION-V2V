"""Token-tagging location extractor trained on annotated spans. Owner: P2 (P2-06).

    python -m v2v.nlp.span_tagger train --train data/splits/train --val data/splits/val
    python -m v2v.nlp.span_tagger evaluate --split data/splits/val

Learns BIO tags (Begin / Inside / Outside a location span) per token from
P1's span labels, with logistic regression over hand-built token features:
the word, its affixes, whether it is a place word from the rule-based
extractor, and the words around it (Hinglish postpositions such as "ke paas"
and English prepositions such as "near" sit next to locations). ``evaluate``
scores it against the rule-based extractor (P2-03) on the same split, so the
choice between them is made on validation, never on test.

Like the rules, it only ever returns a span copied from the text: it can
miss a location, but it cannot invent one.
"""

import argparse
import json
import pickle
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from v2v.data.annotate import GoldRecord, read_gold
from v2v.nlp import extract, preprocess

DEFAULT_MODEL_PATH = Path("models/span_tagger.pkl")
SPLIT_FILE = "reports.jsonl"
_TOKEN = re.compile(r"[\w']+")
WINDOW = 2
"""Neighbouring tokens on each side used as features."""

Token = tuple[str, int, int]
"""(text, start, end) with character offsets into the original string."""


def tokenize(text: str) -> list[Token]:
    return [(m.group(0), m.start(), m.end()) for m in _TOKEN.finditer(text)]


def bio_labels(tokens: Sequence[Token], spans: Sequence[tuple[int, int]]) -> list[str]:
    """B/I/O per token from character spans. A token belongs to a span if it
    overlaps it; the first such token is B."""
    labels = []
    for _, start, end in tokens:
        inside = [s for s in spans if start < s[1] and end > s[0]]
        if not inside:
            labels.append("O")
        elif labels and labels[-1] != "O" and any(start > s[0] for s in inside):
            labels.append("I")
        else:
            labels.append("B")
    return labels


def _place(word: str) -> str:
    if word in extract.STRONG_PLACE_WORDS:
        return "strong"
    if word in extract.WEAK_PLACE_WORDS:
        return "weak"
    return "-"


def token_features(tokens: Sequence[Token], i: int) -> dict[str, Any]:
    words = [preprocess.normalise(t[0]) for t in tokens]
    w = words[i]
    feats: dict[str, Any] = {
        "bias": 1.0,
        "w": w,
        "pre3": w[:3],
        "suf3": w[-3:],
        "digit": w.isdigit(),
        "place": _place(w),
        "len>3": len(w) > 3,
    }
    for d in range(1, WINDOW + 1):
        for sign, name in ((-1, "L"), (1, "R")):
            j = i + sign * d
            if 0 <= j < len(words):
                feats[f"{name}{d}"] = words[j]
                feats[f"{name}{d}_place"] = _place(words[j])
            else:
                feats[f"{name}{d}"] = "<s>" if sign < 0 else "</s>"
    if i + 1 < len(words):  # two-word place names ("petrol pump", "bus stop")
        feats["place2"] = _place(f"{w} {words[i + 1]}")
    if i > 0:
        feats["place2_prev"] = _place(f"{words[i - 1]} {w}")
    return feats


def decode(tokens: Sequence[Token], tags: Sequence[str], text: str) -> list[str]:
    """Spans (copied from ``text``) from BIO tags; a stray I starts a span."""
    spans: list[tuple[int, int]] = []
    current: tuple[int, int] | None = None
    for (_, start, end), tag in zip(tokens, tags, strict=True):
        if tag == "I" and current is not None:
            current = (current[0], end)
            continue
        if current is not None:
            spans.append(current)
        current = (start, end) if tag in ("B", "I") else None
    if current is not None:
        spans.append(current)
    return [text[a:b] for a, b in spans]


class SpanTagger:
    name = "span_tagger_lr"

    def __init__(self, seed: int = 0) -> None:
        from sklearn.feature_extraction import DictVectorizer  # type: ignore[import-untyped, import-not-found, unused-ignore]
        from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped, import-not-found, unused-ignore]
        from sklearn.pipeline import Pipeline  # type: ignore[import-untyped, import-not-found, unused-ignore]

        self._model: Any = Pipeline(
            [
                ("vec", DictVectorizer()),
                (
                    "lr",
                    LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
                ),
            ]
        )

    def fit(self, records: Sequence[GoldRecord]) -> None:
        feats: list[dict[str, Any]] = []
        labels: list[str] = []
        for r in records:
            tokens = tokenize(r.text)
            spans = [(s.start, s.end) for s in r.locations]
            labels.extend(bio_labels(tokens, spans))
            feats.extend(token_features(tokens, i) for i in range(len(tokens)))
        if len(set(labels)) < 2:
            raise ValueError("training data needs both location and non-location tokens")
        self._model.fit(feats, labels)

    def tag(self, text: str) -> list[str]:
        tokens = tokenize(text)
        if not tokens:
            return []
        return [
            str(t)
            for t in self._model.predict([token_features(tokens, i) for i in range(len(tokens))])
        ]

    def find_locations(self, text: str) -> list[str]:
        return decode(tokenize(text), self.tag(text), text)

    def extract_location(self, text: str) -> str | None:
        """Longest predicted span, like the rule-based extractor; None if none."""
        spans = self.find_locations(text)
        return max(spans, key=lambda s: len(s.split())) if spans else None

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(self))

    @staticmethod
    def load(path: Path) -> "SpanTagger":
        model = pickle.loads(path.read_bytes())  # our own trained artefact
        if not isinstance(model, SpanTagger):
            raise TypeError(f"{path} does not hold a SpanTagger")
        return model


# --- Evaluation ---------------------------------------------------------------------


def _norm(span: str) -> str:
    return " ".join(span.lower().split())


@dataclass
class _Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def scores(self) -> dict[str, float]:
        p = self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0
        r = self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0
        return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0}


def span_scores(
    gold: Sequence[Sequence[str]], predicted: Sequence[Sequence[str]]
) -> dict[str, Any]:
    """Exact span match (case- and space-insensitive), token overlap, and
    whether a report with no location is correctly left at None."""
    exact, overlap = _Counts(), _Counts()
    none_right = none_total = 0
    for g, p in zip(gold, predicted, strict=True):
        gs, ps = {_norm(s) for s in g}, {_norm(s) for s in p}
        exact.tp += len(gs & ps)
        exact.fp += len(ps - gs)
        exact.fn += len(gs - ps)
        gt = {w for s in gs for w in s.split()}
        pt = {w for s in ps for w in s.split()}
        overlap.tp += len(gt & pt)
        overlap.fp += len(pt - gt)
        overlap.fn += len(gt - pt)
        if not gs:
            none_total += 1
            none_right += not ps
    return {
        "exact": exact.scores(),
        "token_overlap": overlap.scores(),
        "no_location_kept_none": none_right / none_total if none_total else None,
        "n": len(gold),
    }


def compare(records: Sequence[GoldRecord], tagger: SpanTagger | None) -> dict[str, Any]:
    """Rule-based extractor vs the tagger (if given) on the same records."""
    gold = [[s.text for s in r.locations] for r in records]
    rules = [extract.find_locations(preprocess.normalise(r.text)) for r in records]
    out: dict[str, Any] = {"rules": span_scores(gold, rules)}
    if tagger is not None:
        out["tagger"] = span_scores(gold, [tagger.find_locations(r.text) for r in records])
    return out


def _read_split(split_dir: Path, allow_test: bool = False) -> list[GoldRecord]:
    if not allow_test and "test" in split_dir.resolve().parts:
        raise ValueError(f"refusing to read the frozen test split: {split_dir}")
    return read_gold(split_dir / SPLIT_FILE)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train")
    t.add_argument("--train", type=Path, default=Path("data/splits/train"))
    t.add_argument("--val", type=Path, default=Path("data/splits/val"))
    t.add_argument("--out", type=Path, default=DEFAULT_MODEL_PATH)
    e = sub.add_parser("evaluate")
    e.add_argument("--split", type=Path, default=Path("data/splits/val"))
    e.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    e.add_argument("--allow-test", action="store_true", help="final RQ1 run only")
    args = parser.parse_args(argv)
    if args.command == "train":
        tagger = SpanTagger()
        tagger.fit(_read_split(args.train))
        tagger.save(args.out)
        print(json.dumps(compare(_read_split(args.val), tagger), indent=2))
        return
    tagger_or_none = SpanTagger.load(args.model) if args.model.exists() else None
    records = _read_split(args.split, allow_test=args.allow_test)
    print(json.dumps(compare(records, tagger_or_none), indent=2))


if __name__ == "__main__":
    main()
