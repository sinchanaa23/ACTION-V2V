"""Fine-tune MuRIL / IndicBERT / XLM-R on the frozen splits. Owner: P2 (P2-05).

Runs on Colab/Kaggle (GPU) as a script, not a notebook:

    python -m v2v.nlp.train_transformer select --models muril indicbert xlmr
    python -m v2v.nlp.train_transformer train --model muril --out models/transformer
    python -m v2v.nlp.train_transformer predict --model-dir models/transformer \
        --split data/splits/val --out val_transformer.jsonl

``select`` fine-tunes each candidate (3 epochs at most, as the plan caps it),
keeps each run's best epoch by validation macro-F1, picks the best candidate
and saves it with its temperature fitted on validation. The test split is
never read here; ``predict --allow-test`` is for the one final RQ1 run.

The saved directory holds the Hugging Face model and tokenizer plus
``v2v_meta.json`` (classes, temperature, base model, validation scores).
Point ``V2V_CLASSIFIER`` at it and ``classifier.interpret`` uses it.
"""

import argparse
import json
import random
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from v2v.contracts import HazardType
from v2v.nlp import calibrate, preprocess
from v2v.nlp.classifier import LabelledReport, load_split, macro_f1

CANDIDATES: dict[str, str] = {
    "muril": "google/muril-base-cased",
    "indicbert": "ai4bharat/IndicBERTv2-MLM-only",
    "xlmr": "xlm-roberta-base",
}
"""Short name -> Hugging Face model id. All three cover Romanised Hindi."""
META_FILE = "v2v_meta.json"
MAX_EPOCHS = 3
MAX_LENGTH = 128


class TransformerHazardClassifier:
    """A fine-tuned sequence classifier with the same interface as Baseline B
    (``predict``, ``logits``, ``calibrate_on``, ``save``/``load``, ``name``)."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        classes: Sequence[HazardType],
        base_model: str,
        temperature: float = 1.0,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.classes = tuple(classes)
        self.base_model = base_model
        self.temperature = temperature
        self.name = f"transformer:{base_model}"
        self.val_scores: dict[str, float] = {}

    @classmethod
    def from_pretrained(
        cls, base_model: str, classes: Sequence[HazardType]
    ) -> "TransformerHazardClassifier":
        from transformers import (  # type: ignore[import-not-found, import-untyped, unused-ignore]
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        tokenizer = AutoTokenizer.from_pretrained(base_model)
        model = AutoModelForSequenceClassification.from_pretrained(
            base_model,
            num_labels=len(classes),
            id2label={i: c.value for i, c in enumerate(classes)},
            label2id={c.value: i for i, c in enumerate(classes)},
        )
        return cls(model, tokenizer, classes, base_model)

    def _encode(self, texts: Sequence[str]) -> Any:
        return self.tokenizer(
            [preprocess.normalise(t) for t in texts],
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

    def logits(self, texts: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        import torch  # type: ignore[import-not-found, unused-ignore]

        self.model.eval()
        out: list[list[float]] = []
        device = next(self.model.parameters()).device
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = {
                    k: v.to(device) for k, v in self._encode(texts[i : i + batch_size]).items()
                }
                out.extend(self.model(**batch).logits.cpu().tolist())
        return out

    def calibrate_on(self, texts: Sequence[str], labels: Sequence[HazardType]) -> None:
        index = {c: i for i, c in enumerate(self.classes)}
        self.temperature = calibrate.fit_temperature(self.logits(texts), [index[h] for h in labels])

    def predict(self, texts: Sequence[str]) -> list[tuple[HazardType, float]]:
        out = []
        for row in self.logits(texts):
            probs = calibrate.softmax(row, self.temperature)
            best = max(range(len(probs)), key=probs.__getitem__)
            out.append((self.classes[best], probs[best]))
        return out

    def save(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(out_dir)
        self.tokenizer.save_pretrained(out_dir)
        meta = {
            "base_model": self.base_model,
            "classes": [c.value for c in self.classes],
            "temperature": self.temperature,
            "val_scores": self.val_scores,
        }
        (out_dir / META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, model_dir: Path) -> "TransformerHazardClassifier":
        from transformers import (  # type: ignore[import-not-found, import-untyped, unused-ignore]
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        meta = json.loads((model_dir / META_FILE).read_text(encoding="utf-8"))
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        loaded = cls(
            model,
            tokenizer,
            [HazardType(c) for c in meta["classes"]],
            meta["base_model"],
            float(meta["temperature"]),
        )
        loaded.val_scores = meta.get("val_scores", {})
        return loaded


def class_weights(labels: Sequence[HazardType], classes: Sequence[HazardType]) -> list[float]:
    """Inverse-frequency weights (mean 1 over the classes present), as
    ``class_weight="balanced"`` does for Baseline B."""
    counts = Counter(labels)
    n = len(labels)
    return [n / (len(classes) * counts[c]) if counts[c] else 0.0 for c in classes]


def evaluate(
    model: TransformerHazardClassifier, records: Sequence[LabelledReport]
) -> dict[str, float]:
    preds = model.predict([r.text for r in records])
    gold = [r.hazard for r in records]
    return {
        "n": float(len(records)),
        "macro_f1": macro_f1(gold, [p for p, _ in preds]),
        "ece": calibrate.expected_calibration_error(
            [c for _, c in preds], [p == g for (p, _), g in zip(preds, gold, strict=True)]
        ),
    }


def fine_tune(
    clf: TransformerHazardClassifier,
    train_set: Sequence[LabelledReport],
    val_set: Sequence[LabelledReport],
    epochs: int = MAX_EPOCHS,
    lr: float = 2e-5,
    batch_size: int = 16,
    seed: int = 0,
    log: Any = print,
) -> dict[str, float]:
    """Train with class-weighted cross-entropy, keep the epoch with the best
    validation macro-F1, then fit the temperature on validation."""
    import torch  # type: ignore[import-not-found, unused-ignore]

    if not 1 <= epochs <= MAX_EPOCHS:
        raise ValueError(f"epochs must be 1..{MAX_EPOCHS}")
    random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clf.model.to(device)
    index = {c: i for i, c in enumerate(clf.classes)}
    weights = torch.tensor(class_weights([r.hazard for r in train_set], clf.classes), device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    optimiser = torch.optim.AdamW(clf.model.parameters(), lr=lr)
    best_f1, best_state = -1.0, None
    order = list(range(len(train_set)))
    for epoch in range(1, epochs + 1):
        clf.model.train()
        random.Random(seed + epoch).shuffle(order)
        total = 0.0
        for i in range(0, len(order), batch_size):
            batch_records = [train_set[j] for j in order[i : i + batch_size]]
            enc = {k: v.to(device) for k, v in clf._encode([r.text for r in batch_records]).items()}
            target = torch.tensor([index[r.hazard] for r in batch_records], device=device)
            loss = loss_fn(clf.model(**enc).logits, target)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total += loss.item() * len(batch_records)
        scores = evaluate(clf, val_set)
        log(
            f"epoch {epoch}: train loss {total / len(order):.4f}, "
            f"val macro-F1 {scores['macro_f1']:.4f}"
        )
        if scores["macro_f1"] > best_f1:
            best_f1 = scores["macro_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in clf.model.state_dict().items()}
    assert best_state is not None
    clf.model.load_state_dict(best_state)
    before = evaluate(clf, val_set)
    clf.calibrate_on([r.text for r in val_set], [r.hazard for r in val_set])
    after = evaluate(clf, val_set)
    clf.val_scores = {
        "val_n": after["n"],
        "val_macro_f1": after["macro_f1"],
        "val_ece_uncalibrated": before["ece"],
        "val_ece_calibrated": after["ece"],
        "temperature": clf.temperature,
    }
    return clf.val_scores


def train_one(
    name: str, train_dir: Path, val_dir: Path, epochs: int, seed: int
) -> TransformerHazardClassifier:
    train_set, val_set = load_split(train_dir), load_split(val_dir)
    classes = sorted({r.hazard for r in train_set}, key=lambda h: h.value)
    clf = TransformerHazardClassifier.from_pretrained(CANDIDATES.get(name, name), classes)
    fine_tune(clf, train_set, val_set, epochs=epochs, seed=seed)
    return clf


def select(
    names: Sequence[str], train_dir: Path, val_dir: Path, out_dir: Path, epochs: int, seed: int
) -> dict[str, Any]:
    """Fine-tune every candidate and keep the best by validation macro-F1."""
    results: dict[str, dict[str, float]] = {}
    best_name, best = None, None
    for name in names:
        print(f"== {name}")
        clf = train_one(name, train_dir, val_dir, epochs, seed)
        results[name] = clf.val_scores
        if best is None or clf.val_scores["val_macro_f1"] > best.val_scores["val_macro_f1"]:
            best_name, best = name, clf
    assert best is not None
    best.save(out_dir)
    summary = {"selected": best_name, "candidates": results}
    (out_dir / "selection.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def write_predictions(
    model: TransformerHazardClassifier, records: Sequence[LabelledReport], out: Path
) -> None:
    """Same JSONL as Baseline B's predict (report_id, hazard, p_class)."""
    preds = model.predict([r.text for r in records])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "".join(
            json.dumps({"report_id": r.report_id, "hazard": h.value, "p_class": round(p, 6)}) + "\n"
            for r, (h, p) in zip(records, preds, strict=True)
        ),
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("select", "train"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--train", type=Path, default=Path("data/splits/train"))
        cmd.add_argument("--val", type=Path, default=Path("data/splits/val"))
        cmd.add_argument("--out", type=Path, default=Path("models/transformer"))
        cmd.add_argument("--epochs", type=int, default=MAX_EPOCHS)
        cmd.add_argument("--seed", type=int, default=0)
    sub.choices["select"].add_argument(
        "--models", nargs="+", default=sorted(CANDIDATES), help="short names or HF ids"
    )
    sub.choices["train"].add_argument("--model", required=True, help="short name or HF id")
    pred = sub.add_parser("predict")
    pred.add_argument("--model-dir", type=Path, default=Path("models/transformer"))
    pred.add_argument("--split", type=Path, required=True)
    pred.add_argument("--out", type=Path, required=True)
    pred.add_argument("--allow-test", action="store_true", help="final RQ1 run only")
    args = parser.parse_args(argv)

    if args.command == "select":
        print(
            json.dumps(
                select(args.models, args.train, args.val, args.out, args.epochs, args.seed),
                indent=2,
            )
        )
    elif args.command == "train":
        clf = train_one(args.model, args.train, args.val, args.epochs, args.seed)
        clf.save(args.out)
        print(json.dumps(clf.val_scores, indent=2))
    else:
        records = load_split(args.split, allow_test=args.allow_test)
        write_predictions(TransformerHazardClassifier.load(args.model_dir), records, args.out)
        print(f"wrote {len(records)} predictions to {args.out}")


if __name__ == "__main__":
    main()
