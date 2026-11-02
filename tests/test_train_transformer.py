"""P2-05. A tiny BERT built locally (no download) trained on fixture strings:
checks code paths only, the scores mean nothing."""

import json
from pathlib import Path

import pytest

from v2v.contracts import HazardType
from v2v.nlp import classifier, train_transformer
from v2v.nlp.classifier import LabelledReport

_ROWS = [
    ("accident hua hai aage", HazardType.ACCIDENT),
    ("big crash near the signal", HazardType.ACCIDENT),
    ("oil on the road very slippery", HazardType.SLIPPERY),
    ("road slippery after rain", HazardType.SLIPPERY),
    ("reached office on time", HazardType.NON_ACTIONABLE),
    ("traffic is fine today", HazardType.NON_ACTIONABLE),
]


def _records() -> list[LabelledReport]:
    return [
        LabelledReport(report_id=f"f{i}", text=t, source="fixture", hazard=h)
        for i, (t, h) in enumerate(_ROWS)
    ]


def test_class_weights_balance_classes() -> None:
    a, s = HazardType.ACCIDENT, HazardType.SLIPPERY
    weights = train_transformer.class_weights([a, a, a, s], [a, s])
    assert weights == pytest.approx([4 / 6, 2.0])
    assert train_transformer.class_weights([a], [a, s]) == [0.5, 0.0]


def test_candidates_cover_the_plan() -> None:
    assert set(train_transformer.CANDIDATES) == {"muril", "indicbert", "xlmr"}


def _tiny(tmp_path: Path) -> train_transformer.TransformerHazardClassifier:
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    words = sorted({w for t, _ in _ROWS for w in t.split()})
    vocab = tmp_path / "vocab.txt"
    vocab.write_text(
        "\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *words]), encoding="utf-8"
    )
    tokenizer = transformers.BertTokenizerFast(vocab_file=str(vocab))
    classes = [HazardType.ACCIDENT, HazardType.NON_ACTIONABLE, HazardType.SLIPPERY]
    config = transformers.BertConfig(
        vocab_size=len(words) + 5,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        num_labels=len(classes),
    )
    model = transformers.BertForSequenceClassification(config)
    return train_transformer.TransformerHazardClassifier(model, tokenizer, classes, "tiny-bert")


def test_fine_tune_save_load_predict(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clf = _tiny(tmp_path)
    records = _records()
    logs: list[str] = []
    scores = train_transformer.fine_tune(
        clf, records, records, epochs=2, lr=1e-3, batch_size=4, log=logs.append
    )
    assert len(logs) == 2 and scores["temperature"] > 0
    assert 0.0 <= scores["val_macro_f1"] <= 1.0
    hazard, p = clf.predict(["accident hua"])[0]
    assert hazard in clf.classes and 0.0 < p <= 1.0

    out = tmp_path / "model"
    clf.save(out)
    meta = json.loads((out / train_transformer.META_FILE).read_text(encoding="utf-8"))
    assert meta["base_model"] == "tiny-bert" and meta["val_scores"]
    loaded = train_transformer.TransformerHazardClassifier.load(out)
    assert loaded.temperature == pytest.approx(clf.temperature)
    assert loaded.predict(["accident hua"])[0][0] == hazard

    pred = tmp_path / "pred.jsonl"
    train_transformer.write_predictions(loaded, records, pred)
    rows = [json.loads(line) for line in pred.read_text(encoding="utf-8").splitlines()]
    assert [r["report_id"] for r in rows] == [r.report_id for r in records]

    # The pipeline's interpret() picks the transformer up from a directory.
    monkeypatch.setenv(classifier.MODEL_ENV_VAR, str(out))
    classifier._load_cached.cache_clear()
    interp = classifier.interpret("r", "accident hua hai")
    assert interp.details["model"] == "transformer:tiny-bert"
    classifier._load_cached.cache_clear()


def test_epochs_are_capped(tmp_path: Path) -> None:
    clf = _tiny(tmp_path)
    with pytest.raises(ValueError):
        train_transformer.fine_tune(clf, _records(), _records(), epochs=4)


def test_empty_model_dir_is_no_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(classifier.MODEL_ENV_VAR, str(tmp_path))
    classifier._load_cached.cache_clear()
    assert classifier.interpret("r", "accident hua").details["model"] == "keyword_fallback"
    classifier._load_cached.cache_clear()
