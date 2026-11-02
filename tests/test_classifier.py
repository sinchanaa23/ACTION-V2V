"""P2-01. Hand-written code-path strings with source="fixture"; never data.

Training here only checks that the code runs; the scores mean nothing.
"""

import json
from pathlib import Path

import pytest

from tests.fixtures import ACCIDENT_HINGLISH
from v2v.contracts import HazardInterpretation, HazardType
from v2v.nlp import classifier

_FIXTURE_ROWS = [
    ("accident hua hai aage", HazardType.ACCIDENT),
    ("big crash near the signal", HazardType.ACCIDENT),
    ("gaadi palat gayi accident", HazardType.ACCIDENT),
    ("oil on the road very slippery", HazardType.SLIPPERY),
    ("road fisalan hai paani", HazardType.SLIPPERY),
    ("slippery road after rain", HazardType.SLIPPERY),
    ("reached office on time", HazardType.NON_ACTIONABLE),
    ("traffic is fine today", HazardType.NON_ACTIONABLE),
    ("good morning everyone", HazardType.NON_ACTIONABLE),
]


def _write_split(directory: Path, source: str = "fixture") -> Path:
    directory.mkdir(parents=True)
    lines = [
        json.dumps({"report_id": f"f{i}", "text": t, "source": source, "hazard": h.value})
        for i, (t, h) in enumerate(_FIXTURE_ROWS)
    ]
    (directory / "part.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return directory


@pytest.fixture(autouse=True)
def _no_cached_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(classifier.MODEL_ENV_VAR, str(tmp_path / "absent.pkl"))
    classifier._load_cached.cache_clear()


def test_load_split_rejects_fixture_source_by_default(tmp_path: Path) -> None:
    split = _write_split(tmp_path / "train")
    with pytest.raises(ValueError, match="not allowed"):
        classifier.load_split(split)
    assert len(classifier.load_split(split, allowed_sources={"fixture"})) == len(_FIXTURE_ROWS)


def test_load_split_refuses_test_split(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="frozen test split"):
        classifier.load_split(tmp_path / "splits" / "test")


def test_load_split_needs_records(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="no records"):
        classifier.load_split(tmp_path / "empty", allowed_sources={"fixture"})


def test_interpret_falls_back_to_keywords_without_a_model() -> None:
    interp = classifier.interpret("r0", ACCIDENT_HINGLISH.lower())
    assert interp.hazard is HazardType.ACCIDENT
    assert interp.details["model"] == "keyword_fallback"
    assert not interp.negated and not interp.hedged


def test_interpretation_confidence_penalties() -> None:
    base = HazardInterpretation(
        report_id="r", hazard=HazardType.ACCIDENT, p_class=0.8, location_text="bridge"
    )
    assert classifier.interpretation_confidence(base) == 0.8
    hedged = base.model_copy(update={"hedged": True})
    assert classifier.interpretation_confidence(hedged) == pytest.approx(0.8 * 0.7)
    nowhere = base.model_copy(update={"location_text": None})
    assert classifier.interpretation_confidence(nowhere) == pytest.approx(0.8 * 0.85)


def test_macro_f1() -> None:
    a, s = HazardType.ACCIDENT, HazardType.SLIPPERY
    assert classifier.macro_f1([a, s], [a, s]) == 1.0
    assert classifier.macro_f1([a, a, s, s], [a, a, a, a]) == pytest.approx((2 / 3) / 2)


def test_train_save_load_and_interpret(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("sklearn")
    texts = [t for t, _ in _FIXTURE_ROWS]
    labels = [h for _, h in _FIXTURE_ROWS]
    model = classifier.TfidfHazardClassifier(seed=0)
    model.fit(texts, labels)
    model.calibrate_on(texts, labels)
    assert model.temperature > 0
    hazard, p = model.predict(["accident hua"])[0]
    assert hazard in model.classes and 0.0 < p <= 1.0
    assert 0.0 <= classifier.evaluate(model, _as_records())["ece"] <= 1.0

    path = tmp_path / "m.pkl"
    model.save(path)
    monkeypatch.setenv(classifier.MODEL_ENV_VAR, str(path))
    classifier._load_cached.cache_clear()
    interp = classifier.interpret("r1", "accident hua hai bridge ke paas")
    assert interp.details["model"] == "tfidf_lr"


def test_fit_needs_two_classes() -> None:
    pytest.importorskip("sklearn")
    with pytest.raises(ValueError):
        classifier.TfidfHazardClassifier().fit(["a", "b"], [HazardType.ACCIDENT] * 2)


def test_load_rejects_other_objects(tmp_path: Path) -> None:
    import pickle

    path = tmp_path / "x.pkl"
    path.write_bytes(pickle.dumps({"not": "a model"}))
    with pytest.raises(TypeError):
        classifier.TfidfHazardClassifier.load(path)


def _as_records() -> list[classifier.LabelledReport]:
    return [
        classifier.LabelledReport(report_id=f"f{i}", text=t, source="fixture", hazard=h)
        for i, (t, h) in enumerate(_FIXTURE_ROWS)
    ]


def test_load_split_reads_p1_gold_records(tmp_path: Path) -> None:
    # P1's GoldRecord adds negated, hedged and location spans; they are ignored here.
    split = tmp_path / "val"
    split.mkdir()
    row = {
        "report_id": "f0",
        "text": "accident hua hai bridge ke paas",
        "source": "fixture",
        "hazard": "accident",
        "negated": False,
        "hedged": False,
        "locations": [{"start": 17, "end": 23, "text": "bridge"}],
    }
    (split / "reports.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    [record] = classifier.load_split(split, allowed_sources={"fixture"})
    assert record.hazard is HazardType.ACCIDENT


def test_test_split_needs_explicit_flag(tmp_path: Path) -> None:
    split = _write_split(tmp_path / "splits" / "test")
    assert classifier.load_split(split, allowed_sources={"fixture"}, allow_test=True)


def test_write_predictions_matches_error_analysis_format(tmp_path: Path) -> None:
    pytest.importorskip("sklearn")
    model = classifier.TfidfHazardClassifier(seed=0)
    model.fit([t for t, _ in _FIXTURE_ROWS], [h for _, h in _FIXTURE_ROWS])
    out = tmp_path / "pred.jsonl"
    classifier.write_predictions(model, _as_records(), out)
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [r["report_id"] for r in rows] == [f"f{i}" for i in range(len(_FIXTURE_ROWS))]
    assert all(HazardType(r["hazard"]) in model.classes for r in rows)
    assert all(0.0 < r["p_class"] <= 1.0 for r in rows)


def test_predict_cli_refuses_test_split_without_flag(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="frozen test split"):
        classifier.main(
            ["predict", "--split", str(tmp_path / "splits" / "test"), "--out", "x.jsonl"]
        )
