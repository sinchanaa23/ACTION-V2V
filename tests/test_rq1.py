"""P2-08 RQ1 tooling on fixture records (source="fixture"): code paths only."""

from pathlib import Path
from typing import Any

import pytest

from v2v.contracts import HazardType
from v2v.data.annotate import GoldRecord, Span
from v2v.nlp import rq1

A, S, N = HazardType.ACCIDENT, HazardType.SLIPPERY, HazardType.NON_ACTIONABLE


def _rec(
    rid: str, text: str, hazard: HazardType, place: str | None = None, **flags: bool
) -> GoldRecord:
    spans = ()
    if place:
        start = text.index(place)
        spans = (Span(start=start, end=start + len(place), text=place),)
    return GoldRecord(
        report_id=rid,
        text=text,
        source="fixture",
        hazard=hazard,
        negated=flags.get("negated", False),
        hedged=flags.get("hedged", False),
        locations=spans,
    )


RECORDS = [
    _rec("1", "accident near the signal", A, "signal"),
    _rec("2", "shayad accident hua hai", A, hedged=True),
    _rec("3", "accident nahi hua", A, negated=True),
    _rec("4", "oil spill, slippery road near petrol pump", S, "petrol pump"),
    _rec("5", "reached office on time", N),
]


class _FakeModel:
    """Always right, with fixed confidences."""

    def __init__(self, records: list[GoldRecord], p: float = 0.9) -> None:
        self.answers = {r.text: r.hazard for r in records}
        self.p = p

    def predict(self, texts: list[str]) -> list[tuple[HazardType, float]]:
        return [(self.answers[t], self.p) for t in texts]


def _write(split: Path) -> Path:
    split.mkdir(parents=True)
    (split / "reports.jsonl").write_text(
        "".join(r.model_dump_json() + "\n" for r in RECORDS), encoding="utf-8"
    )
    return split


def test_refuses_test_split(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="frozen test split"):
        rq1.read_split(tmp_path / "splits" / "test")


def test_classification_report_with_and_without_probabilities() -> None:
    preds = rq1.model_predictions(_FakeModel(RECORDS), RECORDS)
    report = rq1.classification_report(RECORDS, preds)
    assert report["macro_f1"] == 1.0 and report["ece"] == pytest.approx(0.1)
    keyword = rq1.classification_report(RECORDS, rq1.keyword_predictions(RECORDS))
    assert keyword["ece"] is None
    lo, hi = keyword["accuracy_ci"]
    assert 0.0 <= lo <= hi <= 1.0


def test_flag_and_extractor_scores() -> None:
    assert rq1.flag_scores([True, False, True], [True, True, False]) == {
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
        "support": 2,
    }
    ex = rq1.extractor_report(RECORDS)
    assert ex["negation"]["recall"] == 1.0 and ex["hedging"]["recall"] == 1.0
    loc = rq1.location_report(RECORDS, None)
    assert set(loc) == {"rules"} and loc["rules"]["exact"]["f1"] == 1.0


def test_tune_factors_lowers_ece_when_flagged_reports_are_wrong() -> None:
    class Wrong(_FakeModel):
        def predict(self, texts: list[str]) -> list[tuple[HazardType, float]]:
            # Wrong exactly on the hedged and the location-less reports.
            return [
                (HazardType.STOPPED if "shayad" in t or "near" not in t else self.answers[t], 0.9)
                for t in texts
            ]

    preds = rq1.model_predictions(Wrong(RECORDS), RECORDS)
    tuned = rq1.tune_factors(RECORDS, preds)
    assert tuned["ece_tuned"] <= tuned["ece_current"] <= tuned["ece_no_factors"]
    assert tuned["hedged_factor"] in rq1.FACTOR_GRID
    with pytest.raises(ValueError):
        rq1.tune_factors(RECORDS, rq1.keyword_predictions(RECORDS))


def test_run_writes_predictions_and_report(tmp_path: Path) -> None:
    split = _write(tmp_path / "val")

    def loader(_: Path) -> Any:
        return _FakeModel(RECORDS)

    path = rq1.run(split, {"fake": Path("unused")}, tmp_path / "out", loader=loader)
    text = path.read_text(encoding="utf-8")
    assert "| fake | 1.000" in text and "| keyword |" in text
    assert "## Location extraction" in text
    for name in ("keyword", "fake"):
        assert (tmp_path / "out" / f"val_{name}.jsonl").exists()


def test_cli_rejects_tuning_on_test(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        rq1.main(["--tune-factors", "x=y", "--allow-test", "--split", str(tmp_path)])
    with pytest.raises(ValueError):
        rq1._named(["no-equals-sign"])
    with pytest.raises(FileNotFoundError):
        rq1.load_model(tmp_path / "absent.pkl")
