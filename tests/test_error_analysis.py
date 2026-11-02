"""Hand-written code-path strings only. Never training data."""

import json
from pathlib import Path

import pytest

from v2v.contracts import HazardType
from v2v.data import annotate, error_analysis, splits
from v2v.data.annotate import GoldRecord, Span
from v2v.data.error_analysis import read_predictions, render, subset_row, top_confusions
from v2v.nlp import keyword_baseline

A = HazardType.ACCIDENT
N = HazardType.NON_ACTIONABLE


def _gold(
    report_id: str,
    hazard: HazardType,
    text: str = "truck hit the wall",
    negated: bool = False,
    hedged: bool = False,
    located: bool = True,
) -> GoldRecord:
    return GoldRecord(
        report_id=report_id,
        text=text,
        source="fixture",
        hazard=hazard,
        negated=negated,
        hedged=hedged,
        locations=(Span(start=0, end=1, text="x"),) if located else (),
    )


def test_read_predictions(tmp_path: Path) -> None:
    path = tmp_path / "p.jsonl"
    path.write_text('{"report_id": "1", "hazard": "accident"}\n\n', encoding="utf-8")
    assert read_predictions(path) == {"1": A}


def test_subset_row_scores_and_empty_subset() -> None:
    assert subset_row("hedged", [], {}) == "| hedged | 0 | - | - | - |"
    row = subset_row("all", [_gold("1", A), _gold("2", A)], {"1": A, "2": N})
    assert row.startswith("| all | 2 | ")
    assert "0.500" in row


def test_top_confusions_counts_mistakes_with_ids() -> None:
    records = [_gold("1", A), _gold("2", A), _gold("3", N)]
    rows = top_confusions(records, {"1": N, "2": N, "3": N})
    assert rows == ["| accident | non_actionable | 2 | 1, 2 |"]


def test_render_splits_hard_subsets() -> None:
    records = [
        _gold("1", N, "accident nahi hua", negated=True, located=False),
        _gold("2", A, "shayad accident hua hai", hedged=True),
        _gold("3", A, "truck hit the wall"),
    ]
    text = render(records, {"keyword": {"1": A, "2": A, "3": A}})
    assert "| negated | 1 | 0.000 | 0.000 |" in text
    assert "| hedged | 1 | 1.000 | 1.000 |" in text
    assert "| no location | 1 | 0.000 |" in text
    assert "| non_actionable | accident | 1 | 1 |" in text


def test_render_refuses_missing_predictions() -> None:
    with pytest.raises(ValueError, match="1 predictions missing"):
        render([_gold("1", A)], {"keyword": {}})


def test_subsets_use_the_language_tag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(error_analysis, "language_mix", lambda _: "hi-en")
    assert error_analysis.SUBSETS["code-mixed"](_gold("1", A))
    assert not error_analysis.SUBSETS["english"](_gold("1", A))


def test_gold_to_report_end_to_end(tmp_path: Path) -> None:
    """The offline path from labelled records to the RQ1 report."""
    records = [
        _gold(f"{h.value}-{i}", h, "takkar on the bridge") for h in HazardType for i in range(8)
    ]
    gold_path = tmp_path / "gold.jsonl"
    annotate.write_gold(records, gold_path)
    out = tmp_path / "splits"
    splits.freeze_splits(gold_path, out, 0, {"fixture"})

    val = annotate.read_gold(out / "val" / "reports.jsonl")
    predictions = tmp_path / "pred.jsonl"
    rows = keyword_baseline.predict(val)
    predictions.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    report = render(val, {"keyword": read_predictions(predictions)})

    assert f"Records: {len(val)}" in report
    assert "## keyword" in report
    (out / "test" / "reports.jsonl").chmod(0o644)
