"""Hand-written code-path strings only. Never training data."""

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from v2v.contracts import HazardType
from v2v.data import annotate, datasheet, splits
from v2v.data.annotate import GoldRecord, cohen_kappa

FIXTURE = {"fixture"}


def _result(hazard: str, flags: tuple[str, ...] = (), span: tuple[int, int] | None = None) -> Any:
    result: list[dict[str, Any]] = [{"from_name": "hazard", "value": {"choices": [hazard]}}]
    if flags:
        result.append({"from_name": "flags", "value": {"choices": list(flags)}})
    if span:
        value = {"start": span[0], "end": span[1], "text": "bridge", "labels": ["LOCATION"]}
        result.append({"from_name": "span", "value": value})
    return result


def _task(report_id: str, result: Any, cancelled: bool = False, source: str = "fixture") -> Any:
    return {
        "data": {"report_id": report_id, "text": "near the bridge", "source": source},
        "annotations": [{"was_cancelled": cancelled, "result": result}],
    }


def _gold(report_id: str, hazard: HazardType, source: str = "fixture") -> GoldRecord:
    return GoldRecord(
        report_id=report_id, text="x", source=source, hazard=hazard, negated=False, hedged=False
    )


def test_assign_tasks_shares_evenly_with_overlap() -> None:
    records = [
        {"report_id": str(i), "text": "x", "source": "fixture", "language": "en"}
        for i in range(100)
    ]
    tasks = annotate.assign_tasks(records, ["a", "b", "c", "d"], seed=1)
    ids = Counter(t["data"]["report_id"] for items in tasks.values() for t in items)
    assert len(ids) == 100
    assert Counter(ids.values()) == {1: 80, 2: 20}
    assert {len(items) for items in tasks.values()} == {30}


def test_assign_tasks_needs_two_people() -> None:
    with pytest.raises(ValueError, match="two annotators"):
        annotate.assign_tasks([], ["a"])


def test_parse_export_reads_labels_and_skips_cancelled() -> None:
    tasks = [
        _task("1", _result("accident", ("hedged",), (9, 15))),
        _task("2", _result("accident"), cancelled=True),
        _task("3", []),
    ]
    parsed = annotate.parse_export(tasks, "p1", FIXTURE)
    assert len(parsed) == 1
    assert parsed[0].hazard is HazardType.ACCIDENT
    assert (parsed[0].hedged, parsed[0].negated) == (True, False)
    assert parsed[0].locations[0].text == "bridge"
    assert parsed[0].annotator == "p1"


def test_parse_export_rejects_unknown_source() -> None:
    with pytest.raises(ValueError, match="not allowed"):
        annotate.parse_export([_task("1", _result("accident"), source="generated")], "p1")


def test_cohen_kappa_known_values() -> None:
    assert cohen_kappa(["a", "b"], ["a", "b"]) == 1.0
    assert cohen_kappa(["a", "a"], ["a", "a"]) is None
    first = ["y"] * 25 + ["n"] * 25
    second = ["y"] * 20 + ["n"] * 5 + ["y"] * 10 + ["n"] * 15
    assert cohen_kappa(first, second) == pytest.approx(0.4)


def test_cohen_kappa_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        cohen_kappa(["a"], [])


def _pair(path: Path) -> list[annotate.Annotation]:
    one = [_task("1", _result("accident")), _task("2", _result("accident"))]
    two = [_task("1", _result("accident")), _task("2", _result("ambiguous"))]
    (path / "p1.json").write_text(json.dumps(one), encoding="utf-8")
    (path / "p2.json").write_text(json.dumps(two), encoding="utf-8")
    return annotate.load_exports([path / "p1.json", path / "p2.json"], FIXTURE)


def _settle(path: Path, decision: str) -> None:
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n").rstrip("\n")
    path.write_text(text.rstrip(",") + decision + "\n", encoding="utf-8")


def test_kappa_report_counts_pairs(tmp_path: Path) -> None:
    report = annotate.kappa_report(_pair(tmp_path))
    assert report["n_double_annotated_pairs"] == 2
    assert report["kappa_hazard"] == pytest.approx(0.0)
    assert annotate.kappa_report([]) == {"n_double_annotated_pairs": 0}


def test_merge_refuses_unsettled_disagreement(tmp_path: Path) -> None:
    annotations = _pair(tmp_path)
    csv_path = tmp_path / "d.csv"
    assert annotate.write_disagreements(annotations, csv_path) == 1
    with pytest.raises(ValueError, match="1 unsettled"):
        annotate.merge(annotations, csv_path)


def test_merge_uses_adjudicated_label(tmp_path: Path) -> None:
    annotations = _pair(tmp_path)
    csv_path = tmp_path / "d.csv"
    annotate.write_disagreements(annotations, csv_path)
    _settle(csv_path, ",ambiguous,true,false,p2")

    gold = annotate.merge(annotations, csv_path)

    assert [r.hazard.value for r in gold] == ["accident", "ambiguous"]
    assert gold[1].negated is True


def test_merge_rejects_unknown_span_annotator(tmp_path: Path) -> None:
    annotations = _pair(tmp_path)
    csv_path = tmp_path / "d.csv"
    annotate.write_disagreements(annotations, csv_path)
    _settle(csv_path, ",ambiguous,false,false,nobody")
    with pytest.raises(ValueError, match="final_spans_from"):
        annotate.merge(annotations, csv_path)


def test_gold_round_trip_keeps_label_field(tmp_path: Path) -> None:
    path = tmp_path / "gold.jsonl"
    annotate.write_gold([_gold("1", HazardType.SLIPPERY)], path)
    assert json.loads(path.read_text(encoding="utf-8"))["hazard"] == "slippery_road"
    assert annotate.read_gold(path)[0].report_id == "1"


def _many(per_class: int) -> list[GoldRecord]:
    return [_gold(f"{h.value}-{i}", h) for h in HazardType for i in range(per_class)]


def test_stratified_split_is_stable_and_proportional() -> None:
    records = _many(100)
    first = splits.stratified_split(records, seed=3)
    assert first == splits.stratified_split(list(reversed(records)), seed=3)
    assert first != splits.stratified_split(records, seed=4)
    sizes = {name: len(items) for name, items in first.items()}
    assert sizes == {"train": 420, "val": 105, "test": 175}
    assert Counter(r.hazard for r in first["test"]) == dict.fromkeys(HazardType, 25)


def test_thin_test_classes_are_reported() -> None:
    assert splits.thin_test_classes(_many(39)) == {h.value: 39 for h in HazardType}
    assert splits.thin_test_classes(_many(40)) == {}


def test_freeze_writes_splits_and_refuses_refreeze(tmp_path: Path) -> None:
    gold = tmp_path / "gold.jsonl"
    annotate.write_gold(_many(20), gold)
    out = tmp_path / "splits"

    splits.freeze_splits(gold, out, 0, FIXTURE)

    files = {name: out / name / "reports.jsonl" for name in ("train", "val", "test")}
    assert all(path.exists() for path in files.values())
    assert not os.access(files["test"], os.W_OK)
    with pytest.raises(FileExistsError):
        splits.freeze_splits(gold, out, 0, FIXTURE)
    files["test"].chmod(0o644)


@pytest.mark.parametrize("source", ["news_dataset", "fixture"])
def test_freeze_rejects_news_and_fixture_text(tmp_path: Path, source: str) -> None:
    gold = tmp_path / "gold.jsonl"
    annotate.write_gold([_gold("1", HazardType.ACCIDENT, source)], gold)
    with pytest.raises(ValueError, match="not allowed"):
        splits.freeze_splits(gold, tmp_path / "s", 0)


def test_freeze_rejects_duplicates_and_empty_gold(tmp_path: Path) -> None:
    gold = tmp_path / "gold.jsonl"
    annotate.write_gold([_gold("1", HazardType.ACCIDENT)] * 2, gold)
    with pytest.raises(ValueError, match="duplicate"):
        splits.freeze_splits(gold, tmp_path / "s", 0, FIXTURE)
    gold.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="no records"):
        splits.freeze_splits(gold, tmp_path / "s", 0, FIXTURE)


def test_datasheet_reports_measured_counts_only() -> None:
    text = datasheet.render(
        [_gold("1", HazardType.ACCIDENT)], {"kappa_hazard": 0.5, "n_double_annotated_pairs": 1}
    )
    assert "Total records: 1" in text
    assert "| accident | 1 |" in text
    assert "| slippery_road | 0 |" in text
    assert "| No location span | 1 |" in text
    assert "kappa_hazard: 0.5" in text
