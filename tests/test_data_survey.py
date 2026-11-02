"""Hand-written code-path strings only. Never training data."""

from pathlib import Path

import pytest

from v2v.contracts import DriverReport
from v2v.data import collect, survey

DOCS = Path(__file__).parent.parent / "docs"


def _write_survey(path: Path, rows: list[dict[str, str]]) -> None:
    columns = sorted(survey._REQUIRED) + ["Timestamp"]
    lines = [",".join(columns)]
    for row in rows:
        lines.append(",".join(row.get(c, "") for c in columns))
    path.write_text("\n".join(lines), encoding="utf-8-sig")


def test_survey_keeps_only_consenting_adults(tmp_path: Path) -> None:
    good = {
        survey.CONSENT_COLUMN: survey.CONSENT_YES,
        survey.ADULT_COLUMN: survey.ADULT_YES,
        survey.CITY_COLUMN: "Pune",
        "Hazard report 1": "call 9876543210",
        "Warning message": "second",
        "False alarm report": "third",
    }
    path = tmp_path / "r.csv"
    _write_survey(
        path,
        [
            good,
            {**good, survey.CONSENT_COLUMN: "I do not agree"},
            {**good, survey.ADULT_COLUMN: "No"},
        ],
    )

    records = list(survey.read_responses(path))

    assert len(records) == 3
    assert {r.source for r in records} == {"survey"}
    assert records[0].city == "Pune"
    assert "[PHONE]" in records[0].text


def test_fixed_options_are_marked_in_report_ids(tmp_path: Path) -> None:
    row = {
        survey.CONSENT_COLUMN: survey.CONSENT_YES,
        survey.ADULT_COLUMN: survey.ADULT_YES,
        "Hazard report 1": "one",
    }
    path = tmp_path / "r.csv"
    _write_survey(path, [row])
    marked = list(survey.read_responses(path, fixed_options=True))
    plain = list(survey.read_responses(path))
    assert marked[0].report_id.startswith(survey.FIXED_OPTION_REPORT_PREFIX)
    assert not plain[0].report_id.startswith(survey.FIXED_OPTION_REPORT_PREFIX)


def test_survey_rejects_renamed_columns(tmp_path: Path) -> None:
    path = tmp_path / "r.csv"
    path.write_text("Consent,Other\nI agree,x\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing columns"):
        list(survey.read_responses(path))


def test_form_spec_names_every_importer_column() -> None:
    spec = (DOCS / "survey_form.md").read_text(encoding="utf-8")
    for name in survey._REQUIRED | {survey.CONSENT_YES, survey.ADULT_YES}:
        assert f"`{name}`" in spec


def test_collect_survey_returns_driver_reports(tmp_path: Path) -> None:
    row = {
        survey.CONSENT_COLUMN: survey.CONSENT_YES,
        survey.ADULT_COLUMN: survey.ADULT_YES,
        "Hazard report 1": "one",
    }
    path = tmp_path / "r.csv"
    _write_survey(path, [row])
    reports = collect.survey(str(path))
    assert [r.text for r in reports] == ["one"]
    assert isinstance(reports[0], DriverReport)
