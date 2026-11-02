"""P3-07: tables from runner rows (toy world, fixture text)."""

from pathlib import Path

import pytest

from v2v.confidence import policy
from v2v.eval import report, runner
from v2v.experiments import CONFIGS


@pytest.fixture(scope="module")
def toy_rows() -> list[dict[str, object]]:
    pool = runner.ReportPool(runner.DEMO_POOL)
    grid = policy.threshold_grid((0.6, 0.9), (0.75, 0.9))
    rows: list[dict[str, object]] = []
    for seed in (0, 1, 2):
        rows.extend(runner.run_toy(seed, pool, list(CONFIGS.values()), grid, {"git_sha": "t"}))
    return rows


def test_default_thresholds_and_per_run(toy_rows: list[dict[str, object]]) -> None:
    chosen = report.at_default_thresholds(toy_rows)
    assert {r["config"] for r in chosen} == set(CONFIGS)
    runs = report.per_run(chosen)
    assert len(runs) == len(CONFIGS) * 3  # one row per (config, toy scenario)


def test_tables(toy_rows: list[dict[str, object]]) -> None:
    runs = report.per_run(report.at_default_thresholds(toy_rows))
    main = report.main_table(runs)
    assert all(r["runs"] == 3 for r in main)
    paired = report.paired_table(runs)
    assert {p["comparison"] for p in paired} == {
        f"proposed - {c}" for c in CONFIGS if c != "proposed"
    }
    curves = report.tradeoff(toy_rows)
    assert len(curves["proposed"]) == 4 and len(curves["baseline_a"]) == 2
    sim = report.classification_in_sim(toy_rows)
    assert {r["reading"] for r in sim} == {"keyword", "nlp"}
    assert all(0.0 <= r["ece"] <= 1.0 for r in sim)


def test_build_writes_markdown_with_fixture_warning(
    toy_rows: list[dict[str, object]], tmp_path: Path
) -> None:
    path = report.build(toy_rows, tmp_path)
    text = path.read_text(encoding="utf-8")
    assert "smoke test, not results" in text
    assert "## Paired comparisons" in text
    with pytest.raises(ValueError):
        report.build([], tmp_path)


def test_cli(toy_rows: list[dict[str, object]], tmp_path: Path) -> None:
    import json

    rows_path = tmp_path / "rows.jsonl"
    rows_path.write_text("".join(json.dumps(r) + "\n" for r in toy_rows), encoding="utf-8")
    assert len(report.read_rows(rows_path)) == len(toy_rows)
    report.main([str(rows_path), "--out", str(tmp_path / "out")])
    assert (tmp_path / "out" / "results.md").exists()
