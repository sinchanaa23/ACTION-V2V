"""P3-05 runner on the toy world (no SUMO). Fixture text only."""

import json
from pathlib import Path

import pytest

from v2v.confidence import policy
from v2v.contracts import HazardType, ResolvedLocation
from v2v.data.annotate import GoldRecord
from v2v.eval import runner
from v2v.experiments import CONFIGS

GRID = policy.threshold_grid((0.5, 0.9), (0.5, 0.9))


def _loc() -> ResolvedLocation:
    return ResolvedLocation(edge_id="e", pos_m=0, x=0, y=0, approx=False)


def _rec(rid: str, hazard: HazardType, negated: bool = False) -> GoldRecord:
    return GoldRecord(
        report_id=rid, text="x", source="fixture", hazard=hazard, negated=negated, hedged=False
    )


def test_pool_matches_slot_needs() -> None:
    pool = runner.ReportPool(
        [
            _rec("a", HazardType.ACCIDENT),
            _rec("b", HazardType.ACCIDENT, negated=True),
            _rec("c", HazardType.SLIPPERY, negated=True),
            _rec("d", HazardType.NON_ACTIONABLE),
        ]
    )
    assert [r.report_id for r in pool.candidates("hazard", HazardType.ACCIDENT)] == ["a"]
    assert [r.report_id for r in pool.candidates("negated_hazard", HazardType.SLIPPERY)] == ["c"]
    assert {r.report_id for r in pool.candidates("negated_hazard", HazardType.STOPPED)} == {
        "b",
        "c",
    }
    assert [r.report_id for r in pool.candidates("non_actionable", HazardType.ACCIDENT)] == ["d"]
    assert pool.pick("hazard", HazardType.SLIPPERY, "k", set()) is None
    assert pool.pick("hazard", HazardType.ACCIDENT, "k", {"a"}) is None
    with pytest.raises(ValueError):
        pool.candidates("nonsense", HazardType.ACCIDENT)
    with pytest.raises(ValueError):
        runner.ReportPool([])


def test_pick_is_reproducible() -> None:
    pool = runner.ReportPool([_rec(str(i), HazardType.ACCIDENT) for i in range(20)])
    first = pool.pick("hazard", HazardType.ACCIDENT, "s1-slot_0-0", set())
    assert first == pool.pick("hazard", HazardType.ACCIDENT, "s1-slot_0-0", set())


def test_toy_run_scores_every_config_and_threshold() -> None:
    pool = runner.ReportPool(runner.DEMO_POOL)
    rows = runner.run_toy(0, pool, list(CONFIGS.values()), GRID, {"git_sha": "x"})
    by_config: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_config.setdefault(str(row["config"]), []).append(row)
    assert set(by_config) == set(CONFIGS)
    assert len(by_config["proposed"]) == len(GRID)
    assert len(by_config["baseline_a"]) == 2  # theta_i only
    for row in rows:
        assert row["text_source"] == "fixture" and row["git_sha"] == "x"
        assert row["event_active"] is True and row["false_alert"] == 0.0
        if row["warned"]:
            assert row["messages_sent"] and row["warning_text"]
    proposed = {(r["theta_i"], r["theta_e"]): r for r in by_config["proposed"]}
    assert proposed[(0.5, 0.5)]["status"] == "unverified"  # one report: evidence 0.35
    strict = proposed[(0.9, 0.9)]  # theta_i decides: P2's confidence (location penalty)
    expected = "unverified" if strict["interp_conf"] >= 0.9 else "suppressed"  # type: ignore[operator]
    assert strict["status"] == expected
    broadcast = by_config["baseline_a"][0]
    directional = by_config["proposed"][0]
    assert broadcast["messages_sent"] >= directional["messages_sent"]  # type: ignore[operator]


def test_cli_toy_writes_rows_and_meta(tmp_path: Path) -> None:
    runner.main(["--toy", "--seeds", "0", "1", "--out", str(tmp_path), "--name", "t"])
    lines = (tmp_path / "t.jsonl").read_text(encoding="utf-8").splitlines()
    meta = json.loads((tmp_path / "t.meta.json").read_text(encoding="utf-8"))
    assert len(lines) == meta["n_rows"] > 0
    assert meta["reports"].startswith("fixture")
    assert {json.loads(line)["seed"] for line in lines} == {0, 1}


def test_cli_needs_scenarios_or_toy(tmp_path: Path) -> None:
    pool_file = tmp_path / "pool.jsonl"
    pool_file.write_text(runner.DEMO_POOL[0].model_dump_json() + "\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        runner.main(["--reports", str(pool_file), "--out", str(tmp_path)])


def test_model_hash(tmp_path: Path) -> None:
    assert runner.model_hash(tmp_path / "absent") == "none"
    (tmp_path / "m.pkl").write_bytes(b"abc")
    h = runner.model_hash(tmp_path)
    assert len(h) == 16
    (tmp_path / "m.pkl").write_bytes(b"abd")
    assert runner.model_hash(tmp_path) != h


def test_git_sha_is_a_string() -> None:
    assert isinstance(runner.git_sha(), str)


def test_run_info_event_window() -> None:
    info = runner.RunInfo("s", "noisy", "low", 0, _loc(), True, 300, 900)
    assert not info.event_active(100) and info.event_active(500) and not info.event_active(1000)
    never = runner.RunInfo("s", "misleading", "low", 0, _loc(), False, 300, 1500)
    assert not never.event_active(500)
