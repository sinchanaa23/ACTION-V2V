"""P3-06: the runner on the real Hebbal network through SUMO.

Opt-in like P4's network tests: needs the [sim] extras and network access
(the corridor is downloaded from Overpass). Fixture text only: these check
code paths, never results.
"""

import importlib.util
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("traci") is None or not os.environ.get("V2V_NETWORK_TESTS"),
    reason="needs the [sim] extras and network access; set V2V_NETWORK_TESTS=1",
)


@pytest.fixture(scope="module")
def noisy_spec(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from v2v.sim.osm_network import CORRIDORS
    from v2v.sim.scenarios import generate

    out = tmp_path_factory.mktemp("sim") / "networks"
    (spec,) = generate(CORRIDORS["hebbal"], ["noisy"], ["low"], [0], out)
    return Path(spec.sumocfg).parent / "spec.json"


def test_smoke_first_60_simulated_seconds(noisy_spec: Path) -> None:
    from v2v.confidence import policy
    from v2v.eval import runner
    from v2v.experiments import PROPOSED

    logs: list[str] = []
    rows = runner.run_sumo_scenario(
        noisy_spec,
        runner.ReportPool(runner.DEMO_POOL, allow_reuse=True),
        [PROPOSED],
        policy.threshold_grid((0.6,), (0.75,)),
        {"git_sha": "test"},
        log=logs.append,
        max_time_s=60.0,
    )
    assert rows == []  # no report falls due before the event starts at 300 s
    assert any("never fell due" in line for line in logs)


def test_noisy_scenario_scores_every_report(noisy_spec: Path) -> None:
    from v2v.confidence import policy
    from v2v.eval import runner
    from v2v.experiments import CONFIGS

    rows = runner.run_sumo_scenario(
        noisy_spec,
        runner.ReportPool(runner.DEMO_POOL, allow_reuse=True),
        list(CONFIGS.values()),
        policy.threshold_grid((0.6,), (0.75,)),
        {"git_sha": "test"},
    )
    roles = {r["role"] for r in rows}
    assert roles == {"original", "duplicate", "stale", "contradiction"}
    proposed = {r["slot_id"]: r for r in rows if r["config"] == "proposed"}
    by_role = {r["role"]: r for r in proposed.values()}
    assert by_role["original"]["event_active"] is True
    assert by_role["stale"]["event_active"] is False  # sent after the event cleared
    assert by_role["contradiction"]["status"] == "suppressed"
    assert by_role["contradiction"]["missed_hazard"] == 0.0
    # Duplicates from other drivers raise the evidence of the same cluster.
    dups = sorted(
        (r for r in proposed.values() if r["role"] in {"original", "duplicate"}),
        key=lambda r: r["t_s"],
    )
    assert [r["n_reports"] for r in dups] == sorted(r["n_reports"] for r in dups)
    assert dups[-1]["event_evidence"] > dups[0]["event_evidence"]
    for row in rows:
        assert row["text_source"] == "fixture"
        if row["warned"]:
            assert row["messages_sent"] > 0
