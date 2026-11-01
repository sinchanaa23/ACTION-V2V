import importlib
import pkgutil

import pytest

import v2v
from tests.fixtures import ACCIDENT_HINGLISH, ACCIDENT_NO_LOCATION, NON_ACTIONABLE
from v2v import pipeline
from v2v.eval.metrics import summarise
from v2v.eval.runner import run_grid
from v2v.experiments import CONFIGS


def test_every_module_imports() -> None:
    for module in pkgutil.walk_packages(v2v.__path__, prefix="v2v."):
        importlib.import_module(module.name)


@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_every_config_runs_end_to_end(name: str) -> None:
    result = pipeline.run(CONFIGS[name], ACCIDENT_HINGLISH, seed=0)
    assert result.config_name == name
    assert result.report.source == "fixture"
    assert result.warning is not None
    assert result.warning.status == "unverified"
    assert "unverified" in result.warning.text
    assert len(result.deliveries) == len(result.warning.recipients)
    assert result.relevant_vehicle_ids  # vehicles behind the hazard exist


def test_non_actionable_report_sends_nothing() -> None:
    result = pipeline.run(CONFIGS["proposed"], NON_ACTIONABLE, seed=0)
    assert result.assessed.status == "suppressed"
    assert result.warning is None
    assert summarise(result)["missed_hazard"] == 1.0


def test_unstated_location_uses_flagged_fallback() -> None:
    result = pipeline.run(CONFIGS["proposed"], ACCIDENT_NO_LOCATION, seed=0)
    assert result.interpretation.location_text is None
    assert result.assessed.location is not None
    assert result.assessed.location.approx


def test_stated_location_is_resolved_not_approximate() -> None:
    result = pipeline.run(CONFIGS["proposed"], ACCIDENT_HINGLISH, seed=0)
    assert result.interpretation.location_text == "bridge"
    assert result.assessed.location is not None
    assert not result.assessed.location.approx


def test_runner_rows() -> None:
    rows = run_grid([CONFIGS["baseline_a"], CONFIGS["proposed"]], [0, 1], ACCIDENT_HINGLISH)
    assert len(rows) == 4
    assert {r["config"] for r in rows} == {"baseline_a", "proposed"}


def test_cli_prints_summary(capsys: pytest.CaptureFixture[str]) -> None:
    pipeline.main(["--config", "baseline_a", "--seed", "0"])
    assert '"config": "baseline_a"' in capsys.readouterr().out
