"""Experiment runner: configs x seeds. Owner: P3 (P3-05)."""

from collections.abc import Iterable

from v2v import pipeline
from v2v.contracts import ExperimentConfig
from v2v.eval import metrics


def run_grid(
    configs: Iterable[ExperimentConfig], seeds: Iterable[int], report_text: str
) -> list[dict[str, object]]:
    """One metrics row per (config, seed).

    STUB(P3-05): single report on the toy world; the real runner covers the full
    grid on the real network and records git SHA and model hash per row.
    """
    rows: list[dict[str, object]] = []
    seed_list = list(seeds)
    for config in configs:
        for seed in seed_list:
            result = pipeline.run(config, report_text, seed=seed)
            rows.append({"config": config.name, "seed": seed, **metrics.summarise(result)})
    return rows
