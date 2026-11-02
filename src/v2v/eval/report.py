"""Tables and figures from runner results. Owner: P3 (P3-07).

    python -m v2v.eval.report experiments/results/<name>.jsonl --out docs/report/results

One command regenerates every RQ2/RQ3 table and figure from a results file:

- main table: each config's mean metric per (condition, density) at the
  default thresholds, with a 95% bootstrap CI across scenario runs;
- paired comparisons: the proposed config against every other config, paired
  by scenario run, with mean difference, CI and permutation p-value;
- threshold trade-off: mean false-alert vs missed-hazard rate per threshold
  pair, with Pareto points, as a table and a figure;
- RQ1 view inside the simulation: macro-F1 and ECE of each reading over the
  distinct reports the scenarios used.

The unit of analysis is one scenario run: per-report rows are averaged per
(config, scenario, thresholds) first, so a scenario with five reports does not
weigh five times as much as one with a single report.
"""

import argparse
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from v2v.contracts import HazardType
from v2v.eval import metrics, stats
from v2v.experiments import PROPOSED
from v2v.nlp import calibrate

DEFAULT_THETA_I = PROPOSED.theta_i
DEFAULT_THETA_E = PROPOSED.theta_e
MAIN_METRICS: tuple[str, ...] = (
    "false_alert",
    "missed_hazard",
    "relevant_delivery_rate",
    "irrelevant_recipient_rate",
    "transmissions",
    "mean_latency_ms",
)
N_RESAMPLES = 2_000


def read_rows(path: Path) -> list[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def at_default_thresholds(
    rows: Iterable[Mapping[str, Any]],
    theta_i: float = DEFAULT_THETA_I,
    theta_e: float = DEFAULT_THETA_E,
) -> list[Mapping[str, Any]]:
    """Rows at the configs' own thresholds (baselines have no theta_e)."""
    return [r for r in rows if r["theta_i"] == theta_i and r["theta_e"] in (None, theta_e)]


def per_run(
    rows: Iterable[Mapping[str, Any]], extra_keys: Sequence[str] = ()
) -> list[dict[str, Any]]:
    """Average report rows into one row per (config, scenario run, extra keys)."""
    keys = ["config", "scenario_id", "condition", "density", "seed", *extra_keys]
    return metrics.aggregate(rows, keys, MAIN_METRICS)


def main_table(run_rows: Sequence[Mapping[str, Any]], seed: int = 0) -> list[dict[str, Any]]:
    groups: defaultdict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for r in run_rows:
        groups[(r["config"], r["condition"], r["density"])].append(r)
    table = []
    for (config, condition, density), members in sorted(groups.items()):
        row: dict[str, Any] = {
            "config": config,
            "condition": condition,
            "density": density,
            "runs": len(members),
        }
        for m in MAIN_METRICS:
            values = [float(x[m]) for x in members if x.get(m) is not None]
            row[m] = stats.mean_with_ci(values, N_RESAMPLES, seed) if values else None
        table.append(row)
    return table


def paired_table(
    run_rows: Sequence[Mapping[str, Any]], reference: str = PROPOSED.name, seed: int = 0
) -> list[dict[str, Any]]:
    others = sorted({r["config"] for r in run_rows} - {reference})
    out = []
    for other in others:
        for m in MAIN_METRICS:
            a, b = stats.pair_rows(run_rows, m, reference, other, ("scenario_id", "seed"))
            if not a:
                continue
            out.append(
                {
                    "comparison": f"{reference} - {other}",
                    "metric": m,
                    **stats.paired_comparison(a, b, N_RESAMPLES, seed),
                }
            )
    return out


def tradeoff(rows: Iterable[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Trade-off curve per config, from per-run means at each threshold pair."""
    run_rows = per_run(rows, ["theta_i", "theta_e"])
    by_config: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in run_rows:
        by_config[r["config"]].append(r)
    return {
        config: stats.tradeoff_curve(members, by=("theta_i", "theta_e"))
        for config, members in sorted(by_config.items())
    }


def classification_in_sim(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Macro-F1 and ECE per reading (keyword / NLP), over distinct reports.

    Configs that share a reading see identical predictions, so each
    (reading, report) counts once."""
    seen: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for r in rows:
        reading = "keyword" if r["config"] == "baseline_a" else "nlp"
        seen.setdefault((reading, r["scenario_id"], r["slot_id"]), r)
    out = []
    for reading in sorted({k[0] for k in seen}):
        members = [r for k, r in seen.items() if k[0] == reading]
        gold = [HazardType(r["gold_hazard"]) for r in members]
        pred = [HazardType(r["pred_hazard"]) for r in members]
        scores = metrics.classification(gold, pred)
        ece = calibrate.expected_calibration_error(
            [float(r["p_class"]) for r in members],
            [g == p for g, p in zip(gold, pred, strict=True)],
        )
        out.append(
            {"reading": reading, "n": len(members), "macro_f1": scores["macro_f1"], "ece": ece}
        )
    return out


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, dict):
        return f"{value['mean']:.3f} [{value['ci_low']:.3f}, {value['ci_high']:.3f}]"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def markdown(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(_fmt(r.get(c)) for c in columns) + " |" for r in rows]
    return "\n".join(lines)


def build(rows: Sequence[Mapping[str, Any]], out_dir: Path, seed: int = 0) -> Path:
    """Write results.md (+ tradeoff.png when matplotlib is installed)."""
    if not rows:
        raise ValueError("no result rows")
    out_dir.mkdir(parents=True, exist_ok=True)
    sources = sorted({r["text_source"] for r in rows})
    run_rows = per_run(at_default_thresholds(rows))
    curves = tradeoff(rows)
    sections = [
        "# Experiment results",
        "",
        f"Rows: {len(rows)}. Report text sources: {', '.join(sources)}.",
        f"Git SHA: {rows[0].get('git_sha', '?')}. Model hash: {rows[0].get('model_hash', '?')}.",
    ]
    if "fixture" in sources:
        sections += [
            "",
            "> **Fixture text: a smoke test, not results. Do not report these numbers.**",
        ]
    sections += [
        "",
        f"## Main table (theta_i = {DEFAULT_THETA_I}, theta_e = {DEFAULT_THETA_E})",
        "",
        "Mean per scenario run with 95% bootstrap CI.",
        "",
        markdown(
            main_table(run_rows, seed), ["config", "condition", "density", "runs", *MAIN_METRICS]
        ),
        "",
        f"## Paired comparisons against {PROPOSED.name}",
        "",
        markdown(
            paired_table(run_rows, seed=seed),
            ["comparison", "metric", "mean_diff", "ci_low", "ci_high", "p_value", "n_pairs"],
        ),
        "",
        "## Threshold trade-off (false alert vs missed hazard)",
        "",
    ]
    for config, curve in curves.items():
        sections += [
            f"### {config}",
            "",
            markdown(curve, ["theta_i", "theta_e", "false_alert", "missed_hazard", "pareto", "n"]),
            "",
        ]
    sections += [
        "## Classification inside the simulation",
        "",
        "Over the distinct reports the scenarios used, not the whole test split; RQ1 proper",
        "is P1's error analysis on the full split.",
        "",
        markdown(classification_in_sim(rows), ["reading", "n", "macro_f1", "ece"]),
        "",
    ]
    path = out_dir / "results.md"
    path.write_text("\n".join(sections), encoding="utf-8")
    try:
        stats.plot_tradeoff(curves, out_dir / "tradeoff.png")
    except ImportError:
        pass
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--out", type=Path, default=Path("docs/report/results"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    print("wrote", build(read_rows(args.results), args.out, args.seed))


if __name__ == "__main__":
    main()
