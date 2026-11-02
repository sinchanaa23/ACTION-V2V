# P3 runbook: from frozen splits to RQ2/RQ3 tables

Run these in your own terminal. Install with `pip install -e ".[dev,sim,nlp]"`.
The full grid runs overnight (ALL-02); the results are committed by 09:00.

Claude must not run the steps that read `data/splits/test/`, per `CLAUDE.md`.

> **Windows:** if `import pyproj` or `mypy` fails with "The filename or
> extension is too long", the virtualenv's path is too long for Windows to
> load its DLLs. Create the venv at a short path such as `C:\v2v-venv`.

## 1. Prerequisites

- P1 has frozen the splits (`data/splits/{train,val,test}/reports.jsonl`).
- P2 has trained and frozen the classifier (`models/tfidf_lr.pkl`, and the
  transformer if it wins on validation). The runner hashes everything under
  `models/` into every row, so freeze the models before the grid runs.

## 2. Network and scenarios (P4's code)

```bash
python -m v2v.sim.osm_network --corridor hebbal
python -m v2v.sim.scenarios --corridor hebbal --seeds 0 1 2 3 4 5 6 7 8 9
```

This writes 3 conditions x 3 densities x 10 seeds = 90 scenarios under
`sim_out/scenarios/hebbal/`. Validation plays each one in SUMO, so this step
takes a while.

## 3. Smoke test first (fixture text, a few minutes)

```bash
python -m v2v.eval.runner sim_out/scenarios/hebbal/noisy-low-s0 --fixtures --name smoke
python -m v2v.eval.report experiments/results/smoke.jsonl --out sim_out/smoke_report
```

The report says "smoke test, not results" at the top. Check that every slot
fired and that the contradiction slot is suppressed.

## 4. The full grid (overnight)

```bash
python -m v2v.eval.runner sim_out/scenarios/hebbal/*-s* --reports data/splits/test/reports.jsonl --name grid-v1
```

- Every config and the whole threshold grid (7 x 6 pairs for the confidence
  configs, 7 for the baselines) are scored from one SUMO run per scenario.
- A scenario whose slots can't all be filled with real text of the right class
  is skipped and logged. Report how many were skipped; never pad.
- The results file is `experiments/results/grid-v1.jsonl` plus
  `grid-v1.meta.json` (git SHA, model hash, grid, Python version).

## 5. Tables and figures

```bash
python -m v2v.eval.report experiments/results/grid-v1.jsonl --out docs/report/results
```

Writes `docs/report/results/results.md` and `tradeoff.png`:

- main table at theta_i = 0.6, theta_e = 0.75, with 95% bootstrap CIs across
  scenario runs;
- proposed vs every other config, paired by scenario run (difference, CI,
  permutation p-value);
- false-alert vs missed-hazard trade-off per threshold pair, Pareto points marked;
- macro-F1 and ECE of each reading over the reports the scenarios used.

Commit the results file, the meta file and the report. Tune nothing after
looking at them.

## 6. What to write up

- RQ2 (confidence): `proposed` vs `ablation_confidence`, and the trade-off curves.
- RQ3 (direction): `proposed` vs `ablation_direction`, irrelevant-recipient
  rate and transmissions.
- Both baselines for context. Report negative or mixed results as measured.
- Delivery numbers are from P4's simplified delivery model: label them as
  simulation, never as radio measurements.
- The V2X field note (`docs/v2x_fields.md`) supports the discussion: DENM has
  one `informationQuality`, while the framework keeps two confidences apart.
