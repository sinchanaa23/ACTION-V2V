# A.C.T.I.O.N-V2V

From informal Indian driver reports to reliable V2V hazard warnings: a confidence-aware
NLP and direction-sensitive geocasting framework, evaluated in SUMO on a real
OpenStreetMap road network. Python only. Real data only.

## Quick start

```bash
pip install -e ".[dev]"          # add ,data / ,nlp / ,sim / ,dashboard for your module
pytest
python -m v2v.pipeline --config proposed
```

Real road network (needs the `sim` extras; output in git-ignored `sim_out/`):

```bash
pip install -e ".[dev,sim]"
python -m v2v.sim.osm_network --corridor hebbal     # OSM snapshot 2026-10-01, left-hand traffic
python -m v2v.sim.gazetteer --corridor hebbal      # real OSM landmarks -> network edges
python -m v2v.sim.scenarios --corridor hebbal --seeds 0 1   # 3 conditions x 3 densities, SUMO-validated
python -m v2v.sim.traci_loop sim_out/scenarios/hebbal/noisy-medium-s0   # play one, log reports
V2V_NETWORK_TESTS=1 pytest tests/test_osm_network.py tests/test_gazetteer.py \
    tests/test_scenarios.py tests/test_traci_loop.py
```

Configs: `baseline_a`, `baseline_b`, `proposed`, `ablation_direction`,
`ablation_confidence`. Right now the pipeline runs on stubs and a toy two-way road
(`src/v2v/sim/toy_world.py`); each stub is replaced by its owner's PR.

## Pipeline

| Stage | Module | Owner |
|---|---|---|
| 1 Report input | `nlp/preprocess.py` | P1 |
| 2 NLP interpretation | `nlp/classifier.py`, `extract.py`, `calibrate.py` | P2 |
| 3 Confidence & context | `confidence/` | P3 |
| 4 Direction-sensitive geocast | `geocast/` | P4 |
| 5 Structured warning | `messages/warning.py` | P2 |
| Simulation & delivery | `sim/` | P4 |
| Orchestration & evaluation | `pipeline.py`, `eval/`, `experiments.py` | P3 |
| Data | `data/` | P1 |

Modules only exchange the types in `src/v2v/contracts.py`.

## Working rules

- Branch `p<n>/<pr-id>-<slug>`, small PRs, one review from your pair (P1↔P2, P3↔P4).
  Changes to `contracts.py` need all four approvals. Never merge your own PR.
- CI must be green: ruff, mypy, pytest, Python share ≥ 95%, real data sources only,
  frozen test split untouched, pipeline smoke run.
- Data: real sources only (`survey`, `reddit`, `youtube`, `news_dataset`, `x`, `fir`).
  No generated text, ever. `data/raw/` is git-ignored; never commit credentials.
- Full plan: [`docs/build-plan.pdf`](docs/build-plan.pdf). Rules for Claude: [`CLAUDE.md`](CLAUDE.md).

Map data © OpenStreetMap contributors (ODbL).
