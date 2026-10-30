# A.C.T.I.O.N-V2V: rules for Claude

## Project
Informal English/Hinglish driver reports -> NLP hazard interpretation -> confidence
assessment -> direction-sensitive geocast -> structured V2V warning, evaluated in
SUMO on a real OpenStreetMap network. The build plan is `docs/build-plan.pdf`.

## Hard rules
- Python 3.11 only. Do not add code in any other language. The dashboard is Streamlit.
- All shared data shapes live in `src/v2v/contracts.py`. Do not edit it unless the task
  says "contract-change"; such PRs need all four members' approval.
- Stay inside the module named in the task (ownership below).
- NEVER generate, paraphrase, translate or augment dataset text. All data is real and
  human-written, with `source` set to its real platform.
- Never read or write `data/splits/test/`.
- Strings in `tests/fixtures/` test code paths only: use `source="fixture"`, never train
  or report results on them.
- Unknown hazard details stay `None`. Never invent a location.
- Never claim SAE J2735 / ETSI DENM compliance in code or docs.

## Ownership
- P1 Data & Language: `src/v2v/data/`, `nlp/preprocess.py`, `nlp/keyword_baseline.py`
- P2 NLP Modelling: `nlp/classifier.py`, `nlp/extract.py`, `nlp/calibrate.py`,
  `nlp/train_transformer.py`, `messages/`
- P3 Confidence & Evaluation: `contracts.py`, `confidence/`, `pipeline.py`, `eval/`,
  `experiments.py`
- P4 Simulation & Geocast: `sim/`, `geocast/`, `dashboard/`

## Stubs
Unfinished code is marked `STUB(<PR id>)` and returns the simplest valid output so the
pipeline always runs. Replace a stub behind the same signature; list what's left with
`git grep "STUB("`.

## Before you finish
```
ruff check . && ruff format --check .
mypy
pytest
python -m v2v.pipeline --config proposed
```
Every new function gets a pytest test. Branch names: `p<n>/<pr-id>-<slug>`.
Never merge your own PR.
