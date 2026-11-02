# P1 runbook: from raw data to RQ1

Run these yourself in your own terminal, once all P1 PRs are merged. Install with
`pip install -e ".[dev,data]"`. Data stays out of git until the freeze step.

Claude must not run the steps that touch `data/splits/test/`, per `CLAUDE.md`.

## 1. Collect

```bash
python -m v2v.data.youtube --out data/raw/youtube.jsonl
python -m v2v.data.reddit --out data/raw/reddit.jsonl
python -m v2v.data.survey data/raw/responses.csv --out data/raw/survey.jsonl
```

Reddit needs an approved app. Skip it if you have none: never scrape around it.

## 2. Filter and prepare

```bash
python -m v2v.data.filter data/raw/survey.jsonl data/raw/youtube.jsonl data/raw/reddit.jsonl
python -m v2v.nlp.preprocess data/raw/candidates.jsonl
```

Review `data/raw/variant_candidates.json`. Copy only true respellings into
`src/v2v/nlp/variants.json`, and never map to or from an English word.

## 3. Annotate

```bash
python -m v2v.data.annotate tasks data/raw/candidates.jsonl --annotators p1 p2 p3 p4
```

Each person imports `data/raw/tasks/<name>.json` into Label Studio, labels it, and
exports JSON to `data/annotated/exports/<name>.json`.

```bash
python -m v2v.data.annotate report data/annotated/exports/p1.json data/annotated/exports/p2.json data/annotated/exports/p3.json data/annotated/exports/p4.json
```

Fill the `final_*` columns in `data/annotated/disagreements.csv`, then:

```bash
python -m v2v.data.annotate merge data/annotated/exports/p1.json data/annotated/exports/p2.json data/annotated/exports/p3.json data/annotated/exports/p4.json
```

## 4. Freeze

This runs once. A second run refuses to replace the test split.

```bash
python -m v2v.data.splits data/annotated/gold.jsonl --out data/splits --seed 0
python -m v2v.data.datasheet data/annotated/gold.jsonl data/annotated/exports/p1.json data/annotated/exports/p2.json data/annotated/exports/p3.json data/annotated/exports/p4.json
```

The first command prints any class under 40 test items. Report those with their
wide intervals. Do not pad them.

## 5. Baseline A

```bash
python -m v2v.nlp.keyword_baseline build --train data/splits/train
python -m v2v.nlp.keyword_baseline score --split data/splits/val --out val_keyword.json --predictions val_keyword.jsonl
python -m v2v.nlp.keyword_baseline score --split data/splits/test --out test_keyword.json --predictions test_keyword.jsonl
```

Commit `src/v2v/nlp/keywords.json`. Tune nothing after looking at test scores.

## 6. Crash dataset

Check the column names in the dataset's data dictionary first.

```bash
python -m v2v.data.news_crash crashes.csv --text-column ARTICLE_TEXT --category-column CRASH_TYPE
```

Send `data/raw/landmark_phrases.json` to P4 for the landmark list.

## 7. RQ1

Once P2 has model predictions as JSONL with `report_id` and `hazard`:

```bash
python -m v2v.data.error_analysis --gold data/splits/test/reports.jsonl --pred keyword=test_keyword.jsonl --pred transformer=test_transformer.jsonl --out docs/rq1_error_analysis.md
```

Report the results as measured, including any negative findings.
