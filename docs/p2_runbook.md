# P2 runbook: from frozen splits to RQ1 results

Run these yourself, in order, once P1 has frozen `data/splits/{train,val,test}`.
Install with `pip install -e ".[dev,nlp]"`. Claude must not run the final step,
which reads the test split.

On Windows keep the repo in a short folder (for example `C:\v2v`); see
`docs/p3_runbook.md`.

## 1. Baseline B (TF-IDF + logistic regression), minutes on a laptop

```bash
python -m v2v.nlp.classifier train --train data/splits/train --val data/splits/val
```

This saves `models/tfidf_lr.pkl` and prints validation macro-F1 and ECE before
and after temperature scaling.

## 2. Transformers, on Colab or Kaggle with a GPU (needs PR #25 merged)

```bash
pip install -e ".[nlp]"
python -m v2v.nlp.train_transformer select --models muril indicbert xlmr
```

This saves the best model (by validation macro-F1) to `models/transformer/`, with
`selection.json` listing every candidate's scores. Copy the folder back to the repo
machine through shared Drive: models are git-ignored.

## 3. Location tagger, minutes

```bash
python -m v2v.nlp.span_tagger train --train data/splits/train --val data/splits/val
```

This saves `models/span_tagger.pkl` and prints rules vs tagger on validation. Use
the tagger only if it wins: `export V2V_SPAN_TAGGER=models/span_tagger.pkl` (Git Bash).

## 4. Confidence factors, on validation

```bash
python -m v2v.nlp.rq1 --split data/splits/val --tune-factors tfidf=models/tfidf_lr.pkl
```

Copy the printed `hedged_factor` and `no_location_factor` into
`HEDGED_FACTOR` / `NO_LOCATION_FACTOR` in `src/v2v/nlp/classifier.py` (PR).

## 5. Choose on validation

```bash
python -m v2v.nlp.rq1 --split data/splits/val --model tfidf=models/tfidf_lr.pkl --model transformer=models/transformer --tagger models/span_tagger.pkl --out docs/report/rq1_val
```

Pick the classifier and the location extractor from `docs/report/rq1_val/rq1_val.md`.
Point the pipeline at the chosen classifier with `V2V_CLASSIFIER` (a `.pkl` or the
transformer folder). Then **freeze**: no more changes after this point.

## 6. Final RQ1, once, on test (run it yourself)

```bash
python -m v2v.nlp.rq1 --split data/splits/test --allow-test --model tfidf=models/tfidf_lr.pkl --model transformer=models/transformer --tagger models/span_tagger.pkl --out docs/report/rq1_test
python -m v2v.data.error_analysis --gold data/splits/test/reports.jsonl --pred keyword=docs/report/rq1_test/test_keyword.jsonl --pred tfidf=docs/report/rq1_test/test_tfidf.jsonl --pred transformer=docs/report/rq1_test/test_transformer.jsonl --out docs/report/rq1_error_analysis.md
```

Commit the reports, then hand the frozen models to P3 for the experiment grid
(`docs/p3_runbook.md`). Report the numbers as measured, including any class the
models get wrong.
