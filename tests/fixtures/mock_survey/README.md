# Mock survey responses (generated, not real data)

`test_data.csv` is a 600-row mock export of the driver survey form, supplied by a
team member. The text is template-generated, not written by real drivers.

Use it only to check that code paths run at survey volume:

- Load it with `source="fixture"`, never `source="survey"`.
- Never train, calibrate, annotate or report accuracy on it.
- Never copy it into `data/` or mix it into the splits.

Do not run `python -m v2v.data.survey` on this file: that importer labels every
answer `source="survey"` and writes to `data/raw/survey.jsonl`. Read the CSV
yourself and pass each answer to the pipeline as a `source="fixture"` record.
Write any output under `sim_out/`, which git ignores.
