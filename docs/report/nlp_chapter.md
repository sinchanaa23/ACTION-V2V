# Chapter draft: Interpreting informal driver reports (P2, RQ1)

> Draft by P2. The method sections describe the code as built. Sections marked
> **[RESULTS]** wait for the frozen splits; no number is filled in until it is
> measured on real data (`docs/p2_runbook.md`).

## RQ1

How accurately can informal English and Hinglish driver reports be turned into
a structured hazard reading (hazard class, negation, hedging, location) with a
probability that can be trusted?

## Task

Input: one report as written or said by a driver, for example *"Bhai, aage
accident hua hai, bridge ke paas. Slow down!"*. Output: a `HazardInterpretation`
with

- a hazard class from seven: `accident`, `road_obstruction`, `slippery_road`,
  `stopped_vehicle`, `speeding_concern`, `non_actionable`, `ambiguous`;
- `p_class`, the calibrated probability of that class;
- `negated` ("accident nahi hua", "road clear ho gaya") and `hedged`
  ("shayad", "I couldn't tell");
- `location_text`, a span copied from the report, or `None`. A location is never
  guessed.

The two non-hazard classes and negated readings never produce a warning
(`confidence.policy.is_actionable`).

## Pipeline

1. **Normalisation (P1).** Unicode and case folding, stretched letters
   ("bhaiiii"), and a reviewed list of Romanised Hindi respellings ("nhi" →
   "nahi"). Words are respelled, never translated.
2. **Classification.** Three models of increasing capacity:
   - *Baseline A (P1):* keyword rules, the first match wins. It has no probability.
   - *Baseline B (P2-01):* TF-IDF over word 1–2-grams and character 2–5-grams
     (character n-grams absorb Hinglish spelling variation), with class-balanced
     logistic regression.
   - *Transformer (P2-05):* MuRIL, IndicBERT v2 and XLM-R base, each fine-tuned for
     at most 3 epochs with class-weighted cross-entropy; the best epoch and the
     best model are chosen by validation macro-F1.
3. **Calibration (P2-04).** Temperature scaling fitted on validation by negative
   log-likelihood (golden-section search on log T). It changes how confident a
   prediction is, never which class wins. Reported as expected calibration
   error (ECE, 15 equal-width bins) with reliability diagrams.
4. **Negation and hedging (P2-02).** Cue lists for English and Romanised Hindi.
   - English negators scope forwards ("no accident") and Hindi negators
     backwards ("accident nahi hua"), within 4 tokens and never across a clause.
   - Phrases that contain a negator but don't deny a hazard are ignored: "no
     entry", "pata nahi", "nahi to", "wasn't sure".
   - "Cleared" phrasings ("has moved", "hata diya", "working again") are a
     separate kind of negation, so the confidence stage can treat them as
     contradicting an earlier report.
5. **Location (P2-03, P2-06).**
   - *Rules:* place words (bridge, flyover, signal, petrol pump, …), with up to 3
     name tokens to their left, plus proximity phrases ("esteem ke paas", "near
     esteem mall"). Generic words like "road" need a name. Names never include
     hazard words, hedges or verbs, and never cross punctuation.
   - *Tagger:* BIO token tagging learned from the annotated spans (logistic
     regression over word, affix, place-word and ±2-context features).
   - Both copy spans from the text. The world model (P4's landmark list) turns a
     span into a point on the road network; a span it can't resolve falls back
     to the reporter's position, flagged "location approximate".
6. **Interpretation confidence.** `p_class` lowered by a factor for hedged
   reports and for reports with no stated location. Both factors are tuned on
   validation by minimising the ECE of interpretation confidence against
   correctness. The warning policy compares this, not `p_class`, with θi.
7. **Warning text (P2-07).** Wording comes from templates only, with an action
   and an explicit status ("Status: unverified report", "confirmed (3 reports)",
   "Location approximate"). No report text reaches other drivers except the
   capped place phrase.

## Design choices worth defending

- **Missing details are never invented.** Location stays `None` rather than
  guessed. This is measured as "no-location kept None".
- **Calibration is evaluated, not assumed.** A class probability is only useful
  to the warning policy if it means what it says.
- **Interpretation confidence and event evidence are kept apart.** P2 answers
  "how sure are we of what the driver *said*?". P3 answers "how sure are we that
  it *happened*?". The V2X field study (`docs/v2x_fields.md`) shows that ETSI
  DENM has one `informationQuality` field for both.
- **Real data only.** No generated or paraphrased text is used for training or
  evaluation (`CLAUDE.md`). The progress-review demo on the mock survey is kept
  out of the results (`demo/README.md`).

## Evaluation protocol

- Splits stratified by class and source: 60/15/25 (P1-06). The test split is
  frozen and read once, after models and factors are frozen.
- Classes: macro-F1 and per-class precision, recall and F1, with a 95% Wilson
  interval on accuracy (P1's shared scoring), for Baseline A, Baseline B and the
  transformer.
- Calibration: ECE and reliability diagrams.
- Extractors: precision, recall and F1 for negation and hedging against the gold
  flags; exact-span and token-overlap F1 for location, plus no-location kept None.
- Hard subsets (P1-09): negated, hedged, no location, code-mixed vs English.

## Results **[RESULTS]**

- Table: classifiers on test (macro-F1, accuracy and CI, ECE).
- Table: per-class scores for the best model.
- Figure: reliability diagrams before and after temperature scaling.
- Table: negation, hedging and location (rules vs tagger).
- Table: hard subsets (from `docs/report/rq1_error_analysis.md`).
- Tuned confidence factors and the ECE they give on validation.

## Limitations (to confirm against the results)

- The cue lists are hand-built. Recall depends on how varied real phrasings
  are, and the error analysis will show which cues are missing.
- Romanised Hindi only. Devanagari is tagged as `hi-en` but has no dedicated
  cues, and other Indian languages are out of scope.
- The tagger is a linear model with local context. Long or unusual place
  names may be cut short.
- Speech is out of the core build (stretch goal S1, Whisper). Recognition errors
  would add noise that these text-only results don't measure.
