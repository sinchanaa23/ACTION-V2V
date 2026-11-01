# Annotation guide v1 (P1-03)

Status: draft for the team to try on 30 items before Day 3. Freeze it on Day 2.
After that, changes go through P1 and are logged at the bottom.

Example sentences here explain the rules. Never copy them into any dataset.

## What you are doing

For each report, decide:

1. **Hazard class**: exactly one of 7.
2. **Flags**: `negated`, `hedged`, both or neither.
3. **Location span**: highlight the place words, if there are any.

Label only what the text says. Do not use outside knowledge, and do not guess
the location. No location is a valid answer and must stay empty.

## Decision order

Go through these in order and stop at the first that fits.

1. Does the text report a current situation on a road? If not:
   `non_actionable`.
2. Does it say the hazard is absent, over or cleared? `non_actionable`, and tick
   `negated`.
3. Is it clearly reporting a hazard, but you cannot tell which type?
   `ambiguous`.
4. Otherwise pick the matching hazard class. If unsure it happened, keep the
   class and tick `hedged`.

## The 7 classes

| Class | Use when the report says | Not this class if |
|---|---|---|
| `accident` | A collision, crash, or a vehicle hit, overturned or a person run over, on the road now or just now | Only a jam with no stated crash |
| `road_obstruction` | Something blocks or narrows the road: fallen tree, debris, construction, barricade, deep pothole, animals, a crowd or procession | The blockage is a stopped vehicle (`stopped_vehicle`) |
| `slippery_road` | Loss of grip or water on the road: oil, mud, waterlogging, flooding, gravel, wet surface | Debris or a tree is the problem rather than the surface (`road_obstruction`). A flooded, closed road stays `slippery_road` |
| `stopped_vehicle` | A broken-down, parked or stationary vehicle in a lane | The vehicle stopped because of a crash (`accident`) |
| `speeding_concern` | Dangerous driving by others: speeding, racing, wrong-side driving, rash overtaking | A general complaint with no current situation |
| `non_actionable` | Not a hazard report: opinions, jokes, past stories, general complaints about traffic or government, questions, or a negated hazard | A real current hazard is described, however rudely |
| `ambiguous` | A real warning whose hazard type cannot be determined, for example "be careful, something is wrong ahead" | The type is clear and only certainty is low (keep the class, tick `hedged`) |

### Tie-breaks

- **Two hazards in one report:** take the one a following driver must react to
  first. If equal, use the table order above.
- **Plain traffic jam:** `non_actionable` unless a cause in the classes is
  stated.
- **Old events:** a story about a past incident is `non_actionable`. "Just now"
  or "still there" counts as current.
- **Questions:** "is there an accident near X?" is `non_actionable`, because the
  writer is asking rather than reporting.
- **Sarcasm:** label the literal situation if one is real. Otherwise
  `non_actionable`.

## Negation

Tick `negated` when the text says the hazard is not present, for example the
plan's "accident nahi hua". This includes "cleared", "road clear now" and
"khatam ho gaya".

- A negated report is `non_actionable`. The flag records why, and the hard-subset
  analysis in RQ1 depends on it.
- Scope matters. Negating something else does not count: if the report says
  there was a crash and nobody was hurt, the class is `accident` with no flag.

## Hedging

Tick `hedged` when the writer signals doubt about whether it happened: shayad,
lagta hai, maybe, I think, not sure, someone said, sunne mein aaya.

- Hedged reports keep their hazard class. The hedge lowers confidence later, and
  that is P2's job, not the class's.
- `hedged` is about doubt. `ambiguous` is about not knowing the type. A report
  can be both if it is a doubtful, type-unknown warning.
- Politeness or advice ("maybe slow down") is not a hedge on the event.

## Location spans

Highlight the shortest words that name the place, including landmark words such
as bridge, flyover, signal, underpass or junction.

- Include: "bridge ke paas" highlights `bridge`. Keep multi-word names whole, as
  in `Outer Ring Road`.
- Exclude: "aage", "ahead", "here", "this road", "nearby". They give no place,
  so they stay unlabelled.
- Two separate places: label both.
- Never add a location the text does not contain, even if the post title or city
  makes it obvious.

## Practical rules

- Bad spelling, slang and script mixing are not reasons to mark `ambiguous`. Ask
  what the writer means.
- Tags such as `[PLATE]` and `[USER]` are scrubbed PII. Ignore them. If you see
  real PII left in the text, skip the item and tell P1.
- Do not fix or rewrite text. Do not add items. All text stays as written.
- If you are truly stuck, pick your best answer and add a Label Studio comment.
  P1 reviews these when settling disagreements.

## Double annotation

Twenty percent of items go to a second annotator, who does not see the first
label. P1 computes Cohen's κ and settles disagreements. Do not discuss items
with each other until κ is computed.

## Label Studio setup

```bash
pip install label-studio
label-studio start
```

Create a project, paste `docs/label_studio_config.xml` into Labeling Setup, and
import tasks as JSON with a `text` field and the `report_id`.

## Change log

- v1 draft: first version.
