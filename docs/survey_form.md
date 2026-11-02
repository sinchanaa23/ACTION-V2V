# Driver survey: Google Form spec (P1-01)

Build this form exactly. `src/v2v/data/survey.py` matches these question titles
and the two answers marked below, so renaming one makes the importer stop with
an error.

**Form settings:** "Collect email addresses" = Do not collect. "Limit to 1
response" = off (it forces a Google sign-in). No file uploads.

## Page 1: Consent

**Title:** Road hazards in your own words

Replace the `[...]` parts before sending.

> We are students at [college] working on a research project about warning
> drivers of road hazards using vehicle-to-vehicle communication. We are asking
> how drivers really describe hazards, in the way they would actually say them.
>
> **What you do:** write a few short messages. It takes about 5 minutes.
>
> **What we keep:** your written answers and your city. We do not collect your
> name, phone number or email address. Please do not include names, phone
> numbers or vehicle numbers in your answers. We remove any that appear.
>
> **How it is used:** to train and test a language model, and quoted in our
> project report. Your answers may be published in a research dataset.
>
> **Your choice:** taking part is voluntary and you must be 18 or older. You can
> stop at any time before submitting. Because answers are anonymous, we cannot
> find or remove them after you submit.
>
> **Questions:** [team contact email]. Supervisor: [name, email].

| Question title (exact) | Type | Required | Answers |
|---|---|---|---|
| `Consent` | Multiple choice | Yes | **`I agree`** / `I do not agree` |
| `Are you 18 or older?` | Multiple choice | Yes | **`Yes`** / `No` |

Send `I do not agree` and `No` to "Submit form" using "Go to section based on
answer".

## Page 2: Your reports

**Help text:** Write the way you would really text or say it. Hindi, English,
Hinglish or a mix are all fine. Spelling does not matter. Only describe things
you really saw. If you cannot think of one, leave it blank.

| Question title (exact) | Type | Required | Shown wording |
|---|---|---|---|
| `Which city do you mostly drive in?` | Short answer | No | as titled |
| `Hazard report 1` | Paragraph | No | Think of a real road hazard you met recently. What would you have told other drivers? |
| `Hazard report 2` | Paragraph | No | Another real hazard, a different kind if you can. |
| `Warning message` | Paragraph | No | What did you last send or say to warn a friend or family member about a road? |
| `Unsure report` | Paragraph | No | Have you seen something on the road and not been sure what it was? How would you describe it? |
| `All clear report` | Paragraph | No | Have you told someone a road problem had cleared, or that the road was fine after all? What did you say? |
| `False alarm report` | Paragraph | No | Think of something you said or heard about the road that turned out not to be a real hazard. How was it worded? |

Put the "Shown wording" in each question's description. The titles stay exact.

Do not add example answers to the form: they would prime people to copy the
wording, and the point is their own words.

## After it closes

Responses tab, then "Download responses (.csv)". Save it as
`data/raw/responses.csv`, which git ignores, then:

```bash
python -m v2v.data.survey data/raw/responses.csv --out data/raw/survey.jsonl
```

Delete `responses.csv` once imported. The export holds unscrubbed answers.
