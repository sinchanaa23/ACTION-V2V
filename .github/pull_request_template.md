## What
<!-- PR id from the build plan (e.g. P4-05) and one line on what it does -->

## Checklist
- [ ] Only touches my module's folders (or is labelled `contract-change`)
- [ ] New functions have pytest tests; `ruff`, `mypy`, `pytest` pass locally
- [ ] No generated, paraphrased or augmented dataset text; every record has a real `source`
- [ ] Nothing under `data/splits/test/` changed (unless this is the first freeze, P1-06)
- [ ] No credentials or raw scraped dumps committed
- [ ] Replaced `STUB(...)` markers for this PR, if any
