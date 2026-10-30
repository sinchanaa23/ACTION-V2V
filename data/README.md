# Data (owner: P1)

Real, human-written data only. Every record carries `source` in
`survey | reddit | youtube | news_dataset | x | fir` (checked in CI).

```
data/
├── raw/          git-ignored scraped dumps (PII scrubbed before anything leaves here)
├── annotated/    Label Studio exports, JSONL
└── splits/
    ├── train/
    ├── val/
    └── test/     read-only once committed (P1-06); CI fails on any change
```

Release only post/comment IDs for Reddit and YouTube, never raw scraped text.
