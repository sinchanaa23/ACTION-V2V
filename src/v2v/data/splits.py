"""Annotation import and frozen train/val/test splits. Owner: P1 (P1-06).

Once ``data/splits/test/`` is committed it is read-only; CI enforces this.

    python -m v2v.data.splits data/annotated/gold.jsonl --out data/splits --seed 0
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from collections.abc import Collection, Sequence
from pathlib import Path

from v2v.contracts import REAL_SOURCES
from v2v.data.annotate import GoldRecord, read_gold

VAL_SHARE = 0.15
TEST_SHARE = 0.25
MIN_TEST_PER_CLASS = 40
SPLIT_SOURCES = REAL_SOURCES - {"news_dataset"}
"""News text is for analysis and landmarks only and never enters a split."""


def stratified_split(records: Sequence[GoldRecord], seed: int) -> dict[str, list[GoldRecord]]:
    """Split each (class, source) group by the same shares."""
    groups: defaultdict[tuple[str, str], list[GoldRecord]] = defaultdict(list)
    for record in sorted(records, key=lambda r: r.report_id):
        groups[(record.hazard.value, record.source)].append(record)
    rng = random.Random(seed)
    splits: dict[str, list[GoldRecord]] = {"train": [], "val": [], "test": []}
    for _, group in sorted(groups.items()):
        rng.shuffle(group)
        n_test = round(len(group) * TEST_SHARE)
        n_val = round(len(group) * VAL_SHARE)
        splits["test"].extend(group[:n_test])
        splits["val"].extend(group[n_test : n_test + n_val])
        splits["train"].extend(group[n_test + n_val :])
    return splits


def thin_test_classes(test: Sequence[GoldRecord]) -> dict[str, int]:
    """Classes under the plan's floor. These are reported, never padded."""
    counts = Counter(record.hazard.value for record in test)
    return {name: n for name, n in sorted(counts.items()) if n < MIN_TEST_PER_CLASS}


def freeze_splits(
    annotated_jsonl: Path,
    out_dir: Path,
    seed: int,
    allowed_sources: Collection[str] = SPLIT_SOURCES,
) -> None:
    """Stratified split by class and source. Refuses to replace a frozen test split."""
    records = read_gold(annotated_jsonl)
    if not records:
        raise ValueError(f"no records in {annotated_jsonl}")
    bad = sorted({r.source for r in records} - set(allowed_sources))
    if bad:
        raise ValueError(f"sources not allowed in a split: {bad}")
    if len({r.report_id for r in records}) != len(records):
        raise ValueError("duplicate report_id in the gold file")
    if (out_dir / "test").exists():
        raise FileExistsError(f"{out_dir / 'test'} exists: the test split is frozen")
    splits = stratified_split(records, seed)
    for name, items in splits.items():
        target = out_dir / name / "reports.jsonl"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            "".join(record.model_dump_json() + "\n" for record in items), encoding="utf-8"
        )
    (out_dir / "test" / "reports.jsonl").chmod(0o444)
    sizes = {name: len(items) for name, items in splits.items()}
    print(json.dumps({"sizes": sizes, "thin_test_classes": thin_test_classes(splits["test"])}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("gold", type=Path)
    parser.add_argument("--out", type=Path, default=Path("data/splits"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    freeze_splits(args.gold, args.out, args.seed)


if __name__ == "__main__":
    main()
