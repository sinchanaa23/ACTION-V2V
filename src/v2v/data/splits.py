"""Annotation import and frozen train/val/test splits. Owner: P1 (P1-06).

Once ``data/splits/test/`` is committed it is read-only; CI enforces this.
"""

from pathlib import Path


def freeze_splits(annotated_jsonl: Path, out_dir: Path, seed: int) -> None:
    """STUB(P1-06): stratified split by class and source."""
    raise NotImplementedError("P1-06")
