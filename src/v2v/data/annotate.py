"""Annotation workflow (P1-06): tasks, Label Studio import, kappa, adjudication.

    python -m v2v.data.annotate tasks data/raw/candidates.jsonl --annotators p1 p2 p3 p4
    python -m v2v.data.annotate report data/annotated/exports/*.json
    python -m v2v.data.annotate merge data/annotated/exports/*.json

Each person exports their project as JSON to ``<their name>.json``. The file
name identifies the annotator, because each laptop has its own Label Studio ids.
``report`` writes the disagreements. P1 fills the ``final_*`` columns, then
``merge`` writes the gold file. Text is copied through exactly as stored.
"""

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from collections.abc import Collection, Iterable, Sequence
from itertools import combinations
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from v2v.contracts import REAL_SOURCES, HazardType

OVERLAP = 0.2
DEFAULT_ADJUDICATION = Path("data/annotated/disagreements.csv")
DEFAULT_GOLD = Path("data/annotated/gold.jsonl")


class Span(BaseModel):
    model_config = ConfigDict(frozen=True)

    start: int
    end: int
    text: str


class GoldRecord(BaseModel):
    """One settled record. ``hazard`` is P2's LABEL_FIELD."""

    model_config = ConfigDict(frozen=True)

    report_id: str
    text: str
    source: str
    hazard: HazardType
    negated: bool
    hedged: bool
    locations: tuple[Span, ...] = ()


class Annotation(GoldRecord):
    """One person's labels for one report."""

    annotator: str


def assign_tasks(
    records: Sequence[dict[str, Any]],
    annotators: Sequence[str],
    seed: int = 0,
    overlap: float = OVERLAP,
) -> dict[str, list[dict[str, Any]]]:
    """Share records evenly, giving ``overlap`` of them to a second person."""
    if len(annotators) < 2:
        raise ValueError("double annotation needs at least two annotators")
    shuffled = sorted(records, key=lambda r: str(r["report_id"]))
    random.Random(seed).shuffle(shuffled)
    tasks: dict[str, list[dict[str, Any]]] = {name: [] for name in annotators}
    double_count = round(len(shuffled) * overlap)
    for index, record in enumerate(shuffled):
        task = {"data": {k: record[k] for k in ("report_id", "text", "source", "language")}}
        tasks[annotators[index % len(annotators)]].append(task)
        if index < double_count:
            tasks[annotators[(index + 1) % len(annotators)]].append(task)
    return tasks


def _parse_result(results: Iterable[dict[str, Any]]) -> tuple[str | None, set[str], list[Span]]:
    hazard = None
    flags: set[str] = set()
    spans: list[Span] = []
    for result in results:
        value = result["value"]
        if result["from_name"] == "hazard":
            hazard = value["choices"][0]
        elif result["from_name"] == "flags":
            flags = set(value["choices"])
        elif result["from_name"] == "span":
            spans.append(Span(start=value["start"], end=value["end"], text=value["text"]))
    return hazard, flags, sorted(spans, key=lambda s: s.start)


def parse_export(
    tasks: Iterable[dict[str, Any]],
    annotator: str,
    allowed_sources: Collection[str] = REAL_SOURCES,
) -> list[Annotation]:
    """Read a Label Studio JSON export. Cancelled and unlabelled items are skipped."""
    parsed: list[Annotation] = []
    for task in tasks:
        data = task["data"]
        if data["source"] not in allowed_sources:
            raise ValueError(f"{data['report_id']}: source={data['source']!r} is not allowed")
        for entry in task.get("annotations", []):
            if entry.get("was_cancelled"):
                continue
            hazard, flags, spans = _parse_result(entry["result"])
            if hazard is None:
                continue
            parsed.append(
                Annotation(
                    report_id=data["report_id"],
                    text=data["text"],
                    source=data["source"],
                    annotator=annotator,
                    hazard=HazardType(hazard),
                    negated="negated" in flags,
                    hedged="hedged" in flags,
                    locations=tuple(spans),
                )
            )
    return parsed


def load_exports(
    paths: Iterable[Path], allowed_sources: Collection[str] = REAL_SOURCES
) -> list[Annotation]:
    return [
        annotation
        for path in paths
        for annotation in parse_export(
            json.loads(path.read_text(encoding="utf-8")), path.stem, allowed_sources
        )
    ]


def cohen_kappa(first: Sequence[object], second: Sequence[object]) -> float | None:
    """Cohen's kappa, or None if chance agreement is 1 and kappa is undefined."""
    if len(first) != len(second) or not first:
        raise ValueError("need two equal-length, non-empty label lists")
    total = len(first)
    observed = sum(a == b for a, b in zip(first, second, strict=True)) / total
    first_counts, second_counts = Counter(first), Counter(second)
    expected = sum(first_counts[k] * second_counts[k] for k in first_counts) / total**2
    return None if expected == 1 else (observed - expected) / (1 - expected)


def _by_report(annotations: Iterable[Annotation]) -> dict[str, list[Annotation]]:
    grouped: defaultdict[str, list[Annotation]] = defaultdict(list)
    for annotation in annotations:
        grouped[annotation.report_id].append(annotation)
    return dict(grouped)


def kappa_report(annotations: Iterable[Annotation]) -> dict[str, Any]:
    """Kappa per field, pooled over every doubly annotated pair of labels."""
    pairs = [
        pair
        for group in _by_report(annotations).values()
        for pair in combinations(sorted(group, key=lambda a: a.annotator), 2)
    ]
    report: dict[str, Any] = {"n_double_annotated_pairs": len(pairs)}
    if pairs:
        for field in ("hazard", "negated", "hedged"):
            first = [getattr(a, field) for a, _ in pairs]
            second = [getattr(b, field) for _, b in pairs]
            report[f"kappa_{field}"] = cohen_kappa(first, second)
    return report


def _agreed(group: Sequence[Annotation]) -> bool:
    return (
        len({(a.hazard, a.negated, a.hedged) for a in group}) == 1
        and len({tuple((s.start, s.end) for s in a.locations) for a in group}) == 1
    )


def write_disagreements(annotations: Iterable[Annotation], path: Path) -> int:
    """Write the items P1 must settle. ``final_spans_from`` names an annotator."""
    groups = [g for g in _by_report(annotations).values() if not _agreed(g)]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(
            [
                "report_id",
                "text",
                "labels",
                "final_hazard",
                "final_negated",
                "final_hedged",
                "final_spans_from",
            ]
        )
        for group in groups:
            labels = {
                a.annotator: {
                    "hazard": a.hazard.value,
                    "negated": a.negated,
                    "hedged": a.hedged,
                    "locations": [s.text for s in a.locations],
                }
                for a in group
            }
            first = group[0]
            label_text = json.dumps(labels, ensure_ascii=False)
            writer.writerow([first.report_id, first.text, label_text, "", "", "", ""])
    return len(groups)


def _read_adjudication(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as src:
        return {row["report_id"]: row for row in csv.DictReader(src)}


def _settled(group: Sequence[Annotation], row: dict[str, str]) -> GoldRecord:
    span_source = next((a for a in group if a.annotator == row["final_spans_from"]), None)
    if span_source is None:
        raise ValueError(f"{group[0].report_id}: final_spans_from must name an annotator")
    return GoldRecord(
        report_id=group[0].report_id,
        text=group[0].text,
        source=group[0].source,
        hazard=HazardType(row["final_hazard"]),
        negated=row["final_negated"].strip().lower() == "true",
        hedged=row["final_hedged"].strip().lower() == "true",
        locations=span_source.locations,
    )


def merge(annotations: Iterable[Annotation], adjudication: Path) -> list[GoldRecord]:
    """Build gold records. Every disagreement must have been settled by P1."""
    settled = _read_adjudication(adjudication)
    gold: list[GoldRecord] = []
    unsettled: list[str] = []
    for report_id, group in sorted(_by_report(annotations).items()):
        if _agreed(group):
            gold.append(GoldRecord(**group[0].model_dump(exclude={"annotator"})))
        elif settled.get(report_id, {}).get("final_hazard", "").strip():
            gold.append(_settled(group, settled[report_id]))
        else:
            unsettled.append(report_id)
    if unsettled:
        raise ValueError(f"{len(unsettled)} unsettled disagreements, e.g. {unsettled[:3]}")
    return gold


def write_gold(gold: Iterable[GoldRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for record in gold:
            out.write(record.model_dump_json() + "\n")


def read_gold(path: Path) -> list[GoldRecord]:
    with path.open(encoding="utf-8") as src:
        return [GoldRecord.model_validate_json(line) for line in src if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    tasks = commands.add_parser("tasks")
    tasks.add_argument("candidates", type=Path)
    tasks.add_argument("--annotators", nargs="+", required=True)
    tasks.add_argument("--out", type=Path, default=Path("data/raw/tasks"))
    tasks.add_argument("--seed", type=int, default=0)
    for name in ("report", "merge"):
        sub = commands.add_parser(name)
        sub.add_argument("exports", nargs="+", type=Path)
        sub.add_argument("--adjudication", type=Path, default=DEFAULT_ADJUDICATION)
        sub.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    args = parser.parse_args()

    if args.command == "tasks":
        lines = args.candidates.read_text(encoding="utf-8").splitlines()
        records = [json.loads(line) for line in lines if line.strip()]
        args.out.mkdir(parents=True, exist_ok=True)
        for name, items in assign_tasks(records, args.annotators, args.seed).items():
            target = args.out / f"{name}.json"
            target.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
            print(target, len(items), "tasks")
        return
    annotations = load_exports(args.exports)
    if args.command == "report":
        print(json.dumps(kappa_report(annotations), indent=2))
        print(write_disagreements(annotations, args.adjudication), "disagreements to settle")
    else:
        gold = merge(annotations, args.adjudication)
        write_gold(gold, args.gold)
        print(len(gold), "gold records in", args.gold)


if __name__ == "__main__":
    main()
