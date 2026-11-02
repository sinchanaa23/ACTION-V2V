"""Indian crash media dataset (D4, P1-08): class coverage and landmark phrases.

    python -m v2v.data.news_crash crashes.csv --text-column TEXT --category-column TYPE

Column names are arguments: take them from the dataset's data dictionary. News
text is not driver reporting, so it is only counted here. Article text is never
written out and never enters a classifier split. The JSON output holds totals
and place phrases only, and the phrases go to P4 for the landmark list.
"""

import argparse
import csv
import json
import re
import sys
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from v2v.nlp import keyword_baseline
from v2v.nlp.preprocess import normalise

MIN_PHRASE_COUNT = 2

_LANDMARK_WORDS = (
    "Bridge|Bypass|Chowk|Circle|Cross|Crossing|Expressway|Flyover|Highway|Junction|"
    "Road|Signal|Tunnel|Underpass"
)
_NAME_WORD = r"[A-Z][\w'-]*"
_PHRASE = re.compile(
    rf"\b(?:near|at|on|in|around|opposite|by)\s+(?:the\s+)?"
    rf"((?:{_NAME_WORD}\s+){{1,4}}(?:{_LANDMARK_WORDS}))\b"
)


def read_articles(
    path: Path, text_column: str, category_column: str | None = None
) -> Iterator[tuple[str, str | None]]:
    """Yield (text, category) pairs, stopping if a column name is wrong."""
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    with path.open(encoding="utf-8-sig", newline="") as src:
        reader = csv.DictReader(src)
        needed = {text_column, category_column} - {None}
        missing = needed - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"columns {sorted(map(str, missing))} not in {reader.fieldnames}")
        for row in reader:
            text = row[text_column].strip()
            if text:
                yield text, row[category_column].strip() if category_column else None


def location_phrases(texts: Iterable[str], min_count: int = MIN_PHRASE_COUNT) -> dict[str, int]:
    """Named places ending in a road word, most frequent first."""
    counts = Counter(" ".join(match.split()) for text in texts for match in _PHRASE.findall(text))
    return {
        phrase: count
        for phrase, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        if count >= min_count
    }


def class_coverage(texts: Iterable[str]) -> dict[str, int]:
    """Baseline A's class for each article. A rough guide to coverage, not a label."""
    return dict(
        Counter(
            keyword_baseline.classify(str(index), normalise(text)).hazard.value
            for index, text in enumerate(texts)
        )
    )


def analyse(articles: Iterable[tuple[str, str | None]]) -> dict[str, Any]:
    pairs = list(articles)
    texts = [text for text, _ in pairs]
    categories = Counter(category for _, category in pairs if category)
    return {
        "articles": len(pairs),
        "dataset_categories": dict(categories.most_common()),
        "keyword_class_coverage": class_coverage(texts),
        "landmark_phrases": location_phrases(texts),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--text-column", required=True)
    parser.add_argument("--category-column")
    parser.add_argument("--out", type=Path, default=Path("data/raw/landmark_phrases.json"))
    args = parser.parse_args()
    report = analyse(read_articles(args.csv_path, args.text_column, args.category_column))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(report["articles"], "articles analysed, wrote", args.out)


if __name__ == "__main__":
    main()
