"""Hand-written code-path strings only. Never training data."""

from pathlib import Path

import pytest

from v2v.data import news_crash
from v2v.data.news_crash import analyse, class_coverage, location_phrases, read_articles


def _csv(path: Path) -> Path:
    target = path / "news.csv"
    target.write_text(
        'TEXT,TYPE,REPORTER\n"A crash near Hebbal Flyover, police said",head-on,x\n'
        ",rollover,y\nTwo hurt on the Outer Ring Road,hit,z\n",
        encoding="utf-8-sig",
    )
    return target


def test_read_articles_skips_blank_text(tmp_path: Path) -> None:
    rows = list(read_articles(_csv(tmp_path), "TEXT", "TYPE"))
    assert [category for _, category in rows] == ["head-on", "hit"]


def test_read_articles_without_category(tmp_path: Path) -> None:
    rows = list(read_articles(_csv(tmp_path), "TEXT"))
    assert {category for _, category in rows} == {None}


def test_read_articles_rejects_wrong_column(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="BODY"):
        list(read_articles(_csv(tmp_path), "BODY"))


def test_location_phrases_keeps_named_road_places() -> None:
    texts = [
        "Crash near Hebbal Flyover today",
        "Two hurt at the Hebbal Flyover",
        "A bus hit a wall on Outer Ring Road",
        "It happened in Bengaluru on Monday",
    ]
    assert location_phrases(texts) == {"Hebbal Flyover": 2}
    assert location_phrases(texts, min_count=1) == {
        "Hebbal Flyover": 2,
        "Outer Ring Road": 1,
    }


def test_class_coverage_counts_keyword_classes() -> None:
    assert class_coverage(["bus crash", "road is slippery", "market opened"]) == {
        "accident": 1,
        "slippery_road": 1,
        "non_actionable": 1,
    }


def test_analyse_holds_no_article_text(tmp_path: Path) -> None:
    report = analyse(read_articles(_csv(tmp_path), "TEXT", "TYPE"))
    assert report["articles"] == 2
    assert report["dataset_categories"] == {"head-on": 1, "hit": 1}
    assert "police said" not in str(report)


def test_module_does_not_import_split_writer() -> None:
    assert not hasattr(news_crash, "freeze_splits")
