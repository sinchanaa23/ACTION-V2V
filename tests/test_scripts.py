from pathlib import Path

from scripts.check_data_sources import find_problems
from scripts.check_python_share import python_share


def test_data_check_rejects_fixture_and_missing_source(tmp_path: Path) -> None:
    split = tmp_path / "splits" / "train"
    split.mkdir(parents=True)
    (split / "a.jsonl").write_text(
        '{"text": "a", "source": "reddit"}\n{"text": "b", "source": "fixture"}\n{"text": "c"}\n',
        encoding="utf-8",
    )
    problems = find_problems(tmp_path)
    assert len(problems) == 2
    assert all("a.jsonl" in p for p in problems)


def test_data_check_skips_raw(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "dump.jsonl").write_text('{"text": "x"}\n', encoding="utf-8")
    assert find_problems(tmp_path) == []


def test_python_share_ignores_config_counts_code(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x = 1\n\ny = 2\n", encoding="utf-8")
    (tmp_path / "ci.yml").write_text("on: push\n", encoding="utf-8")
    (tmp_path / "map.js").write_text("let a = 1;\n", encoding="utf-8")
    assert python_share(sorted(tmp_path.iterdir())) == (2, 1)
