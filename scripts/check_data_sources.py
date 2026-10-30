"""Fail if any record under data/ has a missing or non-real ``source``.

Scans every ``.jsonl`` file under the data directory. Each record must carry a
``source`` from v2v.contracts.REAL_SOURCES; ``fixture`` is not allowed here.

    python scripts/check_data_sources.py [data_dir]
"""

import json
import sys
from pathlib import Path

from v2v.contracts import REAL_SOURCES


def find_problems(data_dir: Path) -> list[str]:
    problems: list[str] = []
    for path in sorted(data_dir.rglob("*.jsonl")):
        if "raw" in path.relative_to(data_dir).parts:
            continue  # raw dumps are git-ignored and checked at import time
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                source = json.loads(line).get("source")
            except json.JSONDecodeError:
                problems.append(f"{path}:{lineno}: not valid JSON")
                continue
            if source not in REAL_SOURCES:
                problems.append(f"{path}:{lineno}: source={source!r} is not a real source")
    return problems


def main() -> int:
    data_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "data")
    if not data_dir.exists():
        print(f"{data_dir}/ does not exist yet; nothing to check")
        return 0
    problems = find_problems(data_dir)
    for problem in problems:
        print(problem, file=sys.stderr)
    print(f"Data source check: {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
