"""Fail if Python is less than ``--min`` percent of the source code we write.

Counts non-blank lines of tracked files in programming languages. Config and
docs formats (YAML, TOML, JSON, Markdown) are not code and are not counted;
generated SUMO XML is git-ignored.

    python scripts/check_python_share.py --min 95
"""

import argparse
import subprocess
import sys
from pathlib import Path

NON_PYTHON_CODE = {
    ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".html", ".htm", ".css", ".scss", ".vue",
    ".java", ".kt", ".scala", ".c", ".h", ".cpp", ".hpp", ".cc", ".cs", ".go", ".rs", ".rb",
    ".php", ".r", ".jl", ".m", ".sh", ".bash", ".ps1", ".bat", ".cmd", ".ipynb", ".sql",
}  # fmt: skip


def count_lines(path: Path) -> int:
    try:
        return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    except (UnicodeDecodeError, FileNotFoundError):
        return 0


def python_share(files: list[Path]) -> tuple[int, int]:
    """Return (python_lines, non_python_code_lines)."""
    py = other = 0
    for path in files:
        suffix = path.suffix.lower()
        if suffix == ".py":
            py += count_lines(path)
        elif suffix in NON_PYTHON_CODE:
            other += count_lines(path)
    return py, other


def tracked_files() -> list[Path]:
    out = subprocess.run(["git", "ls-files"], capture_output=True, text=True, check=True).stdout
    return [Path(line) for line in out.splitlines() if line]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min", type=float, default=95.0)
    args = parser.parse_args()
    py, other = python_share(tracked_files())
    total = py + other
    share = 100.0 if total == 0 else 100.0 * py / total
    print(f"Python share: {share:.1f}% ({py} Python lines, {other} other code lines)")
    if share < args.min:
        print(f"FAIL: below {args.min}%", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
