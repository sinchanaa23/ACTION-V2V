"""Fail if a PR changes an already-frozen test split.

The first PR that adds data/splits/test/ (P1-06) is allowed; after that, any
change to files that exist there on the base branch fails.

    python scripts/check_test_split.py origin/main
"""

import subprocess
import sys

TEST_DIR = "data/splits/test/"


def git(*args: str) -> list[str]:
    out = subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout
    return [line for line in out.splitlines() if line]


def main() -> int:
    base = sys.argv[1] if len(sys.argv) > 1 else "origin/main"
    frozen = set(git("ls-tree", "-r", "--name-only", base, "--", TEST_DIR))
    if not frozen:
        print("No frozen test split on the base branch yet")
        return 0
    changed = git("diff", "--name-only", f"{base}...HEAD", "--", TEST_DIR)
    if changed:
        for path in changed:
            print(f"FAIL: frozen test split changed: {path}", file=sys.stderr)
        return 1
    print("Frozen test split untouched")
    return 0


if __name__ == "__main__":
    sys.exit(main())
