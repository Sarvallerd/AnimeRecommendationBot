#!/usr/bin/env python3
"""Check that secrets, data, and local tool state stay out of Git."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import NoReturn

ROOT = Path(__file__).resolve().parents[1]
IGNORE_CASES = {
    ".env": True,
    ".env.local": True,
    ".env.example": False,
    ".venv/lib/file.py": True,
    ".tools/codex": True,
    ".worktrees/task/file.py": True,
    ".arb/acceptance/secrets/bot.env": True,
    ".codex/config.toml": True,
    "__pycache__/file.pyc": True,
    ".pytest_cache/state": True,
    ".ruff_cache/state": True,
    ".vscode/settings.json": True,
    "target/file": True,
    "bot/target/file": True,
    "data.csv": True,
    "archive.gz": True,
    "vectors.npz": True,
    "glove.6B.300d.txt": True,
    "Cargo.lock": False,
    "uv.lock": False,
}


def fail(message: str) -> NoReturn:
    raise ValueError(message)


def require_equal(actual: object, expected: object, context: str) -> None:
    if actual != expected:
        fail(f"{context}: expected {expected!r}, got {actual!r}")


def check_ignore() -> None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        fail(f"cannot locate Git root: {exc}")
    require_equal(Path(result.stdout.strip()).resolve(), ROOT, "Git root")
    for path, expected_ignored in IGNORE_CASES.items():
        result = subprocess.run(
            ["git", "check-ignore", "--no-index", "--quiet", path],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode not in (0, 1):
            fail(f"git check-ignore failed for {path}: {result.stderr.strip()}")
        require_equal(result.returncode == 0, expected_ignored, f"ignore behavior for {path}")


def main() -> int:
    try:
        check_ignore()
    except ValueError as exc:
        print(f"repository hygiene check failed: {exc}", file=sys.stderr)
        return 1
    print("repository hygiene check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
