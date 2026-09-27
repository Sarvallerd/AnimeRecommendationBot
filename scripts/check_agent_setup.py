#!/usr/bin/env python3
"""Check the repository's Codex roles and bootstrap file hygiene."""

from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / ".codex" / "config.toml"
EXPECTED_ROOT = {
    "model": "gpt-6-astra",
    "model_reasoning_effort": "ultra",
    "sandbox_mode": "workspace-write",
}
EXPECTED_AGENT_SETTINGS = {
    "enabled": True,
    "max_concurrent_threads_per_session": 3,
    "default_subagent_model": "gpt-6-sol",
    "default_subagent_reasoning_effort": "high",
}
ROLES = {
    "architect": ("gpt-6-astra", "ultra", "read-only"),
    "implementer": ("gpt-6-sol", "high", "workspace-write"),
    "reviewer": ("gpt-5.6-sol", "high", "read-only"),
}
COMMON_INSTRUCTIONS = ("AGENTS.md", "worktree", "branch", "baseline", "scope", "delegate")
ROLE_INSTRUCTIONS = {
    "architect": ("architecture", "Do not edit"),
    "implementer": ("test", "Commit only when the root explicitly authorizes"),
    "reviewer": ("exact commit SHA", "PASS", "CHANGES_REQUESTED", "severity", "Do not edit"),
}
IGNORE_CASES = {
    ".env": True,
    ".env.local": True,
    ".env.example": False,
    ".venv/lib/file.py": True,
    ".tools/codex": True,
    ".worktrees/task/file.py": True,
    "__pycache__/file.pyc": True,
    ".pytest_cache/state": True,
    ".ruff_cache/state": True,
    ".vscode/settings.json": True,
    "bot/target/file": True,
    "data.csv": True,
    "archive.gz": True,
    "vectors.npz": True,
    "glove.6B.300d.txt": True,
    "bot/Cargo.lock": False,
    ".codex/config.toml": False,
    ".codex/agents/architect.toml": False,
    ".codex/agents/implementer.toml": False,
    ".codex/agents/reviewer.toml": False,
    ".codex/agents/unlisted.toml": True,
    ".codex/secrets.toml": True,
}


def fail(message: str) -> None:
    raise ValueError(message)


def load_toml(path: Path) -> dict:
    try:
        with path.open("rb") as stream:
            return tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        fail(f"{path.relative_to(ROOT)}: {exc}")


def require_equal(actual: object, expected: object, context: str) -> None:
    if actual != expected:
        fail(f"{context}: expected {expected!r}, got {actual!r}")


def check_config() -> None:
    config = load_toml(CONFIG)
    require_equal(set(config), set(EXPECTED_ROOT) | {"agents"}, ".codex/config.toml keys")
    for key, expected in EXPECTED_ROOT.items():
        require_equal(config[key], expected, f".codex/config.toml {key}")

    agents = config["agents"]
    require_equal(set(agents), set(EXPECTED_AGENT_SETTINGS) | set(ROLES), "agent settings and roles")
    for key, expected in EXPECTED_AGENT_SETTINGS.items():
        require_equal(agents[key], expected, f"agents.{key}")

    for role, (model, effort, sandbox) in ROLES.items():
        entry = agents[role]
        require_equal(set(entry), {"description", "config_file"}, f"agents.{role} fields")
        if not isinstance(entry["description"], str) or not entry["description"].strip():
            fail(f"agents.{role}.description must be nonempty")
        relative = f"agents/{role}.toml"
        require_equal(entry["config_file"], relative, f"agents.{role}.config_file")
        role_config = load_toml(CONFIG.parent / relative)
        require_equal(
            set(role_config),
            {"name", "description", "model", "model_reasoning_effort", "sandbox_mode", "developer_instructions", "agents"},
            f"{relative} fields",
        )
        for key, expected in {
            "name": role,
            "model": model,
            "model_reasoning_effort": effort,
            "sandbox_mode": sandbox,
            "agents": {"enabled": False},
        }.items():
            require_equal(role_config[key], expected, f"{relative} {key}")
        if not isinstance(role_config["description"], str) or not role_config["description"].strip():
            fail(f"{relative} description must be nonempty")
        instructions = role_config["developer_instructions"]
        if not isinstance(instructions, str):
            fail(f"{relative} developer_instructions must be a string")
        for phrase in COMMON_INSTRUCTIONS + ROLE_INSTRUCTIONS[role]:
            if phrase.lower() not in instructions.lower():
                fail(f"{relative} developer_instructions missing {phrase!r}")


def check_guide() -> None:
    try:
        guide = (ROOT / "AGENTS.md").read_text(encoding="utf-8").lower()
    except OSError as exc:
        fail(f"AGENTS.md: {exc}")
    for phrase in (
        "rust", "russian", "private-chat", "python", "mal 2020", "glove", "json", "mal_id",
        "postgresql", "1–10", "0–5", "personalization", "worktree", "pull request",
        "notion", "squash merge", "reviewed sha", "ci", "three child agents",
    ):
        if phrase not in guide:
            fail(f"AGENTS.md missing {phrase!r}")


def check_ignore() -> None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"], cwd=ROOT, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        fail(f"cannot locate Git root: {exc}")
    require_equal(Path(result.stdout.strip()).resolve(), ROOT, "Git root")
    for path, expected_ignored in IGNORE_CASES.items():
        result = subprocess.run(
            ["git", "check-ignore", "--no-index", "--quiet", path], cwd=ROOT,
            capture_output=True, text=True, check=False,
        )
        if result.returncode not in (0, 1):
            fail(f"git check-ignore failed for {path}: {result.stderr.strip()}")
        require_equal(result.returncode == 0, expected_ignored, f"ignore behavior for {path}")


def main() -> int:
    try:
        check_config()
        check_guide()
        check_ignore()
    except ValueError as exc:
        print(f"agent setup check failed: {exc}", file=sys.stderr)
        return 1
    print("agent setup check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
