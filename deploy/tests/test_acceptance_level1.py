"""Fail-closed checks for the local, non-Telegram acceptance runner."""

import importlib.util
import sys
import unittest
from pathlib import Path
from typing import cast
from unittest.mock import patch

from deploy.tests.acceptance_types import Level1Module

MODULE = Path(__file__).with_name("level1-acceptance.py")
SPEC = importlib.util.spec_from_file_location("level1_acceptance", MODULE)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"cannot load level 1 runner from {MODULE}")
loaded = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(loaded)
for export in ("Runner", "require_cases", "require_markers"):
    if not callable(getattr(loaded, export, None)):
        raise ImportError(f"level 1 runner missing {export}")
level1 = cast("Level1Module", loaded)


class Level1RunnerTests(unittest.TestCase):
    def test_required_tests_must_all_pass_once(self) -> None:
        required = {"level1_one", "level1_two"}
        level1.require_cases("test level1_one ... ok\ntest level1_two ... ok\n", required)
        for text in (
            "",
            "test level1_one ... ok\n",
            "test level1_one ... ignored\ntest level1_two ... ok\n",
            "test level1_one ... ok\ntest level1_one ... ok\ntest level1_two ... ok\n",
            "test level1_one ... FAILED\ntest level1_two ... ok\n",
        ):
            with self.subTest(text=text), self.assertRaises(RuntimeError):
                level1.require_cases(text, required)

    def test_compose_markers_must_be_complete_and_final(self) -> None:
        good = "\n".join(
            sorted(level1.COMPOSE_MARKERS - {"LEVEL1_COMPOSE_CLEANUP_PASS"})
            + ["LEVEL1_COMPOSE_CLEANUP_PASS"]
        )
        level1.require_markers(good)
        for bad in (
            good.replace("LEVEL1_COMPOSE_BUILD_PASS\n", ""),
            good + "\nLEVEL1_COMPOSE_CLEANUP_PASS",
            good + "\nunchecked tail",
        ):
            with self.subTest(bad=bad), self.assertRaises(RuntimeError):
                level1.require_markers(bad)

    def test_child_failure_and_timeout_are_fatal(self) -> None:
        runner = level1.Runner("dialogue")
        with self.assertRaises(RuntimeError):
            runner.command("negative-exit", [sys.executable, "-c", "raise SystemExit(7)"], 5)
        with self.assertRaises(Exception):
            runner.command(
                "negative-timeout", [sys.executable, "-c", "import time; time.sleep(30)"], 0.1
            )

    def test_cleanup_failure_prevents_pass(self) -> None:
        runner = level1.Runner("dialogue")
        runner.started_pg = True
        with patch.object(runner, "compose", side_effect=RuntimeError("synthetic cleanup failure")):
            runner.cleanup()
        self.assertFalse(runner.cleanup_ok)
        self.assertEqual(runner.cases["cleanup"]["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
