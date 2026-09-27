"""Check the installed command-line interfaces from another directory."""

import importlib.metadata
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class CliTests(unittest.TestCase):
    def run_cli(self, *args: str, console: bool = False) -> subprocess.CompletedProcess[str]:
        command = (
            [str(Path(sys.executable).with_name("recsys"))]
            if console
            else [sys.executable, "-m", "recsys"]
        )
        with tempfile.TemporaryDirectory() as directory:
            return subprocess.run(
                [*command, *args],
                cwd=directory,
                capture_output=True,
                text=True,
                check=False,
            )

    def test_help_and_no_arguments(self) -> None:
        for args in ((), ("--help",)):
            with self.subTest(args=args):
                result = self.run_cli(*args)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage: recsys", result.stdout)
                self.assertEqual(result.stderr, "")

    def test_version_matches_installed_distribution(self) -> None:
        version = importlib.metadata.version("anime-recommendation-recsys")
        for console in (False, True):
            with self.subTest(console=console):
                result = self.run_cli("--version", console=console)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout, f"recsys {version}\n")
                self.assertEqual(result.stderr, "")

    def test_obtain_offline_uses_installed_registry_outside_cwd(self) -> None:
        result = self.run_cli("obtain", "--offline", "--data-dir", "missing", console=True)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, "")
        self.assertIn("missing source", result.stderr)
        self.assertIn("anime.csv", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_unknown_arguments_and_commands_fail(self) -> None:
        for args in (("--unknown",), ("build",), ("--ver",)):
            with self.subTest(args=args):
                result = self.run_cli(*args)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(result.stdout, "")
                self.assertIn("error:", result.stderr)


if __name__ == "__main__":
    unittest.main()
