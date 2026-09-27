"""Check the installed command-line interfaces from another directory."""

import importlib.metadata
import json
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

    def test_quality_help_and_input_error_outside_cwd(self) -> None:
        help_result = self.run_cli("quality", "baseline", "--help", console=True)
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn("--catalog", help_result.stdout)
        self.assertIn("--output-dir", help_result.stdout)
        result = self.run_cli("quality", "baseline", "--catalog", "missing.json",
                              "--output-dir", "out", console=True)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, "")
        self.assertIn("recsys quality baseline:", result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        with tempfile.TemporaryDirectory() as directory:
            catalog = Path(directory) / "catalog.json"
            catalog.write_text("{}\n")
            loaded = self.run_cli("quality", "baseline", "--catalog", str(catalog),
                                  "--output-dir", str(Path(directory) / "out"), console=True)
        self.assertEqual(loaded.returncode, 1)
        self.assertIn("catalog SHA256 mismatch", loaded.stderr)
        self.assertNotIn("Traceback", loaded.stderr)

    def test_build_help_and_input_error_outside_cwd(self) -> None:
        help_result = self.run_cli("build", "--help", console=True)
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn("--normalization-report", help_result.stdout)
        result = self.run_cli("build", "--catalog", "missing", "--normalization-report",
                              "missing-report", "--glove", "missing-glove", "--output-dir", "out",
                              console=True)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, "")
        self.assertIn("recsys build:", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_build_oversized_integer_score_reports_clean_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = {"schema_version": 1, "anime": {"1": {
                "title": "Example", "aliases": [], "genres": [], "score": 10 ** 1000,
                "year": None, "type": None, "episodes": None, "synopsis": None}}}
            path = root / "catalog.json"
            path.write_text(json.dumps(catalog, ensure_ascii=False, sort_keys=True,
                                       separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")
            result = self.run_cli("build", "--catalog", str(path),
                                  "--normalization-report", str(root / "missing-report.json"),
                                  "--glove", str(root / "missing-glove.txt"),
                                  "--output-dir", str(root / "out"), console=True)
            self.assertEqual(result.returncode, 1)
            self.assertEqual(result.stdout, "")
            self.assertIn("recsys build: invalid score: 1", result.stderr)
            self.assertNotIn("Traceback", result.stderr)
            self.assertFalse((root / "out").exists())

    def test_unknown_arguments_and_commands_fail(self) -> None:
        for args in (("--unknown",), ("build",), ("--ver",)):
            with self.subTest(args=args):
                result = self.run_cli(*args)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(result.stdout, "")
                self.assertIn("error:", result.stderr)


if __name__ == "__main__":
    unittest.main()
