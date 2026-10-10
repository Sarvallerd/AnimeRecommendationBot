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

    def test_obtain_snapshot_defaults_and_explicit_directory(self) -> None:
        for console in (False, True):
            with self.subTest(console=console):
                legacy = self.run_cli("obtain", "--offline", console=console)
                self.assertEqual(legacy.returncode, 1)
                self.assertIn("data/raw/anime.csv", legacy.stderr)
                fresh = self.run_cli("obtain", "--snapshot", "neelagiri-2025-v1",
                                     "--offline", console=console)
                self.assertEqual(fresh.returncode, 1)
                self.assertIn("data/raw/neelagiri-2025-v1/details.csv", fresh.stderr)
                explicit = self.run_cli("obtain", "--snapshot", "neelagiri-2025-v1",
                                        "--data-dir", "chosen", "--offline", console=console)
                self.assertEqual(explicit.returncode, 1)
                self.assertIn("chosen/details.csv", explicit.stderr)
                self.assertNotIn("chosen/neelagiri-2025-v1", explicit.stderr)

    def test_invalid_snapshot_fails_before_data_directory_creation(self) -> None:
        for console in (False, True):
            with self.subTest(console=console), tempfile.TemporaryDirectory() as directory:
                command = ([str(Path(sys.executable).with_name("recsys"))] if console
                           else [sys.executable, "-m", "recsys"])
                result = subprocess.run([*command, "obtain", "--snapshot", "latest"],
                                        cwd=directory, capture_output=True, text=True, check=False)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(result.stdout, "")
                self.assertIn("invalid choice", result.stderr)
                self.assertFalse((Path(directory) / "data").exists())

    def test_isolated_installed_package_contains_new_registry(self) -> None:
        code = ("import json; from recsys.sources import load_registry; "
                "r,h=load_registry('neelagiri-2025-v1'); "
                "print(json.dumps([len(r['sources']),r['provenance']['dataset']['snapshot_id'],len(h)]))")
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, "-I", "-c", code], cwd=directory,
                                    capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), [2, "neelagiri-2025-v1", 64])
        self.assertEqual(result.stderr, "")

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

    def test_comparison_and_summary_help_outside_cwd(self) -> None:
        for command, option in (("compare", "--repetitions"),
                                ("summarize", "--assessment")):
            result = self.run_cli("quality", command, "--help", console=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(option, result.stdout)
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_cli("quality", "compare", "--bundle-dir", "missing",
                                  "--normalization-report", "missing", "--build-report", "missing",
                                  "--anime-csv", "missing", "--synopsis-csv", "missing",
                                  "--glove", "missing", "--lockfile", "missing",
                                  "--code-revision", "a" * 40, "--output-dir", directory,
                                  "--repetitions", "0", console=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn("repetitions must be 1..10", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

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

    def test_export_validate_help_and_errors_outside_checkout(self) -> None:
        for command in ("export", "validate"):
            help_result = self.run_cli(command, "--help", console=True)
            self.assertEqual(help_result.returncode, 0, help_result.stderr)
            self.assertIn("usage: recsys", help_result.stdout)
            self.assertEqual(help_result.stderr, "")
        fixture = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "bundle"
        for console in (False, True):
            result = self.run_cli("validate", str(fixture), console=console)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("records=8", result.stdout)
        invalid = self.run_cli("validate", "missing", console=True)
        self.assertEqual(invalid.returncode, 1)
        self.assertNotIn("Traceback", invalid.stderr)
        invalid = self.run_cli("export", "--catalog", "missing", "--neighbors", "missing",
                               "--build-report", "missing", "--normalization-report", "missing",
                               "--output-dir", "out", console=True)
        self.assertEqual(invalid.returncode, 1)
        self.assertNotIn("Traceback", invalid.stderr)
        self.assertIn("missing", invalid.stderr)

    def test_unknown_arguments_and_commands_fail(self) -> None:
        for args in (("--unknown",), ("build",), ("--ver",)):
            with self.subTest(args=args):
                result = self.run_cli(*args)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(result.stdout, "")
                self.assertIn("error:", result.stderr)


if __name__ == "__main__":
    unittest.main()
