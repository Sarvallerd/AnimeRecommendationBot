import contextlib
import io
import unittest

from recsys.cli.main import build_parser, main


class ResearchCliTests(unittest.TestCase):
    def test_commands_exist_without_eager_research_import(self) -> None:
        parser = build_parser()
        for command in (
            "prepare",
            "validate",
            "model-fit",
            "predict-validation",
            "gpu-preflight",
            "evaluate",
            "compare",
            "finalize",
            "test-queries",
        ):
            help_text = parser.format_help()
            self.assertIn("research", help_text)
            with contextlib.redirect_stdout(io.StringIO()) as output:
                with self.assertRaises(SystemExit) as stopped:
                    parser.parse_args(["research", command, "--help"])
            self.assertEqual(stopped.exception.code, 0)
            self.assertIn(command, output.getvalue())
        parsed = parser.parse_args(
            [
                "research",
                "model-fit",
                "--prepared-dir",
                "p",
                "--expected-prepared-identity",
                "sha256:" + "0" * 64,
                "--variant",
                "puresvd",
                "--code-revision",
                "a" * 40,
                "--output-dir",
                "o",
                "--backend",
                "gpu",
            ]
        )
        self.assertEqual(parsed.backend, "gpu")

    def test_safe_error_code(self) -> None:
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            status = main(
                [
                    "research",
                    "validate",
                    "/private/secret/input",
                    "--expected-identity",
                    "sha256:" + "0" * 64,
                ]
            )
        self.assertEqual(status, 1)
        self.assertIn("research_unsafe_path", err.getvalue())
        self.assertNotIn("/private/secret/input", err.getvalue())
