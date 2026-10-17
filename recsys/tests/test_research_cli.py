import contextlib
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from test_training_artifacts import _identity, _inputs, _tiny_safetensors
from test_training_spec import valid_spec

from recsys.cli.main import build_parser, main
from recsys.research.training.artifacts import publish_training_artifact
from recsys.research.training.spec import parse_training_spec


class ResearchCliTests(unittest.TestCase):
    def test_commands_exist_without_eager_research_import(self) -> None:
        parser = build_parser()
        for command in (
            "prepare",
            "validate",
            "training-verify",
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

    def test_training_verify_reports_integrity_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root / "weights.safetensors"
            weights.write_bytes(_tiny_safetensors())
            model = root / "model"
            manifest = publish_training_artifact(
                model,
                spec=parse_training_spec(valid_spec()),
                inputs=_inputs(),
                weights_path=weights,
                supported_mal_ids=(1,),
                code_revision="a" * 40,
                framework_version="2.0.0",
            )
            identity = _identity(manifest.read_bytes())
            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                status = main(
                    ["research", "training-verify", str(model), "--expected-identity", identity]
                )
            self.assertEqual(status, 0)
            self.assertEqual(err.getvalue(), "")
            self.assertEqual(
                out.getvalue(),
                f"{identity} integrity_verified family=seed-embedding "
                "framework=pytorch weights_format=safetensors-v1\n",
            )
            (model / "weights.safetensors").write_bytes(b"tampered")
            with contextlib.redirect_stderr(err):
                status = main(
                    ["research", "training-verify", str(model), "--expected-identity", identity]
                )
            self.assertEqual(status, 1)
            self.assertIn("training_artifact_invalid", err.getvalue())
            self.assertNotIn(str(model), err.getvalue())

    def test_training_verify_subprocess_has_no_private_path_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root / "weights.safetensors"
            weights.write_bytes(_tiny_safetensors())
            model = root / "model"
            manifest = publish_training_artifact(
                model,
                spec=parse_training_spec(valid_spec()),
                inputs=_inputs(),
                weights_path=weights,
                supported_mal_ids=(1,),
                code_revision="a" * 40,
                framework_version="2.0.0",
            )
            identity = _identity(manifest.read_bytes())
            command = [
                sys.executable,
                "-m",
                "recsys",
                "research",
                "training-verify",
                str(model),
                "--expected-identity",
                identity,
            ]
            success = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertEqual(success.returncode, 0, success.stderr)
            self.assertIn("integrity_verified", success.stdout)
            self.assertNotIn(str(model), success.stdout + success.stderr)
            (model / "weights.safetensors").write_bytes(b"tampered")
            invalid = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertEqual(invalid.returncode, 1)
            self.assertIn("training_artifact_invalid", invalid.stderr)
            self.assertNotIn(str(model), invalid.stdout + invalid.stderr)
