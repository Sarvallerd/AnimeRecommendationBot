import contextlib
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from test_training_artifacts import _identity, _inputs, _tiny_safetensors
from test_training_spec import valid_spec

from recsys.cli.main import build_parser, main
from recsys.research.training.artifacts import publish_training_artifact
from recsys.research.training.spec import parse_training_spec


class ResearchCliTests(unittest.TestCase):
    def test_campaign_profile_flags_are_closed_and_forwarded(self) -> None:
        parser = build_parser()
        arguments = [
            "research",
            "campaign-run",
            "--prepared-dir",
            "prepared",
            "--expected-prepared-identity",
            "sha256:" + "0" * 64,
            "--code-revision",
            "a" * 40,
            "--glove",
            "glove",
            "--output-dir",
            "output",
        ]
        legacy = parser.parse_args(arguments)
        self.assertEqual(legacy.campaign, "ml02-cpu-v1")
        self.assertIsNone(legacy.native_train_device)
        self.assertIsNone(legacy.physical_storage_root)
        native = parser.parse_args(
            [
                *arguments,
                "--campaign",
                "ml02-native-v3",
                "--native-train-device",
                "cpu",
                "--physical-storage-root",
                "/mnt/c",
            ]
        )
        self.assertEqual((native.campaign, native.native_train_device), ("ml02-native-v3", "cpu"))
        with patch(
            "recsys.research.campaign.runner.run_campaign", return_value=Path("result")
        ) as run:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    main(
                        [
                            *arguments,
                            "--campaign",
                            "ml02-native-v3",
                            "--native-train-device",
                            "cpu",
                            "--physical-storage-root",
                            "/mnt/c",
                        ]
                    ),
                    0,
                )
            self.assertEqual(run.call_args.kwargs["campaign_name"], "ml02-native-v3")
            self.assertEqual(run.call_args.kwargs["native_train_device"], "cpu")
            self.assertEqual(run.call_args.kwargs["physical_storage_root"], Path("/mnt/c"))
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as rejected:
                parser.parse_args([*arguments, "--campaign", "arbitrary"])
        self.assertEqual(rejected.exception.code, 2)

    def test_commands_exist_without_eager_research_import(self) -> None:
        parser = build_parser()
        for command in (
            "prepare",
            "validate",
            "training-verify",
            "training-fit",
            "training-predict",
            "model-fit",
            "predict-validation",
            "gpu-preflight",
            "campaign-run",
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

    def test_training_commands_expose_only_bounded_options(self) -> None:
        parser = build_parser()
        fit = parser.parse_args(
            [
                "research",
                "training-fit",
                "--candidate",
                "item-seed-mean-bpr-v1",
                "--prepared-dir",
                "prepared",
                "--expected-prepared-identity",
                "sha256:" + "0" * 64,
                "--output-dir",
                "new-run",
                "--code-revision",
                "a" * 40,
            ]
        )
        self.assertEqual((fit.train_device, fit.inference_device), ("cuda:0", "cpu"))
        predict = parser.parse_args(
            [
                "research",
                "training-predict",
                "--prepared-dir",
                "prepared",
                "--expected-prepared-identity",
                "sha256:" + "0" * 64,
                "--model-dir",
                "model",
                "--expected-model-identity",
                "sha256:" + "1" * 64,
                "--output-dir",
                "prediction",
                "--code-revision",
                "a" * 40,
                "--seed-mal-id",
                "1",
                "--seed-mal-id",
                "2",
            ]
        )
        self.assertEqual((predict.seed_mal_id, predict.limit), ([1, 2], 5))
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as rejected:
                parser.parse_args(
                    [
                        "research",
                        "training-fit",
                        "--candidate",
                        "item-seed-mean-bpr-v1",
                        "--prepared-dir",
                        "prepared",
                        "--expected-prepared-identity",
                        "sha256:" + "0" * 64,
                        "--output-dir",
                        "new-run",
                        "--code-revision",
                        "a" * 40,
                        "--optimizer",
                        "SGD",
                    ]
                )
        self.assertEqual(rejected.exception.code, 2)

    def test_retired_gpu_routes_do_not_touch_private_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            private = Path(directory) / "private-input"
            output = Path(directory) / "unpublished"
            common = [
                "--prepared-dir",
                str(private),
                "--expected-prepared-identity",
                "sha256:" + "0" * 64,
                "--output-dir",
                str(output),
            ]
            routes = (
                [
                    "model-fit",
                    *common,
                    "--variant",
                    "puresvd",
                    "--code-revision",
                    "a" * 40,
                    "--backend",
                    "gpu",
                ],
                [
                    "predict-validation",
                    *common,
                    "--model-dir",
                    str(private),
                    "--backend",
                    "gpu",
                    "--measurement-output",
                    str(output / "measurement.json"),
                ],
                [
                    "campaign-run",
                    *common,
                    "--code-revision",
                    "a" * 40,
                    "--glove",
                    str(private),
                    "--backend",
                    "gpu",
                ],
                ["gpu-preflight"],
                ["gpu-preflight", *common, "--model-dir", str(private)],
            )
            for route in routes:
                with self.subTest(route=route[0]):
                    out, err = io.StringIO(), io.StringIO()
                    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                        status = main(["research", *route])
                    self.assertEqual(status, 1)
                    self.assertEqual(out.getvalue(), "")
                    self.assertIn("recsys research: research_gpu_scoring_retired", err.getvalue())
                    self.assertIn("fa983eaeeeeb41b41ac893b7de9827a40c25d27d", err.getvalue())
                    self.assertNotIn(str(private), err.getvalue())
                    self.assertNotIn(str(output), err.getvalue())
                    self.assertFalse(private.exists())
                    self.assertFalse(output.exists())

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
