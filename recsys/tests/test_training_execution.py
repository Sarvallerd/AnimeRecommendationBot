"""Parent publication preflight and failure cleanup without a model run."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import platform
import subprocess
import tempfile
import unittest
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast
from unittest.mock import patch

from research_model_support import patch_release_info
from test_training_artifacts import _inputs
from training_support import SeedTrainingFixture

from recsys.research.protocol import ResearchError
from recsys.research.resources import JobHandle, ProcessResult, ResourceSupervisor
from recsys.research.training import execution
from recsys.research.training.artifacts import TrainingManifest, VerifiedTrainingArtifact
from recsys.research.training.candidate import candidate_identity, load_candidate
from recsys.research.training.execution import fit_training_model
from recsys.research.training.spec import DeviceSpec
from recsys.research.training.types import (
    FitWorkerRequest,
    FitWorkerResult,
    PredictWorkerRequest,
    PredictWorkerResult,
)

_ID = "sha256:" + "a" * 64
_OTHER = "sha256:" + "b" * 64
_REVISION = "b" * 40


def _context() -> tuple[
    FitWorkerRequest,
    FitWorkerResult,
    PredictWorkerRequest,
    PredictWorkerResult,
    VerifiedTrainingArtifact,
]:
    inputs = _inputs()
    spec = load_candidate(
        train_device=DeviceSpec("cpu", None), inference_device=DeviceSpec("cpu", None)
    )
    manifest = cast(
        "TrainingManifest",
        {
            "config": spec.wire(),
            "config_identity": _ID,
            "data_identity": _ID,
            "source_identity": _ID,
            "framework_version": "2.13.0+cpu",
            "producer": {"code_revision": _REVISION},
        },
    )
    verified = VerifiedTrainingArtifact(
        _ID, manifest, spec, inputs.mal_ids, (1,), Path("/tmp/weights.safetensors")
    )
    base = {
        "schema_version": 1,
        "kind": "arb-training-worker-request",
        "request_token": "a" * 64,
        "candidate": "item-seed-mean-bpr-v1",
        "candidate_resource_identity": candidate_identity(),
        "prepared_dir": "/tmp/prepared",
        "expected_prepared_identity": inputs.prepared_identity,
        "code_revision": _REVISION,
    }
    fit_request = cast("FitWorkerRequest", {**base, "operation": "fit", "spec": spec.wire()})
    runtime = {
        "framework_version": "2.13.0+cpu",
        "python_version": platform.python_version(),
        "numpy_version": "2.5.3",
        "safetensors_version": "0.8.0",
        "device": {"kind": "cpu", "index": None},
        "torch_cuda_version": None,
        "device_name": None,
        "cuda_peak_allocated_bytes": None,
        "cuda_peak_reserved_bytes": None,
    }
    result_base = {
        "schema_version": 1,
        "kind": "arb-training-worker-result",
        "status": "success",
        "candidate": base["candidate"],
        "candidate_resource_identity": base["candidate_resource_identity"],
        "request_identity": _ID,
        "prepared_identity": inputs.prepared_identity,
        "fit_identity": inputs.fit_identity,
        "data_identity": _ID,
        "source_identity": _ID,
        "runtime": runtime,
    }
    fit_result = cast(
        "FitWorkerResult",
        {
            **result_base,
            "operation": "fit",
            "model": {"relative_path": "model", "identity": _ID},
            "config_identity": _ID,
            "training": {
                "epochs": [],
                "examples": 3,
                "optimizer_steps": 3,
                "supported_items": 1,
                "initial_weights_sha256": "a" * 64,
                "final_weights_sha256": "b" * 64,
                "changed_weights": True,
            },
        },
    )
    predict_request = cast(
        "PredictWorkerRequest",
        {
            **base,
            "operation": "predict",
            "model_dir": "/tmp/model",
            "expected_model_identity": _ID,
            "seed_mal_ids": [1],
            "limit": 5,
            "inference_device": None,
        },
    )
    predict_result = cast(
        "PredictWorkerResult",
        {
            **result_base,
            "operation": "predict",
            "model_identity": _ID,
            "inference_producer": {"code_revision": _REVISION},
            "stored_inference_device": runtime["device"],
            "requested_inference_device": None,
            "effective_inference_device": runtime["device"],
            "prediction": {
                "relative_path": "prediction.json",
                "descriptor": {"bytes": 1, "sha256": "a" * 64},
                "identity": _ID,
            },
        },
    )
    return fit_request, fit_result, predict_request, predict_result, verified


class TrainingExecutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.prepared = self.root / "prepared"
        self.prepared.mkdir()
        self.output = self.root / "new-run"

    def _fit(self) -> Path:
        return fit_training_model(
            self.prepared,
            self.output,
            expected_prepared_identity="sha256:" + "a" * 64,
            candidate_id="item-seed-mean-bpr-v1",
            code_revision="b" * 40,
            train_device=DeviceSpec("cpu", None),
            inference_device=DeviceSpec("cpu", None),
        )

    def test_existing_target_rejected_before_input_or_worker(self) -> None:
        self.output.mkdir()
        with patch("recsys.research.training.execution.read_scoring_inputs") as reader:
            with self.assertRaises(ResearchError):
                self._fit()
        reader.assert_not_called()

    def test_timeout_cleans_tentative_attempt_without_publication(self) -> None:
        process = ProcessResult("timeout", None, 1.0, 0, 0, 0, 1, 0.1, True)
        with (
            patch("recsys.research.training.execution.read_scoring_inputs", return_value=_inputs()),
            patch("recsys.research.training.execution._supervised", return_value=process),
        ):
            with self.assertRaisesRegex(ResearchError, "training_timeout"):
                self._fit()
        self.assertFalse(self.output.exists())
        self.assertEqual({path.name for path in self.root.iterdir()}, {"prepared"})

    def test_invalid_candidate_and_revision_do_not_open_input(self) -> None:
        with patch("recsys.research.training.execution.read_scoring_inputs") as reader:
            with self.assertRaises(ResearchError):
                fit_training_model(
                    self.prepared,
                    self.output,
                    expected_prepared_identity="sha256:" + "a" * 64,
                    candidate_id="arbitrary",
                    code_revision="bad",
                )
        reader.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_path_separation_and_symlink_rejected_before_input(self) -> None:
        link = self.root / "link"
        link.symlink_to(self.prepared, target_is_directory=True)
        for target in (self.prepared / "nested", link / "run"):
            with self.subTest(target=target):
                with patch("recsys.research.training.execution.read_scoring_inputs") as reader:
                    with self.assertRaises(ResearchError):
                        fit_training_model(
                            self.prepared,
                            target,
                            expected_prepared_identity=_ID,
                            candidate_id="item-seed-mean-bpr-v1",
                            code_revision=_REVISION,
                            train_device=DeviceSpec("cpu", None),
                        )
                reader.assert_not_called()

    def test_process_outcomes_never_publish(self) -> None:
        for outcome, expected in (
            ("timeout", "training_timeout"),
            ("cancelled", "training_cancelled"),
            ("rss_limit", "training_resource_limit"),
            ("exit_nonzero", "training_worker_failed"),
        ):
            with self.subTest(outcome=outcome):
                process = ProcessResult(outcome, 1, 1.0, 0, 0, 0, 1, 0.1, True)
                with (
                    patch(
                        "recsys.research.training.execution.read_scoring_inputs",
                        return_value=_inputs(),
                    ),
                    patch("recsys.research.training.execution._supervised", return_value=process),
                    self.assertRaisesRegex(ResearchError, expected),
                ):
                    self._fit()
                self.assertFalse(self.output.exists())
                self.assertEqual({path.name for path in self.root.iterdir()}, {"prepared"})

    def test_cleanup_incomplete_retains_private_attempt_without_publication(self) -> None:
        process = ProcessResult("rss_limit", None, 1.0, 0, 0, 0, 1, 0.1, False)
        with (
            patch("recsys.research.training.execution.read_scoring_inputs", return_value=_inputs()),
            patch("recsys.research.training.execution._supervised", return_value=process),
            self.assertRaisesRegex(ResearchError, "training_resource_limit"),
        ):
            self._fit()
        self.assertFalse(self.output.exists())
        self.assertEqual(len(list(self.root.glob(".new-run.attempt.*"))), 1)

    def test_stale_identity_and_nonfinite_worker_failure_do_not_publish(self) -> None:
        for request_identity, expected in (
            (_OTHER, "training_worker_failed"),
            (None, "training_nonfinite_loss"),
        ):
            with self.subTest(request_identity=request_identity):

                def worker(attempt: Path, _cancel: object) -> ProcessResult:
                    raw = (attempt / "request.json").read_bytes()
                    actual = "sha256:" + hashlib.sha256(raw).hexdigest()
                    execution.write_json(
                        attempt / "result.json",
                        {
                            "schema_version": 1,
                            "kind": "arb-training-worker-result",
                            "operation": "fit",
                            "status": "failure",
                            "request_identity": request_identity or actual,
                            "code": "training_nonfinite_loss",
                        },
                    )
                    return ProcessResult("exit_nonzero", 1, 1.0, 0, 0, 0, 1, 0.1, True)

                with (
                    patch(
                        "recsys.research.training.execution.read_scoring_inputs",
                        return_value=_inputs(),
                    ),
                    patch("recsys.research.training.execution._supervised", side_effect=worker),
                    self.assertRaisesRegex(ResearchError, expected),
                ):
                    self._fit()
                self.assertFalse(self.output.exists())
                self.assertEqual({path.name for path in self.root.iterdir()}, {"prepared"})

    def test_oversized_malformed_and_wrong_operation_results_do_not_publish(self) -> None:
        for content, expected in (
            ("oversized", "training_worker_failed"),
            ("malformed", "training_worker_failed"),
            ("wrong_operation", "training_worker_result_invalid"),
        ):
            with self.subTest(content=content):

                def worker(attempt: Path, _cancel: object) -> ProcessResult:
                    path = attempt / "result.json"
                    if content == "oversized":
                        path.write_bytes(b"x" * 65537)
                    elif content == "malformed":
                        path.write_bytes(b'{ "not":"canonical" }')
                    else:
                        actual = (
                            "sha256:"
                            + hashlib.sha256((attempt / "request.json").read_bytes()).hexdigest()
                        )
                        execution.write_json(
                            path,
                            {
                                "schema_version": 1,
                                "kind": "arb-training-worker-result",
                                "operation": "predict",
                                "status": "failure",
                                "request_identity": actual,
                                "code": "training_native_failed",
                            },
                        )
                    return ProcessResult("ok", 0, 1.0, 0, 0, 0, 1, 0.1, True)

                with (
                    patch(
                        "recsys.research.training.execution.read_scoring_inputs",
                        return_value=_inputs(),
                    ),
                    patch("recsys.research.training.execution._supervised", side_effect=worker),
                    self.assertRaisesRegex(ResearchError, expected),
                ):
                    self._fit()
                self.assertFalse(self.output.exists())
                self.assertEqual({path.name for path in self.root.iterdir()}, {"prepared"})

    def test_fit_context_binding_rejects_each_mismatched_field(self) -> None:
        request, result, _predict_request, _predict_result, verified = _context()
        inputs = _inputs()
        execution._bind_fit(result, request, inputs, verified)
        for path in (
            ("candidate",),
            ("candidate_resource_identity",),
            ("prepared_identity",),
            ("fit_identity",),
            ("data_identity",),
            ("source_identity",),
            ("model", "identity"),
            ("config_identity",),
            ("runtime", "framework_version"),
            ("runtime", "device"),
            ("runtime", "numpy_version"),
            ("runtime", "safetensors_version"),
            ("runtime", "python_version"),
            ("training", "supported_items"),
        ):
            with self.subTest(path=path):
                tampered = copy.deepcopy(result)
                current = cast("dict[str, object]", tampered)
                for key in path[:-1]:
                    current = cast("dict[str, object]", current[key])
                current[path[-1]] = _OTHER
                with self.assertRaisesRegex(ResearchError, "training_worker_result_invalid"):
                    execution._bind_fit(tampered, request, inputs, verified)
        wrong_manifest = copy.deepcopy(verified.manifest)
        wrong_manifest["producer"]["code_revision"] = "c" * 40
        wrong_verified = VerifiedTrainingArtifact(
            verified.identity,
            wrong_manifest,
            verified.spec,
            verified.mal_ids,
            verified.supported_mal_ids,
            verified.weights_path,
        )
        with self.assertRaisesRegex(ResearchError, "training_worker_result_invalid"):
            execution._bind_fit(result, request, inputs, wrong_verified)

    def test_prediction_context_and_query_binding_rejects_tampering(self) -> None:
        _fit_request, _fit_result, request, result, verified = _context()
        inputs = _inputs()
        prediction = {
            "schema_version": 1,
            "kind": "arb-seed-prediction",
            "model_identity": _ID,
            "prepared_identity": inputs.prepared_identity,
            "seed_mal_ids": [1],
            "supported_seed_mal_ids": [1],
            "skipped_seed_mal_ids": [],
            "limit": 5,
            "status": "ok",
            "source": "seed-embedding",
            "recommendations": [],
        }
        with patch(
            "recsys.research.training.execution.producer", return_value=result["inference_producer"]
        ):
            execution._bind_predict(result, request, inputs, verified, prediction)
            for key in (
                "candidate",
                "candidate_resource_identity",
                "prepared_identity",
                "fit_identity",
                "model_identity",
                "data_identity",
                "source_identity",
                "inference_producer",
                "stored_inference_device",
                "requested_inference_device",
                "effective_inference_device",
            ):
                with self.subTest(result_key=key):
                    bad = copy.deepcopy(result)
                    bad[key] = _OTHER
                    with self.assertRaisesRegex(ResearchError, "training_worker_result_invalid"):
                        execution._bind_predict(bad, request, inputs, verified, prediction)
            for key in ("device", "framework_version", "numpy_version", "safetensors_version"):
                with self.subTest(runtime_key=key):
                    bad = copy.deepcopy(result)
                    cast("dict[str, object]", bad["runtime"])[key] = _OTHER
                    with self.assertRaisesRegex(ResearchError, "training_worker_result_invalid"):
                        execution._bind_predict(bad, request, inputs, verified, prediction)
            for key, value in (
                ("model_identity", _OTHER),
                ("prepared_identity", _OTHER),
                ("seed_mal_ids", [2]),
                ("supported_seed_mal_ids", []),
                ("skipped_seed_mal_ids", [1]),
                ("limit", 1),
                ("status", "abstained"),
                ("source", "none"),
                ("recommendations", [{"mal_id": 1, "score": 1.0}]),
                ("recommendations", [{"mal_id": 2, "score": float("nan")}]),
            ):
                with self.subTest(prediction_key=key, value=value):
                    bad_prediction = copy.deepcopy(prediction)
                    bad_prediction[key] = value
                    with self.assertRaisesRegex(ResearchError, "training_prediction_invalid"):
                        execution._bind_predict(result, request, inputs, verified, bad_prediction)


@unittest.skipUnless(os.environ.get("ARB_TEST_TORCH") == "1", "opt-in supervised CPU fit")
class SupervisedCpuTrainingTests(unittest.TestCase):
    def test_real_supervised_fit_and_one_prediction_publish_atomically(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        fixture = SeedTrainingFixture(temporary.name)
        prepared, prepared_identity = fixture.make_prepared()
        lock_path = fixture.root / "fixture-lock.json"
        lock_path.write_text(json.dumps(fixture.lock), encoding="utf-8")
        wrapper = fixture.root / "fixture-worker.py"
        wrapper.write_text(
            "import json, sys\n"
            "from pathlib import Path\n"
            "sys.path.insert(0, sys.argv[3])\n"
            "from research_model_support import patch_release_info\n"
            "from recsys.research.training.worker import main\n"
            "with patch_release_info(json.loads(Path(sys.argv[2]).read_text())):\n"
            "    raise SystemExit(main([sys.argv[1]]))\n",
            encoding="utf-8",
        )
        original_start = ResourceSupervisor.start
        tests_dir = str(Path(__file__).resolve().parent)

        def fixture_start(
            supervisor: ResourceSupervisor,
            argv: Sequence[str],
            *,
            cwd: Path,
            log_dir: Path,
            remaining_seconds: float,
            env: Mapping[str, str] | None = None,
        ) -> JobHandle:
            self.assertEqual(argv[1:4], ["-I", "-m", "recsys.research.training.worker"])
            return original_start(
                supervisor,
                [argv[0], "-I", str(wrapper), argv[4], str(lock_path), tests_dir],
                cwd=cwd,
                log_dir=log_dir,
                remaining_seconds=remaining_seconds,
                env=env,
            )

        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        model_run = fixture.root / "model-run"
        prediction_run = fixture.root / "prediction-run"
        with (
            patch_release_info(fixture.lock),
            patch.object(ResourceSupervisor, "start", new=fixture_start),
        ):
            receipt_path = fit_training_model(
                prepared,
                model_run,
                expected_prepared_identity=prepared_identity,
                candidate_id="item-seed-mean-bpr-v1",
                code_revision=revision,
                train_device=DeviceSpec("cpu", None),
                inference_device=DeviceSpec("cpu", None),
            )
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            self.assertEqual(receipt["training"]["optimizer_steps"], 3)
            self.assertEqual(receipt["training"]["supported_items"], 6)
            self.assertTrue(receipt["process"]["cleanup_complete"])
            model = model_run / "model"
            self.assertEqual({path.name for path in model_run.iterdir()}, {"model", "run.json"})
            prediction_path = execution.predict_training_model(
                prepared,
                model,
                prediction_run,
                expected_prepared_identity=prepared_identity,
                expected_model_identity=receipt["model"]["identity"],
                seed_mal_ids=(1, 9),
                code_revision=revision,
                limit=5,
            )
            prediction_receipt = json.loads(prediction_path.read_text(encoding="utf-8"))
            prediction = json.loads(
                (prediction_run / "prediction.json").read_text(encoding="utf-8")
            )
            self.assertEqual(prediction["supported_seed_mal_ids"], [1])
            self.assertEqual(prediction["skipped_seed_mal_ids"], [9])
            self.assertEqual(prediction["status"], "ok")
            self.assertEqual(prediction_receipt["model_identity"], receipt["model"]["identity"])
            self.assertEqual(
                {path.name for path in prediction_run.iterdir()}, {"prediction.json", "run.json"}
            )
        self.assertFalse(list(fixture.root.glob(".*.attempt.*")))


if __name__ == "__main__":
    unittest.main()
