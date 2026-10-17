"""Exact and bounded native worker/receipt wires."""

import copy
import tempfile
import unittest
from pathlib import Path
from typing import cast

from recsys.contracts.bundle import canonical_bytes
from recsys.research.protocol import ResearchError
from recsys.research.training.candidate import candidate_identity, load_candidate
from recsys.research.training.receipts import (
    read_bounded_canonical,
    read_worker_request,
    read_worker_result,
    validate_prediction_receipt,
    validate_training_receipt,
    write_worker_request,
)
from recsys.research.training.spec import DeviceSpec
from recsys.research.training.types import FitWorkerRequest, PredictWorkerRequest


def _fit_result(identity: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": "arb-training-worker-result",
        "operation": "fit",
        "status": "success",
        "candidate": "item-seed-mean-bpr-v1",
        "candidate_resource_identity": candidate_identity(),
        "request_identity": identity,
        "prepared_identity": "sha256:" + "a" * 64,
        "fit_identity": "sha256:" + "b" * 64,
        "data_identity": "sha256:" + "c" * 64,
        "source_identity": "sha256:" + "d" * 64,
        "runtime": {
            "framework_version": "2.13.0+cpu",
            "python_version": "3.12.14",
            "numpy_version": "2.5.3",
            "safetensors_version": "0.8.0",
            "device": {"kind": "cpu", "index": None},
            "torch_cuda_version": None,
            "device_name": None,
            "cuda_peak_allocated_bytes": None,
            "cuda_peak_reserved_bytes": None,
        },
        "model": {"relative_path": "model", "identity": "sha256:" + "e" * 64},
        "config_identity": "sha256:" + "f" * 64,
        "training": {
            "epochs": [
                {
                    "epoch": index,
                    "examples": 3,
                    "optimizer_steps": 1,
                    "mean_loss": 0.7,
                    "max_abs_gradient": 0.1,
                }
                for index in range(3)
            ],
            "examples": 9,
            "optimizer_steps": 3,
            "supported_items": 6,
            "initial_weights_sha256": "1" * 64,
            "final_weights_sha256": "2" * 64,
            "changed_weights": True,
        },
    }


def _limits() -> dict[str, int]:
    return {
        "workers": 1,
        "timeout_seconds": 300,
        "rss_limit_bytes": 4294967296,
        "artifact_max_bytes": 67108864,
        "cuda_min_free_bytes": 2147483648,
        "request_max_bytes": 65536,
        "worker_result_max_bytes": 65536,
        "receipt_max_bytes": 65536,
        "prediction_max_bytes": 32768,
    }


def _predict_result(identity: str) -> dict[str, object]:
    fit = _fit_result(identity)
    result = {
        key: value
        for key, value in fit.items()
        if key not in ("model", "config_identity", "training")
    }
    result["operation"] = "predict"
    result["model_identity"] = "sha256:" + "e" * 64
    result["inference_producer"] = {
        "code_revision": "a" * 40,
        "code_files_sha256": {"src/recsys/research/training/predict.py": "b" * 64},
        "source_layout": "arb054-recsys-layout-v2",
        "python_version": "3.12.14",
        "package_version": "0.4.0",
        "numpy_version": "2.5.3",
    }
    result["stored_inference_device"] = {"kind": "cpu", "index": None}
    result["requested_inference_device"] = None
    result["effective_inference_device"] = {"kind": "cpu", "index": None}
    result["prediction"] = {
        "relative_path": "prediction.json",
        "descriptor": {"bytes": 10, "sha256": "f" * 64},
        "identity": "sha256:" + "f" * 64,
    }
    return result


def _process() -> dict[str, object]:
    return {
        "outcome": "ok",
        "returncode": 0,
        "elapsed_seconds": 1.0,
        "sampled_peak_rss_bytes": 1024,
        "wait4_peak_rss_bytes": 1024,
        "observed_peak_rss_bytes": 1024,
        "rss_samples": 2,
        "max_sample_gap_seconds": 0.1,
        "cleanup_complete": True,
    }


class ReceiptTests(unittest.TestCase):
    def test_cpu_runtime_accepts_either_pinned_torch_build_with_null_cuda_metadata(self) -> None:
        identity = "sha256:" + "a" * 64
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "result.json"
            for framework in ("2.13.0+cpu", "2.13.0+cu129"):
                for operation, template, kind, validate in (
                    ("fit", _fit_result, "arb-native-training-run", validate_training_receipt),
                    (
                        "predict",
                        _predict_result,
                        "arb-native-prediction-run",
                        validate_prediction_receipt,
                    ),
                ):
                    with self.subTest(framework=framework, operation=operation):
                        result = template(identity)
                        runtime = cast("dict[str, object]", result["runtime"])
                        runtime["framework_version"] = framework
                        self.assertEqual(runtime["device"], {"kind": "cpu", "index": None})
                        self.assertTrue(
                            all(
                                runtime[key] is None
                                for key in (
                                    "torch_cuda_version",
                                    "device_name",
                                    "cuda_peak_allocated_bytes",
                                    "cuda_peak_reserved_bytes",
                                )
                            )
                        )
                        path.write_bytes(canonical_bytes(result))
                        self.assertEqual(
                            read_worker_result(path, expected_request_identity=identity), result
                        )
                        receipt = {
                            key: value
                            for key, value in result.items()
                            if key not in ("operation", "status")
                        }
                        receipt.update(kind=kind, limits=_limits(), process=_process())
                        self.assertEqual(validate(receipt), receipt)

    def test_cuda_runtime_rejects_cpu_build_and_malformed_cuda_metadata(self) -> None:
        identity = "sha256:" + "a" * 64
        valid = _fit_result(identity)
        runtime = cast("dict[str, object]", valid["runtime"])
        runtime.update(
            framework_version="2.13.0+cu129",
            device={"kind": "cuda", "index": 0},
            torch_cuda_version="12.9",
            device_name="fixture CUDA device",
            cuda_peak_allocated_bytes=1024,
            cuda_peak_reserved_bytes=2048,
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "result.json"
            path.write_bytes(canonical_bytes(valid))
            self.assertEqual(read_worker_result(path, expected_request_identity=identity), valid)
            receipt = {
                key: value for key, value in valid.items() if key not in ("operation", "status")
            }
            receipt.update(kind="arb-native-training-run", limits=_limits(), process=_process())
            self.assertEqual(validate_training_receipt(receipt), receipt)
            for key, value in (
                ("framework_version", "2.13.0+cpu"),
                ("torch_cuda_version", None),
                ("device_name", ""),
                ("cuda_peak_allocated_bytes", None),
                ("cuda_peak_allocated_bytes", True),
                ("cuda_peak_reserved_bytes", -1),
            ):
                with self.subTest(key=key, value=value):
                    bad = copy.deepcopy(valid)
                    cast("dict[str, object]", bad["runtime"])[key] = value
                    path.write_bytes(canonical_bytes(bad))
                    with self.assertRaisesRegex(ResearchError, "training_worker_result_invalid"):
                        read_worker_result(path, expected_request_identity=identity)
                    bad_receipt = {
                        field: entry
                        for field, entry in bad.items()
                        if field not in ("operation", "status")
                    }
                    bad_receipt.update(
                        kind="arb-native-training-run", limits=_limits(), process=_process()
                    )
                    with self.assertRaisesRegex(ResearchError, "training_receipt_invalid"):
                        validate_training_receipt(bad_receipt)
            cpu_with_cuda_fields = _fit_result(identity)
            cast("dict[str, object]", cpu_with_cuda_fields["runtime"]).update(
                framework_version="2.13.0+cu129", cuda_peak_allocated_bytes=1
            )
            path.write_bytes(canonical_bytes(cpu_with_cuda_fields))
            with self.assertRaisesRegex(ResearchError, "training_worker_result_invalid"):
                read_worker_result(path, expected_request_identity=identity)

    def test_request_roundtrip_and_bounds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request: FitWorkerRequest = {
                "schema_version": 1,
                "kind": "arb-training-worker-request",
                "request_token": "a" * 64,
                "candidate": "item-seed-mean-bpr-v1",
                "candidate_resource_identity": candidate_identity(),
                "prepared_dir": str(root),
                "expected_prepared_identity": "sha256:" + "b" * 64,
                "code_revision": "c" * 40,
                "operation": "fit",
                "spec": load_candidate(
                    train_device=DeviceSpec("cpu", None),
                    inference_device=DeviceSpec("cpu", None),
                ).wire(),
            }
            path = root / "request.json"
            identity = write_worker_request(path, request)
            self.assertEqual(read_worker_request(path), request)
            self.assertTrue(identity.startswith("sha256:"))
            with self.assertRaises(ResearchError):
                read_bounded_canonical(path, max_bytes=20)
            with self.assertRaises(ResearchError):
                write_worker_request(path, request)
            link = root / "link.json"
            link.symlink_to(path)
            with self.assertRaises(ResearchError):
                read_worker_request(link)

    def test_result_and_receipt_reject_unknown_or_wrong_nested_types(self) -> None:
        identity = "sha256:" + "a" * 64
        result = _fit_result(identity)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "result.json"
            path.write_bytes(canonical_bytes(result))
            self.assertEqual(read_worker_result(path, expected_request_identity=identity), result)
            with self.assertRaises(ResearchError):
                read_worker_result(path, expected_request_identity="sha256:" + "b" * 64)
            receipt = {
                key: value for key, value in result.items() if key not in ("operation", "status")
            }
            receipt["kind"] = "arb-native-training-run"
            receipt["limits"] = _limits()
            receipt["process"] = _process()
            self.assertEqual(validate_training_receipt(receipt), receipt)
            bad = copy.deepcopy(receipt)
            training = cast("dict[str, object]", bad["training"])
            epochs = cast("list[dict[str, object]]", training["epochs"])
            epochs[0]["mean_loss"] = float("nan")
            with self.assertRaises(ResearchError):
                validate_training_receipt(bad)
            bad = copy.deepcopy(receipt)
            cast("dict[str, object]", bad["limits"])["workers"] = True
            with self.assertRaises(ResearchError):
                validate_training_receipt(bad)
            bad = copy.deepcopy(receipt)
            bad["request_token"] = "secret"
            with self.assertRaises(ResearchError):
                validate_training_receipt(bad)
            bad = copy.deepcopy(receipt)
            cast("dict[str, object]", bad["training"])["examples"] = 8
            with self.assertRaises(ResearchError):
                validate_training_receipt(bad)
            bad = copy.deepcopy(receipt)
            cast("dict[str, object]", bad["training"])["changed_weights"] = False
            with self.assertRaises(ResearchError):
                validate_training_receipt(bad)
            bad = copy.deepcopy(receipt)
            cast("dict[str, object]", bad["process"])["cleanup_complete"] = False
            with self.assertRaises(ResearchError):
                validate_training_receipt(bad)
            bad = copy.deepcopy(receipt)
            cast("dict[str, object]", bad["runtime"])["cuda_peak_allocated_bytes"] = 0
            with self.assertRaises(ResearchError):
                validate_training_receipt(bad)

    def test_prediction_receipt_reference_and_device_binding(self) -> None:
        result = _predict_result("sha256:" + "a" * 64)
        receipt = {
            key: value
            for key, value in result.items()
            if key
            not in (
                "operation",
                "status",
            )
        }
        receipt["kind"] = "arb-native-prediction-run"
        receipt["limits"] = _limits()
        receipt["process"] = _process()
        self.assertEqual(validate_prediction_receipt(receipt), receipt)
        bad = copy.deepcopy(receipt)
        cast("dict[str, object]", bad["prediction"])["identity"] = "sha256:" + "a" * 64
        with self.assertRaises(ResearchError):
            validate_prediction_receipt(bad)

    def test_predict_request_result_and_failure_wires(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request: PredictWorkerRequest = {
                "schema_version": 1,
                "kind": "arb-training-worker-request",
                "request_token": "a" * 64,
                "candidate": "item-seed-mean-bpr-v1",
                "candidate_resource_identity": candidate_identity(),
                "prepared_dir": str(root),
                "expected_prepared_identity": "sha256:" + "b" * 64,
                "code_revision": "c" * 40,
                "operation": "predict",
                "model_dir": str(root / "model"),
                "expected_model_identity": "sha256:" + "d" * 64,
                "seed_mal_ids": [1, 2],
                "limit": 5,
                "inference_device": {"kind": "cpu", "index": None},
            }
            request_path = root / "request.json"
            identity = write_worker_request(request_path, request)
            self.assertEqual(read_worker_request(request_path), request)
            result = _predict_result(identity)
            result_path = root / "result.json"
            result_path.write_bytes(canonical_bytes(result))
            self.assertEqual(
                read_worker_result(result_path, expected_request_identity=identity), result
            )
            failure = {
                "schema_version": 1,
                "kind": "arb-training-worker-result",
                "operation": "predict",
                "status": "failure",
                "request_identity": identity,
                "code": "training_native_failed",
            }
            result_path.write_bytes(canonical_bytes(failure))
            self.assertEqual(
                read_worker_result(result_path, expected_request_identity=identity), failure
            )
            failure["code"] = "/private/path"
            result_path.write_bytes(canonical_bytes(failure))
            with self.assertRaises(ResearchError):
                read_worker_result(result_path, expected_request_identity=identity)

    def test_unknown_fields_and_exact_numeric_types_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline: dict[str, object] = {
                "schema_version": 1,
                "kind": "arb-training-worker-request",
                "request_token": "a" * 64,
                "candidate": "item-seed-mean-bpr-v1",
                "candidate_resource_identity": candidate_identity(),
                "prepared_dir": str(root),
                "expected_prepared_identity": "sha256:" + "b" * 64,
                "code_revision": "c" * 40,
                "operation": "predict",
                "model_dir": str(root / "model"),
                "expected_model_identity": "sha256:" + "d" * 64,
                "seed_mal_ids": [1, 2],
                "limit": 5,
                "inference_device": None,
            }
            for change in (
                {"schema_version": True},
                {"limit": True},
                {"seed_mal_ids": [1, True]},
                {"extra": "forbidden"},
            ):
                with self.subTest(change=change):
                    path = root / "invalid.json"
                    path.write_bytes(canonical_bytes({**baseline, **change}))
                    with self.assertRaises(ResearchError):
                        read_worker_request(path)
            result = _predict_result("sha256:" + "a" * 64)
            for change in (
                {"extra": "forbidden"},
                {"schema_version": True},
                {"candidate_resource_identity": "sha256:" + "0" * 64},
            ):
                with self.subTest(change=change):
                    path = root / "result.json"
                    path.write_bytes(canonical_bytes({**result, **change}))
                    with self.assertRaises(ResearchError):
                        read_worker_result(path, expected_request_identity="sha256:" + "a" * 64)
            wrong_device = copy.deepcopy(result)
            cast("dict[str, object]", wrong_device["runtime"])["device"] = {
                "kind": "cuda",
                "index": True,
            }
            path.write_bytes(canonical_bytes(wrong_device))
            with self.assertRaises(ResearchError):
                read_worker_result(path, expected_request_identity="sha256:" + "a" * 64)


if __name__ == "__main__":
    unittest.main()
