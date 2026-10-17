"""Strict, bounded request, result, and run-receipt wire validation."""

from __future__ import annotations

import hashlib
import math
import os
import re
import stat
from pathlib import Path
from typing import cast

from recsys.contracts.bundle import canonical_bytes, parse_canonical
from recsys.research.protocol import ResearchError
from recsys.research.training.candidate import candidate_identity, validate_candidate
from recsys.research.training.spec import parse_training_spec
from recsys.research.training.types import (
    EpochMetricsWire,
    FitWorkerRequest,
    FitWorkerResult,
    PredictionRunReceipt,
    PredictWorkerRequest,
    PredictWorkerResult,
    TrainingRunReceipt,
    WorkerFailure,
)

_HEX = re.compile(r"[0-9a-f]{64}\Z")
_IDENTITY = re.compile(r"sha256:[0-9a-f]{64}\Z")
_REVISION = re.compile(r"[0-9a-f]{40}\Z")
_ERROR_CODE = re.compile(r"(?:training|research)_[a-z0-9_]{1,63}\Z")
_REQUEST_FIELDS = {
    "schema_version",
    "kind",
    "request_token",
    "candidate",
    "candidate_resource_identity",
    "prepared_dir",
    "expected_prepared_identity",
    "code_revision",
    "operation",
}
_RESULT_FIELDS = {
    "schema_version",
    "kind",
    "operation",
    "status",
    "candidate",
    "candidate_resource_identity",
    "request_identity",
    "prepared_identity",
    "fit_identity",
    "data_identity",
    "source_identity",
    "runtime",
}
_RECEIPT_FIELDS = _RESULT_FIELDS - {"operation", "status"} | {"limits", "process"}
_LIMITS = {
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


def _fail(code: str) -> None:
    raise ResearchError(code)


def _record(value: object, fields: set[str], code: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != fields:
        _fail(code)
    return cast("dict[str, object]", value)


def _literal(value: object, expected: object, code: str) -> None:
    if type(value) is not type(expected) or value != expected:
        _fail(code)


def _identity(value: object, code: str) -> None:
    if type(value) is not str or _IDENTITY.fullmatch(value) is None:
        _fail(code)


def _hex(value: object, code: str) -> None:
    if type(value) is not str or _HEX.fullmatch(value) is None:
        _fail(code)


def _number(value: object, code: str) -> None:
    if type(value) is not float or not math.isfinite(value) or value < 0:
        _fail(code)


def _integer(value: object, code: str, *, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        _fail(code)


def _device(value: object, code: str) -> str:
    device = _record(value, {"kind", "index"}, code)
    if device["kind"] == "cpu" and device["index"] is None:
        return "cpu"
    if device["kind"] == "cuda" and type(device["index"]) is int and 0 <= device["index"] <= 31:
        return "cuda"
    _fail(code)
    raise AssertionError("unreachable")


def _text(value: object, code: str) -> None:
    if (
        type(value) is not str
        or not 1 <= len(value) <= 128
        or not value.isascii()
        or not value.isprintable()
    ):
        _fail(code)


def _runtime(value: object, code: str) -> None:
    runtime = _record(
        value,
        {
            "framework_version",
            "python_version",
            "numpy_version",
            "safetensors_version",
            "device",
            "torch_cuda_version",
            "device_name",
            "cuda_peak_allocated_bytes",
            "cuda_peak_reserved_bytes",
        },
        code,
    )
    for key in ("framework_version", "python_version", "numpy_version", "safetensors_version"):
        _text(runtime[key], code)
    kind = _device(runtime["device"], code)
    if (
        runtime["framework_version"] != ("2.13.0+cpu" if kind == "cpu" else "2.13.0+cu129")
        or runtime["numpy_version"] != "2.5.3"
        or runtime["safetensors_version"] != "0.8.0"
    ):
        _fail(code)
    if kind == "cpu":
        if any(
            runtime[key] is not None
            for key in (
                "torch_cuda_version",
                "device_name",
                "cuda_peak_allocated_bytes",
                "cuda_peak_reserved_bytes",
            )
        ):
            _fail(code)
    else:
        _text(runtime["torch_cuda_version"], code)
        _text(runtime["device_name"], code)
        _integer(runtime["cuda_peak_allocated_bytes"], code)
        _integer(runtime["cuda_peak_reserved_bytes"], code)


def _metrics(value: object, code: str) -> None:
    metrics = _record(
        value,
        {
            "epochs",
            "examples",
            "optimizer_steps",
            "supported_items",
            "initial_weights_sha256",
            "final_weights_sha256",
            "changed_weights",
        },
        code,
    )
    epochs = metrics["epochs"]
    if type(epochs) is not list or len(epochs) != 3:
        _fail(code)
    total_examples = 0
    total_steps = 0
    for index, value in enumerate(cast("list[object]", epochs)):
        epoch = _record(value, set(EpochMetricsWire.__annotations__), code)
        _literal(epoch["epoch"], index, code)
        _integer(epoch["examples"], code, minimum=1)
        _integer(epoch["optimizer_steps"], code, minimum=1)
        _number(epoch["mean_loss"], code)
        _number(epoch["max_abs_gradient"], code)
        total_examples += cast("int", epoch["examples"])
        total_steps += cast("int", epoch["optimizer_steps"])
    _literal(metrics["examples"], total_examples, code)
    _literal(metrics["optimizer_steps"], total_steps, code)
    _integer(metrics["supported_items"], code, minimum=2)
    _hex(metrics["initial_weights_sha256"], code)
    _hex(metrics["final_weights_sha256"], code)
    if (
        metrics["initial_weights_sha256"] == metrics["final_weights_sha256"]
        or metrics["changed_weights"] is not True
    ):
        _fail(code)


def _descriptor(value: object, code: str) -> dict[str, object]:
    record = _record(value, {"bytes", "sha256"}, code)
    _integer(record["bytes"], code, minimum=1)
    _hex(record["sha256"], code)
    return record


def _model(value: object, code: str) -> None:
    model = _record(value, {"relative_path", "identity"}, code)
    _literal(model["relative_path"], "model", code)
    _identity(model["identity"], code)


def _prediction(value: object, code: str) -> None:
    prediction = _record(value, {"relative_path", "descriptor", "identity"}, code)
    _literal(prediction["relative_path"], "prediction.json", code)
    descriptor = _descriptor(prediction["descriptor"], code)
    _identity(prediction["identity"], code)
    _literal(prediction["identity"], "sha256:" + cast("str", descriptor["sha256"]), code)


def _producer(value: object, code: str) -> None:
    producer = _record(
        value,
        {
            "code_revision",
            "code_files_sha256",
            "source_layout",
            "python_version",
            "package_version",
            "numpy_version",
        },
        code,
    )
    revision = producer["code_revision"]
    if type(revision) is not str or _REVISION.fullmatch(revision) is None:
        _fail(code)
    hashes = producer["code_files_sha256"]
    if type(hashes) is not dict or not hashes:
        _fail(code)
    for path, digest in cast("dict[object, object]", hashes).items():
        if type(path) is not str or not path or len(path) > 256:
            _fail(code)
        _hex(digest, code)
    for key in ("source_layout", "python_version", "package_version", "numpy_version"):
        _text(producer[key], code)


def _limits(value: object, code: str) -> None:
    limits = _record(value, set(_LIMITS), code)
    for key, expected in _LIMITS.items():
        _literal(limits[key], expected, code)


def _process(value: object, code: str) -> None:
    process = _record(
        value,
        {
            "outcome",
            "returncode",
            "elapsed_seconds",
            "sampled_peak_rss_bytes",
            "wait4_peak_rss_bytes",
            "observed_peak_rss_bytes",
            "rss_samples",
            "max_sample_gap_seconds",
            "cleanup_complete",
        },
        code,
    )
    _literal(process["outcome"], "ok", code)
    _literal(process["returncode"], 0, code)
    _literal(process["cleanup_complete"], True, code)
    for key in ("elapsed_seconds", "max_sample_gap_seconds"):
        _number(process[key], code)
    for key in (
        "sampled_peak_rss_bytes",
        "wait4_peak_rss_bytes",
        "observed_peak_rss_bytes",
        "rss_samples",
    ):
        _integer(process[key], code)
    if cast("int", process["observed_peak_rss_bytes"]) < max(
        cast("int", process["sampled_peak_rss_bytes"]),
        cast("int", process["wait4_peak_rss_bytes"]),
    ):
        _fail(code)


def _request(value: object) -> FitWorkerRequest | PredictWorkerRequest:
    code = "training_worker_request_invalid"
    if type(value) is not dict:
        _fail(code)
    request = cast("dict[str, object]", value)
    operation = request.get("operation")
    if operation == "fit":
        _record(request, _REQUEST_FIELDS | {"spec"}, code)
    elif operation == "predict":
        _record(
            request,
            _REQUEST_FIELDS
            | {
                "model_dir",
                "expected_model_identity",
                "seed_mal_ids",
                "limit",
                "inference_device",
            },
            code,
        )
    else:
        _fail(code)
    _literal(request["schema_version"], 1, code)
    _literal(request["kind"], "arb-training-worker-request", code)
    _literal(request["candidate"], "item-seed-mean-bpr-v1", code)
    _literal(request["candidate_resource_identity"], candidate_identity(), code)
    _hex(request["request_token"], code)
    prepared = request["prepared_dir"]
    if (
        type(prepared) is not str
        or not Path(prepared).is_absolute()
        or ".." in Path(prepared).parts
    ):
        _fail(code)
    _identity(request["expected_prepared_identity"], code)
    revision = request["code_revision"]
    if type(revision) is not str or _REVISION.fullmatch(revision) is None:
        _fail(code)
    if operation == "fit":
        try:
            validate_candidate(parse_training_spec(request["spec"]))
        except ResearchError as exc:
            raise ResearchError(code) from exc
        return cast("FitWorkerRequest", request)
    model = request["model_dir"]
    if type(model) is not str or not Path(model).is_absolute() or ".." in Path(model).parts:
        _fail(code)
    _identity(request["expected_model_identity"], code)
    seeds = request["seed_mal_ids"]
    if (
        type(seeds) is not list
        or len(seeds) > 10
        or any(type(seed) is not int or not 1 <= seed <= 2_147_483_647 for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        _fail(code)
    limit = request["limit"]
    if type(limit) is not int or not 1 <= limit <= 20:
        _fail(code)
    if request["inference_device"] is not None:
        _device(request["inference_device"], code)
    return cast("PredictWorkerRequest", request)


def _result(
    value: object, expected_request_identity: str
) -> FitWorkerResult | PredictWorkerResult | WorkerFailure:
    code = "training_worker_result_invalid"
    if type(value) is not dict:
        _fail(code)
    result = cast("dict[str, object]", value)
    operation, status = result.get("operation"), result.get("status")
    if operation not in ("fit", "predict"):
        _fail(code)
    if status == "failure":
        _record(
            result,
            {"schema_version", "kind", "operation", "status", "request_identity", "code"},
            code,
        )
        _literal(result["schema_version"], 1, code)
        _literal(result["kind"], "arb-training-worker-result", code)
        _literal(result["request_identity"], expected_request_identity, code)
        failure_code = result["code"]
        if type(failure_code) is not str or _ERROR_CODE.fullmatch(failure_code) is None:
            _fail(code)
        return cast("WorkerFailure", result)
    if status != "success":
        _fail(code)
    _record(
        result,
        _RESULT_FIELDS
        | (
            {"model", "config_identity", "training"}
            if operation == "fit"
            else {
                "model_identity",
                "inference_producer",
                "stored_inference_device",
                "requested_inference_device",
                "effective_inference_device",
                "prediction",
            }
        ),
        code,
    )
    _literal(result["schema_version"], 1, code)
    _literal(result["kind"], "arb-training-worker-result", code)
    _literal(result["request_identity"], expected_request_identity, code)
    _literal(result["candidate"], "item-seed-mean-bpr-v1", code)
    _literal(result["candidate_resource_identity"], candidate_identity(), code)
    for key in ("prepared_identity", "fit_identity", "data_identity", "source_identity"):
        _identity(result[key], code)
    _runtime(result["runtime"], code)
    if operation == "fit":
        _model(result["model"], code)
        _identity(result["config_identity"], code)
        _metrics(result["training"], code)
        return cast("FitWorkerResult", result)
    _identity(result["model_identity"], code)
    _producer(result["inference_producer"], code)
    _device(result["stored_inference_device"], code)
    if result["requested_inference_device"] is not None:
        _device(result["requested_inference_device"], code)
    _device(result["effective_inference_device"], code)
    _prediction(result["prediction"], code)
    return cast("PredictWorkerResult", result)


def read_bounded_canonical(path: Path, *, max_bytes: int) -> tuple[object, bytes]:
    """Read one regular inode without following a final symlink or exceeding its cap."""
    if type(max_bytes) is not int or max_bytes < 1:
        _fail("training_resource_limit")
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode):
                _fail("training_worker_wire_invalid")
            if not 0 < info.st_size <= max_bytes:
                _fail("training_resource_limit")
            with os.fdopen(fd, "rb", closefd=False) as stream:
                raw = stream.read(max_bytes + 1)
            if len(raw) != info.st_size or len(raw) > max_bytes:
                _fail("training_resource_limit")
        finally:
            os.close(fd)
    except OSError as exc:
        raise ResearchError("training_worker_wire_invalid") from exc
    try:
        return parse_canonical(raw, path.name), raw
    except (TypeError, ValueError, UnicodeError, OverflowError, RecursionError) as exc:
        raise ResearchError("training_worker_wire_invalid") from exc


def read_worker_request(path: Path) -> FitWorkerRequest | PredictWorkerRequest:
    value, _ = read_bounded_canonical(path, max_bytes=_LIMITS["request_max_bytes"])
    return _request(value)


def write_worker_request(path: Path, request: FitWorkerRequest | PredictWorkerRequest) -> str:
    checked = _request(request)
    raw = canonical_bytes(checked)
    if len(raw) > _LIMITS["request_max_bytes"]:
        _fail("training_resource_limit")
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        try:
            with os.fdopen(fd, "wb", closefd=False) as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
        finally:
            os.close(fd)
    except OSError as exc:
        raise ResearchError("training_worker_wire_invalid") from exc
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def read_worker_result(
    path: Path, *, expected_request_identity: str
) -> FitWorkerResult | PredictWorkerResult | WorkerFailure:
    _identity(expected_request_identity, "training_worker_result_invalid")
    value, _ = read_bounded_canonical(path, max_bytes=_LIMITS["worker_result_max_bytes"])
    return _result(value, expected_request_identity)


def _receipt(value: object, *, prediction: bool) -> TrainingRunReceipt | PredictionRunReceipt:
    code = "training_receipt_invalid"
    fields = _RECEIPT_FIELDS | (
        {
            "model_identity",
            "inference_producer",
            "stored_inference_device",
            "requested_inference_device",
            "effective_inference_device",
            "prediction",
        }
        if prediction
        else {"model", "config_identity", "training"}
    )
    record = _record(value, fields, code)
    _literal(record["schema_version"], 1, code)
    _literal(
        record["kind"],
        "arb-native-prediction-run" if prediction else "arb-native-training-run",
        code,
    )
    _literal(record["candidate"], "item-seed-mean-bpr-v1", code)
    _literal(record["candidate_resource_identity"], candidate_identity(), code)
    for key in (
        "request_identity",
        "prepared_identity",
        "fit_identity",
        "data_identity",
        "source_identity",
    ):
        _identity(record[key], code)
    _runtime(record["runtime"], code)
    _limits(record["limits"], code)
    _process(record["process"], code)
    if prediction:
        _identity(record["model_identity"], code)
        _producer(record["inference_producer"], code)
        _device(record["stored_inference_device"], code)
        if record["requested_inference_device"] is not None:
            _device(record["requested_inference_device"], code)
        _device(record["effective_inference_device"], code)
        _prediction(record["prediction"], code)
        return cast("PredictionRunReceipt", record)
    _model(record["model"], code)
    _identity(record["config_identity"], code)
    _metrics(record["training"], code)
    return cast("TrainingRunReceipt", record)


def validate_training_receipt(value: object) -> TrainingRunReceipt:
    return cast("TrainingRunReceipt", _receipt(value, prediction=False))


def validate_prediction_receipt(value: object) -> PredictionRunReceipt:
    return cast("PredictionRunReceipt", _receipt(value, prediction=True))
