"""Bounded, atomic parent orchestration for one native fit or prediction."""

from __future__ import annotations

import json
import math
import os
import platform
import re
import secrets
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Never, cast

from recsys.contracts.bundle import canonical_bytes
from recsys.contracts.resources import resource_bytes
from recsys.research.artifacts import (
    cleanup,
    descriptor,
    open_stage,
    producer,
    publish,
    safe_path,
    separate,
    write_json,
)
from recsys.research.models.inputs import read_scoring_inputs
from recsys.research.models.types import ScoringInputs
from recsys.research.protocol import ResearchError
from recsys.research.resources import ProcessResult, ResourceSupervisor
from recsys.research.training.artifacts import (
    FILES,
    TrainingManifest,
    VerifiedTrainingArtifact,
    verify_training_artifact,
)
from recsys.research.training.candidate import (
    candidate_identity,
    load_candidate,
    validate_candidate,
)
from recsys.research.training.predict import validate_seed_query
from recsys.research.training.receipts import (
    read_bounded_canonical,
    read_worker_result,
    validate_prediction_receipt,
    validate_training_receipt,
    write_worker_request,
)
from recsys.research.training.spec import DeviceSpec
from recsys.research.training.types import (
    CandidateId,
    ExecutionLimitsWire,
    FitWorkerRequest,
    FitWorkerResult,
    NativeRuntimeWire,
    PredictionRunReceipt,
    PredictWorkerRequest,
    PredictWorkerResult,
    ProcessResultWire,
    TrainingRunReceipt,
)

_ID = re.compile(r"sha256:[0-9a-f]{64}\Z")
_REVISION = re.compile(r"[0-9a-f]{40}\Z")
_CANDIDATE: CandidateId = "item-seed-mean-bpr-v1"
_MAX_ARTIFACT = 67108864
_MAX_PREDICTION = 32768
_MAX_RECEIPT = 65536
_TIMEOUT = 300
_RSS = 4294967296
_CUDA_MIN_FREE = 2147483648


def _fail(code: str = "training_execution_invalid") -> Never:
    raise ResearchError(code)


def _identity(value: object) -> None:
    if type(value) is not str or _ID.fullmatch(value) is None:
        _fail("training_identity_mismatch")


def _revision(value: object) -> None:
    if type(value) is not str or _REVISION.fullmatch(value) is None:
        _fail()


def _limits() -> ExecutionLimitsWire:
    candidate_identity()  # Checks the pinned raw resource bytes before parsing them.
    value = json.loads(resource_bytes("campaigns", "research_seed_embedding_v1.json"))
    if type(value) is not dict or type(value.get("execution_limits")) is not dict:
        _fail("training_candidate_invalid")
    return cast("ExecutionLimitsWire", value["execution_limits"])


def _paths(
    prepared_dir: str | Path, output_dir: str | Path, model_dir: str | Path | None = None
) -> tuple[Path, Path, Path | None]:
    prepared = Path(prepared_dir).absolute()
    target = Path(output_dir).absolute()
    model = Path(model_dir).absolute() if model_dir is not None else None
    try:
        safe_path(prepared, directory=True)
        safe_path(target.parent, directory=True)
        if model is not None:
            safe_path(model, directory=True)
            separate(prepared, model, target)
        else:
            separate(prepared, target)
        if target.exists() or target.is_symlink() or not target.name:
            _fail("training_output_exists")
    except OSError as exc:
        raise ResearchError("research_unsafe_path") from exc
    return prepared, target, model


def _private_attempt(target: Path) -> Path:
    path = Path(tempfile.mkdtemp(prefix=f".{target.name}.attempt.", dir=target.parent))
    os.chmod(path, 0o700)
    return path


def _worker_env(device: DeviceSpec) -> dict[str, str]:
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    environment.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    environment.pop("PYTORCH_ALLOC_CONF", None)
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        environment[key] = "1"
    environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    if device.kind == "cpu":
        environment["CUDA_VISIBLE_DEVICES"] = ""
    return environment


def _supervised(
    attempt: Path, cancel_requested: Callable[[], bool] | None, *, device: DeviceSpec
) -> ProcessResult:
    supervisor = ResourceSupervisor(rss_limit_bytes=_RSS)
    with supervisor:
        # Preserve the venv executable symlink so isolated mode retains its site-packages.
        handle = supervisor.start(
            [
                str(Path(sys.executable).absolute()),
                "-I",
                "-m",
                "recsys.research.training.worker",
                str(attempt / "request.json"),
            ],
            cwd=attempt,
            log_dir=attempt / "logs",
            remaining_seconds=_TIMEOUT,
            env=_worker_env(device),
        )
        cancelled = False
        while True:
            if cancel_requested is not None and cancel_requested() and not cancelled:
                supervisor.cancel(handle)
                cancelled = True
            supervisor.poll(wait_seconds=0.1)
            result = supervisor.result(handle)
            if result is not None:
                if not result.cleanup_complete:
                    _fail("training_cleanup_failed")
                break
    return result


def _cuda_admission(device: DeviceSpec) -> None:
    if device.kind != "cuda":
        return
    if device.index is None:
        _fail("training_device_invalid")
    try:
        check = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
                "-i",
                str(device.index),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        if check.returncode != 0 or re.fullmatch(r"[0-9]+\s*", check.stdout) is None:
            _fail("training_device_unavailable")
        if int(check.stdout.strip()) * 1024**2 < _CUDA_MIN_FREE:
            _fail("training_cuda_memory_unavailable")
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ResearchError("training_device_unavailable") from exc


def _process(result: ProcessResult) -> ProcessResultWire:
    return {
        "outcome": result.outcome,
        "returncode": result.returncode,
        "elapsed_seconds": result.elapsed_seconds,
        "sampled_peak_rss_bytes": result.sampled_peak_rss_bytes,
        "wait4_peak_rss_bytes": result.wait4_peak_rss_bytes,
        "observed_peak_rss_bytes": result.observed_peak_rss_bytes,
        "rss_samples": result.rss_samples,
        "max_sample_gap_seconds": result.max_sample_gap_seconds,
        "cleanup_complete": result.cleanup_complete,
    }


def _complete_result(
    attempt: Path, request_identity: str, process: ProcessResult, operation: str
) -> FitWorkerResult | PredictWorkerResult:
    if process.outcome == "cancelled":
        _fail("training_cancelled")
    if process.outcome == "timeout":
        _fail("training_timeout")
    if process.outcome == "rss_limit":
        _fail("training_resource_limit")
    try:
        result = read_worker_result(
            attempt / "result.json", expected_request_identity=request_identity
        )
    except ResearchError:
        _fail("training_worker_failed")
    if result["operation"] != operation:
        _fail("training_worker_result_invalid")
    if result["status"] == "failure":
        code = result["code"]
        if re.fullmatch(r"(?:research|training)_[a-z0-9_]{1,100}", code) is None:
            _fail("training_worker_failed")
        _fail(code)
    if process.outcome != "ok" or process.returncode != 0 or not process.cleanup_complete:
        _fail("training_worker_failed")
    return result


def _model_size(model: Path) -> None:
    if {child.name for child in model.iterdir()} != FILES:
        _fail("training_artifact_invalid")
    if sum((model / name).stat().st_size for name in FILES) > _MAX_ARTIFACT:
        _fail("training_resource_limit")


def _bind_runtime(runtime: NativeRuntimeWire, device: object) -> None:
    versions = json.loads(resource_bytes("campaigns", "research_seed_embedding_v1.json"))[
        "runtime_versions"
    ]
    framework = runtime["framework_version"]
    if (
        runtime["device"] != device
        or runtime["python_version"] != platform.python_version()
        or runtime["numpy_version"] != versions["numpy"]
        or runtime["safetensors_version"] != versions["safetensors"]
        or framework not in (versions["pytorch_cpu"], versions["pytorch_cuda"])
        or (runtime["device"]["kind"] == "cuda" and framework != versions["pytorch_cuda"])
    ):
        _fail("training_worker_result_invalid")


def _bind_fit(
    result: FitWorkerResult,
    request: FitWorkerRequest,
    inputs: ScoringInputs,
    verified: VerifiedTrainingArtifact,
) -> None:
    manifest: TrainingManifest = verified.manifest
    if (
        result["candidate"] != request["candidate"]
        or result["candidate_resource_identity"] != request["candidate_resource_identity"]
        or result["prepared_identity"] != request["expected_prepared_identity"]
        or result["fit_identity"] != inputs.fit_identity
        or result["model"] != {"relative_path": "model", "identity": verified.identity}
        or result["data_identity"] != manifest["data_identity"]
        or result["source_identity"] != manifest["source_identity"]
        or result["config_identity"] != manifest["config_identity"]
        or manifest["config"] != request["spec"]
        or manifest["producer"]["code_revision"] != request["code_revision"]
        or result["runtime"]["framework_version"] != manifest["framework_version"]
        or result["training"]["supported_items"] != len(verified.supported_mal_ids)
    ):
        _fail("training_worker_result_invalid")
    _bind_runtime(result["runtime"], verified.spec.train_device.wire())


def _bind_predict(
    result: PredictWorkerResult,
    request: PredictWorkerRequest,
    inputs: ScoringInputs,
    verified: VerifiedTrainingArtifact,
    prediction: object,
) -> None:
    stored = verified.spec.inference_device.wire()
    requested = request["inference_device"]
    effective = requested if requested is not None else stored
    if (
        result["candidate"] != request["candidate"]
        or result["candidate_resource_identity"] != request["candidate_resource_identity"]
        or result["prepared_identity"] != request["expected_prepared_identity"]
        or result["fit_identity"] != inputs.fit_identity
        or result["model_identity"] != verified.identity
        or result["data_identity"] != verified.manifest["data_identity"]
        or result["source_identity"] != verified.manifest["source_identity"]
        or result["inference_producer"] != producer(request["code_revision"])
        or result["stored_inference_device"] != stored
        or result["requested_inference_device"] != requested
        or result["effective_inference_device"] != effective
    ):
        _fail("training_worker_result_invalid")
    _bind_runtime(result["runtime"], effective)
    if type(prediction) is not dict or set(prediction) != {
        "schema_version",
        "kind",
        "model_identity",
        "prepared_identity",
        "seed_mal_ids",
        "supported_seed_mal_ids",
        "skipped_seed_mal_ids",
        "limit",
        "status",
        "source",
        "recommendations",
    }:
        _fail("training_prediction_invalid")
    doc = cast("dict[str, object]", prediction)
    seeds = request["seed_mal_ids"]
    supported = doc["supported_seed_mal_ids"]
    skipped = doc["skipped_seed_mal_ids"]
    if (
        type(doc["schema_version"]) is not int
        or doc["schema_version"] != 1
        or doc["kind"] != "arb-seed-prediction"
        or doc["model_identity"] != verified.identity
        or doc["prepared_identity"] != inputs.prepared_identity
        or doc["seed_mal_ids"] != seeds
        or doc["limit"] != request["limit"]
        or type(supported) is not list
        or type(skipped) is not list
        or any(type(seed) is not int for seed in supported + skipped)
        or sorted(supported + skipped) != seeds
        or supported != [seed for seed in seeds if seed in verified.supported_mal_ids]
        or skipped != [seed for seed in seeds if seed not in verified.supported_mal_ids]
    ):
        _fail("training_prediction_invalid")
    status, source, rows = doc["status"], doc["source"], doc["recommendations"]
    if (
        status not in ("ok", "abstained")
        or source not in ("seed-embedding", "train-popularity", "none")
        or type(rows) is not list
        or len(rows) > request["limit"]
    ):
        _fail("training_prediction_invalid")
    if (
        (not seeds and (source != "train-popularity" or status != "ok"))
        or (seeds and not supported and (source != "none" or status != "abstained" or rows))
        or (supported and (source != "seed-embedding" or status != "ok"))
    ):
        _fail("training_prediction_invalid")
    seen: set[int] = set()
    ranked: list[tuple[float, int]] = []
    eligible = set(verified.supported_mal_ids) - set(seeds) if supported else set()
    for row in cast("list[object]", rows):
        if type(row) is not dict or set(row) != {"mal_id", "score"}:
            _fail("training_prediction_invalid")
        item, score = row["mal_id"], row["score"]
        if (
            type(item) is not int
            or item not in inputs.mal_ids
            or item in seen
            or item in seeds
            or type(score) is not float
            or not math.isfinite(score)
        ):
            _fail("training_prediction_invalid")
        seen.add(item)
        ranked.append((-score, item))
        if supported and item not in eligible:
            _fail("training_prediction_invalid")
    if supported:
        if len(rows) != min(request["limit"], len(eligible)) or ranked != sorted(ranked):
            _fail("training_prediction_invalid")
    elif not seeds:
        expected = sorted(
            (
                {
                    "mal_id": item,
                    "score": float(inputs.train_stats["items"][str(item)]["positive_users"]),
                }
                for item in inputs.mal_ids
            ),
            key=lambda row: (-row["score"], row["mal_id"]),
        )[: request["limit"]]
        if rows != expected:
            _fail("training_prediction_invalid")


def _publish_fit(
    attempt: Path,
    target: Path,
    result: FitWorkerResult,
    request: FitWorkerRequest,
    inputs: ScoringInputs,
    limits: ExecutionLimitsWire,
    process: ProcessResult,
) -> Path:
    model = safe_path(attempt / "model", directory=True)
    _model_size(model)
    verified = verify_training_artifact(
        model, expected_identity=result["model"]["identity"], inputs=inputs
    )
    validate_candidate(verified.spec)
    _bind_fit(result, request, inputs, verified)
    receipt: TrainingRunReceipt = {
        "schema_version": 1,
        "kind": "arb-native-training-run",
        "candidate": result["candidate"],
        "candidate_resource_identity": result["candidate_resource_identity"],
        "request_identity": result["request_identity"],
        "prepared_identity": result["prepared_identity"],
        "fit_identity": result["fit_identity"],
        "data_identity": result["data_identity"],
        "source_identity": result["source_identity"],
        "runtime": result["runtime"],
        "model": result["model"],
        "config_identity": result["config_identity"],
        "training": result["training"],
        "limits": limits,
        "process": _process(process),
    }
    validate_training_receipt(receipt)
    if len(canonical_bytes(receipt)) > _MAX_RECEIPT:
        _fail("training_resource_limit")
    stage = open_stage(target)
    try:
        os.rename(model, stage / "model")
        write_json(stage / "run.json", receipt)
        if {entry.name for entry in stage.iterdir()} != {"model", "run.json"}:
            _fail("training_artifact_invalid")
        publish(stage, target)
        return target / "run.json"
    except BaseException:
        cleanup(stage)
        raise


def _publish_predict(
    attempt: Path,
    target: Path,
    result: PredictWorkerResult,
    request: PredictWorkerRequest,
    inputs: ScoringInputs,
    verified: VerifiedTrainingArtifact,
    limits: ExecutionLimitsWire,
    process: ProcessResult,
) -> Path:
    value, raw = read_bounded_canonical(attempt / "prediction.json", max_bytes=_MAX_PREDICTION)
    if result["prediction"]["descriptor"] != descriptor(raw):
        _fail("training_prediction_invalid")
    _bind_predict(result, request, inputs, verified, value)
    receipt: PredictionRunReceipt = {
        "schema_version": 1,
        "kind": "arb-native-prediction-run",
        "candidate": result["candidate"],
        "candidate_resource_identity": result["candidate_resource_identity"],
        "request_identity": result["request_identity"],
        "prepared_identity": result["prepared_identity"],
        "fit_identity": result["fit_identity"],
        "data_identity": result["data_identity"],
        "source_identity": result["source_identity"],
        "runtime": result["runtime"],
        "model_identity": result["model_identity"],
        "inference_producer": result["inference_producer"],
        "stored_inference_device": result["stored_inference_device"],
        "requested_inference_device": result["requested_inference_device"],
        "effective_inference_device": result["effective_inference_device"],
        "prediction": result["prediction"],
        "limits": limits,
        "process": _process(process),
    }
    validate_prediction_receipt(receipt)
    if len(canonical_bytes(receipt)) > _MAX_RECEIPT:
        _fail("training_resource_limit")
    stage = open_stage(target)
    try:
        os.rename(attempt / "prediction.json", stage / "prediction.json")
        write_json(stage / "run.json", receipt)
        if {entry.name for entry in stage.iterdir()} != {"prediction.json", "run.json"}:
            _fail("training_prediction_invalid")
        publish(stage, target)
        return target / "run.json"
    except BaseException:
        cleanup(stage)
        raise


def fit_training_model(
    prepared_dir: str | Path,
    output_dir: str | Path,
    *,
    expected_prepared_identity: str,
    candidate_id: str,
    code_revision: str,
    train_device: DeviceSpec = DeviceSpec("cuda", 0),
    inference_device: DeviceSpec = DeviceSpec("cpu", None),
    cancel_requested: Callable[[], bool] | None = None,
) -> Path:
    _identity(expected_prepared_identity)
    _revision(code_revision)
    if candidate_id != _CANDIDATE:
        _fail("training_candidate_invalid")
    spec = load_candidate(train_device=train_device, inference_device=inference_device)
    prepared, target, _ = _paths(prepared_dir, output_dir)
    _cuda_admission(spec.train_device)
    inputs = read_scoring_inputs(prepared, expected_prepared_identity=expected_prepared_identity)
    limits = _limits()
    request: FitWorkerRequest = {
        "schema_version": 1,
        "kind": "arb-training-worker-request",
        "request_token": secrets.token_hex(32),
        "candidate": _CANDIDATE,
        "candidate_resource_identity": candidate_identity(),
        "prepared_dir": str(prepared),
        "expected_prepared_identity": expected_prepared_identity,
        "code_revision": code_revision,
        "operation": "fit",
        "spec": spec.wire(),
    }
    attempt = _private_attempt(target)
    cleanup_allowed = False
    try:
        request_identity = write_worker_request(attempt / "request.json", request)
        process = _supervised(attempt, cancel_requested, device=spec.train_device)
        cleanup_allowed = process.cleanup_complete
        result = _complete_result(attempt, request_identity, process, "fit")
        if result["operation"] != "fit":
            _fail("training_worker_result_invalid")
        return _publish_fit(attempt, target, result, request, inputs, limits, process)
    finally:
        if cleanup_allowed:
            cleanup(attempt)


def predict_training_model(
    prepared_dir: str | Path,
    model_dir: str | Path,
    output_dir: str | Path,
    *,
    expected_prepared_identity: str,
    expected_model_identity: str,
    seed_mal_ids: tuple[int, ...],
    code_revision: str,
    limit: int = 5,
    inference_device: DeviceSpec | None = None,
    cancel_requested: Callable[[], bool] | None = None,
) -> Path:
    _identity(expected_prepared_identity)
    _identity(expected_model_identity)
    _revision(code_revision)
    prepared, target, model = _paths(prepared_dir, output_dir, model_dir)
    if model is None:
        _fail()
    inputs = read_scoring_inputs(prepared, expected_prepared_identity=expected_prepared_identity)
    verified = verify_training_artifact(
        model, expected_identity=expected_model_identity, inputs=inputs
    )
    validate_candidate(verified.spec)
    seeds, count = validate_seed_query(seed_mal_ids, mal_ids=inputs.mal_ids, limit=limit)
    effective = inference_device or verified.spec.inference_device
    _cuda_admission(effective)
    limits = _limits()
    request: PredictWorkerRequest = {
        "schema_version": 1,
        "kind": "arb-training-worker-request",
        "request_token": secrets.token_hex(32),
        "candidate": _CANDIDATE,
        "candidate_resource_identity": candidate_identity(),
        "prepared_dir": str(prepared),
        "expected_prepared_identity": expected_prepared_identity,
        "code_revision": code_revision,
        "operation": "predict",
        "model_dir": str(model),
        "expected_model_identity": expected_model_identity,
        "seed_mal_ids": list(seeds),
        "limit": count,
        "inference_device": inference_device.wire() if inference_device else None,
    }
    attempt = _private_attempt(target)
    cleanup_allowed = False
    try:
        request_identity = write_worker_request(attempt / "request.json", request)
        process = _supervised(attempt, cancel_requested, device=effective)
        cleanup_allowed = process.cleanup_complete
        result = _complete_result(attempt, request_identity, process, "predict")
        if result["operation"] != "predict":
            _fail("training_worker_result_invalid")
        return _publish_predict(attempt, target, result, request, inputs, verified, limits, process)
    finally:
        if cleanup_allowed:
            cleanup(attempt)
