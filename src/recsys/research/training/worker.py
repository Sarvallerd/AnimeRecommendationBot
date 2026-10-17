"""Fixed native fit and prediction operations in one supervised child process."""

from __future__ import annotations

import hashlib
import os
import re
import sys
from pathlib import Path

from recsys.contracts.bundle import canonical_bytes
from recsys.research.artifacts import producer, safe_path, write_json
from recsys.research.models.inputs import read_scoring_inputs, read_training_inputs
from recsys.research.protocol import ResearchError
from recsys.research.training.artifacts import (
    MANIFEST,
    publish_training_artifact,
    verify_training_artifact,
)
from recsys.research.training.candidate import validate_candidate
from recsys.research.training.predict import predict_one, validate_seed_query
from recsys.research.training.receipts import read_worker_request
from recsys.research.training.spec import DeviceSpec, parse_training_spec
from recsys.research.training.types import (
    EpochMetricsWire,
    FitWorkerRequest,
    FitWorkerResult,
    NativeRuntime,
    NativeRuntimeWire,
    PredictWorkerRequest,
    PredictWorkerResult,
    TrainingMetricsWire,
    WorkerFailure,
)

_MAX_RESULT = 65536
_MAX_PREDICTION = 32768
_MAX_ARTIFACT = 67108864


def _runtime(value: NativeRuntime) -> NativeRuntimeWire:
    return {
        "framework_version": value.framework_version,
        "python_version": value.python_version,
        "numpy_version": value.numpy_version,
        "safetensors_version": value.safetensors_version,
        "device": value.device.wire(),
        "torch_cuda_version": value.torch_cuda_version,
        "device_name": value.device_name,
        "cuda_peak_allocated_bytes": value.cuda_peak_allocated_bytes,
        "cuda_peak_reserved_bytes": value.cuda_peak_reserved_bytes,
    }


def _write_result(path: Path, value: FitWorkerResult | PredictWorkerResult | WorkerFailure) -> None:
    if len(canonical_bytes(value)) > _MAX_RESULT:
        raise ResearchError("training_resource_limit")
    write_json(path, value)


def _fit(request: FitWorkerRequest, directory: Path, request_identity: str) -> FitWorkerResult:
    from recsys.research.training.examples import validate_training_rows
    from recsys.research.training.torch_model import fit_native, load_native

    spec = validate_candidate(parse_training_spec(request["spec"]))
    inputs = read_training_inputs(
        request["prepared_dir"], expected_prepared_identity=request["expected_prepared_identity"]
    )
    validate_training_rows(inputs)
    raw = directory / "raw"
    raw.mkdir(mode=0o700)
    fitted = fit_native(inputs, spec, raw / "weights.safetensors")
    model = directory / "model"
    publish_training_artifact(
        model,
        spec=spec,
        inputs=inputs,
        weights_path=fitted.weights_path,
        supported_mal_ids=fitted.supported_mal_ids,
        code_revision=request["code_revision"],
        framework_version=fitted.runtime.framework_version,
    )
    identity = "sha256:" + hashlib.sha256((model / MANIFEST).read_bytes()).hexdigest()
    verified = verify_training_artifact(model, expected_identity=identity, inputs=inputs)
    if (
        verified.spec != spec
        or verified.supported_mal_ids != fitted.supported_mal_ids
        or len(verified.supported_mal_ids) < 2
        or sum(
            (model / name).stat().st_size
            for name in ("training-manifest.json", "items.json", "weights.safetensors")
        )
        > _MAX_ARTIFACT
    ):
        raise ResearchError("training_artifact_invalid")
    scorer = load_native(
        model,
        expected_identity=identity,
        inputs=inputs,
        inference_device=DeviceSpec("cpu", None),
    )
    scorer.close()
    epochs: list[EpochMetricsWire] = [
        {
            "epoch": epoch.epoch,
            "examples": epoch.examples,
            "optimizer_steps": epoch.optimizer_steps,
            "mean_loss": epoch.mean_loss,
            "max_abs_gradient": epoch.max_abs_gradient,
        }
        for epoch in fitted.epochs
    ]
    training: TrainingMetricsWire = {
        "epochs": epochs,
        "examples": sum(epoch["examples"] for epoch in epochs),
        "optimizer_steps": sum(epoch["optimizer_steps"] for epoch in epochs),
        "supported_items": len(fitted.supported_mal_ids),
        "initial_weights_sha256": fitted.initial_weights_sha256,
        "final_weights_sha256": fitted.final_weights_sha256,
        "changed_weights": True,
    }
    manifest = verified.manifest
    return {
        "schema_version": 1,
        "kind": "arb-training-worker-result",
        "operation": "fit",
        "status": "success",
        "candidate": request["candidate"],
        "candidate_resource_identity": request["candidate_resource_identity"],
        "request_identity": request_identity,
        "prepared_identity": inputs.prepared_identity,
        "fit_identity": inputs.fit_identity,
        "data_identity": manifest["data_identity"],
        "source_identity": manifest["source_identity"],
        "runtime": _runtime(fitted.runtime),
        "model": {"relative_path": "model", "identity": identity},
        "config_identity": manifest["config_identity"],
        "training": training,
    }


def _predict(
    request: PredictWorkerRequest, directory: Path, request_identity: str
) -> PredictWorkerResult:
    from recsys.research.training.torch_model import load_native

    inputs = read_scoring_inputs(
        request["prepared_dir"], expected_prepared_identity=request["expected_prepared_identity"]
    )
    verified = verify_training_artifact(
        request["model_dir"], expected_identity=request["expected_model_identity"], inputs=inputs
    )
    validate_candidate(verified.spec)
    seeds, limit = validate_seed_query(
        request["seed_mal_ids"], mal_ids=inputs.mal_ids, limit=request["limit"]
    )
    device_wire = request["inference_device"]
    device = (
        DeviceSpec(device_wire["kind"], device_wire["index"]) if device_wire is not None else None
    )
    scorer = load_native(
        Path(request["model_dir"]),
        expected_identity=request["expected_model_identity"],
        inputs=inputs,
        inference_device=device,
    )
    try:
        document = predict_one(
            inputs,
            scorer,
            model_identity=verified.identity,
            seed_mal_ids=seeds,
            limit=limit,
        )
        runtime = _runtime(scorer.runtime)
    finally:
        scorer.close()
    raw = canonical_bytes(document)
    if len(raw) > _MAX_PREDICTION:
        raise ResearchError("training_resource_limit")
    prediction_path = directory / "prediction.json"
    prediction_descriptor = write_json(prediction_path, document)
    manifest = verified.manifest
    return {
        "schema_version": 1,
        "kind": "arb-training-worker-result",
        "operation": "predict",
        "status": "success",
        "candidate": request["candidate"],
        "candidate_resource_identity": request["candidate_resource_identity"],
        "request_identity": request_identity,
        "prepared_identity": inputs.prepared_identity,
        "fit_identity": inputs.fit_identity,
        "data_identity": manifest["data_identity"],
        "source_identity": manifest["source_identity"],
        "runtime": runtime,
        "model_identity": verified.identity,
        "inference_producer": producer(request["code_revision"]),
        "stored_inference_device": verified.spec.inference_device.wire(),
        "requested_inference_device": device.wire() if device is not None else None,
        "effective_inference_device": (device or verified.spec.inference_device).wire(),
        "prediction": {
            "relative_path": "prediction.json",
            "descriptor": prediction_descriptor,
            "identity": "sha256:" + prediction_descriptor["sha256"],
        },
    }


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 1:
        return 1
    request_path = Path(arguments[0])
    if request_path.name != "request.json" or not request_path.is_absolute():
        return 1
    try:
        safe_path(request_path, directory=False)
        directory = safe_path(request_path.parent, directory=True)
        if directory.stat().st_uid != os.getuid() or directory.stat().st_mode & 0o077:
            return 1
        request = read_worker_request(request_path)
        request_identity = "sha256:" + hashlib.sha256(canonical_bytes(request)).hexdigest()
    except (ResearchError, OSError, TypeError, ValueError):
        return 1
    try:
        result = (
            _fit(request, directory, request_identity)
            if request["operation"] == "fit"
            else _predict(request, directory, request_identity)
        )
        _write_result(directory / "result.json", result)
        return 0
    except BaseException as exc:
        code = str(exc) if isinstance(exc, ResearchError) else "training_native_failed"
        if re.fullmatch(r"(?:research|training)_[a-z0-9_]{1,100}", code) is None:
            code = "training_native_failed"
        failure: WorkerFailure = {
            "schema_version": 1,
            "kind": "arb-training-worker-result",
            "operation": request["operation"],
            "status": "failure",
            "request_identity": request_identity,
            "code": code,
        }
        try:
            _write_result(directory / "result.json", failure)
        except (ResearchError, OSError, ValueError, TypeError):
            pass
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
