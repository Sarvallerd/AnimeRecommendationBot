"""The fixed PyTorch arm of the versioned validation campaign."""

from __future__ import annotations

import hashlib
import math
import os
import re
import stat
import tempfile
from collections.abc import Callable
from pathlib import Path

from recsys.contracts.bundle import canonical_bytes
from recsys.research.artifacts import cleanup, safe_path, separate, write_json
from recsys.research.campaign.spec import NATIVE_VARIANT
from recsys.research.models.inputs import read_scoring_inputs, read_training_inputs
from recsys.research.predict import PredictionModelMetadata, write_validation_scores
from recsys.research.protocol import ResearchError
from recsys.research.training.artifacts import (
    FILES,
    publish_training_artifact,
    verify_training_artifact,
)
from recsys.research.training.candidate import candidate_identity, validate_candidate
from recsys.research.training.spec import DeviceSpec, TrainingSpec
from recsys.research.training.types import (
    EpochMetricsWire,
    NativeRuntime,
    NativeRuntimeWire,
    TrainingMetricsWire,
)

_IDENTITY = re.compile(r"sha256:[0-9a-f]{64}\Z")
_REVISION = re.compile(r"[0-9a-f]{40}\Z")
_MAX_ARTIFACT = 64 * 1024**2
_MAX_EVIDENCE = 64 * 1024


def _identity(value: str) -> None:
    if type(value) is not str or _IDENTITY.fullmatch(value) is None:
        raise ResearchError("research_campaign_native_identity_invalid")


def _evidence_target(
    prepared_dir: Path, output_dir: Path, evidence_path: Path, model_dir: Path | None = None
) -> None:
    paths = (prepared_dir, output_dir, evidence_path)
    separate(*paths, *((model_dir,) if model_dir is not None else ()))
    parent = safe_path(evidence_path.parent, directory=True)
    if (
        not evidence_path.is_absolute()
        or evidence_path.name != "measurement.json"
        or evidence_path.exists()
        or evidence_path.is_symlink()
        or parent.stat().st_uid != os.getuid()
        or stat.S_IMODE(parent.stat().st_mode) != 0o700
    ):
        raise ResearchError("research_unsafe_path")


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


def _training(result: object, support_count: int) -> TrainingMetricsWire:
    from recsys.research.training.types import NativeFitResult

    if type(result) is not NativeFitResult:
        raise ResearchError("research_campaign_native_fit_invalid")
    fitted = result
    epochs: list[EpochMetricsWire] = []
    for ordinal, row in enumerate(fitted.epochs):
        if (
            row.epoch != ordinal
            or type(row.examples) is not int
            or row.examples < 1
            or type(row.optimizer_steps) is not int
            or row.optimizer_steps < 1
            or not math.isfinite(row.mean_loss)
            or not math.isfinite(row.max_abs_gradient)
            or row.mean_loss <= 0
            or row.max_abs_gradient <= 0
        ):
            raise ResearchError("research_campaign_native_fit_invalid")
        epochs.append(
            {
                "epoch": row.epoch,
                "examples": row.examples,
                "optimizer_steps": row.optimizer_steps,
                "mean_loss": row.mean_loss,
                "max_abs_gradient": row.max_abs_gradient,
            }
        )
    if (
        len(epochs) != 3
        or support_count < 2
        or not re.fullmatch(r"[0-9a-f]{64}", fitted.initial_weights_sha256)
        or not re.fullmatch(r"[0-9a-f]{64}", fitted.final_weights_sha256)
        or fitted.initial_weights_sha256 == fitted.final_weights_sha256
    ):
        raise ResearchError("research_campaign_native_fit_invalid")
    return {
        "epochs": epochs,
        "examples": sum(row["examples"] for row in epochs),
        "optimizer_steps": sum(row["optimizer_steps"] for row in epochs),
        "supported_items": support_count,
        "initial_weights_sha256": fitted.initial_weights_sha256,
        "final_weights_sha256": fitted.final_weights_sha256,
        "changed_weights": True,
    }


def fit_native_campaign_arm(
    prepared_dir: Path,
    output_dir: Path,
    evidence_path: Path,
    *,
    expected_prepared_identity: str,
    code_revision: str,
    spec: TrainingSpec,
    plan_identity: str,
    request_identity: str,
) -> Path:
    """Train in the already supervised campaign worker; publish only the three model files."""
    _identity(expected_prepared_identity)
    _identity(plan_identity)
    _identity(request_identity)
    if type(code_revision) is not str or _REVISION.fullmatch(code_revision) is None:
        raise ResearchError("research_campaign_native_fit_invalid")
    spec = validate_candidate(spec)
    if spec.inference_device != DeviceSpec("cpu", None) or spec.train_device not in (
        DeviceSpec("cpu", None),
        DeviceSpec("cuda", 0),
    ):
        raise ResearchError("research_campaign_native_device_invalid")
    prepared_dir, output_dir, evidence_path = map(Path, (prepared_dir, output_dir, evidence_path))
    _evidence_target(prepared_dir, output_dir, evidence_path)
    if output_dir.exists() or output_dir.is_symlink():
        raise ResearchError("training_output_exists")
    safe_path(output_dir.parent, directory=True)
    inputs = read_training_inputs(
        prepared_dir, expected_prepared_identity=expected_prepared_identity
    )
    from recsys.research.training.torch_model import fit_native, load_native

    raw_dir = Path(tempfile.mkdtemp(prefix=".native-weights-", dir=output_dir.parent))
    os.chmod(raw_dir, 0o700)
    published = False
    try:
        fitted = fit_native(inputs, spec, raw_dir / "weights.safetensors")
        metrics = _training(fitted, len(fitted.supported_mal_ids))
        manifest_path = publish_training_artifact(
            output_dir,
            spec=spec,
            inputs=inputs,
            weights_path=fitted.weights_path,
            supported_mal_ids=fitted.supported_mal_ids,
            code_revision=code_revision,
            framework_version=fitted.runtime.framework_version,
        )
        published = True
        if {child.name for child in output_dir.iterdir()} != FILES or sum(
            (output_dir / name).stat().st_size for name in FILES
        ) > _MAX_ARTIFACT:
            raise ResearchError("training_resource_limit")
        model_identity = "sha256:" + hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        verified = verify_training_artifact(
            output_dir, expected_identity=model_identity, inputs=inputs
        )
        if (
            verified.spec != spec
            or verified.supported_mal_ids != fitted.supported_mal_ids
            or verified.manifest["producer"]["code_revision"] != code_revision
        ):
            raise ResearchError("research_campaign_native_fit_invalid")
        scorer = load_native(
            output_dir,
            expected_identity=model_identity,
            inputs=inputs,
            inference_device=DeviceSpec("cpu", None),
        )
        try:
            if scorer.runtime.device != DeviceSpec("cpu", None):
                raise ResearchError("research_campaign_native_device_invalid")
        finally:
            scorer.close()
        evidence = {
            "schema_version": 1,
            "kind": "arb-ml02-native-fit-evidence",
            "plan_identity": plan_identity,
            "request_identity": request_identity,
            "candidate_resource_identity": candidate_identity(),
            "model_identity": model_identity,
            "config_identity": verified.manifest["config_identity"],
            "data_identity": verified.manifest["data_identity"],
            "source_identity": verified.manifest["source_identity"],
            "runtime": _runtime(fitted.runtime),
            "training": metrics,
        }
        if len(canonical_bytes(evidence)) > _MAX_EVIDENCE:
            raise ResearchError("training_resource_limit")
        write_json(evidence_path, evidence)
        return manifest_path
    except BaseException:
        if published:
            cleanup(output_dir)
        raise
    finally:
        cleanup(raw_dir)


def predict_native_validation(
    prepared_dir: Path,
    model_dir: Path,
    output_dir: Path,
    *,
    expected_prepared_identity: str,
    expected_model_identity: str,
    code_revision: str,
    measurement_output: Path,
    storage_guard: Callable[[int], None],
) -> Path:
    """Score every validation query with verified CPU inference and the shared encoder."""
    _identity(expected_prepared_identity)
    _identity(expected_model_identity)
    if type(code_revision) is not str or _REVISION.fullmatch(code_revision) is None:
        raise ResearchError("research_campaign_native_prediction_invalid")
    prepared_dir, model_dir, output_dir, measurement_output = map(
        Path, (prepared_dir, model_dir, output_dir, measurement_output)
    )
    _evidence_target(prepared_dir, output_dir, measurement_output, model_dir)
    if not callable(storage_guard):
        raise ResearchError("research_campaign_native_prediction_invalid")
    inputs = read_scoring_inputs(
        prepared_dir, expected_prepared_identity=expected_prepared_identity
    )
    verified = verify_training_artifact(
        model_dir, expected_identity=expected_model_identity, inputs=inputs
    )
    validate_candidate(verified.spec)
    if (
        verified.manifest["producer"]["code_revision"] != code_revision
        or verified.spec.inference_device != DeviceSpec("cpu", None)
        or verified.spec.train_device not in (DeviceSpec("cpu", None), DeviceSpec("cuda", 0))
    ):
        raise ResearchError("research_campaign_native_prediction_invalid")
    from recsys.research.training.torch_model import load_native

    scorer = load_native(
        model_dir,
        expected_identity=expected_model_identity,
        inputs=inputs,
        inference_device=DeviceSpec("cpu", None),
    )
    published = False
    try:
        if scorer.runtime.device != DeviceSpec("cpu", None):
            raise ResearchError("research_campaign_native_device_invalid")
        measurements: dict[str, object] = {}
        manifest_path = write_validation_scores(
            prepared_dir,
            output_dir,
            inputs=inputs,
            model=PredictionModelMetadata(
                NATIVE_VARIANT, expected_model_identity.removeprefix("sha256:"), code_revision
            ),
            scorer=scorer,
            measurements=measurements,
            storage_guard=storage_guard,
        )
        published = True
        measurements.update(
            {
                "schema_version": 1,
                "kind": "arb-ml02-native-prediction-measurement",
                "model_identity": expected_model_identity,
                "stored_inference_device": verified.spec.inference_device.wire(),
                "effective_inference_device": scorer.runtime.device.wire(),
                "runtime": _runtime(scorer.runtime),
            }
        )
        if len(canonical_bytes(measurements)) > _MAX_EVIDENCE:
            raise ResearchError("training_resource_limit")
        write_json(measurement_output, measurements)
        return manifest_path
    except BaseException:
        if published:
            cleanup(output_dir)
        raise
    finally:
        scorer.close()
