"""Fixed, supervised and resumable validation-only ML-02 campaign.

The worker entry uses only the standard library until the generation gate has
been acquired and checked. All expensive scientific work runs in workers.
"""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import NotRequired, TypedDict, cast

from recsys.research.campaign.common import _digest, _fail, _read_private
from recsys.research.campaign.journal import CampaignEvent, PhaseKey
from recsys.research.campaign.plan import CampaignPlan
from recsys.research.campaign.spec import NATIVE_CAMPAIGN, NATIVE_VARIANT
from recsys.research.models.types import ScoringInputs


def _native_variant(plan: CampaignPlan) -> dict[str, object]:
    entries = [item for item in plan["variants"] if item["variant_id"] == NATIVE_VARIANT]
    if len(entries) != 1:
        _fail("research_campaign_plan_mismatch")
    return cast("dict[str, object]", entries[0])


def _check_native_training(value: object, supported: int) -> None:
    if type(value) is not dict or set(value) != {
        "epochs",
        "examples",
        "optimizer_steps",
        "supported_items",
        "initial_weights_sha256",
        "final_weights_sha256",
        "changed_weights",
    }:
        _fail("research_campaign_native_fit_invalid")
    metrics = cast("dict[str, object]", value)
    epochs = metrics["epochs"]
    if (
        type(epochs) is not list
        or len(epochs) != 3
        or metrics["supported_items"] != supported
        or metrics["changed_weights"] is not True
        or any(
            type(metrics[key]) is not str
            or len(cast("str", metrics[key])) != 64
            or any(c not in "0123456789abcdef" for c in cast("str", metrics[key]))
            for key in ("initial_weights_sha256", "final_weights_sha256")
        )
        or metrics["initial_weights_sha256"] == metrics["final_weights_sha256"]
    ):
        _fail("research_campaign_native_fit_invalid")
    total_examples = total_steps = 0
    for expected, item in enumerate(epochs):
        if type(item) is not dict or set(item) != {
            "epoch",
            "examples",
            "optimizer_steps",
            "mean_loss",
            "max_abs_gradient",
        }:
            _fail("research_campaign_native_fit_invalid")
        row = cast("dict[str, object]", item)
        if (
            type(row["epoch"]) is not int
            or row["epoch"] != expected
            or type(row["examples"]) is not int
            or row["examples"] < 1
            or type(row["optimizer_steps"]) is not int
            or row["optimizer_steps"] < 1
            or type(row["mean_loss"]) not in (float, int)
            or not math.isfinite(cast("float", row["mean_loss"]))
            or cast("float", row["mean_loss"]) <= 0
            or type(row["max_abs_gradient"]) not in (float, int)
            or not math.isfinite(cast("float", row["max_abs_gradient"]))
            or cast("float", row["max_abs_gradient"]) <= 0
        ):
            _fail("research_campaign_native_fit_invalid")
        total_examples += row["examples"]
        total_steps += row["optimizer_steps"]
    if metrics["examples"] != total_examples or metrics["optimizer_steps"] != total_steps:
        _fail("research_campaign_native_fit_invalid")


def _verify_native_fit(
    artifact: Path,
    attempt_dir: Path,
    plan: CampaignPlan,
    inputs: ScoringInputs,
    manifest_identity: str,
    receipt: WorkerReceipt,
) -> None:
    from recsys.research.training.artifacts import FILES, verify_training_artifact
    from recsys.research.training.candidate import candidate_identity, validate_candidate
    from recsys.research.training.spec import DeviceSpec, parse_training_spec
    from recsys.research.training.torch_model import _hash_weight, load_native

    if {child.name for child in artifact.iterdir()} != FILES or sum(
        (artifact / name).stat().st_size for name in FILES
    ) > 64 * 1024**2:
        _fail("research_campaign_native_fit_invalid")
    verified = verify_training_artifact(
        artifact, expected_identity=manifest_identity, inputs=inputs
    )
    variant = _native_variant(plan)
    expected_spec = parse_training_spec(variant["training_spec"])
    if (
        verified.spec != expected_spec
        or validate_candidate(verified.spec) != expected_spec
        or verified.manifest["producer"]["code_revision"] != plan["code_revision"]
        or verified.spec.inference_device != DeviceSpec("cpu", None)
        or variant["candidate_resource_identity"] != candidate_identity()
    ):
        _fail("research_campaign_native_fit_invalid")
    evidence, _ = _read_private(attempt_dir / "resources" / "measurement.json", limit=64 * 1024)
    if type(evidence) is not dict or set(evidence) != {
        "schema_version",
        "kind",
        "plan_identity",
        "request_identity",
        "candidate_resource_identity",
        "model_identity",
        "config_identity",
        "data_identity",
        "source_identity",
        "runtime",
        "training",
    }:
        _fail("research_campaign_native_fit_invalid")
    record = cast("dict[str, object]", evidence)
    if (
        record["schema_version"] != 1
        or record["kind"] != "arb-ml02-native-fit-evidence"
        or record["plan_identity"] != receipt["plan_identity"]
        or record["request_identity"] != receipt["request_identity"]
        or record["candidate_resource_identity"] != candidate_identity()
        or record["model_identity"] != manifest_identity
        or record["config_identity"] != verified.manifest["config_identity"]
        or record["data_identity"] != verified.manifest["data_identity"]
        or record["source_identity"] != verified.manifest["source_identity"]
    ):
        _fail("research_campaign_native_fit_invalid")
    runtime = record["runtime"]
    if (
        type(runtime) is not dict
        or set(runtime)
        != {
            "framework_version",
            "python_version",
            "numpy_version",
            "safetensors_version",
            "device",
            "torch_cuda_version",
            "device_name",
            "cuda_peak_allocated_bytes",
            "cuda_peak_reserved_bytes",
        }
        or runtime["framework_version"] != verified.manifest["framework_version"]
        or runtime["device"] != verified.spec.train_device.wire()
    ):
        _fail("research_campaign_native_fit_invalid")
    for key in ("cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes"):
        value = runtime[key]
        if value is not None and (type(value) is not int or value < 0):
            _fail("research_campaign_native_fit_invalid")
    if verified.spec.train_device.kind == "cpu" and any(
        runtime[key] is not None
        for key in ("device_name", "cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes")
    ):
        _fail("research_campaign_native_fit_invalid")
    _check_native_training(record["training"], len(verified.supported_mal_ids))
    scorer = load_native(
        artifact,
        expected_identity=manifest_identity,
        inputs=inputs,
        inference_device=DeviceSpec("cpu", None),
    )
    try:
        if scorer.runtime.device != DeviceSpec("cpu", None) or scorer._weights is None:
            _fail("research_campaign_native_fit_invalid")
        if (
            _hash_weight(scorer._weights)
            != cast("dict[str, object]", record["training"])["final_weights_sha256"]
        ):
            _fail("research_campaign_native_fit_invalid")
    finally:
        scorer.close()


class WorkerReceipt(TypedDict):
    schema_version: int
    kind: str
    phase: str
    variant: str
    attempt: int
    generation: str
    status: str
    manifest_sha256: str
    plan_identity: NotRequired[str]
    request_identity: NotRequired[str]
    measurement_descriptor: NotRequired[dict[str, object] | None]


def _phase_directory(root: Path, variant: str, phase: str, attempt: int) -> Path:
    return root / "variants" / variant / "attempts" / phase / str(attempt)


def _phase_path(
    root: Path, successes: dict[PhaseKey, CampaignEvent], variant: str, phase: str
) -> Path | None:
    event = successes.get((variant, phase))
    if event is None:
        return None
    return _phase_directory(root, variant, phase, event["attempt"]) / "artifact"


def _read_receipt(
    attempt_dir: Path,
    variant: str,
    phase: str,
    attempt: int,
    generation: str,
    *,
    native_campaign: bool = False,
) -> WorkerReceipt:
    result, _ = _read_private(attempt_dir / "result.json", limit=4096)
    common = {
        "schema_version": 2 if native_campaign else 1,
        "kind": "arb-ml02-campaign-worker-receipt",
        "phase": phase,
        "variant": variant,
        "attempt": attempt,
        "generation": generation,
    }
    if (
        type(result) is not dict
        or type(result.get("schema_version")) is not int
        or type(result.get("attempt")) is not int
        or any(
            type(result.get(key)) is not str
            for key in ("kind", "phase", "variant", "generation", "status", "manifest_sha256")
        )
        or any(result.get(k) != v for k, v in common.items())
        or result.get("status") != "ok"
        or set(result)
        != (
            {
                *common,
                "status",
                "manifest_sha256",
                "plan_identity",
                "request_identity",
                "measurement_descriptor",
            }
            if native_campaign
            else {*common, "status", "manifest_sha256"}
        )
    ):
        _fail("research_campaign_receipt_invalid")
    if native_campaign:
        from recsys.research.artifacts import check_descriptor, file_descriptor

        request, request_raw = _read_private(attempt_dir / "request.json", limit=16 * 1024)
        _, plan_raw = _read_private(attempt_dir.parents[4] / "campaign-plan.json")
        expected_request_fields = {
            "schema_version",
            "kind",
            "root",
            "plan_identity",
            "generation",
            "variant",
            "phase",
            "attempt",
            "prepared_dir",
            "glove",
            "model_dir",
            "prediction_dir",
            "content_model",
            "svd_model",
            "output_dir",
            "campaign_id",
            "request_token",
        }
        if (
            type(request) is not dict
            or set(request) != expected_request_fields
            or request.get("schema_version") != 2
            or request.get("kind") != "arb-ml02-campaign-worker"
            or request.get("campaign_id") != NATIVE_CAMPAIGN
            or request.get("root") != str(attempt_dir.parents[4])
            or request.get("plan_identity") != _digest(plan_raw)
            or request.get("generation") != generation
            or request.get("variant") != variant
            or request.get("phase") != phase
            or type(request.get("attempt")) is not int
            or request.get("attempt") != attempt
            or request.get("output_dir") != str(attempt_dir / "artifact")
            or type(request.get("request_token")) is not str
            or re.fullmatch(r"[0-9a-f]{64}", request["request_token"]) is None
            or result["request_identity"] != _digest(request_raw)
            or result["plan_identity"] != _digest(plan_raw)
        ):
            _fail("research_campaign_receipt_invalid")
        measurement = result["measurement_descriptor"]
        if measurement is not None:
            check_descriptor(measurement)
            if measurement != file_descriptor(attempt_dir / "resources" / "measurement.json"):
                _fail("research_campaign_receipt_invalid")
        if (
            variant == "item-seed-mean-bpr-v1"
            and phase in ("fit", "predict")
            and measurement is None
        ):
            _fail("research_campaign_receipt_invalid")
    return cast("WorkerReceipt", result)


def _verify_phase(
    root: Path,
    plan: CampaignPlan,
    successes: dict[PhaseKey, CampaignEvent],
    variant: str,
    phase: str,
    attempt: int,
    generation: str,
    prepared_dir: Path,
) -> tuple[str, str]:
    """Validate one artifact and its chain after every active process has stopped."""
    from recsys.research.artifacts import file_descriptor, read_json, safe_path
    from recsys.research.models.manifest import _read_model_package
    from recsys.research.models.package import load_model, read_scoring_inputs

    attempt_dir = _phase_directory(root, variant, phase, attempt)
    artifact = safe_path(attempt_dir / "artifact", directory=True)
    native_campaign = plan.get("campaign_id") == NATIVE_CAMPAIGN
    receipt = _read_receipt(
        attempt_dir,
        variant,
        phase,
        attempt,
        generation,
        native_campaign=native_campaign,
    )
    manifest_name = {
        "fit": "training-manifest.json" if variant == NATIVE_VARIANT else "model-manifest.json",
        "predict": "prediction-manifest.json",
        "evaluate": "evaluation-manifest.json",
        "replay": "evaluation-manifest.json",
    }[phase]
    _, raw = read_json(artifact / manifest_name)
    manifest_identity = _digest(raw)
    if receipt["manifest_sha256"] != manifest_identity:
        _fail("research_campaign_receipt_invalid")
    backend = "cpu"
    inputs = read_scoring_inputs(prepared_dir, expected_prepared_identity=plan["prepared_identity"])
    if phase == "fit":
        if variant == NATIVE_VARIANT and native_campaign:
            _verify_native_fit(artifact, attempt_dir, plan, inputs, manifest_identity, receipt)
            return manifest_identity, manifest_identity
        manifest, validated_raw = _read_model_package(
            artifact, inputs=inputs, backend=backend, expected_code_revision=plan["code_revision"]
        )
        if raw != validated_raw:
            _fail("research_campaign_chain_invalid")
        if (
            manifest["variant_id"] != variant
            or manifest["producer"]["code_revision"] != plan["code_revision"]
        ):
            _fail()
        _, cpu_scorer = load_model(artifact, inputs=inputs, backend=backend)
        close = getattr(cpu_scorer, "close", None)
        if callable(close):
            close()
    elif phase == "predict":
        from recsys.research.evaluation.evaluate import PreparationManifest, _predictions

        prepared_value, _ = read_json(prepared_dir / "prepare-manifest.json")
        prepared_manifest = cast("PreparationManifest", prepared_value)
        prediction, prediction_identity = _predictions(
            artifact,
            prepared_manifest,
            plan["prepared_identity"],
            "validation",
            inputs.fit_identity,
        )
        model_path = _phase_path(root, successes, variant, "fit")
        if model_path is None:
            _fail()
        _, model_raw = read_json(
            model_path
            / ("training-manifest.json" if variant == NATIVE_VARIANT else "model-manifest.json")
        )
        if variant == NATIVE_VARIANT and native_campaign:
            from recsys.research.training.artifacts import verify_training_artifact

            verify_training_artifact(
                model_path, expected_identity=_digest(model_raw), inputs=inputs
            )
        if (
            prediction["model_id"] != variant
            or prediction["config_sha256"] != hashlib.sha256(model_raw).hexdigest()
            or prediction["code_revision"] != plan["code_revision"]
            or file_descriptor(artifact / "scores.jsonl.gz")
            != {key: prediction["scores"][key] for key in ("bytes", "sha256")}
        ):
            _fail("research_campaign_prediction_invalid")
        if variant == NATIVE_VARIANT and native_campaign:
            measurement, _ = _read_private(
                attempt_dir / "resources" / "measurement.json", limit=64 * 1024
            )
            if (
                type(measurement) is not dict
                or measurement.get("schema_version") != 1
                or measurement.get("kind") != "arb-ml02-native-prediction-measurement"
                or measurement.get("model_identity") != _digest(model_raw)
                or measurement.get("stored_inference_device") != {"kind": "cpu", "index": None}
                or measurement.get("effective_inference_device") != {"kind": "cpu", "index": None}
                or type(measurement.get("runtime")) is not dict
                or measurement["runtime"].get("device") != {"kind": "cpu", "index": None}
                or measurement.get("status") != "published"
                or type(measurement.get("queries")) is not int
                or type(measurement.get("score_pairs")) is not int
                or measurement["queries"] != prediction["scores"]["rows"]
                or measurement["score_pairs"] != measurement["queries"] * len(inputs.mal_ids)
            ):
                _fail("research_campaign_prediction_invalid")
        return manifest_identity, prediction_identity
    else:
        from recsys.research.evaluation.compare import _evaluation

        eval_manifest, eval_identity, rows = _evaluation(artifact, plan["prepared_identity"])
        prediction_path = _phase_path(root, successes, variant, "predict")
        if prediction_path is None:
            _fail()
        _, prediction_raw = read_json(prediction_path / "prediction-manifest.json")
        if (
            eval_manifest["model_id"] != variant
            or eval_manifest["prediction_identity"] != _digest(prediction_raw)
            or eval_manifest["prediction_code_revision"] != plan["code_revision"]
            or eval_manifest["fit_identity"] != plan["fit_identity"]
            or len(rows) == 0
        ):
            _fail("research_campaign_evaluation_invalid")
        if phase == "replay":
            original = _phase_path(root, successes, variant, "evaluate")
            if original is None:
                _fail()
            for name in ("report.json", "user-metrics.jsonl.gz", "evaluation-manifest.json"):
                if (artifact / name).read_bytes() != (original / name).read_bytes():
                    _fail("research_campaign_replay_mismatch")
        return manifest_identity, eval_identity
    return manifest_identity, manifest_identity
