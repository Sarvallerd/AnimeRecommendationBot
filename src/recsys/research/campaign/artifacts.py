"""Fixed, supervised and resumable validation-only ML-02 campaign.

The worker entry uses only the standard library until the generation gate has
been acquired and checked. All expensive scientific work runs in workers.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Literal, TypedDict, cast

from recsys.research.campaign.common import AUX_PHASE, VARIANTS, _digest, _fail, _read_private
from recsys.research.campaign.journal import CampaignEvent, PhaseKey
from recsys.research.campaign.plan import CampaignPlan


class WorkerReceipt(TypedDict):
    schema_version: int
    kind: str
    phase: str
    variant: str
    attempt: int
    generation: str
    status: str
    manifest_sha256: str


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
    attempt_dir: Path, variant: str, phase: str, attempt: int, generation: str
) -> WorkerReceipt:
    result, _ = _read_private(attempt_dir / "result.json", limit=4096)
    common = {
        "schema_version": 1,
        "kind": "arb-ml02-campaign-worker-receipt",
        "phase": phase,
        "variant": variant,
        "attempt": attempt,
        "generation": generation,
    }
    if (
        type(result) is not dict
        or any(result.get(k) != v for k, v in common.items())
        or result.get("status") != "ok"
        or set(result) != {*common, "status", "manifest_sha256"}
        or type(result["manifest_sha256"]) is not str
    ):
        _fail("research_campaign_receipt_invalid")
    return cast("WorkerReceipt", result)


def _validate_gpu_measurement(value: object) -> None:
    """Check the retained measurement before campaign reporting reads its fields."""

    def nonnegative_float(item: object) -> bool:
        return type(item) is float and math.isfinite(item) and item >= 0

    def nonnegative_int(item: object) -> bool:
        return type(item) is int and item >= 0

    durations = {
        "session_acquire_seconds",
        "input_validation_seconds",
        "model_validation_upload_seconds",
        "input_model_init_upload_seconds",
        "gpu_admission_seconds",
        "synchronized_scoring_seconds",
        "encoding_gzip_write_seconds",
        "staging_wall_seconds",
        "publication_wall_seconds",
    }
    if type(value) is not dict or set(value) != {
        "backend",
        "status",
        "gpu_admission",
        "dedicated_pool_session_including_admission",
        "memory_scope",
        "queries",
        "score_pairs",
        *durations,
    }:
        _fail("research_campaign_admission_invalid")
    admission_value = value["gpu_admission"]
    pool_value = value["dedicated_pool_session_including_admission"]
    if (
        value["status"] != "published"
        or value["backend"] != "gpu"
        or value["memory_scope"] != "dedicated-pool-only-excludes-context-and-cublas"
        or any(not nonnegative_float(value[key]) for key in durations)
        or any(not nonnegative_int(value[key]) for key in ("queries", "score_pairs"))
        or type(admission_value) is not dict
        or type(pool_value) is not dict
    ):
        _fail("research_campaign_admission_invalid")
    if (
        set(admission_value)
        != {
            "purpose",
            "canonical_queries",
            "nonempty_queries",
            "raw_atol",
            "raw_rtol",
            "raw_max_abs_difference",
            "encoded_score_differences",
            "complete_available_rankings_equal",
            "model_sha256",
            "query_sha256",
            "scoring_provenance",
            "code_files_sha256",
        }
        or admission_value["purpose"] != "parity-only-gpu-admission"
        or admission_value["complete_available_rankings_equal"] is not True
        or any(
            not nonnegative_int(admission_value[key])
            for key in ("canonical_queries", "nonempty_queries", "encoded_score_differences")
        )
        or any(
            not nonnegative_float(admission_value[key])
            for key in ("raw_atol", "raw_rtol", "raw_max_abs_difference")
        )
        or any(type(admission_value[key]) is not str for key in ("model_sha256", "query_sha256"))
        or type(admission_value["scoring_provenance"]) is not dict
        or type(admission_value["code_files_sha256"]) is not dict
        or set(pool_value)
        != {
            "pool_used_bytes",
            "pool_reserved_bytes",
            "pool_limit_bytes",
            "pool_peak_used_bytes",
            "pool_peak_reserved_bytes",
        }
        or any(not nonnegative_int(item) for item in pool_value.values())
    ):
        _fail("research_campaign_admission_invalid")


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
    receipt = _read_receipt(attempt_dir, variant, phase, attempt, generation)
    manifest_name = {
        "fit": "model-manifest.json",
        AUX_PHASE: "model-manifest.json",
        "predict": "prediction-manifest.json",
        "evaluate": "evaluation-manifest.json",
        "replay": "evaluation-manifest.json",
    }[phase]
    _, raw = read_json(artifact / manifest_name)
    manifest_identity = _digest(raw)
    if receipt["manifest_sha256"] != manifest_identity:
        _fail("research_campaign_receipt_invalid")
    backend = "cpu" if phase == AUX_PHASE else plan["variants"][VARIANTS.index(variant)]["backend"]
    inputs = read_scoring_inputs(prepared_dir, expected_prepared_identity=plan["prepared_identity"])
    if phase in ("fit", AUX_PHASE):
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
        if backend == "gpu":
            from recsys.research.gpu import gpu_execution

            with gpu_execution():
                _, gpu_scorer = load_model(
                    artifact, inputs=inputs, backend=cast("Literal['gpu']", backend)
                )
                gpu_scorer.close()
        else:
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
        _, model_raw = read_json(model_path / "model-manifest.json")
        if (
            prediction["model_id"] != variant
            or prediction["config_sha256"] != hashlib.sha256(model_raw).hexdigest()
            or prediction["code_revision"] != plan["code_revision"]
            or file_descriptor(artifact / "scores.jsonl.gz")
            != {key: prediction["scores"][key] for key in ("bytes", "sha256")}
        ):
            _fail("research_campaign_prediction_invalid")
        if backend == "gpu":
            measured_value, _ = _read_private(attempt_dir / "resources" / "measurement.json")
            _validate_gpu_measurement(measured_value)
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
