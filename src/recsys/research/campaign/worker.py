"""Fixed, supervised and resumable validation-only ML-02 campaign.

The worker entry uses only the standard library until the generation gate has
been acquired and checked. All expensive scientific work runs in workers.
"""

from __future__ import annotations

import fcntl
import os
import re
import secrets
import sys
from pathlib import Path
from typing import NotRequired, TypedDict, cast

from recsys.research.campaign.artifacts import _phase_path
from recsys.research.campaign.common import (
    _SAFE_CODE,
    PHASES,
    VARIANTS,
    _canonical,
    _digest,
    _fail,
    _private_file,
    _read_private,
    _touch_lock,
)
from recsys.research.campaign.journal import _Journal, _successes
from recsys.research.campaign.plan import (
    CampaignPlan,
    _expected_nodes,
    _source_receipt,
    _validate_plan_structure,
)
from recsys.research.campaign.spec import NATIVE_CAMPAIGN, NATIVE_VARIANT, CampaignName

_TRAIN_SAFE_CODE = re.compile(r"training_[a-z0-9_]{1,63}\Z")


class WorkerRequest(TypedDict):
    schema_version: int
    kind: str
    root: str
    plan_identity: str
    generation: str
    variant: str
    phase: str
    attempt: int
    prepared_dir: str
    glove: str
    model_dir: str | None
    prediction_dir: str | None
    content_model: str | None
    svd_model: str | None
    output_dir: str
    campaign_id: NotRequired[str]
    request_token: NotRequired[str]


def _validated_worker_request(value: object) -> WorkerRequest:
    wanted = {
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
    }
    if type(value) is not dict:
        _fail()
    raw = cast("dict[str, object]", value)
    if raw.get("schema_version") == 3:
        wanted.update(("campaign_id", "request_token"))
    if set(raw) != wanted:
        _fail()
    if type(raw["schema_version"]) is not int or type(raw["attempt"]) is not int:
        _fail()
    string_fields = (
        "kind",
        "root",
        "plan_identity",
        "generation",
        "variant",
        "phase",
        "prepared_dir",
        "glove",
        "output_dir",
    )
    if any(type(raw[field]) is not str for field in string_fields):
        _fail()
    optional_path_fields = ("model_dir", "prediction_dir", "content_model", "svd_model")
    if any(
        raw[field] is not None and type(raw[field]) is not str for field in optional_path_fields
    ):
        _fail()
    if raw["schema_version"] == 3:
        token = raw["request_token"]
        if (
            raw["campaign_id"] != NATIVE_CAMPAIGN
            or type(token) is not str
            or len(token) != 64
            or any(char not in "0123456789abcdef" for char in token)
        ):
            _fail("research_campaign_request_invalid")
    return cast("WorkerRequest", raw)


def _required_path(value: str | None) -> Path:
    if value is None:
        _fail("research_campaign_request_invalid")
    return Path(value)


def _worker_request(
    root: Path,
    plan_identity: str,
    generation: str,
    variant: str,
    phase: str,
    attempt: int,
    prepared_dir: Path,
    glove: Path,
    model_dir: Path | None,
    prediction_dir: Path | None,
    content_model: Path | None,
    svd_model: Path | None,
    *,
    campaign_name: CampaignName = "ml02-cpu-v1",
) -> WorkerRequest:
    artifact = root / "variants" / variant / "attempts" / phase / str(attempt) / "artifact"
    request: WorkerRequest = {
        "schema_version": 1,
        "kind": "arb-ml02-campaign-worker",
        "root": str(root),
        "plan_identity": plan_identity,
        "generation": generation,
        "variant": variant,
        "phase": phase,
        "attempt": attempt,
        "prepared_dir": str(prepared_dir),
        "glove": str(glove),
        "model_dir": str(model_dir) if model_dir else None,
        "prediction_dir": str(prediction_dir) if prediction_dir else None,
        "content_model": str(content_model) if content_model else None,
        "svd_model": str(svd_model) if svd_model else None,
        "output_dir": str(artifact),
    }
    if campaign_name == NATIVE_CAMPAIGN:
        request["schema_version"] = 3
        request["campaign_id"] = NATIVE_CAMPAIGN
        request["request_token"] = secrets.token_hex(32)
        request["model_dir"] = str(model_dir) if phase == "predict" and model_dir else None
        request["prediction_dir"] = (
            str(prediction_dir) if phase in ("evaluate", "replay") and prediction_dir else None
        )
        request["content_model"] = (
            str(content_model) if phase == "fit" and variant == "hybrid" and content_model else None
        )
        request["svd_model"] = (
            str(svd_model) if phase == "fit" and variant == "hybrid" and svd_model else None
        )
    return request


def _worker_main_native(request_path: Path, request: WorkerRequest) -> int:
    """V2 worker: bind every field to the closed plan before scientific imports."""
    from recsys.research.campaign.spec import campaign_variants

    variant, phase, attempt = request["variant"], request["phase"], request["attempt"]
    if variant not in campaign_variants(NATIVE_CAMPAIGN) or phase not in PHASES or attempt < 1:
        _fail("research_campaign_request_invalid")
    root = Path(request["root"])
    expected = root / "variants" / variant / "attempts" / phase / str(attempt)
    if (
        not root.is_absolute()
        or request_path != expected / "request.json"
        or request["output_dir"] != str(expected / "artifact")
    ):
        _fail("research_campaign_request_invalid")
    gate = _touch_lock(root / ".workers.lock")
    try:
        fcntl.flock(gate, fcntl.LOCK_SH)
        generation, _ = _read_private(root / "generation.json", limit=4096)
        plan_value, plan_raw = _read_private(root / "campaign-plan.json")
        if (
            generation != {"generation": request["generation"]}
            or _digest(plan_raw) != request["plan_identity"]
        ):
            _fail("research_campaign_generation_stale")
        _validate_plan_structure(plan_value)
        plan = cast("CampaignPlan", plan_value)
        if (
            plan.get("campaign_id") != NATIVE_CAMPAIGN
            or plan["producer"] != _source_receipt(native=True)
            or (variant, phase) not in _expected_nodes(plan)
        ):
            _fail("research_campaign_plan_mismatch")
        started, _, _, active, _ = _successes(
            _Journal(root / "journal.jsonl").events, variants=campaign_variants(NATIVE_CAMPAIGN)
        )
        event = active.get((variant, phase))
        if (
            event is None
            or event["attempt"] != attempt
            or event["generation"] != request["generation"]
        ):
            _fail("research_campaign_generation_stale")
        expected_paths = {
            "model_dir": _phase_path(root, started, variant, "fit") if phase == "predict" else None,
            "prediction_dir": _phase_path(root, started, variant, "predict")
            if phase in ("evaluate", "replay")
            else None,
            "content_model": _phase_path(root, started, "content-mean", "fit")
            if phase == "fit" and variant == "hybrid"
            else None,
            "svd_model": _phase_path(root, started, "puresvd", "fit")
            if phase == "fit" and variant == "hybrid"
            else None,
        }
        if any(
            cast("dict[str, object]", request)[key] != (str(path) if path else None)
            for key, path in expected_paths.items()
        ):
            _fail("research_campaign_request_invalid")
        prepared = Path(request["prepared_dir"])
        glove = Path(request["glove"])
        if not prepared.is_absolute() or not glove.is_absolute():
            _fail("research_campaign_request_invalid")
        from recsys.research.campaign.storage import check_binding, score_write_guard

        storage = plan["storage"]
        physical = Path(storage["physical_storage_root"])
        check_binding(
            root,
            physical,
            output_device=storage["output_filesystem_device"],
            physical_device=storage["physical_filesystem_device"],
        )
        inner_guard = score_write_guard(root, physical)

        def bound_score_guard(next_emit_bytes: int) -> None:
            check_binding(
                root,
                physical,
                output_device=storage["output_filesystem_device"],
                physical_device=storage["physical_filesystem_device"],
            )
            inner_guard(next_emit_bytes)

        request_identity = _digest(_canonical(request))
        output = expected / "artifact"
        measurement_path = expected / "resources" / "measurement.json"
        identity = plan["prepared_identity"]
        if phase == "fit" and variant == NATIVE_VARIANT:
            from recsys.research.campaign.native import fit_native_campaign_arm
            from recsys.research.training.candidate import load_candidate
            from recsys.research.training.spec import DeviceSpec

            entry = plan["variants"][-1]
            device = (
                DeviceSpec("cuda", 0)
                if entry["train_device"] == DeviceSpec("cuda", 0).wire()
                else DeviceSpec("cpu", None)
            )
            spec = load_candidate(train_device=device, inference_device=DeviceSpec("cpu", None))
            if spec.wire() != entry["training_spec"]:
                _fail("research_campaign_plan_mismatch")
            manifest_path = fit_native_campaign_arm(
                prepared,
                output,
                measurement_path,
                expected_prepared_identity=identity,
                code_revision=plan["code_revision"],
                spec=spec,
                plan_identity=request["plan_identity"],
                request_identity=request_identity,
            )
        elif phase == "predict" and variant == NATIVE_VARIANT:
            from recsys.research.campaign.native import predict_native_validation

            fit_path = _required_path(request["model_dir"])
            model_identity = _digest((fit_path / "training-manifest.json").read_bytes())
            manifest_path = predict_native_validation(
                prepared,
                fit_path,
                output,
                expected_prepared_identity=identity,
                expected_model_identity=model_identity,
                code_revision=plan["code_revision"],
                measurement_output=measurement_path,
                storage_guard=bound_score_guard,
            )
        elif phase == "fit":
            from recsys.research.models.package import fit_model

            assets = (
                {"glove": glove}
                if variant.startswith("content-")
                else {
                    "content_model": _required_path(request["content_model"]),
                    "puresvd_model": _required_path(request["svd_model"]),
                }
                if variant == "hybrid"
                else None
            )
            manifest_path = fit_model(
                prepared,
                output,
                expected_prepared_identity=identity,
                variant_id=variant,
                code_revision=plan["code_revision"],
                assets=assets,
                backend="cpu",
            )
        elif phase == "predict":
            from recsys.research.predict import predict_validation

            measured: dict[str, object] = {}
            manifest_path = predict_validation(
                prepared,
                _required_path(request["model_dir"]),
                output,
                expected_prepared_identity=identity,
                backend="cpu",
                measurements=measured,
                measurement_output=measurement_path,
                storage_guard=bound_score_guard,
                score_format="binary-v1",
            )
        else:
            from recsys.research.evaluation.evaluate import evaluate_predictions

            manifest_path = evaluate_predictions(
                prepared,
                _required_path(request["prediction_dir"]),
                output,
                split="validation",
                expected_prepared_identity=identity,
            )
        from recsys.research.artifacts import file_descriptor

        measurement = file_descriptor(measurement_path) if measurement_path.is_file() else None
        if variant == NATIVE_VARIANT and phase in ("fit", "predict") and measurement is None:
            _fail("research_campaign_measurement_invalid")
        receipt = {
            "schema_version": 3,
            "kind": "arb-ml02-campaign-worker-receipt",
            "status": "ok",
            "manifest_sha256": _digest(manifest_path.read_bytes()),
            "phase": phase,
            "variant": variant,
            "attempt": attempt,
            "generation": request["generation"],
            "plan_identity": request["plan_identity"],
            "request_identity": request_identity,
            "measurement_descriptor": measurement,
        }
        _private_file(expected / "result.json", _canonical(receipt))
        return 0
    except BaseException as error:
        code = str(error)
        if _SAFE_CODE.fullmatch(code) is None and _TRAIN_SAFE_CODE.fullmatch(code) is None:
            code = "research_campaign_worker_failed"
        receipt = {
            "schema_version": 3,
            "kind": "arb-ml02-campaign-worker-receipt",
            "status": "failed",
            "code": code,
            "phase": phase,
            "variant": variant,
            "attempt": attempt,
            "generation": request["generation"],
            "plan_identity": request["plan_identity"],
            "request_identity": _digest(_canonical(request)),
            "measurement_descriptor": None,
        }
        try:
            _private_file(expected / "result.json", _canonical(receipt))
        except OSError:
            pass
        return 1
    finally:
        fcntl.flock(gate, fcntl.LOCK_UN)
        os.close(gate)


def _worker_main(request_path: Path) -> int:
    """A closed phase dispatcher; acquire the shared generation lock first."""
    request_value, _ = _read_private(request_path, limit=16 * 1024)
    request = _validated_worker_request(request_value)
    if request["schema_version"] == 3:
        return _worker_main_native(request_path, request)
    if (
        request["schema_version"] != 1
        or request["kind"] != "arb-ml02-campaign-worker"
        or request["variant"] not in VARIANTS
        or request["phase"] not in PHASES
        or request["attempt"] < 1
    ):
        _fail()
    root = Path(request["root"])
    expected = (
        root
        / "variants"
        / request["variant"]
        / "attempts"
        / request["phase"]
        / str(request["attempt"])
    )
    if (
        not root.is_absolute()
        or Path(request_path) != expected / "request.json"
        or Path(request["output_dir"]) != expected / "artifact"
    ):
        _fail()
    gate = _touch_lock(root / ".workers.lock")
    try:
        fcntl.flock(gate, fcntl.LOCK_SH)
        generation, _ = _read_private(root / "generation.json", limit=4096)
        plan_value, raw = _read_private(root / "campaign-plan.json")
        if type(plan_value) is not dict:
            _fail("research_campaign_generation_stale")
        if (
            generation != {"generation": request["generation"]}
            or _digest(raw) != request["plan_identity"]
        ):
            _fail("research_campaign_generation_stale")
        _validate_plan_structure(plan_value)
        plan = cast("CampaignPlan", plan_value)
        if (
            plan["variants"][VARIANTS.index(request["variant"])]["variant_id"] != request["variant"]
            or (request["variant"], request["phase"]) not in _expected_nodes()
        ):
            _fail("research_campaign_generation_stale")
        if plan.get("producer") != _source_receipt():
            _fail("research_campaign_plan_mismatch")
        accepted, _, _, running, _ = _successes(_Journal(root / "journal.jsonl").events)
        start = running.get((request["variant"], request["phase"]))
        if (
            start is None
            or start["attempt"] != request["attempt"]
            or start["generation"] != request["generation"]
        ):
            _fail("research_campaign_generation_stale")
        variant = request["variant"]
        phase = request["phase"]
        backend = "cpu"
        prepared = Path(request["prepared_dir"])
        output = Path(request["output_dir"])
        identity = plan["prepared_identity"]
        if not prepared.is_absolute() or not Path(request["glove"]).is_absolute():
            _fail()
        expected_model = _phase_path(root, accepted, variant, "fit")
        expected_prediction = _phase_path(root, accepted, variant, "predict")
        expected_content = _phase_path(root, accepted, "content-mean", "fit")
        expected_svd = _phase_path(root, accepted, "puresvd", "fit")
        required_paths = (
            (("model_dir", expected_model),)
            if phase == "predict"
            else (
                (("prediction_dir", expected_prediction),)
                if phase in ("evaluate", "replay")
                else (("content_model", expected_content), ("svd_model", expected_svd))
                if phase == "fit" and variant == "hybrid"
                else ()
            )
        )
        for field, expected_path in required_paths:
            if cast("dict[str, object]", request)[field] != (
                str(expected_path) if expected_path is not None else None
            ):
                _fail("research_campaign_request_invalid")
        # No scientific import or input read occurs before the gate checks.
        if phase == "fit":
            from recsys.research.models.package import fit_model

            assets = (
                {"glove": Path(request["glove"])}
                if variant.startswith("content-")
                else {
                    "content_model": _required_path(request["content_model"]),
                    "puresvd_model": _required_path(request["svd_model"]),
                }
                if variant == "hybrid"
                else None
            )
            manifest_path = fit_model(
                prepared,
                output,
                expected_prepared_identity=identity,
                variant_id=variant,
                code_revision=plan["code_revision"],
                assets=assets,
                backend=backend,
            )
        elif phase == "predict":
            from recsys.research.predict import predict_validation

            measured: dict[str, object] = {}
            manifest_path = predict_validation(
                prepared,
                _required_path(request["model_dir"]),
                output,
                expected_prepared_identity=identity,
                backend=backend,
                measurements=measured,
                measurement_output=expected / "resources" / "measurement.json",
                score_format="jsonl-v1",
            )
        else:
            from recsys.research.evaluation.evaluate import evaluate_predictions

            manifest_path = evaluate_predictions(
                prepared,
                _required_path(request["prediction_dir"]),
                output,
                split="validation",
                expected_prepared_identity=identity,
            )
        receipt = {
            "schema_version": 1,
            "kind": "arb-ml02-campaign-worker-receipt",
            "status": "ok",
            "manifest_sha256": _digest(manifest_path.read_bytes()),
            "phase": phase,
            "variant": variant,
            "attempt": request["attempt"],
            "generation": request["generation"],
        }
        _private_file(expected / "result.json", _canonical(receipt))
        return 0
    except BaseException as error:
        code = str(error)
        if _SAFE_CODE.fullmatch(code) is None:
            code = "research_campaign_worker_failed"
        receipt = {
            "schema_version": 1,
            "kind": "arb-ml02-campaign-worker-receipt",
            "status": "failed",
            "code": code,
            "phase": request["phase"],
            "variant": request["variant"],
            "attempt": request["attempt"],
            "generation": request["generation"],
        }
        try:
            _private_file(expected / "result.json", _canonical(receipt))
        except OSError:
            pass
        return 1
    finally:
        fcntl.flock(gate, fcntl.LOCK_UN)
        os.close(gate)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker-request":
        try:
            worker_status = _worker_main(Path(sys.argv[2]))
        except BaseException:
            worker_status = 2
        sys.exit(worker_status)
    else:
        sys.exit(2)
