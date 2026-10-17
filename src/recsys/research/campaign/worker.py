"""Fixed, supervised and resumable validation-only ML-02 campaign.

The worker entry uses only the standard library until the generation gate has
been acquired and checked. All expensive scientific work runs in workers.
"""

from __future__ import annotations

import fcntl
import os
import sys
from pathlib import Path
from typing import TypedDict, cast

from recsys.research.campaign.artifacts import _phase_path
from recsys.research.campaign.common import (
    _SAFE_CODE,
    AUX_PHASE,
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
    _component_phase,
    _expected_nodes,
    _source_receipt,
    _validate_plan_structure,
)


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
    if type(value) is not dict or set(value) != wanted:
        _fail()
    raw = cast("dict[str, object]", value)
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
) -> WorkerRequest:
    artifact = root / "variants" / variant / "attempts" / phase / str(attempt) / "artifact"
    return {
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


def _worker_main(request_path: Path) -> int:
    """A closed phase dispatcher; acquire the shared generation lock first."""
    request_value, _ = _read_private(request_path, limit=16 * 1024)
    request = _validated_worker_request(request_value)
    if (
        request["schema_version"] != 1
        or request["kind"] != "arb-ml02-campaign-worker"
        or request["variant"] not in VARIANTS
        or request["phase"] not in (*PHASES, AUX_PHASE)
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
        configured = _validate_plan_structure(plan_value)
        plan = cast("CampaignPlan", plan_value)
        if plan["variants"][VARIANTS.index(request["variant"])]["variant_id"] != request[
            "variant"
        ] or (request["variant"], request["phase"]) not in _expected_nodes(plan):
            _fail("research_campaign_generation_stale")
        if plan.get("producer") != _source_receipt(configured):
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
        backend = (
            "cpu" if phase == AUX_PHASE else plan["variants"][VARIANTS.index(variant)]["backend"]
        )
        prepared = Path(request["prepared_dir"])
        output = Path(request["output_dir"])
        identity = plan["prepared_identity"]
        if not prepared.is_absolute() or not Path(request["glove"]).is_absolute():
            _fail()
        expected_model = _phase_path(root, accepted, variant, "fit")
        expected_prediction = _phase_path(root, accepted, variant, "predict")
        expected_content = _phase_path(root, accepted, "content-mean", _component_phase(plan))
        expected_svd = _phase_path(root, accepted, "puresvd", _component_phase(plan))
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
        if phase in ("fit", AUX_PHASE):
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
