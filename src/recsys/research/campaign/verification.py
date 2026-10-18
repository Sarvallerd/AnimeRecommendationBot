"""Closed supervised verifier and public-report projection for campaign v2."""

from __future__ import annotations

import fcntl
import sys
from pathlib import Path
from typing import cast

from recsys.research.campaign.common import (
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
from recsys.research.campaign.spec import NATIVE_CAMPAIGN, campaign_variants


def _verification_main(request_path: Path) -> int:
    value, _ = _read_private(request_path, limit=16 * 1024)
    if type(value) is not dict:
        _fail("research_campaign_verification_invalid")
    request = cast("dict[str, object]", value)
    operation = request.get("operation")
    common = {
        "schema_version",
        "kind",
        "operation",
        "root",
        "plan_identity",
        "generation",
        "prepared_dir",
        "journal_identity",
    }
    phase_fields = {"variant", "phase", "attempt", "execution_identity"}
    if (
        request.get("schema_version") != 1
        or request.get("kind") != "arb-ml02-campaign-verification-request"
        or operation not in ("verify-phase", "project-report", "resume-verify", "validate-report")
        or set(request)
        != (
            common | phase_fields
            if operation == "verify-phase"
            else common | {"validation_id"}
            if operation == "validate-report"
            else common
        )
        or any(type(request[key]) is not str for key in common - {"schema_version"})
    ):
        _fail("research_campaign_verification_invalid")
    root = Path(cast("str", request["root"]))
    prepared = Path(cast("str", request["prepared_dir"]))
    if not root.is_absolute() or not prepared.is_absolute():
        _fail("research_campaign_verification_invalid")
    if operation == "verify-phase":
        variant = request["variant"]
        phase = request["phase"]
        attempt = request["attempt"]
        if type(variant) is not str or type(phase) is not str or type(attempt) is not int:
            _fail("research_campaign_verification_invalid")
        attempt_dir = root / "variants" / variant / "attempts" / phase / str(attempt)
        if request_path != attempt_dir / "verify-request.json":
            _fail("research_campaign_verification_invalid")
    else:
        if operation == "validate-report":
            validation_id = request["validation_id"]
            if (
                type(validation_id) is not str
                or len(validation_id) != 32
                or any(char not in "0123456789abcdef" for char in validation_id)
            ):
                _fail("research_campaign_verification_invalid")
            expected_request = root / "validations" / validation_id / "request.json"
        else:
            expected_request = (
                root
                / ("projections" if operation == "project-report" else "resumptions")
                / cast("str", request["generation"])
                / "request.json"
            )
        if request_path != expected_request:
            _fail("research_campaign_verification_invalid")
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
        if plan.get("campaign_id") != NATIVE_CAMPAIGN or plan["producer"] != _source_receipt(
            native=True
        ):
            _fail("research_campaign_plan_mismatch")
        journal_path = root / "journal.jsonl"
        journal_raw = journal_path.read_bytes()
        if _digest(journal_raw) != request["journal_identity"]:
            _fail("research_campaign_verification_invalid")
        journal = _Journal(journal_path)
        successes, _, _, starts, failed = _successes(
            journal.events, variants=campaign_variants(NATIVE_CAMPAIGN)
        )
        from recsys.research.campaign.artifacts import _verify_phase

        if operation == "verify-phase":
            variant = cast("str", request["variant"])
            phase = cast("str", request["phase"])
            attempt = cast("int", request["attempt"])
            key = (variant, phase)
            start = starts.get(key)
            if (
                key not in _expected_nodes(plan)
                or start is None
                or start["attempt"] != attempt
                or start["generation"] != request["generation"]
                or key in failed
            ):
                _fail("research_campaign_verification_invalid")
            execution, execution_raw = _read_private(
                attempt_dir / "resources" / "execution.json", limit=4096
            )
            if (
                _digest(execution_raw) != request["execution_identity"]
                or type(execution) is not dict
                or execution.get("outcome") != "ok"
                or execution.get("returncode") != 0
                or execution.get("cleanup_complete") is not True
            ):
                _fail("research_campaign_verification_invalid")
            manifest_identity, output_identity = _verify_phase(
                root,
                plan,
                successes,
                variant,
                phase,
                attempt,
                request["generation"],
                prepared,
            )
            receipt = {
                "schema_version": 1,
                "kind": "arb-ml02-campaign-verification-receipt",
                "plan_identity": request["plan_identity"],
                "request_identity": _digest(_canonical(request)),
                "execution_identity": request["execution_identity"],
                "variant": variant,
                "phase": phase,
                "attempt": attempt,
                "generation": request["generation"],
                "manifest_identity": manifest_identity,
                "output_identity": output_identity,
            }
            _private_file(attempt_dir / "verification-result.json", _canonical(receipt))
            return 0
        if starts or (
            operation in ("project-report", "validate-report")
            and (failed or set(successes) != _expected_nodes(plan))
        ):
            _fail("research_campaign_incomplete")
        for (variant, phase), event in successes.items():
            manifest_identity, _ = _verify_phase(
                root,
                plan,
                successes,
                variant,
                phase,
                event["attempt"],
                event["generation"],
                prepared,
            )
            if manifest_identity != event["output_identity"]:
                _fail("research_campaign_chain_invalid")
        if operation in ("project-report", "validate-report"):
            from recsys.research.campaign.report import _public_projection

            projection = _public_projection(root, plan, successes, prepared)
            projection_raw = _canonical(projection)
            if len(projection_raw) > 256 * 1024:
                _fail("research_campaign_report_invalid")
            if operation == "project-report":
                _private_file(
                    root / "projections" / cast("str", request["generation"]) / "projection.json",
                    projection_raw,
                )
                projection_identity = _digest(projection_raw)
            else:
                from recsys.research.campaign.report import _check_native_report_contents

                projection_identity = _check_native_report_contents(
                    root, plan, projection, request["plan_identity"]
                )
        else:
            projection_identity = None
        receipt = {
            "schema_version": 1,
            "kind": "arb-ml02-campaign-verification-receipt",
            "operation": operation,
            "plan_identity": request["plan_identity"],
            "request_identity": _digest(_canonical(request)),
            "journal_identity": request["journal_identity"],
            "generation": request["generation"],
            "projection_identity": projection_identity,
        }
        result_path = (
            root / "validations" / cast("str", request["validation_id"]) / "result.json"
            if operation == "validate-report"
            else root
            / ("projections" if operation == "project-report" else "resumptions")
            / cast("str", request["generation"])
            / "result.json"
        )
        _private_file(result_path, _canonical(receipt))
        return 0
    finally:
        fcntl.flock(gate, fcntl.LOCK_UN)
        import os

        os.close(gate)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--verification-request":
        try:
            status = _verification_main(Path(sys.argv[2]))
        except BaseException:
            status = 2
        sys.exit(status)
    sys.exit(2)
