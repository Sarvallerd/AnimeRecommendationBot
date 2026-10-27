"""Fixed, supervised and resumable validation-only ML-02 campaign.

The worker entry uses only the standard library until the generation gate has
been acquired and checked. All expensive scientific work runs in workers.
"""

from __future__ import annotations

import fcntl
import hashlib
import os
import sys
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, cast

from recsys.research.campaign.artifacts import _phase_path, _verify_phase
from recsys.research.campaign.common import (
    FIT_SECONDS,
    PHASES,
    VARIANTS,
    _bytes_equal,
    _canonical,
    _digest,
    _ensure_private,
    _fail,
    _private_file,
    _read_native_public,
    _read_private,
    _touch_lock,
)
from recsys.research.campaign.journal import CampaignEvent, PhaseKey, _Journal, _successes
from recsys.research.campaign.plan import (
    CampaignPlan,
    _expected_nodes,
    _source_receipt,
    _validate_plan_structure,
)
from recsys.research.campaign.spec import NATIVE_CAMPAIGN, NATIVE_VARIANT, campaign_variants

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from recsys.research.evaluation.compare import EvaluationRows


def _required_phase_path(
    root: Path, successes: dict[PhaseKey, CampaignEvent], variant: str, phase: str
) -> Path:
    path = _phase_path(root, successes, variant, phase)
    if path is None:
        _fail("research_campaign_incomplete")
    return path


def _public_projection(
    root: Path,
    plan: CampaignPlan,
    successes: dict[PhaseKey, CampaignEvent],
    prepared_dir: Path,
) -> dict[str, object]:
    """Build only anonymous validation aggregates from strictly paired rows."""
    import numpy as np

    from recsys.research.artifacts import file_descriptor, read_json
    from recsys.research.evaluation.compare import _bootstrap, _evaluation, _primary, _quantile
    from recsys.research.evaluation.evaluate import PredictionManifest, _aggregate, _prepared
    from recsys.research.models.manifest import ModelManifest
    from recsys.research.protocol import load_protocol
    from recsys.research.training.artifacts import TrainingManifest

    prepared, protocol, catalog, fit, fit_identity, stats = _prepared(
        prepared_dir, plan["prepared_identity"]
    )
    if fit_identity != plan["fit_identity"] or protocol != load_protocol():
        _fail("research_campaign_input_invalid")
    all_rows: dict[str, EvaluationRows] = {}
    variants: list[dict[str, object]] = []
    journal_events = _Journal(root / "journal.jsonl").events
    native = plan.get("campaign_id") == NATIVE_CAMPAIGN
    names = campaign_variants(NATIVE_CAMPAIGN) if native else VARIANTS
    _, attempts, charged, _, _ = _successes(journal_events, variants=names)
    reference_metadata = None
    reference_n0 = None
    for variant in names:
        maybe_paths = {phase: _phase_path(root, successes, variant, phase) for phase in PHASES}
        if any(value is None for value in maybe_paths.values()):
            _fail("research_campaign_incomplete")
        paths = cast("dict[str, Path]", maybe_paths)
        model_value, model_raw = read_json(
            paths["fit"]
            / ("training-manifest.json" if variant == NATIVE_VARIANT else "model-manifest.json")
        )
        model = cast("ModelManifest", model_value)
        prediction_value, prediction_raw = read_json(paths["predict"] / "prediction-manifest.json")
        prediction = cast("PredictionManifest", prediction_value)
        evaluation, evaluation_identity, rows = _evaluation(
            paths["evaluate"], plan["prepared_identity"]
        )
        initial_report_value, _ = read_json(
            paths["evaluate"] / "report.json", evaluation["files"]["report.json"]
        )
        initial_report = cast("dict[str, object]", initial_report_value)
        recomputed = _aggregate(rows.values(), catalog, stats)
        expected_report = {
            "schema_version": 1,
            "kind": "arb-ml-evaluation-report",
            "split": "validation",
            "prediction_identity": _digest(prediction_raw),
            "prepared_identity": plan["prepared_identity"],
            "queries": len(rows) // 2,
            "groups": recomputed,
        }
        if _canonical(initial_report) != _canonical(expected_report):
            _fail("research_campaign_aggregate_mismatch")
        if (
            (variant != NATIVE_VARIANT and model["variant_id"] != variant)
            or prediction["model_id"] != variant
            or prediction["config_sha256"] != hashlib.sha256(model_raw).hexdigest()
            or evaluation["model_id"] != variant
            or evaluation["prediction_identity"] != _digest(prediction_raw)
            or evaluation["config_sha256"] != prediction["config_sha256"]
            or evaluation["prediction_code_revision"] != plan["code_revision"]
            or evaluation["selection_identity"] is not None
            or prediction["split"] != "validation"
            or file_descriptor(paths["predict"] / ("scores.bin" if native else "scores.jsonl.gz"))
            != {key: prediction["scores"][key] for key in ("bytes", "sha256")}
            or prediction["schema_version"] != (2 if native else 1)
        ):
            _fail("research_campaign_chain_invalid")
        for name in ("report.json", "user-metrics.jsonl.gz", "evaluation-manifest.json"):
            if (paths["evaluate"] / name).read_bytes() != (paths["replay"] / name).read_bytes():
                _fail("research_campaign_replay_mismatch")
        metadata = {
            key: (
                row["user_key"],
                row["split"],
                row["cohort"],
                row["mask"],
                row["order"],
                row["n"],
                row["history_bin"],
                row["seed_band"],
                row["recent_2024_seed_count"],
            )
            for key, row in rows.items()
        }
        n0 = {key: (row["ranking"], row["metrics"]) for key, row in rows.items() if row["n"] == 0}
        if reference_metadata is None:
            reference_metadata, reference_n0 = metadata, n0
        elif metadata != reference_metadata or n0 != reference_n0:
            _fail("research_campaign_unpaired")
        all_rows[variant] = rows
        phase_resources: dict[str, dict[str, object]] = {}
        for phase in PHASES:
            event = successes[(variant, phase)]
            evidence = event["resources"]
            if (
                type(evidence) is not dict
                or evidence.get("outcome") != "ok"
                or evidence.get("cleanup_complete") is not True
            ):
                _fail("research_campaign_resource_invalid")
            phase_resources[phase] = {
                key: evidence[key]
                for key in (
                    "elapsed_seconds",
                    "sampled_peak_rss_bytes",
                    "wait4_peak_rss_bytes",
                    "observed_peak_rss_bytes",
                    "rss_samples",
                    "max_sample_gap_seconds",
                )
            }
            if native:
                verified = event.get("verification_resources")
                if (
                    type(verified) is not dict
                    or verified.get("outcome") != "ok"
                    or verified.get("cleanup_complete") is not True
                ):
                    _fail("research_campaign_resource_invalid")
                phase_resources[phase]["verification"] = {
                    key: verified[key]
                    for key in (
                        "elapsed_seconds",
                        "sampled_peak_rss_bytes",
                        "wait4_peak_rss_bytes",
                        "observed_peak_rss_bytes",
                        "rss_samples",
                        "max_sample_gap_seconds",
                    )
                }
        variant_row: dict[str, object] = {
            "variant_id": variant,
            "backend": plan["variants"][names.index(variant)]["backend"],
            "model_identity": _digest(model_raw),
            "prediction_identity": _digest(prediction_raw),
            "evaluation_identity": evaluation_identity,
            "replay_identity": _digest((paths["replay"] / "evaluation-manifest.json").read_bytes()),
            "evaluation_replay_bytes_equal": True,
            "queries": initial_report["queries"],
            "resources": phase_resources,
            "fit_budget": {
                "primary_charged_seconds": charged.get((variant, "fit"), 0.0),
                "all_attempts_charged_seconds": charged.get((variant, "fit"), 0.0),
                "remaining_seconds": FIT_SECONDS - charged.get((variant, "fit"), 0.0),
            },
            "checked_groups": recomputed,
        }
        if native:
            from recsys.research.score_blocks import BlockScoreDescriptor

            block_desc = cast("BlockScoreDescriptor", prediction["scores"])
            variant_row.update(
                scores_stored_bytes=block_desc["bytes"],
                scores_decoded_bytes=block_desc["uncompressed_bytes"],
                score_format=prediction["score_format"],
                dense_rows=block_desc["dense_rows"],
                query_block_count=block_desc["query_block_count"],
            )
            entry = plan["variants"][names.index(variant)]
            evaluation_charge = sum(charged.get((variant, step), 0.0) for step in PHASES[1:])
            variant_row.update(
                framework=entry["framework"],
                train_device=entry["train_device"],
                inference_device=entry["inference_device"],
                model_kind=entry["model_kind"],
                config_identity=prediction["config_sha256"],
                data_identity=plan["prepared_identity"],
                source_identity=_digest(_canonical(plan["producer"])),
                evaluation_budget={
                    "all_attempts_charged_seconds": evaluation_charge,
                    "remaining_seconds": plan["execution_profile"]["evaluation_seconds_per_arm"]
                    - evaluation_charge,
                },
                attempts_by_phase={step: attempts.get((variant, step), 0) for step in PHASES},
            )
            if variant == NATIVE_VARIANT:
                native_evidence, _ = _read_private(
                    paths["fit"].parent / "resources" / "measurement.json",
                    limit=64 * 1024,
                )
                if type(native_evidence) is not dict:
                    _fail("research_campaign_measurement_invalid")
                runtime = native_evidence.get("runtime")
                training = native_evidence.get("training")
                if type(runtime) is not dict or type(training) is not dict:
                    _fail("research_campaign_measurement_invalid")
                native_manifest = cast("TrainingManifest", model_value)
                variant_row.update(
                    candidate_resource_identity=entry["candidate_resource_identity"],
                    config_identity=native_manifest["config_identity"],
                    data_identity=native_manifest["data_identity"],
                    source_identity=native_manifest["source_identity"],
                    weights_sha256=native_manifest["files"]["weights.safetensors"]["sha256"],
                    framework_version=native_manifest["framework_version"],
                    fit_runtime_actual_device=runtime["device"],
                    fit_cuda_peak_allocated_bytes=runtime["cuda_peak_allocated_bytes"],
                    fit_cuda_peak_reserved_bytes=runtime["cuda_peak_reserved_bytes"],
                    fit_examples=training["examples"],
                    fit_optimizer_steps=training["optimizer_steps"],
                    fit_supported_items=training["supported_items"],
                    initial_weights_sha256=training["initial_weights_sha256"],
                    final_weights_sha256=training["final_weights_sha256"],
                )
        else:
            variant_row.update(
                scores_compressed_bytes=prediction["scores"]["bytes"],
                scores_uncompressed_bytes=prediction["scores"]["uncompressed_bytes"],
            )
        variants.append(variant_row)
    hybrid_path = _required_phase_path(root, successes, "hybrid", "fit")
    hybrid_value, _ = read_json(hybrid_path / "model-manifest.json")
    hybrid_manifest = cast("ModelManifest", hybrid_value)
    if hybrid_manifest["producer"]["execution"]["backend"] != "cpu":
        _fail("research_campaign_component_invalid")
    for part, variant in (("content", "content-mean"), ("svd", "puresvd")):
        origin = _required_phase_path(root, successes, variant, "fit")
        source_value, raw = read_json(origin / "model-manifest.json")
        source = cast("ModelManifest", source_value)
        components = cast("dict[str, dict[str, str]]", hybrid_manifest["metadata"]["components"])
        if components[part]["identity"] != _digest(raw):
            _fail("research_campaign_component_invalid")
        if not _bytes_equal(
            origin / "model-manifest.json", hybrid_path / (part + "_manifest.json")
        ):
            _fail("research_campaign_component_invalid")
        prefix = "content_" if part == "content" else "svd_"
        for logical in source["files"]:
            if not _bytes_equal(origin / logical, hybrid_path / (prefix + logical)):
                _fail("research_campaign_component_invalid")
    reference_users, reference_cells = _primary(all_rows["popularity"])
    main_users = len(reference_users)
    for variant in names[1:]:
        users, cells = _primary(all_rows[variant])
        if users != reference_users or any(
            cells[user][arm][0] != reference_cells[user][arm][0]
            for user in users
            for arm in reference_cells[user]
        ):
            _fail("research_campaign_unpaired")
    primary = paired = None
    if main_users >= 20:
        matrices: dict[str, NDArray[np.float64]] = {}
        for variant in names:
            _, cells = _primary(all_rows[variant])
            matrices[variant] = np.asarray(
                [
                    [
                        sum(cells[user][(f"random-liked-{order}", n)][1] for order in (1, 2, 3)) / 3
                        for n in range(1, 11)
                    ]
                    for user in reference_users
                ],
                dtype=np.float64,
            )
        primary = {
            variant: {
                "users": main_users,
                "mean_ndcg5_by_n": [float(x) for x in matrix.mean(axis=0)],
            }
            for variant, matrix in matrices.items()
        }
        paired = {}
        comparison = protocol["comparison"]
        for variant in names[1:]:
            delta = matrices[variant] - matrices["popularity"]
            sample = _bootstrap(delta, comparison["seed"], comparison["bootstrap_replicates"])
            if sample is None:
                _fail("research_campaign_unpaired")
            paired[variant] = {
                "users": main_users,
                "mean_difference_by_n": [float(x) for x in delta.mean(axis=0)],
                "pointwise_ci95_by_n": [
                    [
                        _quantile(sample[:, index], 0.025, "lower"),
                        _quantile(sample[:, index], 0.975, "higher"),
                    ]
                    for index in range(10)
                ],
                "bootstrap_replicates": comparison["bootstrap_replicates"],
                "bootstrap_seed": comparison["seed"],
            }
    return {
        "variants": variants,
        "execution_config_count": plan["execution_config_count"],
        "planned_nodes": len(_expected_nodes(plan)),
        "actual_completed_model_packages": sum(phase == "fit" for _variant, phase in successes),
        "actual_attempts_started": sum(attempts.values()),
        "actual_attempts_completed": sum(
            event["event"] == "attempt_finished" for event in journal_events
        ),
        "primary_users": main_users,
        "primary_ndcg5": primary,
        "paired_vs_popularity": paired,
        "small_cohort_suppressed": main_users < 20,
    }


def validate_campaign(
    prepared_dir: str | Path,
    campaign_dir: str | Path,
    *,
    expected_prepared_identity: str,
    expected_campaign_identity: str,
) -> dict[str, object]:
    """Revalidate retained artifacts and published anonymous aggregates."""
    from recsys.research.artifacts import safe_path, separate

    prepared_dir = Path(prepared_dir).absolute()
    root = Path(campaign_dir).absolute()
    separate(prepared_dir, root)
    safe_path(prepared_dir, directory=True)
    safe_path(root, directory=True)
    plan_value, raw = _read_private(root / "campaign-plan.json")
    _validate_plan_structure(plan_value)
    plan = cast("CampaignPlan", plan_value)
    if plan.get("campaign_id") == NATIVE_CAMPAIGN:
        return _validate_native_campaign(
            prepared_dir,
            root,
            plan,
            raw,
            expected_prepared_identity=expected_prepared_identity,
            expected_campaign_identity=expected_campaign_identity,
        )
    if (
        _digest(raw) != expected_campaign_identity
        or plan["prepared_identity"] != expected_prepared_identity
        or plan["producer"] != _source_receipt()
        or tuple(entry["variant_id"] for entry in plan["variants"]) != VARIANTS
    ):
        _fail("research_campaign_plan_mismatch")
    journal = _Journal(root / "journal.jsonl")
    successes, attempts, charged, starts, failed = _successes(journal.events)
    if starts or failed or set(successes) != _expected_nodes():
        _fail("research_campaign_incomplete")
    for (variant, phase), event in successes.items():
        identity, _ = _verify_phase(
            root,
            plan,
            successes,
            variant,
            phase,
            event["attempt"],
            event["generation"],
            prepared_dir,
        )
        if identity != event["output_identity"]:
            _fail("research_campaign_chain_invalid")
    projection = _public_projection(root, plan, successes, prepared_dir)
    report_value, report_raw = _read_private(root / "campaign-report.json")
    if type(report_value) is not dict:
        _fail("research_campaign_report_invalid")
    report = cast("dict[str, object]", report_value)
    manifest, _ = _read_private(root / "campaign-report-manifest.json")
    expected_fields = {
        "schema_version": 2,
        "kind": "arb-ml02-campaign-report",
        "plan_identity": expected_campaign_identity,
        "prepared_identity": expected_prepared_identity,
        "status": "complete",
        "failure_code": None,
        "diagnostics": {
            "single_query_latency": "not_measured",
            "batch_throughput": "not_measured",
            "reason": "standalone_benchmark_not_scheduled",
        },
    }
    if (
        any(report.get(key) != value for key, value in expected_fields.items())
        or any(report.get(key) != value for key, value in projection.items())
        or set(report) != {*expected_fields, *projection}
        or manifest
        != {
            "schema_version": 1,
            "kind": "arb-ml02-campaign-report-manifest",
            "plan_identity": expected_campaign_identity,
            "status": "complete",
            "report": {"bytes": len(report_raw), "sha256": hashlib.sha256(report_raw).hexdigest()},
        }
    ):
        _fail("research_campaign_report_invalid")
    return report


def _native_public_allowed(projection: object, plan: CampaignPlan) -> dict[str, object]:
    """Reject unexpected projection fields before publishing anonymous output."""
    if type(projection) is not dict:
        _fail("research_campaign_report_invalid")
    value = cast("dict[str, object]", projection)
    wanted = {
        "variants",
        "execution_config_count",
        "planned_nodes",
        "actual_completed_model_packages",
        "actual_attempts_started",
        "actual_attempts_completed",
        "primary_users",
        "primary_ndcg5",
        "paired_vs_popularity",
        "small_cohort_suppressed",
    }
    if set(value) != wanted or value["execution_config_count"] != 8 or value["planned_nodes"] != 32:
        _fail("research_campaign_report_invalid")
    variants = value["variants"]
    if type(variants) is not list or len(variants) != 8:
        _fail("research_campaign_report_invalid")
    if [item.get("variant_id") for item in variants if type(item) is dict] != [
        entry["variant_id"] for entry in plan["variants"]
    ]:
        _fail("research_campaign_report_invalid")
    common_variant_fields = {
        "variant_id",
        "backend",
        "model_identity",
        "prediction_identity",
        "evaluation_identity",
        "replay_identity",
        "evaluation_replay_bytes_equal",
        "queries",
        "scores_stored_bytes",
        "scores_decoded_bytes",
        "score_format",
        "dense_rows",
        "query_block_count",
        "resources",
        "fit_budget",
        "checked_groups",
        "framework",
        "train_device",
        "inference_device",
        "model_kind",
        "config_identity",
        "data_identity",
        "source_identity",
        "evaluation_budget",
        "attempts_by_phase",
    }
    native_variant_fields = {
        "candidate_resource_identity",
        "weights_sha256",
        "framework_version",
        "fit_runtime_actual_device",
        "fit_cuda_peak_allocated_bytes",
        "fit_cuda_peak_reserved_bytes",
        "fit_examples",
        "fit_optimizer_steps",
        "fit_supported_items",
        "initial_weights_sha256",
        "final_weights_sha256",
    }
    for item in variants:
        if type(item) is not dict or set(item) != common_variant_fields | (
            native_variant_fields if item.get("variant_id") == NATIVE_VARIANT else set()
        ):
            _fail("research_campaign_report_invalid")
        if (
            item.get("backend") != "cpu"
            or item.get("evaluation_replay_bytes_equal") is not True
            or type(item.get("queries")) is not int
            or item["queries"] < 1
            or type(item.get("scores_stored_bytes")) is not int
            or type(item.get("scores_decoded_bytes")) is not int
            or item.get("score_format") != "arb-score-blocks-v1"
            or type(item.get("dense_rows")) is not int
            or type(item.get("query_block_count")) is not int
            or item["scores_stored_bytes"]
            > plan["storage"]["prediction_upper_bytes"][item["variant_id"]]
            or type(item.get("resources")) is not dict
            or set(item["resources"]) != set(PHASES)
        ):
            _fail("research_campaign_report_invalid")
        for phase_resource in item["resources"].values():
            if type(phase_resource) is not dict or set(phase_resource) != {
                "elapsed_seconds",
                "sampled_peak_rss_bytes",
                "wait4_peak_rss_bytes",
                "observed_peak_rss_bytes",
                "rss_samples",
                "max_sample_gap_seconds",
                "verification",
            }:
                _fail("research_campaign_report_invalid")
            verified = phase_resource["verification"]
            if type(verified) is not dict or set(verified) != set(phase_resource) - {
                "verification"
            }:
                _fail("research_campaign_report_invalid")
    for key in (
        "actual_completed_model_packages",
        "actual_attempts_started",
        "actual_attempts_completed",
        "primary_users",
    ):
        if type(value.get(key)) is not int or cast("int", value[key]) < 0:
            _fail("research_campaign_report_invalid")
    if (
        value["actual_completed_model_packages"] != 8
        or cast("int", value["actual_attempts_started"]) < 32
    ):
        _fail("research_campaign_report_invalid")
    if value["small_cohort_suppressed"] is not (cast("int", value["primary_users"]) < 20):
        _fail("research_campaign_report_invalid")
    if cast("int", value["primary_users"]) < 20:
        if value["primary_ndcg5"] is not None or value["paired_vs_popularity"] is not None:
            _fail("research_campaign_report_invalid")
    raw = _canonical(value)
    for forbidden in (
        b"user_key",
        b"query_id",
        b"seed_mal_ids",
        b"request_token",
        b"profile",
        b"private_path",
    ):
        if forbidden in raw:
            _fail("research_campaign_report_invalid")
    return value


def _check_native_report_contents(
    root: Path, plan: CampaignPlan, projection: dict[str, object], plan_identity: str
) -> str:
    """Verifier-side exact public allowlist and published descriptor check."""
    allowed = _native_public_allowed(projection, plan)
    report_value, report_raw = _read_native_public(root / "campaign-report.json")
    if type(report_value) is not dict:
        _fail("research_campaign_report_invalid")
    report = cast("dict[str, object]", report_value)
    expected = {
        "schema_version": 4,
        "kind": "arb-ml02-campaign-report",
        "campaign_id": NATIVE_CAMPAIGN,
        "plan_identity": plan_identity,
        "prepared_identity": plan["prepared_identity"],
        "status": "complete",
        "failure_code": None,
        "diagnostics": {
            "single_query_latency": "not_measured",
            "batch_throughput": "not_measured",
            "reason": "standalone_benchmark_not_scheduled",
        },
    }
    if (
        set(report)
        != {*allowed, *expected, "storage", "aggregate_peak_rss_bytes", "projection_resources"}
        or any(report.get(key) != value for key, value in {**allowed, **expected}.items())
        or type(report["aggregate_peak_rss_bytes"]) is not int
        or not 0
        <= report["aggregate_peak_rss_bytes"]
        <= plan["execution_profile"]["aggregate_rss_limit_bytes"]
    ):
        _fail("research_campaign_report_invalid")
    storage = report["storage"]
    policy = plan["storage"]["policy"]
    if (
        type(storage) is not dict
        or set(storage)
        != {
            "policy",
            "catalog_count",
            "query_count",
            "zero_seed_queries",
            "prediction_upper_bytes",
            "fresh_admission_bytes",
            "minimum_output_available_bytes",
            "minimum_physical_available_bytes",
            "reserve_bytes",
            "stop_margin_bytes",
            "estimated_other_bytes",
        }
        or storage["policy"] != policy["kind"]
        or storage["reserve_bytes"] != policy["reserve_bytes"]
        or storage["stop_margin_bytes"] != policy["stop_margin_bytes"]
        or storage["catalog_count"] != plan["storage"]["prediction_dimensions"]["catalog_count"]
        or storage["query_count"] != plan["storage"]["prediction_dimensions"]["query_count"]
        or storage["zero_seed_queries"]
        != plan["storage"]["prediction_dimensions"]["zero_seed_queries"]
        or storage["prediction_upper_bytes"] != plan["storage"]["prediction_upper_bytes"]
        or storage["fresh_admission_bytes"] != plan["storage"]["fresh_admission_bytes"]
        or storage["estimated_other_bytes"] != policy["estimated_other_bytes"]
        or any(
            type(storage[key]) is not int or storage[key] < 0
            for key in ("minimum_output_available_bytes", "minimum_physical_available_bytes")
        )
    ):
        _fail("research_campaign_report_invalid")
    resources = report["projection_resources"]
    if (
        type(resources) is not dict
        or set(resources)
        != {
            "outcome",
            "returncode",
            "elapsed_seconds",
            "sampled_peak_rss_bytes",
            "wait4_peak_rss_bytes",
            "observed_peak_rss_bytes",
            "rss_samples",
            "max_sample_gap_seconds",
            "cleanup_complete",
        }
        or resources["outcome"] != "ok"
        or resources["returncode"] != 0
        or resources["cleanup_complete"] is not True
    ):
        _fail("research_campaign_report_invalid")
    manifest, _ = _read_private(root / "campaign-report-manifest.json", limit=4096)
    if manifest != {
        "schema_version": 1,
        "kind": "arb-ml02-campaign-report-manifest",
        "plan_identity": plan_identity,
        "status": "complete",
        "report": {"bytes": len(report_raw), "sha256": hashlib.sha256(report_raw).hexdigest()},
    }:
        _fail("research_campaign_report_invalid")
    journal = _Journal(root / "journal.jsonl")
    current_generation, _ = _read_private(root / "generation.json", limit=4096)
    latest_run = next(
        (event for event in reversed(journal.events) if event["event"] == "run_started"), None
    )
    latest_finish = next(
        (event for event in reversed(journal.events) if event["event"] == "campaign_finished"),
        None,
    )
    if (
        type(current_generation) is not dict
        or latest_run is None
        or latest_finish is None
        or latest_run.get("generation") != current_generation.get("generation")
        or latest_finish.get("generation") != current_generation.get("generation")
        or latest_finish["seq"] <= latest_run["seq"]
        or latest_finish.get("status") != "complete"
        or latest_finish.get("code") is not None
    ):
        _fail("research_campaign_report_invalid")
    return _digest(report_raw)


def _audit_worker_env() -> dict[str, str]:
    """Constrain the read-only verifier to the same CPU-only worker environment."""
    environment = dict(os.environ)
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        environment[name] = "1"
    environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    environment["CUDA_VISIBLE_DEVICES"] = ""
    return environment


def _audit_parent_rss_bytes() -> int:
    try:
        pages = int(Path("/proc/self/statm").read_text().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        _fail("research_campaign_rss_probe_failed")


def _validate_native_campaign(
    prepared_dir: Path,
    root: Path,
    plan: CampaignPlan,
    plan_raw: bytes,
    *,
    expected_prepared_identity: str,
    expected_campaign_identity: str,
) -> dict[str, object]:
    """Read bounded public JSON; verify heavy retained artifacts in a child."""
    from recsys.research.resources import ResourceSupervisor

    if (
        _digest(plan_raw) != expected_campaign_identity
        or plan["prepared_identity"] != expected_prepared_identity
        or plan["producer"] != _source_receipt(native=True)
    ):
        _fail("research_campaign_plan_mismatch")
    lock = _touch_lock(root / ".campaign.lock")
    try:
        try:
            fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            _fail("research_campaign_busy")
        generation_value, _ = _read_private(root / "generation.json", limit=4096)
        if (
            type(generation_value) is not dict
            or type(generation_value.get("generation")) is not str
        ):
            _fail("research_campaign_generation_stale")
        generation = cast("str", generation_value["generation"])
        journal = _Journal(root / "journal.jsonl")
        directory = root / "validations" / uuid.uuid4().hex
        _ensure_private(directory)
        request = {
            "schema_version": 1,
            "kind": "arb-ml02-campaign-verification-request",
            "operation": "validate-report",
            "root": str(root),
            "plan_identity": expected_campaign_identity,
            "generation": generation,
            "prepared_dir": str(prepared_dir),
            "journal_identity": _digest(journal.path.read_bytes()),
            "validation_id": directory.name,
        }
        request_path = directory / "request.json"
        _private_file(request_path, _canonical(request))
        with ResourceSupervisor(
            rss_limit_bytes=plan["execution_profile"]["aggregate_rss_limit_bytes"]
        ) as supervisor:
            handle = supervisor.start(
                [
                    str(Path(sys.executable).absolute()),
                    "-I",
                    "-m",
                    "recsys.research.campaign.verification",
                    "--verification-request",
                    str(request_path),
                ],
                cwd=root,
                log_dir=directory / "logs",
                remaining_seconds=3600.0,
                env=_audit_worker_env(),
            )
            while True:
                snapshot = supervisor.poll(wait_seconds=0.1)
                if (
                    snapshot.sampled_rss_bytes + _audit_parent_rss_bytes()
                    > plan["execution_profile"]["aggregate_rss_limit_bytes"]
                ):
                    supervisor.cancel(handle)
                result = supervisor.result(handle)
                if result is not None and result.cleanup_complete:
                    if handle not in snapshot.completed:
                        _fail("research_campaign_supervisor_invalid")
                    break
            if result.outcome != "ok" or result.returncode != 0:
                _fail("research_campaign_verification_failed")
        receipt, _ = _read_private(directory / "result.json", limit=4096)
        report_value, report_raw = _read_native_public(root / "campaign-report.json")
        if (
            type(receipt) is not dict
            or receipt.get("operation") != "validate-report"
            or receipt.get("plan_identity") != expected_campaign_identity
            or receipt.get("request_identity") != _digest(_canonical(request))
            or receipt.get("projection_identity") != _digest(report_raw)
            or type(report_value) is not dict
        ):
            _fail("research_campaign_report_invalid")
        return cast("dict[str, object]", report_value)
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        os.close(lock)
