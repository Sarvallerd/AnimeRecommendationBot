"""Fixed, supervised and resumable validation-only ML-02 campaign.

The worker entry uses only the standard library until the generation gate has
been acquired and checked. All expensive scientific work runs in workers.
"""

from __future__ import annotations

import hashlib
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
    _fail,
    _read_private,
)
from recsys.research.campaign.journal import CampaignEvent, PhaseKey, _Journal, _successes
from recsys.research.campaign.plan import (
    CampaignPlan,
    _expected_nodes,
    _source_receipt,
    _validate_plan_structure,
)

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

    prepared, protocol, catalog, fit, fit_identity, stats = _prepared(
        prepared_dir, plan["prepared_identity"]
    )
    if fit_identity != plan["fit_identity"] or protocol != load_protocol():
        _fail("research_campaign_input_invalid")
    all_rows: dict[str, EvaluationRows] = {}
    variants: list[dict[str, object]] = []
    journal_events = _Journal(root / "journal.jsonl").events
    _, attempts, charged, _, _ = _successes(journal_events)
    reference_metadata = None
    reference_n0 = None
    for variant in VARIANTS:
        maybe_paths = {phase: _phase_path(root, successes, variant, phase) for phase in PHASES}
        if any(value is None for value in maybe_paths.values()):
            _fail("research_campaign_incomplete")
        paths = cast("dict[str, Path]", maybe_paths)
        model_value, model_raw = read_json(paths["fit"] / "model-manifest.json")
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
            model["variant_id"] != variant
            or prediction["model_id"] != variant
            or prediction["config_sha256"] != hashlib.sha256(model_raw).hexdigest()
            or evaluation["model_id"] != variant
            or evaluation["prediction_identity"] != _digest(prediction_raw)
            or evaluation["config_sha256"] != prediction["config_sha256"]
            or evaluation["prediction_code_revision"] != plan["code_revision"]
            or evaluation["selection_identity"] is not None
            or prediction["split"] != "validation"
            or file_descriptor(paths["predict"] / "scores.jsonl.gz")
            != {key: prediction["scores"][key] for key in ("bytes", "sha256")}
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
        variants.append(
            {
                "variant_id": variant,
                "backend": plan["variants"][VARIANTS.index(variant)]["backend"],
                "model_identity": _digest(model_raw),
                "prediction_identity": _digest(prediction_raw),
                "evaluation_identity": evaluation_identity,
                "replay_identity": _digest(
                    (paths["replay"] / "evaluation-manifest.json").read_bytes()
                ),
                "evaluation_replay_bytes_equal": True,
                "queries": initial_report["queries"],
                "scores_compressed_bytes": prediction["scores"]["bytes"],
                "scores_uncompressed_bytes": prediction["scores"]["uncompressed_bytes"],
                "resources": phase_resources,
                "fit_budget": {
                    "primary_charged_seconds": charged.get((variant, "fit"), 0.0),
                    "all_attempts_charged_seconds": charged.get((variant, "fit"), 0.0),
                    "remaining_seconds": FIT_SECONDS - charged.get((variant, "fit"), 0.0),
                },
                "checked_groups": recomputed,
            }
        )
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
    for variant in VARIANTS[1:]:
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
        for variant in VARIANTS:
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
        for variant in VARIANTS[1:]:
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
        "planned_nodes": len(_expected_nodes()),
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
