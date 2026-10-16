"""Fixed, supervised and resumable validation-only ML-02 campaign.

The worker entry uses only the standard library until the generation gate has
been acquired and checked. All expensive scientific work runs in workers.
"""

from __future__ import annotations

import importlib.metadata
import platform
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import NotRequired, TypedDict, cast

from recsys.contracts.provenance import SOURCE_LAYOUT, source_fingerprints
from recsys.research.artifacts import FileDescriptor
from recsys.research.campaign.common import (
    AUX_PHASE,
    AUX_VARIANTS,
    CAMPAIGN_SECONDS,
    EVALUATION_SECONDS,
    FIT_SECONDS,
    PHASES,
    RSS_LIMIT_BYTES,
    VARIANTS,
    _fail,
    _iso,
    _parse_iso,
)


class CampaignVariant(TypedDict):
    variant_id: str
    parameters: object
    backend: str
    components: NotRequired[list[dict[str, str]]]


class CampaignScheduling(TypedDict):
    cpu_workers: int
    gpu_workers: int
    blas_threads: int
    gpu_optional_benchmark: str


class SourceReceipt(TypedDict):
    source_layout: str
    code_files_sha256: dict[str, str]
    python_version: str
    package_versions: dict[str, str]


class CampaignPlan(TypedDict):
    schema_version: int
    kind: str
    prepared_identity: str
    protocol_identity: str
    fit_identity: str
    catalog_descriptor: FileDescriptor
    train_interactions: FileDescriptor
    train_stats: FileDescriptor
    campaign_spec_sha256: str
    code_revision: str
    producer: SourceReceipt
    variants: list[CampaignVariant]
    auxiliary_fits: list[dict[str, str]]
    execution_config_count: int
    assets: dict[str, FileDescriptor]
    budgets: dict[str, float | int]
    scheduling: CampaignScheduling
    score_retention: str
    created_utc: str
    deadline_utc: str


def _backend(variant: str, configured: str) -> str:
    return "cpu" if configured == "cpu" or variant in ("popularity", "itemknn", "hybrid") else "gpu"


def _expected_nodes(plan: CampaignPlan) -> set[tuple[str, str]]:
    nodes = {(variant, phase) for variant in VARIANTS for phase in PHASES}
    if plan["scheduling"]["gpu_workers"] == 1 and any(
        entry["backend"] == "gpu" for entry in plan["variants"]
    ):
        nodes.update((variant, AUX_PHASE) for variant in AUX_VARIANTS)
    return nodes


def _component_phase(plan: CampaignPlan) -> str:
    return AUX_PHASE if plan["auxiliary_fits"] else "fit"


def _string_map(value: object) -> bool:
    return type(value) is dict and all(
        type(key) is str and type(item) is str for key, item in value.items()
    )


def _descriptor(value: object) -> bool:
    return (
        type(value) is dict and type(value.get("bytes")) is int and type(value.get("sha256")) is str
    )


def _plan_fields_valid(plan: dict[str, object]) -> bool:
    producer = plan.get("producer")
    variants = plan.get("variants")
    auxiliary = plan.get("auxiliary_fits")
    assets = plan.get("assets")
    budgets = plan.get("budgets")
    scheduling = plan.get("scheduling")
    if (
        type(producer) is not dict
        or not _string_map(producer.get("code_files_sha256"))
        or type(producer.get("python_version")) is not str
        or not _string_map(producer.get("package_versions"))
        or any(
            type(plan.get(key)) is not str
            for key in (
                "kind",
                "prepared_identity",
                "protocol_identity",
                "fit_identity",
                "campaign_spec_sha256",
                "code_revision",
                "score_retention",
                "created_utc",
                "deadline_utc",
            )
        )
        or any(
            not _descriptor(plan.get(key))
            for key in ("catalog_descriptor", "train_interactions", "train_stats")
        )
        or type(plan.get("execution_config_count")) is not int
        or type(variants) is not list
        or type(auxiliary) is not list
        or type(assets) is not dict
        or type(budgets) is not dict
        or type(scheduling) is not dict
    ):
        return False
    if any(
        type(item) is not dict
        or type(item.get("variant_id")) is not str
        or "parameters" not in item
        or type(item.get("backend")) is not str
        or (
            "components" in item
            and (
                type(item["components"]) is not list
                or not all(_string_map(component) for component in item["components"])
            )
        )
        for item in variants
    ):
        return False
    return (
        all(_string_map(item) for item in auxiliary)
        and all(type(key) is str and _descriptor(item) for key, item in assets.items())
        and all(type(key) is str and type(item) in (int, float) for key, item in budgets.items())
        and all(
            type(scheduling.get(key)) is int
            for key in ("cpu_workers", "gpu_workers", "blas_threads")
        )
        and type(scheduling.get("gpu_optional_benchmark")) is str
    )


def _validate_plan_structure(plan: object) -> str:
    if (
        type(plan) is not dict
        or type(plan.get("schema_version")) is not int
        or plan["schema_version"] != 2
        or type(plan.get("producer")) is not dict
        or plan["producer"].get("source_layout") != SOURCE_LAYOUT
    ):
        _fail("research_source_layout_unsupported")
    from recsys.research.models.spec import campaign

    spec, campaign_sha = campaign()
    if (
        type(plan) is not dict
        or plan.get("schema_version") != 2
        or plan.get("kind") != "arb-ml02-campaign-plan"
        or plan.get("campaign_spec_sha256") != campaign_sha
        or type(plan.get("variants")) is not list
        or len(plan["variants"]) != 7
        or plan.get("score_retention") != "keep"
    ):
        _fail("research_campaign_plan_mismatch")
    if not _plan_fields_valid(cast("dict[str, object]", plan)):
        _fail("research_campaign_plan_mismatch")
    plan = cast("CampaignPlan", plan)
    configured = "gpu" if any(item.get("backend") == "gpu" for item in plan["variants"]) else "cpu"
    expected_aux = (
        [
            {
                "variant_id": variant,
                "phase": AUX_PHASE,
                "backend": "cpu",
                "fit_budget_owner": variant,
                "consumer": "hybrid",
            }
            for variant in AUX_VARIANTS
        ]
        if configured == "gpu"
        else []
    )
    expected_components = [
        {"variant_id": variant, "phase": AUX_PHASE if configured == "gpu" else "fit"}
        for variant in AUX_VARIANTS
    ]
    expected_variants = [
        {
            "variant_id": name,
            "parameters": spec["variants"][name],
            "backend": _backend(name, configured),
            **({"components": expected_components} if name == "hybrid" else {}),
        }
        for name in VARIANTS
    ]
    expected_budgets = {
        "fit_seconds_per_variant": FIT_SECONDS,
        "evaluation_seconds_per_variant": EVALUATION_SECONDS,
        "campaign_wall_seconds": CAMPAIGN_SECONDS,
        "sampled_rss_limit_bytes": RSS_LIMIT_BYTES,
        "score_stream_limit_bytes": 64 * 1024**3,
        "score_row_limit_bytes": 4 * 1024**2,
    }
    schedule = plan.get("scheduling")
    if (
        plan["variants"] != expected_variants
        or plan.get("auxiliary_fits") != expected_aux
        or plan.get("execution_config_count") != (9 if configured == "gpu" else 7)
        or plan.get("budgets") != expected_budgets
        or type(schedule) is not dict
        or schedule.get("cpu_workers") not in (1, 2)
        or schedule
        != {
            "cpu_workers": schedule["cpu_workers"],
            "gpu_workers": 1,
            "blas_threads": 1,
            "gpu_optional_benchmark": "not_scheduled",
        }
        or _parse_iso(plan["deadline_utc"]) - _parse_iso(plan["created_utc"])
        != timedelta(seconds=CAMPAIGN_SECONDS)
    ):
        _fail("research_campaign_plan_mismatch")
    return configured


def _source_receipt(backend: str) -> SourceReceipt:
    names = source_fingerprints()
    dependencies: tuple[str, ...] = ("numpy", "scipy", "scikit-learn", "threadpoolctl")
    if backend == "gpu":
        from recsys.research.models.spec import GPU_DEPENDENCIES

        dependencies += GPU_DEPENDENCIES
    return {
        "source_layout": SOURCE_LAYOUT,
        "code_files_sha256": names,
        "python_version": platform.python_version(),
        "package_versions": {name: importlib.metadata.version(name) for name in dependencies},
    }


def _plan(
    prepared_dir: str | Path,
    expected_prepared_identity: str,
    code_revision: str,
    glove: str | Path,
    backend: str,
    cpu_workers: int,
    created: datetime,
) -> CampaignPlan:
    from recsys.release.lock import release_info
    from recsys.research.artifacts import file_descriptor
    from recsys.research.models.inputs import read_scoring_inputs
    from recsys.research.models.spec import campaign

    if (
        type(code_revision) is not str
        or re.fullmatch(r"[0-9a-f]{40}", code_revision) is None
        or backend not in ("cpu", "gpu")
        or type(backend) is not str
        or type(cpu_workers) is not int
        or cpu_workers not in (1, 2)
    ):
        _fail()
    inputs = read_scoring_inputs(
        prepared_dir, expected_prepared_identity=expected_prepared_identity
    )
    spec, campaign_sha = campaign()
    locked = release_info()["inputs"]["embeddings"]
    glove_desc = file_descriptor(glove)
    if glove_desc != {"bytes": locked["text_size"], "sha256": locked["text_sha256"]}:
        _fail()
    auxiliary = (
        [
            {
                "variant_id": variant,
                "phase": AUX_PHASE,
                "backend": "cpu",
                "fit_budget_owner": variant,
                "consumer": "hybrid",
            }
            for variant in AUX_VARIANTS
        ]
        if backend == "gpu"
        else []
    )
    component_phase = AUX_PHASE if auxiliary else "fit"
    variants: list[CampaignVariant] = []
    for name in VARIANTS:
        item: CampaignVariant = {
            "variant_id": name,
            "parameters": spec["variants"][name],
            "backend": _backend(name, backend),
        }
        if name == "hybrid":
            item["components"] = [
                {"variant_id": variant, "phase": component_phase} for variant in AUX_VARIANTS
            ]
        variants.append(item)
    return {
        "schema_version": 2,
        "kind": "arb-ml02-campaign-plan",
        "prepared_identity": inputs.prepared_identity,
        "protocol_identity": inputs.protocol_identity,
        "fit_identity": inputs.fit_identity,
        "catalog_descriptor": inputs.catalog_descriptor,
        "train_interactions": inputs.train_interactions_descriptor,
        "train_stats": inputs.train_stats_descriptor,
        "campaign_spec_sha256": campaign_sha,
        "code_revision": code_revision,
        "producer": _source_receipt(backend),
        "variants": variants,
        "auxiliary_fits": auxiliary,
        "execution_config_count": 9 if backend == "gpu" else 7,
        "assets": {"glove": glove_desc},
        "budgets": {
            "fit_seconds_per_variant": FIT_SECONDS,
            "evaluation_seconds_per_variant": EVALUATION_SECONDS,
            "campaign_wall_seconds": CAMPAIGN_SECONDS,
            "sampled_rss_limit_bytes": RSS_LIMIT_BYTES,
            "score_stream_limit_bytes": 64 * 1024**3,
            "score_row_limit_bytes": 4 * 1024**2,
        },
        "scheduling": {
            "cpu_workers": cpu_workers,
            "gpu_workers": 1,
            "blas_threads": 1,
            "gpu_optional_benchmark": "not_scheduled",
        },
        "score_retention": "keep",
        "created_utc": _iso(created),
        "deadline_utc": _iso(created + timedelta(seconds=CAMPAIGN_SECONDS)),
    }
