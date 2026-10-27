"""Fixed, supervised and resumable validation-only ML-02 campaign.

The worker entry uses only the standard library until the generation gate has
been acquired and checked. All expensive scientific work runs in workers.
"""

from __future__ import annotations

import importlib.metadata
import os
import platform
import re
import stat
from datetime import datetime, timedelta
from pathlib import Path
from typing import NotRequired, TypedDict, cast

from recsys.contracts.provenance import SOURCE_LAYOUT, source_fingerprints
from recsys.research.artifacts import FileDescriptor, StreamDescriptor, check_descriptor, read_json
from recsys.research.campaign.common import (
    CAMPAIGN_SECONDS,
    EVALUATION_SECONDS,
    FIT_SECONDS,
    HYBRID_COMPONENT_VARIANTS,
    PHASES,
    RSS_LIMIT_BYTES,
    VARIANTS,
    _fail,
    _iso,
    _parse_iso,
)
from recsys.research.campaign.spec import (
    LEGACY_CAMPAIGN,
    NATIVE_CAMPAIGN,
    NATIVE_VARIANT,
    CampaignName,
    ExecutionProfile,
    NativeTrainDevice,
    StoragePolicy,
    campaign_variants,
    native_campaign_spec,
)
from recsys.research.campaign.storage import admission_bytes, prediction_bounds
from recsys.research.score_blocks import ScoreDimensions
from recsys.research.training.candidate import load_candidate
from recsys.research.training.spec import DeviceSpec, DeviceWire, TrainingSpecWire


class CampaignVariant(TypedDict):
    variant_id: str
    parameters: object
    backend: str
    components: NotRequired[list[dict[str, str]]]
    model_kind: NotRequired[str]
    framework: NotRequired[str]
    train_device: NotRequired[DeviceWire]
    inference_device: NotRequired[DeviceWire]
    candidate_resource_identity: NotRequired[str]
    training_spec: NotRequired[TrainingSpecWire]


class CampaignScheduling(TypedDict):
    cpu_workers: int
    blas_threads: int


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
    execution_config_count: int
    assets: dict[str, FileDescriptor]
    budgets: dict[str, float | int]
    scheduling: CampaignScheduling
    score_retention: str
    created_utc: str
    deadline_utc: str
    campaign_id: NotRequired[CampaignName]
    execution_profile: NotRequired[ExecutionProfile]
    candidate_resource_identity: NotRequired[str]
    storage: NotRequired[CampaignStorage]


class CampaignStorage(TypedDict):
    physical_storage_root: str
    output_filesystem_device: int
    physical_filesystem_device: int
    policy: StoragePolicy
    prediction_dimensions: PredictionDimensionReceipt
    prediction_upper_bytes: dict[str, int]
    fresh_admission_bytes: int


class PredictionDimensionReceipt(TypedDict):
    catalog_count: int
    query_count: int
    zero_seed_queries: int
    query_descriptor: StreamDescriptor


def _expected_nodes(plan: CampaignPlan | None = None) -> set[tuple[str, str]]:
    """Use the verified plan's arm set; no argument retains the v1 test API."""
    variants = VARIANTS if plan is None else tuple(item["variant_id"] for item in plan["variants"])
    return {(variant, phase) for variant in variants for phase in PHASES}


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
        all(type(key) is str and _descriptor(item) for key, item in assets.items())
        and all(type(key) is str and type(item) in (int, float) for key, item in budgets.items())
        and all(type(scheduling.get(key)) is int for key in ("cpu_workers", "blas_threads"))
    )


def _validate_plan_structure(plan: object) -> None:
    if type(plan) is dict and plan.get("schema_version") == 5:
        _validate_native_plan_structure(plan)
        return
    if type(plan) is dict and plan.get("schema_version") == 4:
        _fail("research_campaign_profile_retired")
    if (
        type(plan) is not dict
        or type(plan.get("schema_version")) is not int
        or plan["schema_version"] != 3
        or type(plan.get("producer")) is not dict
        or plan["producer"].get("source_layout") != SOURCE_LAYOUT
    ):
        _fail("research_source_layout_unsupported")
    from recsys.research.models.spec import campaign

    spec, campaign_sha = campaign()
    if (
        type(plan) is not dict
        or plan.get("schema_version") != 3
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
    expected_components = [
        {"variant_id": variant, "phase": "fit"} for variant in HYBRID_COMPONENT_VARIANTS
    ]
    expected_variants = [
        {
            "variant_id": name,
            "parameters": spec["variants"][name],
            "backend": "cpu",
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
        or plan.get("execution_config_count") != 7
        or plan.get("budgets") != expected_budgets
        or type(schedule) is not dict
        or schedule.get("cpu_workers") not in (1, 2)
        or schedule
        != {
            "cpu_workers": schedule["cpu_workers"],
            "blas_threads": 1,
        }
        or set(plan)
        != {
            "schema_version",
            "kind",
            "prepared_identity",
            "protocol_identity",
            "fit_identity",
            "catalog_descriptor",
            "train_interactions",
            "train_stats",
            "campaign_spec_sha256",
            "code_revision",
            "producer",
            "variants",
            "execution_config_count",
            "assets",
            "budgets",
            "scheduling",
            "score_retention",
            "created_utc",
            "deadline_utc",
        }
        or _parse_iso(plan["deadline_utc"]) - _parse_iso(plan["created_utc"])
        != timedelta(seconds=CAMPAIGN_SECONDS)
    ):
        _fail("research_campaign_plan_mismatch")


def _source_receipt(*, native: bool = False) -> SourceReceipt:
    names = source_fingerprints()
    dependencies: tuple[str, ...] = ("numpy", "scipy", "scikit-learn", "threadpoolctl")
    if native:
        dependencies += ("torch", "safetensors")
    try:
        versions = {name: importlib.metadata.version(name) for name in dependencies}
    except importlib.metadata.PackageNotFoundError:
        _fail("research_campaign_framework_unavailable")
    return {
        "source_layout": SOURCE_LAYOUT,
        "code_files_sha256": names,
        "python_version": platform.python_version(),
        "package_versions": versions,
    }


def _native_runtime_valid(receipt: SourceReceipt, train_device: NativeTrainDevice) -> bool:
    versions = receipt["package_versions"]
    torch = versions.get("torch")
    return (
        versions.get("numpy") == "2.5.3"
        and versions.get("safetensors") == "0.8.0"
        and (
            torch == "2.13.0+cu129"
            if train_device == "cuda:0"
            else torch in ("2.13.0+cpu", "2.13.0+cu129")
        )
    )


def _plan(
    prepared_dir: str | Path,
    expected_prepared_identity: str,
    code_revision: str,
    glove: str | Path,
    cpu_workers: int,
    created: datetime,
    *,
    campaign_name: CampaignName = LEGACY_CAMPAIGN,
    native_train_device: NativeTrainDevice | None = None,
    physical_storage_root: Path | None = None,
    output_root: Path | None = None,
) -> CampaignPlan:
    from recsys.release.lock import release_info
    from recsys.research.artifacts import file_descriptor
    from recsys.research.models.inputs import read_scoring_inputs
    from recsys.research.models.spec import campaign

    if campaign_name == LEGACY_CAMPAIGN and (
        native_train_device is not None or physical_storage_root is not None
    ):
        _fail("research_campaign_profile_unsupported")
    if campaign_name == "ml02-native-v2":
        _fail("research_campaign_profile_retired")
    if campaign_name not in (LEGACY_CAMPAIGN, NATIVE_CAMPAIGN):
        _fail("research_campaign_profile_unsupported")
    if (
        type(code_revision) is not str
        or re.fullmatch(r"[0-9a-f]{40}", code_revision) is None
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
    variants: list[CampaignVariant] = []
    for name in VARIANTS:
        item: CampaignVariant = {
            "variant_id": name,
            "parameters": spec["variants"][name],
            "backend": "cpu",
        }
        if name == "hybrid":
            item["components"] = [
                {"variant_id": variant, "phase": "fit"} for variant in HYBRID_COMPONENT_VARIANTS
            ]
        variants.append(item)
    legacy_plan: CampaignPlan = {
        "schema_version": 3,
        "kind": "arb-ml02-campaign-plan",
        "prepared_identity": inputs.prepared_identity,
        "protocol_identity": inputs.protocol_identity,
        "fit_identity": inputs.fit_identity,
        "catalog_descriptor": inputs.catalog_descriptor,
        "train_interactions": inputs.train_interactions_descriptor,
        "train_stats": inputs.train_stats_descriptor,
        "campaign_spec_sha256": campaign_sha,
        "code_revision": code_revision,
        "producer": _source_receipt(),
        "variants": variants,
        "execution_config_count": 7,
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
            "blas_threads": 1,
        },
        "score_retention": "keep",
        "created_utc": _iso(created),
        "deadline_utc": _iso(created + timedelta(seconds=CAMPAIGN_SECONDS)),
    }
    if campaign_name == LEGACY_CAMPAIGN:
        return legacy_plan
    from recsys.research.predict import prediction_dimensions

    prep, _ = read_json(Path(prepared_dir) / "prepare-manifest.json")
    if type(prep) is not dict or type(prep.get("files")) is not dict:
        _fail("research_campaign_plan_mismatch")
    query_desc = cast("StreamDescriptor", prep["files"]["development/queries.jsonl.gz"])
    dims = prediction_dimensions(Path(prepared_dir), query_desc, inputs.mal_ids)
    return _native_plan(
        legacy_plan,
        native_train_device=native_train_device,
        physical_storage_root=physical_storage_root,
        output_root=output_root,
        dimensions=dims,
        query_descriptor=query_desc,
    )


def _native_plan(
    legacy_plan: CampaignPlan,
    *,
    native_train_device: NativeTrainDevice | None,
    physical_storage_root: Path | None,
    output_root: Path | None,
    dimensions: ScoreDimensions,
    query_descriptor: StreamDescriptor,
) -> CampaignPlan:
    spec, spec_identity = native_campaign_spec()
    selected = (
        spec["native_execution"]["default_train_device"]
        if native_train_device is None
        else native_train_device
    )
    if selected not in spec["native_execution"]["allowed_train_devices"]:
        _fail("research_campaign_profile_unsupported")
    if physical_storage_root is None or output_root is None:
        _fail("research_campaign_storage_root_required")
    physical = Path(physical_storage_root)
    output = Path(output_root)
    if not physical.is_absolute() or not output.is_absolute():
        _fail("research_campaign_storage_root_invalid")
    try:
        physical_info = os.stat(physical, follow_symlinks=False)
        output_info = os.stat(output, follow_symlinks=False)
        if (
            not stat.S_ISDIR(physical_info.st_mode)
            or not stat.S_ISDIR(output_info.st_mode)
            or physical.resolve(strict=True) != physical
            or output.resolve(strict=True) != output
        ):
            _fail("research_campaign_storage_root_invalid")
    except OSError:
        _fail("research_campaign_storage_root_invalid")
    cpu = DeviceSpec("cpu", None)
    train = cpu if selected == "cpu" else DeviceSpec("cuda", 0)
    training_spec = load_candidate(train_device=train, inference_device=cpu)
    producer = _source_receipt(native=True)
    if not _native_runtime_valid(producer, selected):
        _fail("research_campaign_framework_unsupported")
    variants: list[CampaignVariant] = []
    for entry in legacy_plan["variants"]:
        variants.append(
            {
                **entry,
                "model_kind": "baseline",
                "framework": spec["baseline_execution"]["framework"],
                "train_device": cpu.wire(),
                "inference_device": cpu.wire(),
            }
        )
    variants.append(
        {
            "variant_id": NATIVE_VARIANT,
            "parameters": {},
            "backend": "cpu",
            "model_kind": "native-seed",
            "framework": spec["native_execution"]["framework"],
            "train_device": train.wire(),
            "inference_device": cpu.wire(),
            "candidate_resource_identity": spec["candidate_resource_identity"],
            "training_spec": training_spec.wire(),
        }
    )
    upper = prediction_bounds(dimensions, campaign_variants(NATIVE_CAMPAIGN))
    return {
        **legacy_plan,
        "schema_version": 5,
        "campaign_id": NATIVE_CAMPAIGN,
        "campaign_spec_sha256": spec_identity,
        "candidate_resource_identity": spec["candidate_resource_identity"],
        "producer": producer,
        "variants": variants,
        "execution_config_count": len(campaign_variants(NATIVE_CAMPAIGN)),
        "execution_profile": spec["execution_profile"],
        "storage": {
            "physical_storage_root": str(physical),
            "output_filesystem_device": output_info.st_dev,
            "physical_filesystem_device": physical_info.st_dev,
            "policy": spec["storage_policy"],
            "prediction_dimensions": {
                "catalog_count": dimensions.catalog_count,
                "query_count": dimensions.query_count,
                "zero_seed_queries": dimensions.zero_seed_queries,
                "query_descriptor": query_descriptor,
            },
            "prediction_upper_bytes": upper,
            "fresh_admission_bytes": admission_bytes(set(upper), upper),
        },
    }


def _validate_native_plan_structure(plan: dict[str, object]) -> None:
    spec, spec_identity = native_campaign_spec()
    if type(plan.get("schema_version")) is not int or not _plan_fields_valid(plan):
        _fail("research_campaign_plan_mismatch")
    cast_plan = cast("CampaignPlan", plan)
    if (
        cast_plan.get("kind") != "arb-ml02-campaign-plan"
        or cast_plan.get("campaign_id") != NATIVE_CAMPAIGN
        or cast_plan.get("campaign_spec_sha256") != spec_identity
        or cast_plan.get("candidate_resource_identity") != spec["candidate_resource_identity"]
        or cast_plan.get("execution_profile") != spec["execution_profile"]
        or cast_plan.get("execution_config_count") != 8
        or cast_plan.get("score_retention") != "keep"
        or cast_plan.get("budgets")
        != {
            "fit_seconds_per_variant": FIT_SECONDS,
            "evaluation_seconds_per_variant": EVALUATION_SECONDS,
            "campaign_wall_seconds": CAMPAIGN_SECONDS,
            "sampled_rss_limit_bytes": RSS_LIMIT_BYTES,
            "score_stream_limit_bytes": 64 * 1024**3,
            "score_row_limit_bytes": 4 * 1024**2,
        }
        or cast_plan["producer"].get("source_layout") != SOURCE_LAYOUT
        or set(cast_plan["producer"]["package_versions"])
        != {"numpy", "scipy", "scikit-learn", "threadpoolctl", "torch", "safetensors"}
        or set(cast_plan)
        != {
            "schema_version",
            "kind",
            "prepared_identity",
            "protocol_identity",
            "fit_identity",
            "catalog_descriptor",
            "train_interactions",
            "train_stats",
            "campaign_spec_sha256",
            "code_revision",
            "producer",
            "variants",
            "execution_config_count",
            "assets",
            "budgets",
            "scheduling",
            "score_retention",
            "created_utc",
            "deadline_utc",
            "campaign_id",
            "execution_profile",
            "candidate_resource_identity",
            "storage",
        }
        or _parse_iso(cast_plan["deadline_utc"]) - _parse_iso(cast_plan["created_utc"])
        != timedelta(seconds=CAMPAIGN_SECONDS)
    ):
        _fail("research_campaign_plan_mismatch")
    storage = cast_plan.get("storage")
    if (
        type(storage) is not dict
        or set(storage)
        != {
            "physical_storage_root",
            "output_filesystem_device",
            "physical_filesystem_device",
            "policy",
            "prediction_dimensions",
            "prediction_upper_bytes",
            "fresh_admission_bytes",
        }
        or type(storage.get("physical_storage_root")) is not str
        or not Path(storage["physical_storage_root"]).is_absolute()
        or type(storage.get("output_filesystem_device")) is not int
        or type(storage.get("physical_filesystem_device")) is not int
        or storage.get("policy") != spec["storage_policy"]
    ):
        _fail("research_campaign_plan_mismatch")
    dim = storage["prediction_dimensions"]
    if (
        type(dim) is not dict
        or set(dim) != {"catalog_count", "query_count", "zero_seed_queries", "query_descriptor"}
        or any(
            type(dim[key]) is not int
            for key in ("catalog_count", "query_count", "zero_seed_queries")
        )
    ):
        _fail("research_campaign_plan_mismatch")
    check_descriptor(dim["query_descriptor"], stream=True)
    try:
        dimensions = ScoreDimensions(
            dim["catalog_count"], dim["query_count"], dim["zero_seed_queries"]
        )
        upper = prediction_bounds(dimensions, campaign_variants(NATIVE_CAMPAIGN))
    except (ValueError, TypeError):
        _fail("research_campaign_plan_mismatch")
    if (
        storage["prediction_upper_bytes"] != upper
        or type(storage["fresh_admission_bytes"]) is not int
        or storage["fresh_admission_bytes"] != admission_bytes(set(upper), upper)
    ):
        _fail("research_campaign_plan_mismatch")
    physical = Path(storage["physical_storage_root"])
    try:
        physical_info = os.stat(physical, follow_symlinks=False)
        if (
            not stat.S_ISDIR(physical_info.st_mode)
            or physical.resolve(strict=True) != physical
            or physical_info.st_dev != storage["physical_filesystem_device"]
        ):
            _fail("research_campaign_plan_mismatch")
    except OSError:
        _fail("research_campaign_plan_mismatch")
    schedule = cast_plan["scheduling"]
    if (
        type(schedule) is not dict
        or schedule.get("cpu_workers") not in (1, 2)
        or schedule != {"cpu_workers": schedule["cpu_workers"], "blas_threads": 1}
    ):
        _fail("research_campaign_plan_mismatch")
    from recsys.research.models.spec import campaign

    baseline, _ = campaign()
    cpu = DeviceSpec("cpu", None)
    expected: list[CampaignVariant] = []
    for name in VARIANTS:
        entry: CampaignVariant = {
            "variant_id": name,
            "parameters": baseline["variants"][name],
            "backend": "cpu",
            "model_kind": "baseline",
            "framework": spec["baseline_execution"]["framework"],
            "train_device": cpu.wire(),
            "inference_device": cpu.wire(),
        }
        if name == "hybrid":
            entry["components"] = [
                {"variant_id": item, "phase": "fit"} for item in HYBRID_COMPONENT_VARIANTS
            ]
        expected.append(entry)
    observed = cast_plan["variants"]
    if type(observed) is not list or len(observed) != 8:
        _fail("research_campaign_plan_mismatch")
    selected_device = observed[-1].get("train_device")
    if selected_device == cpu.wire():
        selected = "cpu"
        train = cpu
    elif selected_device == DeviceSpec("cuda", 0).wire():
        selected = "cuda:0"
        train = DeviceSpec("cuda", 0)
    else:
        _fail("research_campaign_plan_mismatch")
    if selected not in spec["native_execution"]["allowed_train_devices"]:
        _fail("research_campaign_plan_mismatch")
    if not _native_runtime_valid(cast_plan["producer"], selected):
        _fail("research_campaign_plan_mismatch")
    expected.append(
        {
            "variant_id": NATIVE_VARIANT,
            "parameters": {},
            "backend": "cpu",
            "model_kind": "native-seed",
            "framework": spec["native_execution"]["framework"],
            "train_device": train.wire(),
            "inference_device": cpu.wire(),
            "candidate_resource_identity": spec["candidate_resource_identity"],
            "training_spec": load_candidate(train_device=train, inference_device=cpu).wire(),
        }
    )
    if observed != expected:
        _fail("research_campaign_plan_mismatch")
