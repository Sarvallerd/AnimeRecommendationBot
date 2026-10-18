"""Closed, versioned campaign profiles without importing a training framework."""

from __future__ import annotations

import hashlib
from typing import Literal, TypedDict, cast

from recsys.contracts.bundle import parse_canonical
from recsys.contracts.resources import resource_bytes
from recsys.research.campaign.common import VARIANTS, _fail
from recsys.research.models.spec import campaign as baseline_campaign
from recsys.research.training.candidate import candidate_identity

type CampaignName = Literal["ml02-cpu-v1", "ml02-native-v2"]
type NativeTrainDevice = Literal["cpu", "cuda:0"]

LEGACY_CAMPAIGN: CampaignName = "ml02-cpu-v1"
NATIVE_CAMPAIGN: CampaignName = "ml02-native-v2"
NATIVE_VARIANT = "item-seed-mean-bpr-v1"
NATIVE_RESOURCE = "research_campaign_native_v2.json"
NATIVE_RESOURCE_SHA256 = "4c395e475dc23e1c3faebb0ed0458fceb1bcc3c5d2af43318d0394acd729a13d"


class ExecutionProfile(TypedDict):
    aggregate_rss_limit_bytes: int
    blas_threads: int
    campaign_wall_seconds: int
    cuda_min_free_bytes: int
    evaluation_seconds_per_arm: int
    fit_seconds_per_arm: int
    gpu_fit_workers: int
    max_execution_configurations: int
    max_workers: int
    native_artifact_max_bytes: int
    profile_id: str
    score_batch_rows: int
    score_compressed_limit_bytes: int
    score_retention: str
    score_row_limit_bytes: int
    score_uncompressed_limit_bytes: int


class StoragePolicy(TypedDict):
    admission_is_completion_guarantee: bool
    estimated_other_bytes: int
    estimated_scores_bytes_per_arm: int
    kind: str
    poll_interval_seconds: int
    reserve_bytes: int
    stop_margin_bytes: int


class NativeExecution(TypedDict):
    allowed_train_devices: list[NativeTrainDevice]
    default_train_device: NativeTrainDevice
    framework: str
    inference_device: str


class BaselineExecution(TypedDict):
    framework: str
    inference_device: str
    train_device: str


class NativeCampaignSpec(TypedDict):
    arms: list[str]
    baseline_campaign_sha256: str
    baseline_execution: BaselineExecution
    campaign_id: CampaignName
    candidate_resource_identity: str
    execution_profile: ExecutionProfile
    kind: str
    native_execution: NativeExecution
    schema_version: int
    score_encoding: dict[str, str]
    storage_policy: StoragePolicy


def is_native_variant(variant: str) -> bool:
    return variant == NATIVE_VARIANT


def native_campaign_spec() -> tuple[NativeCampaignSpec, str]:
    """Verify the raw resource and its immutable parent resources before use."""
    try:
        raw = resource_bytes("campaigns", NATIVE_RESOURCE)
        if len(raw) != 1596 or hashlib.sha256(raw).hexdigest() != NATIVE_RESOURCE_SHA256:
            _fail("research_campaign_spec_invalid")
        value = parse_canonical(raw, NATIVE_RESOURCE)
    except (OSError, TypeError, ValueError, UnicodeError):
        _fail("research_campaign_spec_invalid")
    if type(value) is not dict or set(value) != {
        "arms",
        "baseline_campaign_sha256",
        "baseline_execution",
        "campaign_id",
        "candidate_resource_identity",
        "execution_profile",
        "kind",
        "native_execution",
        "schema_version",
        "score_encoding",
        "storage_policy",
    }:
        _fail("research_campaign_spec_invalid")
    spec = cast("NativeCampaignSpec", value)
    baseline, baseline_sha = baseline_campaign()
    if (
        type(spec["schema_version"]) is not int
        or spec["schema_version"] != 2
        or spec["kind"] != "arb-ml02-campaign"
        or spec["campaign_id"] != NATIVE_CAMPAIGN
        or spec["arms"] != [*VARIANTS, NATIVE_VARIANT]
        or spec["baseline_campaign_sha256"] != baseline_sha.removeprefix("sha256:")
        or spec["candidate_resource_identity"] != candidate_identity()
        or spec["score_encoding"] != baseline["score_encoding"]
    ):
        _fail("research_campaign_spec_invalid")
    return spec, "sha256:" + NATIVE_RESOURCE_SHA256


def campaign_variants(name: CampaignName) -> tuple[str, ...]:
    if name == LEGACY_CAMPAIGN:
        return VARIANTS
    if name == NATIVE_CAMPAIGN:
        return tuple(native_campaign_spec()[0]["arms"])
    _fail("research_campaign_profile_unsupported")
