"""Private ML-02 model inputs, fixed variants, and immutable model packages."""

from __future__ import annotations

from typing import TypedDict, cast

from recsys.contracts.bundle import JSONValue, canonical_bytes
from recsys.contracts.resources import resource_bytes
from recsys.research.protocol import exact, fail, sha

CAMPAIGN = "research_campaign_v1.json"


class CampaignSpec(TypedDict):
    schema_version: int
    kind: str
    score_encoding: dict[str, str]
    variants: dict[str, dict[str, JSONValue]]


GPU_SCORING_RECOVERY_COMMIT = "fa983eaeeeeb41b41ac893b7de9827a40c25d27d"


EXPECTED_VARIANTS = {
    "popularity": {},
    "content-mean": {"aggregation": "mean", "dimension": 300, "positive_only": True},
    "content-max": {"aggregation": "max", "dimension": 300, "positive_only": True},
    "content-union": {"dimension": 300, "positive_only": True, "rrf_k": 60, "top_k": 100},
    "itemknn": {"positive_min_rating": 8, "shrinkage": 10, "source_block": 128, "top_k": 200},
    "puresvd": {
        "flip_sign": True,
        "n_iter": 4,
        "n_oversamples": 16,
        "power_iteration_normalizer": "LU",
        "rank": 64,
        "random_state": 3737,
        "svd_lapack_driver": "gesdd",
        "transpose": False,
    },
    "hybrid": {"content_weight": 0.5, "rrf_k": 60, "svd_weight": 0.5},
}


VARIANT_DEPENDENCIES = {
    "popularity": (),
    "content-mean": (),
    "content-max": (),
    "content-union": (),
    "itemknn": ("scipy", "scikit-learn", "threadpoolctl"),
    "puresvd": ("scipy", "scikit-learn", "threadpoolctl"),
    "hybrid": ("scipy", "scikit-learn", "threadpoolctl"),
}


def campaign() -> tuple[CampaignSpec, str]:
    raw = resource_bytes("campaigns", CAMPAIGN)
    from recsys.contracts.bundle import parse_canonical

    value = parse_canonical(raw, CAMPAIGN)
    exact(value, {"schema_version", "kind", "score_encoding", "variants"})
    spec = cast("CampaignSpec", value)
    if (
        type(spec["schema_version"]) is not int
        or spec["schema_version"] != 1
        or spec["kind"] != "arb-ml02-campaign"
        or spec["score_encoding"]
        != {
            "nonpop_n_gt_0": "int-rint-half-even-times-1e9",
            "popularity_all_n": "raw-train-positive-user-counts",
            "zero_seed_all_variants": "raw-train-positive-user-counts",
        }
        or canonical_bytes(spec["variants"]) != canonical_bytes(EXPECTED_VARIANTS)
    ):
        fail("research_model_invalid")
    return spec, sha(raw)


def require_cpu_backend(requested: object) -> None:
    if type(requested) is str and requested == "cpu":
        return
    if type(requested) is str and requested == "gpu":
        fail("research_gpu_scoring_retired")
    fail("research_backend_unsupported")
