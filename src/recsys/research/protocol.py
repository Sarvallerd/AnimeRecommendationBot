"""Frozen ML-01 policy; overrides are restricted to trusted synthetic tests."""

import hashlib
import math
from collections.abc import Collection
from typing import Never, TypedDict, cast

from recsys.contracts.bundle import canonical_bytes, parse_canonical
from recsys.contracts.resources import resource_bytes

RESOURCE = "research_protocol_v1.json"


class Budget(TypedDict):
    campaign_hours: int
    evaluation_hours_per_variant: int
    fit_hours_per_variant: int
    max_model_config_variants: int
    rss_gib: int
    score_artifact_gib: int


class Cohorts(TypedDict):
    main_min_positives: int
    short_max_positives: int
    short_min_positives: int


class Comparison(TypedDict):
    bootstrap_replicates: int
    confidence: float
    epsilon_ndcg5: float
    seed: int


class Metrics(TypedDict):
    cutoffs: list[int]
    incomplete_ranking: str
    primary: str
    relevance: str
    user_aggregation: str


class Query(TypedDict):
    fixed_targets_across_n: bool
    main_targets: int
    max_n: int
    nested_seeds: bool
    order_seeds: list[int]
    orders: list[str]
    popularity_n0: bool
    sample_main_per_split: int
    sample_seed: int
    sample_short_per_split: int
    split_seed: int
    target_seed: int
    toprated_tie_seed: int


class Selection(TypedDict):
    cohort: str
    minimum_slice_users: int
    n_range: list[int]
    rule: str


class Split(TypedDict):
    test: list[int]
    train: list[int]
    validation: list[int]


class ProtocolSpec(TypedDict):
    budget: Budget
    candidate_policy: dict[str, str]
    catalog_policy: str
    cohorts: Cohorts
    comparison: Comparison
    interaction_policy: dict[str, str | int]
    kind: str
    metrics: Metrics
    query: Query
    schema_version: int
    selection: Selection
    split: Split


class ResearchError(ValueError):
    """Public, fixed-code failure without private input values."""


def fail(code: str = "research_integrity_error") -> Never:
    raise ResearchError(code)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def exact(value: object, fields: Collection[str]) -> None:
    if type(value) is not dict or set(value) != set(fields):
        fail()


def integer(value: object, low: int = 0, high: int | None = None) -> None:
    if type(value) is not int or value < low or (high is not None and value > high):
        fail()


def finite(value: object) -> None:
    if type(value) is int:
        return
    if type(value) is not float or not math.isfinite(value):
        fail()


def load_protocol(
    selector: str = "ml01-v1", *, _protocol_bytes: bytes | None = None
) -> ProtocolSpec:
    if selector != "ml01-v1":
        fail("research_protocol_invalid")
    try:
        trusted = resource_bytes("protocols", RESOURCE)
        raw = trusted if _protocol_bytes is None else _protocol_bytes
        value = parse_canonical(raw, RESOURCE)
        expected = parse_canonical(trusted, RESOURCE)
        exact(value, cast("Collection[str]", expected))  # Trusted frozen resource has object keys.
        # Production is immutable. Test-only overrides may adjust quantitative values,
        # while retaining shape and type of the frozen contract.
        if _protocol_bytes is None and raw != trusted:
            fail()

        def shape(actual: object, reference: object) -> None:
            if type(actual) is not type(reference):
                fail()
            if type(reference) is dict:
                exact(actual, reference)
                actual_map = cast("dict[str, object]", actual)
                reference_map = cast("dict[str, object]", reference)
                for key in reference_map:
                    shape(actual_map[key], reference_map[key])
            elif type(reference) is list:
                actual_list = cast("list[object]", actual)
                reference_list = cast("list[object]", reference)
                if len(actual_list) != len(reference_list):
                    fail()
                for left, right in zip(actual_list, reference_list):
                    shape(left, right)
            elif type(reference) is float:
                finite(actual)
            elif type(reference) is int:
                integer(actual)
            elif actual != reference:
                fail()

        shape(value, expected)
        return cast("ProtocolSpec", value)  # Recursive shape matched the trusted resource.
    except (OSError, ValueError, TypeError, OverflowError) as exc:
        raise ResearchError("research_protocol_invalid") from exc


def protocol_identity(*, _protocol_bytes: bytes | None = None) -> str:
    value = load_protocol(_protocol_bytes=_protocol_bytes)
    return "sha256:" + sha(canonical_bytes(value))


def split_user(user_key: str, protocol: ProtocolSpec | None = None) -> str:
    if type(user_key) is not str or len(user_key) != 67 or not user_key.startswith("u1_"):
        fail()
    if any(char not in "0123456789abcdef" for char in user_key[3:]):
        fail()
    seed = (protocol or load_protocol())["query"]["split_seed"]
    bucket = (
        int.from_bytes(
            hashlib.sha256(
                b"arb-ml-user-split-v1\0" + str(seed).encode() + b"\0" + user_key.encode()
            ).digest()[:8],
            "big",
        )
        % 100
    )
    return "train" if bucket < 80 else "validation" if bucket < 90 else "test"


def order_hash(seed: int, user: str, mal_id: int) -> bytes:
    return hashlib.sha256(
        b"arb-ml-item-order-v1\0"
        + str(seed).encode()
        + b"\0"
        + user.encode()
        + b"\0"
        + str(mal_id).encode()
    ).digest()


def sample_hash(seed: int, split: str, cohort: str, user: str) -> bytes:
    return hashlib.sha256(
        b"arb-ml-user-sample-v1\0"
        + str(seed).encode()
        + b"\0"
        + split.encode()
        + b"\0"
        + cohort.encode()
        + b"\0"
        + user.encode()
    ).digest()
