"""Offline preparation of the named immutable catalog release."""

import hashlib
import re
from collections.abc import Collection
from typing import Never, NotRequired, TypedDict, cast

from recsys.contracts.bundle import parse_canonical
from recsys.contracts.provenance import (
    LEGACY_V1_PACKAGE_VERSION,
    LEGACY_V1_SOURCE_SHA256,
    SOURCE_FILES,
    SOURCE_LAYOUT,
)
from recsys.contracts.resources import resource_bytes

CURRENT_RELEASE = "neelagiri-2025-nafiul-2026-v1"


LEGACY_RELEASE = "mal-2020-glove-v1"


LOCK_RESOURCE = "release_neelagiri_2025_nafiul_2026_v1.json"


HEX = re.compile(r"[0-9a-f]{64}\Z")


SAFE_ID = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*\Z")


VERSION = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+\Z")


COMMIT = re.compile(r"[0-9a-f]{40}\Z")


MAX_ID = 2_147_483_647


class AlgorithmParameters(TypedDict):
    field_order: list[str]
    tokenizer: str
    tokenizer_version: str
    lowercasing: str
    pooling: str
    token_weight: int
    dimension: int
    dtype: str
    normalization: str
    reduction: str
    positive_only: bool
    overshoot_tolerance: float
    tie_break: str


class AlgorithmProfile(TypedDict):
    name: str
    version: str
    max_neighbors: int
    similarity: str
    parameters: AlgorithmParameters


class Snapshot(TypedDict):
    snapshot_id: str
    identity: str


class CatalogRelease(TypedDict):
    identity: str
    catalog_sha256: str
    catalog_count: int


class Embeddings(TypedDict):
    snapshot_id: str
    registry_sha256: str
    text_size: int
    text_sha256: str


class ReleaseInputs(TypedDict):
    base: Snapshot
    aliases: Snapshot
    catalog_release: CatalogRelease
    embeddings: Embeddings
    algorithm: AlgorithmProfile


class ReferenceProducer(TypedDict):
    source_commit: str
    package_version: str
    python_version: str
    numpy_version: str
    code_files_sha256: dict[str, str]
    source_layout: NotRequired[str]


class Reference(TypedDict):
    block_size: int
    build_report_sha256: str
    bundle_identity: str
    catalog_sha256: str
    neighbors_sha256: str
    producer: ReferenceProducer


class Seed(TypedDict):
    mal_id: int
    query: str
    neighbor_ids: NotRequired[list[int]]


class SemanticExpectations(TypedDict):
    catalog_count: int
    vector_status_counts: dict[str, int]
    zero_vector_ids: list[int]
    neighbor_count_histogram: dict[str, int]
    smoke_seed: Seed
    empty_seed: Seed | None


class ReleaseLock(TypedDict):
    schema_version: int
    kind: str
    release_id: str
    inputs: ReleaseInputs
    reference: Reference
    semantic_expectations: SemanticExpectations


ALGORITHM_PROFILE: AlgorithmProfile = {
    "name": "glove-mean-cosine",
    "version": "arb009-v1",
    "max_neighbors": 5,
    "similarity": "cosine",
    "parameters": {
        "field_order": ["title", "aliases", "genres", "synopsis"],
        "tokenizer": "[a-z]+(?:'[a-z]+)?",
        "tokenizer_version": "arb009-v1",
        "lowercasing": "unicode-str.lower",
        "pooling": "arithmetic_mean_known_occurrences",
        "token_weight": 1,
        "dimension": 300,
        "dtype": "float64",
        "normalization": "L2",
        "reduction": "numpy.einsum_ik_jk_ij_optimize_false",
        "positive_only": True,
        "overshoot_tolerance": 1e-12,
        "tie_break": "numeric_MAL_ID_ascending",
    },
}


class PrepareError(ValueError):
    """A fixed-code preparation failure without private input content."""


def _fail(code: str = "release_prepare_integrity_error") -> Never:
    raise PrepareError(code)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _fields(value: object, names: Collection[str]) -> dict[str, object]:
    if type(value) is not dict or set(value) != set(names):
        _fail("release_lock_invalid")
    return cast("dict[str, object]", value)  # Exact object type and key set checked above.


def _digest(value: object) -> None:
    if type(value) is not str or HEX.fullmatch(value) is None:
        _fail("release_lock_invalid")


def _identity(value: object) -> None:
    if type(value) is not str or not value.startswith("sha256:"):
        _fail("release_lock_invalid")
    _digest(value[7:])


def _safe_id(value: object) -> None:
    if type(value) is not str or len(value) > 80 or SAFE_ID.fullmatch(value) is None:
        _fail("release_lock_invalid")


def _count(value: object, low: int = 0, high: int | None = None) -> None:
    if type(value) is not int or value < low or (high is not None and value > high):
        _fail("release_lock_invalid")


def _mal_id(value: object) -> None:
    _count(value, 1, MAX_ID)


def _same_typed(value: object, expected: object) -> bool:
    if type(value) is not type(expected):
        return False
    if type(value) is dict:
        value_map = cast("dict[str, object]", value)
        expected_map = cast("dict[str, object]", expected)
        return set(value_map) == set(expected_map) and all(
            _same_typed(value_map[key], expected_map[key]) for key in expected_map
        )
    if type(value) is list:
        value_list = cast("list[object]", value)
        expected_list = cast("list[object]", expected)
        return len(value_list) == len(expected_list) and all(
            _same_typed(a, b) for a, b in zip(value_list, expected_list)
        )
    return value == expected


def _seed(value: object, *, empty: bool = False) -> None:
    if empty and value is None:
        return
    value = _fields(value, {"mal_id", "query"} if empty else {"mal_id", "query", "neighbor_ids"})
    _mal_id(value["mal_id"])
    if type(value["query"]) is not str or not value["query"].strip():
        _fail("release_lock_invalid")
    if not empty:
        ids = value["neighbor_ids"]
        if type(ids) is not list or len(ids) > ALGORITHM_PROFILE["max_neighbors"]:
            _fail("release_lock_invalid")
        for item in ids:
            _mal_id(item)
        if len(set(ids)) != len(ids) or value["mal_id"] in ids:
            _fail("release_lock_invalid")


def _validate_lock(value: object) -> None:
    value = _fields(
        value,
        {"schema_version", "kind", "release_id", "inputs", "reference", "semantic_expectations"},
    )
    if type(value["schema_version"]) is not int or value["schema_version"] not in (1, 2):
        _fail("release_lock_invalid")
    if type(value["kind"]) is not str or value["kind"] != "arb-release-lock":
        _fail("release_lock_invalid")
    _safe_id(value["release_id"])
    inputs = _fields(
        value["inputs"], {"base", "aliases", "catalog_release", "embeddings", "algorithm"}
    )
    for key, snapshot in (("base", "neelagiri-2025-v1"), ("aliases", "nafiul-2026-v1")):
        item = _fields(inputs[key], {"snapshot_id", "identity"})
        _safe_id(item["snapshot_id"])
        if item["snapshot_id"] != snapshot:
            _fail("release_lock_invalid")
        _identity(item["identity"])
    catalog = _fields(inputs["catalog_release"], {"identity", "catalog_sha256", "catalog_count"})
    _identity(catalog["identity"])
    _digest(catalog["catalog_sha256"])
    _count(catalog["catalog_count"], 1)
    embeddings = _fields(
        inputs["embeddings"], {"snapshot_id", "registry_sha256", "text_size", "text_sha256"}
    )
    _safe_id(embeddings["snapshot_id"])
    if embeddings["snapshot_id"] != "glove-6b-300d-v1":
        _fail("release_lock_invalid")
    _digest(embeddings["registry_sha256"])
    _digest(embeddings["text_sha256"])
    _count(embeddings["text_size"], 1)
    if not _same_typed(inputs["algorithm"], ALGORITHM_PROFILE):
        _fail("release_lock_invalid")

    reference = _fields(
        value["reference"],
        {
            "block_size",
            "build_report_sha256",
            "bundle_identity",
            "catalog_sha256",
            "neighbors_sha256",
            "producer",
        },
    )
    _count(reference["block_size"], 1, 256)
    for key in ("build_report_sha256", "catalog_sha256", "neighbors_sha256"):
        _digest(reference[key])
    _identity(reference["bundle_identity"])
    if reference["catalog_sha256"] != catalog["catalog_sha256"]:
        _fail("release_lock_invalid")
    producer = reference["producer"]
    current_layout = value["schema_version"] == 2
    producer = _fields(
        producer,
        {"source_commit", "package_version", "python_version", "numpy_version", "code_files_sha256"}
        | ({"source_layout"} if current_layout else set()),
    )
    if current_layout and producer["source_layout"] != SOURCE_LAYOUT:
        _fail("release_lock_invalid")
    commit = producer["source_commit"]
    if type(commit) is not str or COMMIT.fullmatch(commit) is None:
        _fail("release_lock_invalid")
    for key in ("package_version", "python_version", "numpy_version"):
        version = producer[key]
        if type(version) is not str or VERSION.fullmatch(version) is None:
            _fail("release_lock_invalid")
    files = _fields(
        producer["code_files_sha256"],
        SOURCE_FILES
        if current_layout
        else {"release.py", "build.py", "release_build.py", "export.py", "bundle.py"},
    )
    for digest in files.values():
        _digest(digest)
    if not current_layout and (
        producer["package_version"] != LEGACY_V1_PACKAGE_VERSION
        or files != {name: LEGACY_V1_SOURCE_SHA256[name] for name in files}
    ):
        _fail("release_lock_invalid")

    semantics = _fields(
        value["semantic_expectations"],
        {
            "catalog_count",
            "vector_status_counts",
            "zero_vector_ids",
            "neighbor_count_histogram",
            "smoke_seed",
            "empty_seed",
        },
    )
    _count(semantics["catalog_count"], 1)
    if semantics["catalog_count"] != catalog["catalog_count"]:
        _fail("release_lock_invalid")
    statuses = _fields(
        semantics["vector_status_counts"], {"ready", "no_tokens", "oov_only", "zero_vector"}
    )
    for count in statuses.values():
        _count(count)
    typed_statuses = cast("dict[str, int]", statuses)  # All values passed _count.
    if sum(typed_statuses.values()) != catalog["catalog_count"]:
        _fail("release_lock_invalid")
    nonready = semantics["zero_vector_ids"]
    if type(nonready) is not list:
        _fail("release_lock_invalid")
    for mal in nonready:
        _mal_id(mal)
    if (
        nonready != sorted(set(nonready))
        or len(nonready) != catalog["catalog_count"] - typed_statuses["ready"]
    ):
        _fail("release_lock_invalid")
    histogram = _fields(
        semantics["neighbor_count_histogram"],
        {str(i) for i in range(ALGORITHM_PROFILE["max_neighbors"] + 1)},
    )
    for count in histogram.values():
        _count(count)
    typed_histogram = cast("dict[str, int]", histogram)  # All values passed _count.
    if sum(typed_histogram.values()) != catalog["catalog_count"] or typed_histogram["0"] < len(
        nonready
    ):
        _fail("release_lock_invalid")
    _seed(semantics["smoke_seed"])
    _seed(semantics["empty_seed"], empty=True)
    empty_seed = semantics["empty_seed"]
    if (
        empty_seed is not None
        and cast("dict[str, object]", empty_seed)["mal_id"]
        == cast("dict[str, object]", semantics["smoke_seed"])["mal_id"]
    ):
        _fail("release_lock_invalid")


def _lock(lock_bytes: bytes | None = None) -> ReleaseLock:
    try:
        raw = resource_bytes("releases", LOCK_RESOURCE) if lock_bytes is None else lock_bytes
        value = parse_canonical(raw, LOCK_RESOURCE)
        _validate_lock(value)
        return cast("ReleaseLock", value)  # Full recursive release validation above.
    except Exception:
        raise PrepareError("release_lock_invalid") from None


def release_info(selector: str = "current", *, _lock_bytes: bytes | None = None) -> ReleaseLock:
    if selector not in ("current", CURRENT_RELEASE):
        _fail("unknown_release")
    lock = _lock(_lock_bytes)
    if _lock_bytes is None and lock["release_id"] != CURRENT_RELEASE:
        _fail("release_lock_invalid")
    return lock
