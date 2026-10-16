"""Offline preparation of the named immutable catalog release."""

import hashlib
import re

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


ALGORITHM_PROFILE = {
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


def _fail(code="release_prepare_integrity_error"):
    raise PrepareError(code)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _fields(value, names):
    if type(value) is not dict or set(value) != set(names):
        _fail("release_lock_invalid")


def _digest(value):
    if type(value) is not str or HEX.fullmatch(value) is None:
        _fail("release_lock_invalid")


def _identity(value):
    if type(value) is not str or not value.startswith("sha256:"):
        _fail("release_lock_invalid")
    _digest(value[7:])


def _safe_id(value):
    if type(value) is not str or len(value) > 80 or SAFE_ID.fullmatch(value) is None:
        _fail("release_lock_invalid")


def _count(value, low=0, high=None):
    if type(value) is not int or value < low or (high is not None and value > high):
        _fail("release_lock_invalid")


def _mal_id(value):
    _count(value, 1, MAX_ID)


def _same_typed(value, expected):
    if type(value) is not type(expected):
        return False
    if type(value) is dict:
        return set(value) == set(expected) and all(
            _same_typed(value[key], expected[key]) for key in expected
        )
    if type(value) is list:
        return len(value) == len(expected) and all(
            _same_typed(a, b) for a, b in zip(value, expected)
        )
    return value == expected


def _seed(value, *, empty=False):
    if empty and value is None:
        return
    _fields(value, {"mal_id", "query"} if empty else {"mal_id", "query", "neighbor_ids"})
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


def _validate_lock(value):
    _fields(
        value,
        {"schema_version", "kind", "release_id", "inputs", "reference", "semantic_expectations"},
    )
    if type(value["schema_version"]) is not int or value["schema_version"] not in (1, 2):
        _fail("release_lock_invalid")
    if type(value["kind"]) is not str or value["kind"] != "arb-release-lock":
        _fail("release_lock_invalid")
    _safe_id(value["release_id"])
    inputs = value["inputs"]
    _fields(inputs, {"base", "aliases", "catalog_release", "embeddings", "algorithm"})
    for key, snapshot in (("base", "neelagiri-2025-v1"), ("aliases", "nafiul-2026-v1")):
        item = inputs[key]
        _fields(item, {"snapshot_id", "identity"})
        _safe_id(item["snapshot_id"])
        if item["snapshot_id"] != snapshot:
            _fail("release_lock_invalid")
        _identity(item["identity"])
    catalog = inputs["catalog_release"]
    _fields(catalog, {"identity", "catalog_sha256", "catalog_count"})
    _identity(catalog["identity"])
    _digest(catalog["catalog_sha256"])
    _count(catalog["catalog_count"], 1)
    embeddings = inputs["embeddings"]
    _fields(embeddings, {"snapshot_id", "registry_sha256", "text_size", "text_sha256"})
    _safe_id(embeddings["snapshot_id"])
    if embeddings["snapshot_id"] != "glove-6b-300d-v1":
        _fail("release_lock_invalid")
    _digest(embeddings["registry_sha256"])
    _digest(embeddings["text_sha256"])
    _count(embeddings["text_size"], 1)
    if not _same_typed(inputs["algorithm"], ALGORITHM_PROFILE):
        _fail("release_lock_invalid")

    reference = value["reference"]
    _fields(
        reference,
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
    _fields(
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
    files = producer["code_files_sha256"]
    _fields(
        files,
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

    semantics = value["semantic_expectations"]
    _fields(
        semantics,
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
    statuses = semantics["vector_status_counts"]
    _fields(statuses, {"ready", "no_tokens", "oov_only", "zero_vector"})
    for count in statuses.values():
        _count(count)
    if sum(statuses.values()) != catalog["catalog_count"]:
        _fail("release_lock_invalid")
    nonready = semantics["zero_vector_ids"]
    if type(nonready) is not list:
        _fail("release_lock_invalid")
    for mal in nonready:
        _mal_id(mal)
    if (
        nonready != sorted(set(nonready))
        or len(nonready) != catalog["catalog_count"] - statuses["ready"]
    ):
        _fail("release_lock_invalid")
    histogram = semantics["neighbor_count_histogram"]
    _fields(histogram, {str(i) for i in range(ALGORITHM_PROFILE["max_neighbors"] + 1)})
    for count in histogram.values():
        _count(count)
    if sum(histogram.values()) != catalog["catalog_count"] or histogram["0"] < len(nonready):
        _fail("release_lock_invalid")
    _seed(semantics["smoke_seed"])
    _seed(semantics["empty_seed"], empty=True)
    if (
        semantics["empty_seed"] is not None
        and semantics["empty_seed"]["mal_id"] == semantics["smoke_seed"]["mal_id"]
    ):
        _fail("release_lock_invalid")


def _lock(lock_bytes=None):
    try:
        raw = resource_bytes("releases", LOCK_RESOURCE) if lock_bytes is None else lock_bytes
        value = parse_canonical(raw, LOCK_RESOURCE)
        _validate_lock(value)
        return value
    except Exception:
        raise PrepareError("release_lock_invalid") from None


def release_info(selector="current", *, _lock_bytes=None):
    if selector not in ("current", CURRENT_RELEASE):
        _fail("unknown_release")
    lock = _lock(_lock_bytes)
    if _lock_bytes is None and lock["release_id"] != CURRENT_RELEASE:
        _fail("release_lock_invalid")
    return lock
