"""Validate pinned build provenance and publish immutable recommendation bundles."""

import fcntl
import hashlib
import os
import re
import shutil
import stat
import tempfile
from collections import Counter
from pathlib import Path

from . import __version__
from .build import (BUILDER_VERSION, DIMENSION, FIELDS, SCORE_BLOCK_BUDGET,
                    TOKEN_PATTERN, _effective_rows, _report, _sources, _tokens)
from .bundle import (ContractError, canonical_bytes, check_bundle, check_bundle_bytes,
                     check_catalog, check_neighbors, exact_fields, integer,
                     read_canonical, require)
from .sources import load_registry, validate_registry

EXPORTER_VERSION = "arb010-v1"
REPORT_FIELDS = {"schema_version", "builder_version", "algorithm", "sources", "registry_sha256",
                 "inputs", "files", "producer", "execution", "coverage", "anime"}
# This literal profile is deliberately frozen to the supported arb009-v1 builder.
ALGORITHM = {"name": "glove-mean-cosine", "version": BUILDER_VERSION, "max_neighbors": 5,
             "similarity": "cosine", "parameters": {
                 "field_order": list(FIELDS), "tokenizer": TOKEN_PATTERN,
                 "tokenizer_version": "arb009-v1", "lowercasing": "unicode-str.lower",
                 "pooling": "arithmetic_mean_known_occurrences", "token_weight": 1,
                 "dimension": DIMENSION, "dtype": "float64", "normalization": "L2",
                 "reduction": "numpy.einsum_ik_jk_ij_optimize_false",
                 "positive_only": True, "overshoot_tolerance": 1e-12,
                 "tie_break": "numeric_MAL_ID_ascending"}}
BUNDLE_FILES = ("catalog.json", "neighbors.json", "manifest.json")


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _strict_equal(actual, expected):
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return (actual.keys() == expected.keys()
                and all(_strict_equal(actual[key], value) for key, value in expected.items()))
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _strict_equal(left, right) for left, right in zip(actual, expected))
    return actual == expected


def _same(actual, expected, location):
    require(_strict_equal(actual, expected), location, "does not match pinned producer or input")


def _code_sha(name):
    return _sha((Path(__file__).parent / name).read_bytes())


def _validate_report(report, catalog, neighbors, raw, registry, registry_sha, normalization, normalization_raw):
    exact_fields(report, REPORT_FIELDS, "build-report.json")
    integer(report["schema_version"], "build-report.json.schema_version", 1, 1)
    _same(report["builder_version"], BUILDER_VERSION, "build-report.json.builder_version")
    _same(report["algorithm"], ALGORITHM, "build-report.json.algorithm")
    _same(report["registry_sha256"], registry_sha, "build-report.json.registry_sha256")
    catalog_sha = _sha(raw["catalog.json"])
    neighbor_sha = _sha(raw["neighbors.json"])
    normalization_sha = _sha(normalization_raw)
    count = len(catalog["anime"])
    # The builder's normalization report check covers the pinned MAL sources and catalog.
    try:
        _report(normalization, registry, registry_sha, catalog_sha, count)
    except ValueError as exc:
        raise ContractError(f"normalization-report.json: {exc}") from exc
    _same(report["files"], {"catalog.json": {"sha256": catalog_sha},
                            "neighbors.json": {"sha256": neighbor_sha}}, "build-report.json.files")
    descriptors = {item["id"]: item for item in registry["sources"]}
    glove = descriptors["glove_300d"]
    glove_info = {"sha256": glove["sha256"], "size": glove["size"]}
    _same(report["inputs"], {"catalog.json": {"sha256": catalog_sha},
                              "normalization-report.json": {"sha256": normalization_sha},
                              "glove.6B.300d.txt": glove_info}, "build-report.json.inputs")
    _same(report["sources"], _sources(registry, registry_sha, catalog_sha,
                                       normalization_sha, glove_info), "build-report.json.sources")
    producer = report["producer"]
    exact_fields(producer, {"package_version", "python_version", "numpy_version", "code_files"},
                 "build-report.json.producer")
    _same(producer["package_version"], __version__, "build-report.json.producer.package_version")
    require(isinstance(producer["python_version"], str)
            and re.fullmatch(r"3\.12\.[0-9]+", producer["python_version"]) is not None,
            "build-report.json.producer.python_version", "expected Python 3.12 release")
    _same(producer["numpy_version"], "2.5.3", "build-report.json.producer.numpy_version")
    _same(producer["code_files"], {name: _code_sha(name) for name in ("build.py", "sources.py")},
          "build-report.json.producer.code_files")
    execution = report["execution"]
    exact_fields(execution, {"requested_block_rows", "effective_block_rows",
                             "similarity_block_budget_bytes"}, "build-report.json.execution")
    requested = execution["requested_block_rows"]
    integer(requested, "build-report.json.execution.requested_block_rows", 1, 256)
    _same(execution, {"requested_block_rows": requested,
                      "effective_block_rows": _effective_rows(count, requested),
                      "similarity_block_budget_bytes": SCORE_BLOCK_BUDGET},
          "build-report.json.execution")

    details = report["anime"]
    require(isinstance(details, dict) and set(details) == set(catalog["anime"]),
            "build-report.json.anime", "MAL_ID keys must exactly match catalog")
    statuses = Counter()
    histogram = Counter()
    vocabulary = set()
    occurrences = matched_occurrences = 0
    nonready = []
    for key, entry in catalog["anime"].items():
        loc = f"build-report.json.anime.{key}"
        detail = details[key]
        exact_fields(detail, {"token_count", "matched_token_count", "vector_status", "neighbor_count"}, loc)
        tokens = list(_tokens(entry))
        vocabulary.update(tokens)
        token_count = len(tokens)
        matched = detail["matched_token_count"]
        integer(detail["token_count"], f"{loc}.token_count", 0)
        integer(matched, f"{loc}.matched_token_count", 0, token_count)
        _same(detail["token_count"], token_count, f"{loc}.token_count")
        status = detail["vector_status"]
        require(status in ("no_tokens", "oov_only", "zero_vector", "ready"),
                f"{loc}.vector_status", "unknown status")
        require((status == "no_tokens" and token_count == 0 and matched == 0)
                or (status == "oov_only" and token_count > 0 and matched == 0)
                or (status in ("zero_vector", "ready") and matched > 0),
                f"{loc}.vector_status", "inconsistent with token counts")
        items = neighbors["neighbors"][key]
        integer(detail["neighbor_count"], f"{loc}.neighbor_count", 0, 5)
        _same(detail["neighbor_count"], len(items), f"{loc}.neighbor_count")
        if status != "ready":
            require(not items, loc, "non-ready source has neighbors")
            nonready.append(int(key))
        statuses[status] += 1
        histogram[len(items)] += 1
        occurrences += token_count
        matched_occurrences += matched
    nonready_set = set(nonready)
    for key, items in neighbors["neighbors"].items():
        for item in items:
            require(item["mal_id"] not in nonready_set,
                    f"neighbors.json.neighbors.{key}", "non-ready target")
    coverage = report["coverage"]
    exact_fields(coverage, {"catalog_count", "vocabulary_count", "matched_vocabulary_count",
                            "token_occurrences", "matched_token_occurrences", "vector_status_counts",
                            "zero_vector_ids", "neighbor_count_histogram"}, "build-report.json.coverage")
    matched_vocab = coverage["matched_vocabulary_count"]
    integer(matched_vocab, "build-report.json.coverage.matched_vocabulary_count",
            1 if matched_occurrences else 0, min(len(vocabulary), matched_occurrences))
    expected = {"catalog_count": count, "vocabulary_count": len(vocabulary),
                "matched_vocabulary_count": matched_vocab,
                "token_occurrences": occurrences, "matched_token_occurrences": matched_occurrences,
                "vector_status_counts": {key: statuses[key] for key in
                                         ("no_tokens", "oov_only", "zero_vector", "ready")},
                "zero_vector_ids": sorted(nonready),
                "neighbor_count_histogram": {str(i): histogram[i] for i in range(6)}}
    _same(coverage, expected, "build-report.json.coverage")


def _verify_existing(target, files, identity, count):
    require(not target.is_symlink() and target.is_dir(), str(target), "unsafe published target")
    try:
        entries = list(target.iterdir())
    except OSError as exc:
        raise ContractError(f"{target}: {exc}") from exc
    require({entry.name for entry in entries} == set(BUNDLE_FILES), str(target),
            "published target must contain exactly three files")
    for entry in entries:
        require(not entry.is_symlink() and entry.is_file(), str(entry), "expected regular file")
    observed = check_bundle(target)
    _same(observed, (identity, count), str(target))
    for name, content in files.items():
        _same((target / name).read_bytes(), content, str(target / name))


def _publish(store, files, identity, count):
    require(os.name == "posix", str(store), "POSIX publication required")
    require(not store.is_symlink() and (not store.exists() or store.is_dir()),
            str(store), "unsafe output directory")
    store.mkdir(parents=True, exist_ok=True)
    lock_path = store / ".publication.lock"
    flags = os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW
    fd = os.open(lock_path, flags, 0o600)
    try:
        require(stat.S_ISREG(os.fstat(fd).st_mode), str(lock_path), "expected regular lock file")
        fcntl.flock(fd, fcntl.LOCK_EX)
        target = store / identity.replace(":", "-")
        if target.exists() or target.is_symlink():
            _verify_existing(target, files, identity, count)
            return target
        stage = Path(tempfile.mkdtemp(prefix=".bundle-", dir=store))
        try:
            for name in BUNDLE_FILES:
                path = stage / name
                with path.open("xb") as stream:
                    stream.write(files[name])
                    stream.flush()
                    os.fsync(stream.fileno())
            _same(check_bundle(stage), (identity, count), str(stage))
            dirfd = os.open(stage, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(dirfd)
            finally:
                os.close(dirfd)
            os.rename(stage, target)
            dirfd = os.open(store, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(dirfd)
            finally:
                os.close(dirfd)
            return target
        finally:
            if stage.exists():
                shutil.rmtree(stage)
    finally:
        os.close(fd)


def export_bundle(catalog_path, neighbors_path, build_report_path, normalization_report_path,
                  output_dir, *, registry=None, registry_sha256=None) -> Path:
    """Validate retained bytes, then publish or reuse their immutable bundle."""
    if registry is None:
        registry, registry_sha256 = load_registry()
    else:
        validate_registry(registry)
        actual_sha = _sha(canonical_bytes(registry))
        if registry_sha256 is None:
            registry_sha256 = actual_sha
        _same(registry_sha256, actual_sha, "registry_sha256")
    catalog, catalog_raw = read_canonical(Path(catalog_path))
    neighbors, neighbors_raw = read_canonical(Path(neighbors_path))
    report, report_raw = read_canonical(Path(build_report_path))
    normalization, normalization_raw = read_canonical(Path(normalization_report_path))
    keys = check_catalog(catalog)
    check_neighbors(neighbors, keys)
    raw = {"catalog.json": catalog_raw, "neighbors.json": neighbors_raw}
    _validate_report(report, catalog, neighbors, raw, registry, registry_sha256,
                     normalization, normalization_raw)
    sources = list(report["sources"])
    sources.extend([
        {"name": "build_report", "version": BUILDER_VERSION, "sha256": _sha(report_raw),
         "metadata": {"filename": "build-report.json", "producer": report["producer"]}},
        {"name": "bundle_exporter", "version": EXPORTER_VERSION, "sha256": _code_sha("export.py"),
         "metadata": {"filename": "export.py", "package_version": __version__,
                      "bundle_sha256": _code_sha("bundle.py"),
                      "sources_sha256": _code_sha("sources.py")}},
    ])
    sources.sort(key=lambda item: (item["name"], item["version"]))
    manifest_raw = canonical_bytes({"schema_version": 1, "algorithm": report["algorithm"],
                                    "sources": sources, "files": report["files"]})
    identity, count = check_bundle_bytes(catalog_raw, neighbors_raw, manifest_raw)
    return _publish(Path(output_dir), {**raw, "manifest.json": manifest_raw}, identity, count)
