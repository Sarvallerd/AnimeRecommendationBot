"""Build and export strict v2 artifacts from a verified catalog release."""

import hashlib
import os
import platform
import re
import shutil
import tempfile
from collections import Counter
from importlib import resources
from pathlib import Path

from . import __version__
from .build import (_effective_rows, _tokens, build_vectors, cosine_neighbors,
                    load_embeddings, SCORE_BLOCK_BUDGET)
from .bundle import (ContractError, canonical_bytes, check_bundle_bytes, check_catalog,
                     check_neighbors, exact_fields, integer, parse_canonical, require)
from .export import (ALGORITHM, _code_sha, _publish, _same, _validate_diagnostics)
from .release import ReleaseError, VerifiedRelease, _sha, _verified_release
from .sources import SourceError, validate_registry

BUILDER_VERSION = "arb031-v1"
REPORT_FIELDS = {"schema_version", "kind", "builder_version", "catalog_release_identity",
                 "algorithm", "sources", "registry_sha256", "inputs", "files", "producer",
                 "execution", "coverage", "anime"}


class ReleaseBuildError(ValueError):
    """A fixed-code v2 build or export failure."""


def _safe_file(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ReleaseBuildError("release_build_integrity_error")
    return path.read_bytes()


def _glove_registry(raw: bytes | None):
    if raw is None:
        raw = resources.files("recsys").joinpath("source_registry_glove_6b_300d_v1.json").read_bytes()
    # Registry is distributed in readable JSON, and its original bytes are the pinned identity.
    from .aliases import _json_unique
    value = _json_unique(raw)
    validate_registry(value)
    if (set(value) != {"schema_version", "provenance", "sources"}
            or type(value["schema_version"]) is not int or value["schema_version"] != 1
            or set(value["provenance"]) != {"glove"}
            or [item["id"] for item in value["sources"]] != ["glove_zip", "glove_300d"]):
        raise ReleaseBuildError("release_build_integrity_error")
    return value, _sha(raw)


def _source(name, version, digest, metadata):
    return {"name": name, "version": version, "sha256": digest, "metadata": metadata}


def _sources(verified: VerifiedRelease, registry: dict, registry_sha: str):
    base = verified.base_manifest
    alias = verified.alias_manifest
    release = verified.manifest
    descriptions = {item["id"]: item for item in registry["sources"]}
    glove = descriptions["glove_300d"]
    provenance = registry["provenance"]["glove"]
    items = [
        _source("research_dataset", base["normalizer"]["version"],
                verified.summary.base_identity.split(":", 1)[1],
                {"filename": "dataset-manifest.json", "snapshot_id": base["snapshot_id"],
                 "registry_sha256": base["registry_sha256"], "provenance": base["provenance"],
                 "manifest_sha256": release["inputs"]["base"]["manifest"]["sha256"]}),
        _source("alias_enrichment", alias["normalizer"]["version"],
                verified.summary.alias_identity.split(":", 1)[1],
                {"filename": "alias-manifest.json", "snapshot_id": alias["snapshot_id"],
                 "policy": alias["normalizer"]["policy"], "base": alias["base"],
                 "manifest_sha256": release["inputs"]["aliases"]["manifest"]["sha256"]}),
        _source("catalog_release", release["normalizer"]["version"],
                verified.summary.identity.split(":", 1)[1],
                {"filename": "release-manifest.json", "policy": release["policy"]}),
        _source("normalized_catalog", "ordered-alias-overlay-v1", verified.summary.catalog_sha256,
                {"filename": "catalog.json"}),
        _source("glove_300d", "6B-300d", glove["sha256"],
                {"filename": glove["filename"], "size": glove["size"],
                 "archive_sha256": descriptions["glove_zip"]["sha256"],
                 "member": glove["member"], "project": provenance["project"],
                 "license": provenance["pretrained_vectors_license"]}),
        _source("glove_source_registry", "1", registry_sha,
                {"filename": "source_registry_glove_6b_300d_v1.json"}),
    ]
    return sorted(items, key=lambda item: (item["name"], item["version"]))


def _verified(release_dir, *, expected_release_identity, base_dir, aliases_dir,
              alias_data_dir, _base_spec, _alias_spec):
    return _verified_release(release_dir, base_dir=base_dir, aliases_dir=aliases_dir,
                             alias_data_dir=alias_data_dir,
                             expected_identity=expected_release_identity,
                             _base_spec=_base_spec, _alias_spec=_alias_spec)


def _write(path: Path, raw: bytes):
    with open(path, "xb", opener=lambda name, flags: os.open(name, flags, 0o600)) as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def _fsync(path: Path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _check_output(output: Path, inputs):
    if (not output.parent.is_dir() or output.parent.is_symlink()
            or output.exists() or output.is_symlink()):
        raise ReleaseBuildError("release_build_integrity_error")
    target = output.resolve(strict=False)
    for item in inputs:
        origin = Path(item).resolve(strict=False)
        if target == origin or target.is_relative_to(origin) or origin.is_relative_to(target):
            raise ReleaseBuildError("release_build_integrity_error")


def build_release(release_dir, glove_path, output_dir, *, expected_release_identity,
                  base_dir, aliases_dir, alias_data_dir, block_size=256,
                  _base_spec=None, _alias_spec=None, _glove_registry_bytes=None) -> Path:
    """Validate once, then publish exactly neighbors.json and the v2 report."""
    import numpy as np
    try:
        output_dir = Path(output_dir)
        _check_output(output_dir, (release_dir, base_dir, aliases_dir, alias_data_dir, glove_path))
        verified = _verified(release_dir, expected_release_identity=expected_release_identity,
                             base_dir=base_dir, aliases_dir=aliases_dir,
                             alias_data_dir=alias_data_dir,
                             _base_spec=_base_spec, _alias_spec=_alias_spec)
        registry, registry_sha = _glove_registry(_glove_registry_bytes)
        descriptions = {item["id"]: item for item in registry["sources"]}
        glove_path = Path(glove_path)
        if glove_path.is_symlink() or not glove_path.is_file():
            raise ReleaseBuildError("release_build_integrity_error")
        anime = verified.catalog["anime"]
        vocabulary = {token for entry in anime.values() for token in _tokens(entry)}
        embeddings, glove_info = load_embeddings(glove_path, vocabulary, descriptions["glove_300d"])
        ids, vectors, diagnostics = build_vectors(anime, embeddings)
        del embeddings, vocabulary
        neighbors = {"schema_version": 1,
                     "neighbors": cosine_neighbors(ids, vectors, block_size=block_size)}
        del vectors
        for key, entry in diagnostics["anime"].items():
            entry["neighbor_count"] = len(neighbors["neighbors"][key])
        histogram = Counter(len(items) for items in neighbors["neighbors"].values())
        diagnostics["coverage"]["neighbor_count_histogram"] = {
            str(key): histogram[key] for key in range(6)}
        check_neighbors(neighbors, check_catalog(verified.catalog))
        neighbor_raw = canonical_bytes(neighbors)
        report = {
            "schema_version": 2, "kind": "arb-release-build",
            "builder_version": BUILDER_VERSION,
            "catalog_release_identity": verified.summary.identity,
            "algorithm": ALGORITHM,
            "sources": _sources(verified, registry, registry_sha),
            "registry_sha256": registry_sha,
            "inputs": {"release-manifest.json": {"bytes": len(verified.manifest_raw),
                                                  "sha256": _sha(verified.manifest_raw)},
                       "catalog.json": {"bytes": len(verified.catalog_raw),
                                        "sha256": verified.summary.catalog_sha256},
                       "glove.6B.300d.txt": glove_info},
            "files": {"catalog.json": {"sha256": verified.summary.catalog_sha256},
                      "neighbors.json": {"sha256": _sha(neighbor_raw)}},
            "producer": {"package_version": __version__, "python_version": platform.python_version(),
                         "numpy_version": np.__version__,
                         "code_files": {name: _code_sha(name) for name in
                                        ("build.py", "release_build.py")}},
            "execution": {"requested_block_rows": block_size,
                          "effective_block_rows": _effective_rows(len(ids), block_size),
                          "similarity_block_budget_bytes": SCORE_BLOCK_BUDGET},
            "coverage": diagnostics["coverage"], "anime": diagnostics["anime"],
        }
        _validate_release_report(report, verified, neighbors, neighbor_raw, registry,
                                 registry_sha, enforce_current=True)
        stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
        os.chmod(stage, 0o700)
        try:
            _write(stage / "neighbors.json", neighbor_raw)
            _write(stage / "build-report.json", canonical_bytes(report))
            _fsync(stage)
            from .dataset import _publish_no_replace
            _publish_no_replace(stage, output_dir)
            return output_dir / "build-report.json"
        finally:
            if stage.exists():
                shutil.rmtree(stage)
    except (ReleaseError, ReleaseBuildError, ContractError, SourceError, OSError,
            ValueError, TypeError, KeyError, AttributeError) as exc:
        raise ReleaseBuildError("release_build_integrity_error") from exc


def _validate_release_report(report, verified, neighbors, neighbor_raw,
                             registry, registry_sha, *, enforce_current):
    exact_fields(report, REPORT_FIELDS, "build-report.json")
    integer(report["schema_version"], "build-report.json.schema_version", 2, 2)
    _same(report["kind"], "arb-release-build", "build-report.json.kind")
    _same(report["builder_version"], BUILDER_VERSION, "build-report.json.builder_version")
    _same(report["catalog_release_identity"], verified.summary.identity,
          "build-report.json.catalog_release_identity")
    _same(report["algorithm"], ALGORITHM, "build-report.json.algorithm")
    _same(report["registry_sha256"], registry_sha, "build-report.json.registry_sha256")
    _same(report["sources"], _sources(verified, registry, registry_sha), "build-report.json.sources")
    _same(report["inputs"], {
        "release-manifest.json": {"bytes": len(verified.manifest_raw),
                                  "sha256": _sha(verified.manifest_raw)},
        "catalog.json": {"bytes": len(verified.catalog_raw),
                         "sha256": verified.summary.catalog_sha256},
        "glove.6B.300d.txt": {"size": registry["sources"][1]["size"],
                             "sha256": registry["sources"][1]["sha256"]}},
          "build-report.json.inputs")
    _same(report["files"], {"catalog.json": {"sha256": verified.summary.catalog_sha256},
                            "neighbors.json": {"sha256": _sha(neighbor_raw)}},
          "build-report.json.files")
    producer = report["producer"]
    exact_fields(producer, {"package_version", "python_version", "numpy_version", "code_files"},
                 "build-report.json.producer")
    _same(producer["package_version"], __version__, "build-report.json.producer.package_version")
    require(isinstance(producer["python_version"], str)
            and re.fullmatch(r"3\.12\.[0-9]+", producer["python_version"]) is not None,
            "build-report.json.producer.python_version", "expected Python 3.12")
    _same(producer["numpy_version"], "2.5.3", "build-report.json.producer.numpy_version")
    _same(producer["code_files"], {name: _code_sha(name) for name in
                                   ("build.py", "release_build.py")},
          "build-report.json.producer.code_files")
    _validate_diagnostics(report, verified.catalog, neighbors)


def export_release_bundle(release_dir, neighbors_path, build_report_path, output_dir, *,
                          expected_release_identity, base_dir, aliases_dir, alias_data_dir,
                          _base_spec=None, _alias_spec=None, _glove_registry_bytes=None) -> Path:
    """Validate every retained input and publish the unchanged v1 runtime wire format."""
    try:
        verified = _verified(release_dir, expected_release_identity=expected_release_identity,
                             base_dir=base_dir, aliases_dir=aliases_dir,
                             alias_data_dir=alias_data_dir,
                             _base_spec=_base_spec, _alias_spec=_alias_spec)
        registry, registry_sha = _glove_registry(_glove_registry_bytes)
        neighbors_raw = _safe_file(Path(neighbors_path))
        report_raw = _safe_file(Path(build_report_path))
        neighbors = parse_canonical(neighbors_raw, "neighbors.json")
        report = parse_canonical(report_raw, "build-report.json")
        check_neighbors(neighbors, check_catalog(verified.catalog))
        _validate_release_report(report, verified, neighbors, neighbors_raw,
                                 registry, registry_sha, enforce_current=True)
        sources = [*report["sources"],
                   _source("build_report", BUILDER_VERSION, _sha(report_raw),
                           {"filename": "build-report.json", "producer": report["producer"]}),
                   _source("bundle_exporter", BUILDER_VERSION, _code_sha("release_build.py"),
                           {"filename": "release_build.py", "package_version": __version__,
                            "export_sha256": _code_sha("export.py"),
                            "bundle_sha256": _code_sha("bundle.py")})]
        sources.sort(key=lambda item: (item["name"], item["version"]))
        manifest_raw = canonical_bytes({"schema_version": 1, "algorithm": report["algorithm"],
                                        "sources": sources, "files": report["files"]})
        identity, count = check_bundle_bytes(verified.catalog_raw, neighbors_raw, manifest_raw)
        return _publish(Path(output_dir), {"catalog.json": verified.catalog_raw,
                                           "neighbors.json": neighbors_raw,
                                           "manifest.json": manifest_raw}, identity, count)
    except (ReleaseError, ReleaseBuildError, ContractError, SourceError, OSError,
            ValueError, TypeError, KeyError, AttributeError) as exc:
        raise ReleaseBuildError("release_build_integrity_error") from exc
