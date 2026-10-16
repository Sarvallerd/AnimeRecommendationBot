"""Build and export strict v2 artifacts from a verified catalog release."""

import os
import platform
import re
import shutil
import tempfile
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import TypedDict, cast

from recsys import __version__
from recsys.content.glove import (
    SCORE_BLOCK_BUDGET,
    AnimeDiagnostics,
    Coverage,
    _effective_rows,
    _tokens,
    build_vectors,
    cosine_neighbors,
    load_embeddings,
)
from recsys.contracts.bundle import (
    ContractError,
    JSONValue,
    ManifestSource,
    Neighbors,
    canonical_bytes,
    check_bundle_bytes,
    check_catalog,
    check_neighbors,
    exact_fields,
    integer,
    parse_canonical,
    require,
)
from recsys.contracts.provenance import SOURCE_LAYOUT, source_fingerprints
from recsys.contracts.resources import resource_bytes
from recsys.data.aliases import AliasSpec
from recsys.data.dataset import DatasetSpec
from recsys.data.sources import SourceError, SourceRegistry, validate_registry
from recsys.release.catalog import ReleaseError, VerifiedRelease, _sha, _verified_release
from recsys.release.export import ALGORITHM, _code_sha, _publish, _same, _validate_diagnostics

BUILDER_VERSION = "arb031-v2"
REPORT_FIELDS = {
    "schema_version",
    "kind",
    "builder_version",
    "catalog_release_identity",
    "algorithm",
    "sources",
    "registry_sha256",
    "inputs",
    "files",
    "producer",
    "execution",
    "coverage",
    "anime",
}


class ReleaseBuildError(ValueError):
    """A fixed-code v2 build or export failure."""


def _safe_file(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ReleaseBuildError("release_build_integrity_error")
    return path.read_bytes()


class BuildProducer(TypedDict):
    package_version: str
    python_version: str
    numpy_version: str
    source_layout: str
    code_files: dict[str, str]


class BuildReport(TypedDict):
    schema_version: int
    kind: str
    builder_version: str
    catalog_release_identity: str
    algorithm: dict[str, object]
    sources: list[ManifestSource]
    registry_sha256: str
    inputs: dict[str, object]
    files: dict[str, object]
    producer: BuildProducer
    execution: dict[str, int]
    coverage: Coverage
    anime: dict[str, AnimeDiagnostics]


def _glove_registry(raw: bytes | None) -> tuple[SourceRegistry, str]:
    if raw is None:
        raw = resource_bytes("registries", "source_registry_glove_6b_300d_v1.json")
    # Registry is distributed in readable JSON, and its original bytes are the pinned identity.
    from recsys.data.aliases import _json_unique

    value = _json_unique(raw)
    validate_registry(value)
    registry = cast("SourceRegistry", value)
    glove_provenance = registry["provenance"].get("glove")
    if (
        set(registry) != {"schema_version", "provenance", "sources"}
        or type(registry["schema_version"]) is not int
        or registry["schema_version"] != 1
        or set(registry["provenance"]) != {"glove"}
        or not isinstance(glove_provenance, dict)
        or [item["id"] for item in registry["sources"]] != ["glove_zip", "glove_300d"]
    ):
        raise ReleaseBuildError("release_build_integrity_error")
    return registry, _sha(raw)


def _source(name: str, version: str, digest: str, metadata: dict[str, JSONValue]) -> ManifestSource:
    return {"name": name, "version": version, "sha256": digest, "metadata": metadata}


def _sources(
    verified: VerifiedRelease, registry: SourceRegistry, registry_sha: str
) -> list[ManifestSource]:
    base = cast("dict[str, JSONValue]", verified.base_manifest)
    alias = cast("dict[str, JSONValue]", verified.alias_manifest)
    release = verified.manifest
    base_normalizer = cast("dict[str, str]", base["normalizer"])
    alias_normalizer = cast("dict[str, JSONValue]", alias["normalizer"])
    descriptions = {item["id"]: item for item in registry["sources"]}
    glove = descriptions["glove_300d"]
    provenance = cast("dict[str, str]", registry["provenance"]["glove"])
    items = [
        _source(
            "research_dataset",
            base_normalizer["version"],
            verified.summary.base_identity.split(":", 1)[1],
            {
                "filename": "dataset-manifest.json",
                "snapshot_id": base["snapshot_id"],
                "registry_sha256": base["registry_sha256"],
                "provenance": base["provenance"],
                "manifest_sha256": release["inputs"]["base"]["manifest"]["sha256"],
            },
        ),
        _source(
            "alias_enrichment",
            cast("str", alias_normalizer["version"]),
            verified.summary.alias_identity.split(":", 1)[1],
            {
                "filename": "alias-manifest.json",
                "snapshot_id": alias["snapshot_id"],
                "policy": alias_normalizer["policy"],
                "base": alias["base"],
                "manifest_sha256": release["inputs"]["aliases"]["manifest"]["sha256"],
            },
        ),
        _source(
            "catalog_release",
            release["normalizer"]["version"],
            verified.summary.identity.split(":", 1)[1],
            {"filename": "release-manifest.json", "policy": release["policy"]},
        ),
        _source(
            "normalized_catalog",
            "ordered-alias-overlay-v1",
            verified.summary.catalog_sha256,
            {"filename": "catalog.json"},
        ),
        _source(
            "glove_300d",
            "6B-300d",
            glove["sha256"],
            {
                "filename": glove["filename"],
                "size": glove["size"],
                "archive_sha256": descriptions["glove_zip"]["sha256"],
                "member": glove["member"],
                "project": provenance["project"],
                "license": provenance["pretrained_vectors_license"],
            },
        ),
        _source(
            "glove_source_registry",
            "1",
            registry_sha,
            {"filename": "resources/registries/source_registry_glove_6b_300d_v1.json"},
        ),
    ]
    return sorted(items, key=lambda item: (item["name"], item["version"]))


def _verified(
    release_dir: str | Path,
    *,
    expected_release_identity: str,
    base_dir: str | Path,
    aliases_dir: str | Path,
    alias_data_dir: str | Path,
    _base_spec: DatasetSpec | None,
    _alias_spec: AliasSpec | None,
) -> VerifiedRelease:
    return _verified_release(
        release_dir,
        base_dir=base_dir,
        aliases_dir=aliases_dir,
        alias_data_dir=alias_data_dir,
        expected_identity=expected_release_identity,
        _base_spec=_base_spec,
        _alias_spec=_alias_spec,
    )


def _write(path: Path, raw: bytes) -> None:
    with open(path, "xb", opener=lambda name, flags: os.open(name, flags, 0o600)) as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def _fsync(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _check_output(output: Path, inputs: Iterable[str | Path]) -> None:
    if (
        not output.parent.is_dir()
        or output.parent.is_symlink()
        or output.exists()
        or output.is_symlink()
    ):
        raise ReleaseBuildError("release_build_integrity_error")
    target = output.resolve(strict=False)
    for item in inputs:
        origin = Path(item).resolve(strict=False)
        if target == origin or target.is_relative_to(origin) or origin.is_relative_to(target):
            raise ReleaseBuildError("release_build_integrity_error")


def _check_export_store(output: Path, input_dirs: Iterable[str | Path]) -> None:
    """Keep publication and its lock outside every immutable input tree."""
    for component in (output, *output.parents):
        if component.is_symlink() or (component.exists() and not component.is_dir()):
            raise ReleaseBuildError("release_build_integrity_error")
    target = output.resolve(strict=False)
    for directory in input_dirs:
        origin = Path(directory).resolve(strict=False)
        if target == origin or target.is_relative_to(origin) or origin.is_relative_to(target):
            raise ReleaseBuildError("release_build_integrity_error")


def build_release(
    release_dir: str | Path,
    glove_path: str | Path,
    output_dir: str | Path,
    *,
    expected_release_identity: str,
    base_dir: str | Path,
    aliases_dir: str | Path,
    alias_data_dir: str | Path,
    block_size: int = 256,
    _base_spec: DatasetSpec | None = None,
    _alias_spec: AliasSpec | None = None,
    _glove_registry_bytes: bytes | None = None,
) -> Path:
    """Validate once, then publish exactly neighbors.json and the v2 report."""
    import numpy as np

    try:
        output_dir = Path(output_dir)
        _check_output(output_dir, (release_dir, base_dir, aliases_dir, alias_data_dir, glove_path))
        verified = _verified(
            release_dir,
            expected_release_identity=expected_release_identity,
            base_dir=base_dir,
            aliases_dir=aliases_dir,
            alias_data_dir=alias_data_dir,
            _base_spec=_base_spec,
            _alias_spec=_alias_spec,
        )
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
        neighbors: Neighbors = {
            "schema_version": 1,
            "neighbors": cosine_neighbors(ids, vectors, block_size=block_size),
        }
        del vectors
        for key, entry in diagnostics["anime"].items():
            entry["neighbor_count"] = len(neighbors["neighbors"][key])
        histogram = Counter(len(items) for items in neighbors["neighbors"].values())
        diagnostics["coverage"]["neighbor_count_histogram"] = {
            str(key): histogram[key] for key in range(6)
        }
        check_neighbors(neighbors, check_catalog(verified.catalog))
        neighbor_raw = canonical_bytes(neighbors)
        report: BuildReport = {
            "schema_version": 3,
            "kind": "arb-release-build",
            "builder_version": BUILDER_VERSION,
            "catalog_release_identity": verified.summary.identity,
            "algorithm": ALGORITHM,
            "sources": _sources(verified, registry, registry_sha),
            "registry_sha256": registry_sha,
            "inputs": {
                "release-manifest.json": {
                    "bytes": len(verified.manifest_raw),
                    "sha256": _sha(verified.manifest_raw),
                },
                "catalog.json": {
                    "bytes": len(verified.catalog_raw),
                    "sha256": verified.summary.catalog_sha256,
                },
                "glove.6B.300d.txt": glove_info,
            },
            "files": {
                "catalog.json": {"sha256": verified.summary.catalog_sha256},
                "neighbors.json": {"sha256": _sha(neighbor_raw)},
            },
            "producer": {
                "package_version": __version__,
                "python_version": platform.python_version(),
                "numpy_version": np.__version__,
                "source_layout": SOURCE_LAYOUT,
                "code_files": source_fingerprints(),
            },
            "execution": {
                "requested_block_rows": block_size,
                "effective_block_rows": _effective_rows(len(ids), block_size),
                "similarity_block_budget_bytes": SCORE_BLOCK_BUDGET,
            },
            "coverage": diagnostics["coverage"],
            "anime": diagnostics["anime"],
        }
        _validate_release_report(
            report, verified, neighbors, neighbor_raw, registry, registry_sha, enforce_current=True
        )
        stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
        os.chmod(stage, 0o700)
        try:
            _write(stage / "neighbors.json", neighbor_raw)
            _write(stage / "build-report.json", canonical_bytes(report))
            _fsync(stage)
            from recsys.contracts.io import publish_no_replace

            publish_no_replace(stage, output_dir)
            return output_dir / "build-report.json"
        finally:
            if stage.exists():
                shutil.rmtree(stage)
    except (
        ReleaseError,
        ReleaseBuildError,
        ContractError,
        SourceError,
        OSError,
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
    ) as exc:
        raise ReleaseBuildError("release_build_integrity_error") from exc


def _validate_release_report(
    report: object,
    verified: VerifiedRelease,
    neighbors: Neighbors,
    neighbor_raw: bytes,
    registry: SourceRegistry,
    registry_sha: str,
    *,
    enforce_current: bool,
) -> None:
    exact_fields(report, REPORT_FIELDS, "build-report.json")
    report = cast("BuildReport", report)
    integer(report["schema_version"], "build-report.json.schema_version", 3, 3)
    _same(report["kind"], "arb-release-build", "build-report.json.kind")
    _same(report["builder_version"], BUILDER_VERSION, "build-report.json.builder_version")
    _same(
        report["catalog_release_identity"],
        verified.summary.identity,
        "build-report.json.catalog_release_identity",
    )
    _same(report["algorithm"], ALGORITHM, "build-report.json.algorithm")
    _same(report["registry_sha256"], registry_sha, "build-report.json.registry_sha256")
    _same(
        report["sources"], _sources(verified, registry, registry_sha), "build-report.json.sources"
    )
    _same(
        report["inputs"],
        {
            "release-manifest.json": {
                "bytes": len(verified.manifest_raw),
                "sha256": _sha(verified.manifest_raw),
            },
            "catalog.json": {
                "bytes": len(verified.catalog_raw),
                "sha256": verified.summary.catalog_sha256,
            },
            "glove.6B.300d.txt": {
                "size": registry["sources"][1]["size"],
                "sha256": registry["sources"][1]["sha256"],
            },
        },
        "build-report.json.inputs",
    )
    _same(
        report["files"],
        {
            "catalog.json": {"sha256": verified.summary.catalog_sha256},
            "neighbors.json": {"sha256": _sha(neighbor_raw)},
        },
        "build-report.json.files",
    )
    producer = report["producer"]
    exact_fields(
        producer,
        {"package_version", "python_version", "numpy_version", "code_files", "source_layout"},
        "build-report.json.producer",
    )
    _same(producer["package_version"], __version__, "build-report.json.producer.package_version")
    require(
        isinstance(producer["python_version"], str)
        and re.fullmatch(r"3\.12\.[0-9]+", producer["python_version"]) is not None,
        "build-report.json.producer.python_version",
        "expected Python 3.12",
    )
    _same(producer["numpy_version"], "2.5.3", "build-report.json.producer.numpy_version")
    _same(producer["source_layout"], SOURCE_LAYOUT, "build-report.json.producer.source_layout")
    _same(producer["code_files"], source_fingerprints(), "build-report.json.producer.code_files")
    _validate_diagnostics(cast("dict[str, object]", report), verified.catalog, neighbors)


def export_release_bundle(
    release_dir: str | Path,
    neighbors_path: str | Path,
    build_report_path: str | Path,
    output_dir: str | Path,
    *,
    expected_release_identity: str,
    base_dir: str | Path,
    aliases_dir: str | Path,
    alias_data_dir: str | Path,
    _base_spec: DatasetSpec | None = None,
    _alias_spec: AliasSpec | None = None,
    _glove_registry_bytes: bytes | None = None,
) -> Path:
    """Validate every retained input and publish the unchanged v1 runtime wire format."""
    try:
        output_dir = Path(output_dir)
        _check_export_store(
            output_dir,
            (
                release_dir,
                base_dir,
                aliases_dir,
                alias_data_dir,
                Path(neighbors_path).parent,
                Path(build_report_path).parent,
            ),
        )
        verified = _verified(
            release_dir,
            expected_release_identity=expected_release_identity,
            base_dir=base_dir,
            aliases_dir=aliases_dir,
            alias_data_dir=alias_data_dir,
            _base_spec=_base_spec,
            _alias_spec=_alias_spec,
        )
        registry, registry_sha = _glove_registry(_glove_registry_bytes)
        neighbors_raw = _safe_file(Path(neighbors_path))
        report_raw = _safe_file(Path(build_report_path))
        neighbors = parse_canonical(neighbors_raw, "neighbors.json")
        report = parse_canonical(report_raw, "build-report.json")
        check_neighbors(neighbors, check_catalog(verified.catalog))
        checked_neighbors = cast("Neighbors", neighbors)
        _validate_release_report(
            report,
            verified,
            checked_neighbors,
            neighbors_raw,
            registry,
            registry_sha,
            enforce_current=True,
        )
        checked_report = cast("BuildReport", report)
        sources = [
            *checked_report["sources"],
            _source(
                "build_report",
                BUILDER_VERSION,
                _sha(report_raw),
                {
                    "filename": "build-report.json",
                    "producer": cast("JSONValue", checked_report["producer"]),
                },
            ),
            _source(
                "bundle_exporter",
                BUILDER_VERSION,
                _code_sha("release/build.py"),
                {
                    "filename": "release/build.py",
                    "package_version": __version__,
                    "export_sha256": _code_sha("release/export.py"),
                    "bundle_sha256": _code_sha("contracts/bundle.py"),
                },
            ),
        ]
        sources.sort(key=lambda item: (item["name"], item["version"]))
        manifest_raw = canonical_bytes(
            {
                "schema_version": 1,
                "algorithm": checked_report["algorithm"],
                "sources": sources,
                "files": checked_report["files"],
            }
        )
        identity, count = check_bundle_bytes(verified.catalog_raw, neighbors_raw, manifest_raw)
        return _publish(
            output_dir,
            {
                "catalog.json": verified.catalog_raw,
                "neighbors.json": neighbors_raw,
                "manifest.json": manifest_raw,
            },
            identity,
            count,
        )
    except (
        ReleaseError,
        ReleaseBuildError,
        ContractError,
        SourceError,
        OSError,
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
    ) as exc:
        raise ReleaseBuildError("release_build_integrity_error") from exc
