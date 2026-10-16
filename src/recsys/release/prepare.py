"""Offline preparation of the named immutable catalog release."""

import os
import platform
import shutil
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import cast

from recsys import __version__
from recsys.contracts.bundle import (
    Manifest,
    ManifestSource,
    Neighbors,
    check_bundle_bytes,
    parse_canonical,
)
from recsys.contracts.provenance import SOURCE_LAYOUT, source_fingerprints
from recsys.contracts.resources import resource_bytes
from recsys.data.aliases import AliasSpec
from recsys.data.dataset import DatasetSpec
from recsys.data.sources import SourceRegistry
from recsys.release.build import BuildReport
from recsys.release.catalog import VerifiedRelease
from recsys.release.lock import (
    CURRENT_RELEASE,
    LEGACY_RELEASE,
    PrepareError,
    ReleaseLock,
    _fail,
    _sha,
    release_info,
)

RUNTIME_FILES = ("catalog.json", "neighbors.json", "manifest.json")


def _read(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        _fail()
    return path.read_bytes()


def _safe_layout(data: Path) -> None:
    """Reject symlink ancestors and nested input/output trees before any mutation."""
    paths = [
        data,
        data / "raw",
        data / "datasets",
        data / "aliases",
        data / "releases",
        data / "work",
        data / "bundles",
    ]
    for path in paths:
        for part in (path, *path.parents):
            if part.is_symlink() or (part.exists() and not part.is_dir()):
                _fail("unsafe_release_path")
    resolved = [path.resolve(strict=False) for path in paths[1:]]
    if len(set(resolved)) != len(resolved):
        _fail("unsafe_release_path")


def _separate(paths: Iterable[Path]) -> None:
    resolved: list[Path] = []
    for path in paths:
        for part in (path, *path.parents):
            if part.is_symlink():
                _fail("unsafe_release_path")
        resolved.append(path.resolve(strict=False))
    for index, first in enumerate(resolved):
        for second in resolved[index + 1 :]:
            if first == second or first.is_relative_to(second) or second.is_relative_to(first):
                _fail("unsafe_release_path")


def _producer(lock: ReleaseLock) -> None:
    """Allow current producers only for an explicit version-2 test lock."""
    if lock["schema_version"] == 1:
        _fail("release_reference_producer_required")
    import numpy as np

    expected = lock["reference"]["producer"]
    if (
        expected["source_layout"] != SOURCE_LAYOUT
        or platform.python_version() != expected["python_version"]
        or np.__version__ != expected["numpy_version"]
        or __version__ != expected["package_version"]
        or source_fingerprints() != expected["code_files_sha256"]
    ):
        _fail("release_producer_mismatch")


def _release_pins(verified: VerifiedRelease, lock: ReleaseLock) -> None:
    source = lock["inputs"]
    summary = verified.summary
    if (
        summary.identity != source["catalog_release"]["identity"]
        or summary.base_identity != source["base"]["identity"]
        or summary.alias_identity != source["aliases"]["identity"]
        or verified.base_manifest["snapshot_id"] != source["base"]["snapshot_id"]
        or verified.alias_manifest["snapshot_id"] != source["aliases"]["snapshot_id"]
        or summary.catalog_sha256 != source["catalog_release"]["catalog_sha256"]
        or summary.catalog_count != source["catalog_release"]["catalog_count"]
        or summary.catalog_count != lock["semantic_expectations"]["catalog_count"]
        or summary.catalog_sha256 != lock["reference"]["catalog_sha256"]
    ):
        _fail("release_reference_mismatch")
    seeds = lock["semantic_expectations"]
    for key in ("smoke_seed", "empty_seed"):
        seed = seeds[key]
        if seed is None:
            continue
        entry = verified.catalog["anime"].get(str(seed["mal_id"]))
        if entry is None or entry["title"] != seed["query"]:
            _fail("release_semantic_mismatch")
    queries = [v["title"] for v in verified.catalog["anime"].values()]
    for seed in (seeds["smoke_seed"], seeds["empty_seed"]):
        if seed is None:
            continue
        if queries.count(seed["query"]) != 1:
            _fail("release_semantic_mismatch")


def _sources_pins(sources: list[ManifestSource], lock: ReleaseLock) -> None:
    by_name = {item["name"]: item for item in sources}
    if len(by_name) != len(sources):
        _fail("release_reference_mismatch")
    inp = lock["inputs"]
    ref = lock["reference"]
    builder_name = "release/build.py" if lock["schema_version"] == 2 else "release_build.py"
    expected = {
        "research_dataset": inp["base"]["identity"][7:],
        "alias_enrichment": inp["aliases"]["identity"][7:],
        "catalog_release": inp["catalog_release"]["identity"][7:],
        "normalized_catalog": inp["catalog_release"]["catalog_sha256"],
        "glove_300d": inp["embeddings"]["text_sha256"],
        "glove_source_registry": inp["embeddings"]["registry_sha256"],
        "build_report": ref["build_report_sha256"],
        "bundle_exporter": ref["producer"]["code_files_sha256"][builder_name],
    }
    if set(by_name) != set(expected) or any(by_name[k]["sha256"] != v for k, v in expected.items()):
        _fail("release_reference_mismatch")
    if lock["schema_version"] == 2:
        hashes = ref["producer"]["code_files_sha256"]
        report_producer = {
            "package_version": ref["producer"]["package_version"],
            "python_version": ref["producer"]["python_version"],
            "numpy_version": ref["producer"]["numpy_version"],
            "source_layout": SOURCE_LAYOUT,
            "code_files": hashes,
        }
        if (
            by_name["build_report"]["version"] != "arb031-v2"
            or by_name["build_report"]["metadata"]
            != {"filename": "build-report.json", "producer": report_producer}
            or by_name["bundle_exporter"]["version"] != "arb031-v2"
            or by_name["bundle_exporter"]["metadata"]
            != {
                "filename": "release/build.py",
                "package_version": ref["producer"]["package_version"],
                "export_sha256": hashes["release/export.py"],
                "bundle_sha256": hashes["contracts/bundle.py"],
            }
        ):
            _fail("release_reference_mismatch")
        return
    if by_name["build_report"]["metadata"]["producer"] != {
        "package_version": ref["producer"]["package_version"],
        "python_version": ref["producer"]["python_version"],
        "numpy_version": ref["producer"]["numpy_version"],
        "code_files": {
            k: ref["producer"]["code_files_sha256"][k] for k in ("build.py", "release_build.py")
        },
    } or by_name["bundle_exporter"]["metadata"] != {
        "filename": "release_build.py",
        "package_version": ref["producer"]["package_version"],
        "export_sha256": ref["producer"]["code_files_sha256"]["export.py"],
        "bundle_sha256": ref["producer"]["code_files_sha256"]["bundle.py"],
    }:
        _fail("release_reference_mismatch")


def _bundle_files(
    directory: Path, verified: VerifiedRelease, lock: ReleaseLock
) -> tuple[dict[str, bytes], str, int]:
    if (
        directory.is_symlink()
        or not directory.is_dir()
        or {p.name for p in directory.iterdir()} != set(RUNTIME_FILES)
    ):
        _fail("release_bundle_invalid")
    files = {name: _read(directory / name) for name in RUNTIME_FILES}
    identity, count = check_bundle_bytes(*(files[name] for name in RUNTIME_FILES))
    ref = lock["reference"]
    if (
        identity != ref["bundle_identity"]
        or count != verified.summary.catalog_count
        or files["catalog.json"] != verified.catalog_raw
        or _sha(files["neighbors.json"]) != ref["neighbors_sha256"]
    ):
        _fail("release_reference_mismatch")
    manifest = cast("Manifest", parse_canonical(files["manifest.json"], "manifest.json"))
    if manifest["algorithm"] != lock["inputs"]["algorithm"]:
        _fail("release_reference_mismatch")
    _sources_pins(manifest["sources"], lock)
    return files, identity, count


def _report_pins(report_raw: bytes, neighbors_raw: bytes, lock: ReleaseLock) -> None:
    ref, inp, sem = lock["reference"], lock["inputs"], lock["semantic_expectations"]
    report = cast("BuildReport", parse_canonical(report_raw, "build-report.json"))
    neighbors = cast("Neighbors", parse_canonical(neighbors_raw, "neighbors.json"))
    glove_input = cast("dict[str, object]", report["inputs"]["glove.6B.300d.txt"])
    if lock["schema_version"] == 2:
        producer = ref["producer"]
        if (
            type(report) is not dict
            or report.get("schema_version") != 3
            or report.get("builder_version") != "arb031-v2"
            or report.get("producer")
            != {
                "package_version": producer["package_version"],
                "python_version": producer["python_version"],
                "numpy_version": producer["numpy_version"],
                "source_layout": SOURCE_LAYOUT,
                "code_files": producer["code_files_sha256"],
            }
        ):
            _fail("release_reference_mismatch")
    elif report.get("schema_version") != 2 or report.get("builder_version") != "arb031-v1":
        _fail("release_reference_mismatch")
    if (
        _sha(report_raw) != ref["build_report_sha256"]
        or _sha(neighbors_raw) != ref["neighbors_sha256"]
        or report["algorithm"] != inp["algorithm"]
        or report["catalog_release_identity"] != inp["catalog_release"]["identity"]
        or report["registry_sha256"] != inp["embeddings"]["registry_sha256"]
        or glove_input
        != {"size": inp["embeddings"]["text_size"], "sha256": inp["embeddings"]["text_sha256"]}
        or report["coverage"]["catalog_count"] != sem["catalog_count"]
        or report["coverage"]["vector_status_counts"] != sem["vector_status_counts"]
        or report["coverage"]["zero_vector_ids"] != sem["zero_vector_ids"]
        or report["coverage"]["neighbor_count_histogram"] != sem["neighbor_count_histogram"]
        or [item["mal_id"] for item in neighbors["neighbors"][str(sem["smoke_seed"]["mal_id"])]]
        != sem["smoke_seed"]["neighbor_ids"]
        or (
            sem["empty_seed"] is not None
            and neighbors["neighbors"][str(sem["empty_seed"]["mal_id"])] != []
        )
    ):
        _fail("release_reference_mismatch")


def _published_permissions(path: Path) -> None:
    os.chmod(path, 0o755)
    for name in RUNTIME_FILES:
        os.chmod(path / name, 0o644)


def _prepare_current(
    data: Path,
    lock: ReleaseLock,
    *,
    _base_spec: DatasetSpec | None = None,
    _alias_spec: AliasSpec | None = None,
    _glove_registry_bytes: bytes | None = None,
) -> tuple[Path, str, int]:
    from recsys.release.build import build_release, export_release_bundle
    from recsys.release.catalog import _verified_release, compose_release
    from recsys.release.export import _publish

    base = data / "datasets" / lock["inputs"]["base"]["identity"].replace(":", "-")
    aliases = data / "aliases" / lock["inputs"]["aliases"]["identity"].replace(":", "-")
    release = data / "releases" / lock["inputs"]["catalog_release"]["identity"].replace(":", "-")
    nafiul = data / "raw" / lock["inputs"]["aliases"]["snapshot_id"]
    glove_dir = data / "raw" / lock["inputs"]["embeddings"]["snapshot_id"]
    glove = glove_dir / "glove.6B.300d.txt"
    target = data / "bundles" / lock["reference"]["bundle_identity"].replace(":", "-")
    work = data / "work"
    _separate((base, aliases, release, nafiul, glove_dir, work, data / "bundles"))
    if not all(p.is_dir() and not p.is_symlink() for p in (base, aliases, nafiul)):
        _fail("release_prerequisite_missing")
    if not release.exists():
        _producer(lock)
        from recsys.contracts.io import publish_no_replace

        work.mkdir(parents=True, exist_ok=True)
        compose_stage = Path(tempfile.mkdtemp(prefix="prepare-release-", dir=work))
        os.chmod(compose_stage, 0o700)
        candidate_release = compose_stage / "release"
        _separate((base, aliases, nafiul, glove_dir, candidate_release, release, data / "bundles"))
        try:
            compose_release(
                base,
                aliases,
                nafiul,
                candidate_release,
                expected_base_identity=lock["inputs"]["base"]["identity"],
                expected_alias_identity=lock["inputs"]["aliases"]["identity"],
                _base_spec=_base_spec,
                _alias_spec=_alias_spec,
            )
            verified = _verified_release(
                candidate_release,
                base_dir=base,
                aliases_dir=aliases,
                alias_data_dir=nafiul,
                expected_identity=lock["inputs"]["catalog_release"]["identity"],
                _base_spec=_base_spec,
                _alias_spec=_alias_spec,
            )
            _release_pins(verified, lock)
            release.parent.mkdir(parents=True, exist_ok=True)
            publish_no_replace(candidate_release, release)
        finally:
            shutil.rmtree(compose_stage)
    else:
        verified = _verified_release(
            release,
            base_dir=base,
            aliases_dir=aliases,
            alias_data_dir=nafiul,
            expected_identity=lock["inputs"]["catalog_release"]["identity"],
            _base_spec=_base_spec,
            _alias_spec=_alias_spec,
        )
        _release_pins(verified, lock)
    if target.exists() or target.is_symlink():
        _, identity, count = _bundle_files(target, verified, lock)
        _published_permissions(target)
        return target, identity, count
    _producer(lock)
    if _glove_registry_bytes is None:
        _glove_registry_bytes = resource_bytes(
            "registries", "source_registry_glove_6b_300d_v1.json"
        )
    if _sha(_glove_registry_bytes) != lock["inputs"]["embeddings"]["registry_sha256"]:
        _fail("release_reference_mismatch")
    if not glove_dir.is_dir() or glove.is_symlink() or not glove.is_file():
        _fail("release_prerequisite_missing")
    if glove.stat().st_size != lock["inputs"]["embeddings"]["text_size"]:
        _fail("release_reference_mismatch")
    work.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="prepare-", dir=work))
    os.chmod(stage, 0o700)
    build_dir, export_store = stage / "build", stage / "export-store"
    _separate(
        (base, aliases, release, nafiul, glove_dir, build_dir, export_store, data / "bundles")
    )
    try:
        report = build_release(
            release,
            glove,
            build_dir,
            expected_release_identity=verified.summary.identity,
            base_dir=base,
            aliases_dir=aliases,
            alias_data_dir=nafiul,
            block_size=lock["reference"]["block_size"],
            _base_spec=_base_spec,
            _alias_spec=_alias_spec,
            _glove_registry_bytes=_glove_registry_bytes,
        )
        neighbors = build_dir / "neighbors.json"
        _report_pins(_read(report), _read(neighbors), lock)
        candidate = export_release_bundle(
            release,
            neighbors,
            report,
            export_store,
            expected_release_identity=verified.summary.identity,
            base_dir=base,
            aliases_dir=aliases,
            alias_data_dir=nafiul,
            _base_spec=_base_spec,
            _alias_spec=_alias_spec,
            _glove_registry_bytes=_glove_registry_bytes,
        )
        files, identity, count = _bundle_files(candidate, verified, lock)
        published = _publish(data / "bundles", files, identity, count)
        _published_permissions(published)
        return published, identity, count
    finally:
        shutil.rmtree(stage)


def _prepare_legacy(
    data: Path, registry: tuple[SourceRegistry, str] | None = None
) -> tuple[Path, str, int]:
    from recsys.content.glove import build
    from recsys.contracts.bundle import check_bundle
    from recsys.data.mal2020 import normalize
    from recsys.data.obtain import obtain
    from recsys.release.export import export_bundle

    raw, work = data / "raw" / LEGACY_RELEASE, data / "work"
    glove = data / "raw" / "glove-6b-300d-v1" / "glove.6B.300d.txt"
    _separate((raw, glove.parent, work, data / "bundles"))
    if not raw.is_dir() or glove.is_symlink() or not glove.is_file():
        _fail("release_prerequisite_missing")
    if registry is None:
        obtain(raw, offline=True, snapshot=LEGACY_RELEASE)
    else:
        obtain(raw, offline=True, registry=registry[0], registry_sha256=registry[1])
    work.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="prepare-legacy-", dir=work))
    os.chmod(stage, 0o700)
    try:
        normalized = stage / "normalized"
        built = stage / "build"
        source_registry = registry[0] if registry is not None else None
        registry_sha = registry[1] if registry is not None else None
        report = normalize(
            raw / "anime.csv",
            raw / "anime_with_synopsis.csv",
            normalized,
            registry=source_registry,
            registry_sha256=registry_sha,
        )
        build_report = build(
            normalized / "catalog.json",
            report,
            glove,
            built,
            registry=source_registry,
            registry_sha256=registry_sha,
        )
        bundle = export_bundle(
            normalized / "catalog.json",
            built / "neighbors.json",
            build_report,
            report,
            data / "bundles",
            registry=source_registry,
            registry_sha256=registry_sha,
        )
        identity, count = check_bundle(bundle)
        _published_permissions(bundle)
        return bundle, identity, count
    finally:
        shutil.rmtree(stage)


def prepare_release(
    data_dir: str | Path,
    selector: str = "current",
    *,
    _lock_bytes: bytes | None = None,
    _base_spec: DatasetSpec | None = None,
    _alias_spec: AliasSpec | None = None,
    _glove_registry_bytes: bytes | None = None,
    _legacy_registry: tuple[SourceRegistry, str] | None = None,
) -> tuple[Path, str, int]:
    """Return (bundle path, identity, count) after complete offline verification."""
    if selector not in ("current", CURRENT_RELEASE, LEGACY_RELEASE):
        _fail("unknown_release")
    from recsys.data.sources import SourceError

    data = Path(data_dir).absolute()
    _safe_layout(data)
    try:
        if selector == LEGACY_RELEASE:
            if any(
                x is not None for x in (_lock_bytes, _base_spec, _alias_spec, _glove_registry_bytes)
            ):
                _fail("release_lock_invalid")
            return _prepare_legacy(data, _legacy_registry)
        if _legacy_registry is not None:
            _fail("release_lock_invalid")
        lock = release_info(selector, _lock_bytes=_lock_bytes)
        if lock["schema_version"] == 1:
            release = (
                data / "releases" / lock["inputs"]["catalog_release"]["identity"].replace(":", "-")
            )
            bundle = data / "bundles" / lock["reference"]["bundle_identity"].replace(":", "-")
            if not release.is_dir() or not bundle.is_dir():
                _fail("release_reference_producer_required")
        return _prepare_current(
            data,
            lock,
            _base_spec=_base_spec,
            _alias_spec=_alias_spec,
            _glove_registry_bytes=_glove_registry_bytes,
        )
    except PrepareError:
        raise
    except (OSError, SourceError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise PrepareError("release_prepare_integrity_error") from exc
