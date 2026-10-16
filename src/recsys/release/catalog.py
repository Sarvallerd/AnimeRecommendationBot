"""Compose and verify an immutable research catalog with pinned title aliases."""

import hashlib
import os
import platform
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import NotRequired, TypedDict, cast

from recsys import __version__
from recsys.contracts.bundle import (
    AnimeRecord,
    Catalog,
    ContractError,
    canonical_bytes,
    check_catalog,
    parse_canonical,
)
from recsys.contracts.provenance import (
    LEGACY_V1_PACKAGE_VERSION,
    LEGACY_V1_SOURCE_SHA256,
    SOURCE_FILES,
    source_fingerprints,
)
from recsys.data.aliases import AliasError, AliasPayload, AliasSpec, AliasSummary, validate_aliases
from recsys.data.aliases import _load_spec as _load_alias_spec
from recsys.data.dataset import DatasetSpec
from recsys.data.dataset import _load_spec as _load_base_spec

POLICY = "ordered-alias-overlay-v1"
IDENTITY = re.compile(r"sha256:[0-9a-f]{64}\Z")
HEX = re.compile(r"[0-9a-f]{64}\Z")


class ReleaseDescriptor(TypedDict):
    bytes: int
    sha256: str


class ReleaseNormalizer(TypedDict):
    name: str
    version: str
    code_files_sha256: dict[str, str]


class ReleaseInput(TypedDict):
    snapshot_id: str
    identity: str
    manifest: ReleaseDescriptor
    catalog: NotRequired[ReleaseDescriptor]
    aliases: NotRequired[ReleaseDescriptor]


class ReleaseProducer(TypedDict):
    package_version: str
    python_version: str


class ReleaseManifest(TypedDict):
    schema_version: int
    kind: str
    normalizer: ReleaseNormalizer
    policy: str
    inputs: dict[str, ReleaseInput]
    producer: ReleaseProducer
    counts: dict[str, int]
    files: dict[str, ReleaseDescriptor]


type LoadedInputs = tuple[
    AliasSummary,
    dict[str, object],
    bytes,
    Catalog,
    bytes,
    dict[str, object],
    bytes,
    AliasPayload,
    bytes,
]


class ReleaseError(ValueError):
    """A fixed-code release integrity or publication error."""


@dataclass(frozen=True)
class ReleaseSummary:
    identity: str
    base_identity: str
    alias_identity: str
    catalog_sha256: str
    catalog_bytes: int
    catalog_count: int
    enrichment_items: int
    catalog_aliases_changed_items: int
    first_alias_changed_items: int
    alias_strings_before: int
    alias_strings_after: int


@dataclass(frozen=True)
class VerifiedRelease:
    summary: ReleaseSummary
    manifest: ReleaseManifest
    manifest_raw: bytes
    catalog: Catalog
    catalog_raw: bytes
    base_manifest: dict[str, object]
    alias_manifest: dict[str, object]


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _equal(left: object, right: object) -> bool:
    return canonical_bytes(left) == canonical_bytes(right)


def _read(path: Path) -> tuple[dict[str, object], bytes]:
    if path.is_symlink() or not path.is_file():
        raise ReleaseError("release_integrity_error")
    raw = path.read_bytes()
    value = parse_canonical(raw, path.name)
    if not isinstance(value, dict):
        raise ReleaseError("release_integrity_error")
    return cast("dict[str, object]", value), raw


def _descriptor(raw: bytes) -> ReleaseDescriptor:
    return {"bytes": len(raw), "sha256": _sha(raw)}


def _valid_descriptor(value: object, raw: bytes) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == {"bytes", "sha256"}
        and type(value["bytes"]) is int
        and value["bytes"] >= 0
        and isinstance(value["sha256"], str)
        and HEX.fullmatch(value["sha256"]) is not None
        and _equal(value, _descriptor(raw))
    )


def _identity(value: str) -> None:
    if not isinstance(value, str) or not IDENTITY.fullmatch(value):
        raise ReleaseError("release_integrity_error")


def _paths(inputs: tuple[Path, ...], output: Path | None = None) -> None:
    for source in inputs:
        if source.is_symlink() or not source.is_dir():
            raise ReleaseError("release_integrity_error")
    if len({p.resolve() for p in inputs}) != len(inputs):
        raise ReleaseError("release_integrity_error")
    if output is not None:
        if (
            not output.parent.is_dir()
            or output.parent.is_symlink()
            or output.exists()
            or output.is_symlink()
        ):
            raise ReleaseError("release_integrity_error")
        target = output.resolve(strict=False)
        for source in inputs:
            origin = source.resolve()
            if target == origin or target.is_relative_to(origin) or origin.is_relative_to(target):
                raise ReleaseError("release_integrity_error")


def _inputs(
    base_dir: Path,
    aliases_dir: Path,
    alias_data_dir: Path,
    base_identity: str,
    alias_identity: str,
    base_spec: DatasetSpec | None,
    alias_spec: AliasSpec | None,
) -> LoadedInputs:
    """One upstream full validation, followed by retained, hash-bound consumed bytes."""
    _identity(base_identity)
    _identity(alias_identity)
    _paths((base_dir, aliases_dir, alias_data_dir))
    summary = validate_aliases(
        aliases_dir,
        data_dir=alias_data_dir,
        base_dir=base_dir,
        expected_base_identity=base_identity,
        expected_identity=alias_identity,
        _spec=alias_spec,
        _base_spec=base_spec,
    )
    base_manifest, base_raw = _read(base_dir / "dataset-manifest.json")
    alias_manifest, alias_raw = _read(aliases_dir / "alias-manifest.json")
    if "sha256:" + _sha(base_raw) != base_identity or "sha256:" + _sha(alias_raw) != alias_identity:
        raise ReleaseError("release_integrity_error")
    base_catalog, base_catalog_raw = _read(base_dir / "catalog.json")
    aliases, aliases_raw = _read(aliases_dir / "aliases.json")
    _, base_registry, _, base_registry_sha, _ = _load_base_spec(base_spec)
    _, alias_registry, _, alias_registry_sha, _ = _load_alias_spec(alias_spec)
    if (
        not _equal(base_manifest.get("provenance"), base_registry["provenance"])
        or not _equal(base_manifest.get("sources"), base_registry["sources"])
        or not _equal(base_manifest.get("registry_sha256"), base_registry_sha)
        or not _equal(alias_manifest.get("provenance"), alias_registry["provenance"])
        or not _equal(alias_manifest.get("sources"), alias_registry["sources"])
        or not _equal(alias_manifest.get("registry_sha256"), alias_registry_sha)
    ):
        raise ReleaseError("release_integrity_error")
    check_catalog(base_catalog)
    base_catalog_typed = cast("Catalog", base_catalog)  # Full wire validation above.
    base_files = cast("dict[str, object]", base_manifest["files"])
    alias_files = cast("dict[str, object]", alias_manifest["files"])
    base_block = alias_manifest.get("base")
    expected_base = {
        "snapshot_id": base_manifest["snapshot_id"],
        "identity": base_identity,
        "catalog": base_files["catalog.json"],
        "catalog_items": len(base_catalog_typed["anime"]),
    }
    if (
        not _valid_descriptor(base_files["catalog.json"], base_catalog_raw)
        or not _valid_descriptor(alias_files["aliases.json"], aliases_raw)
        or not isinstance(base_block, dict)
        or not _equal(base_block, expected_base)
        or type(base_block.get("catalog_items")) is not int
        or not isinstance(aliases, dict)
        or type(aliases.get("schema_version")) is not int
        or aliases["schema_version"] != 1
        or set(aliases) != {"schema_version", "aliases"}
        or not isinstance(aliases["aliases"], dict)
        or len(aliases["aliases"]) != summary.items
    ):
        raise ReleaseError("release_integrity_error")
    return (
        summary,
        base_manifest,
        base_raw,
        base_catalog_typed,
        base_catalog_raw,
        alias_manifest,
        alias_raw,
        cast("AliasPayload", aliases),  # Exact alias shape and count checked above.
        aliases_raw,
    )


def _overlay(
    base: Catalog, aliases: AliasPayload, enrichment_items: int
) -> tuple[Catalog, dict[str, int]]:
    anime: dict[str, AnimeRecord] = {}
    counts = {
        "catalog_count": len(base["anime"]),
        "enrichment_items": enrichment_items,
        "catalog_aliases_changed_items": 0,
        "first_alias_changed_items": 0,
        "alias_strings_before": 0,
        "alias_strings_after": 0,
    }
    for key, entry in base["anime"].items():
        previous = entry["aliases"]
        counts["alias_strings_before"] += len(previous)
        addition = aliases["aliases"].get(key)
        if addition is None:
            anime[key] = entry
            counts["alias_strings_after"] += len(previous)
            continue
        texts = [item["text"] for item in addition]
        combined = list(dict.fromkeys([*texts, *previous]))
        counts["alias_strings_after"] += len(combined)
        counts["catalog_aliases_changed_items"] += combined != previous
        counts["first_alias_changed_items"] += (combined[0] if combined else None) != (
            previous[0] if previous else None
        )
        anime[key] = {**entry, "aliases": combined}
    catalog: Catalog = {"schema_version": 1, "anime": anime}
    check_catalog(catalog)
    return catalog, counts


def _expected(
    base_identity: str,
    alias_identity: str,
    loaded: LoadedInputs,
    catalog_raw: bytes,
    counts: dict[str, int],
) -> ReleaseManifest:
    (_, bm, bm_raw, _, bc_raw, am, am_raw, _, aliases_raw) = loaded
    return {
        "schema_version": 1,
        "kind": "arb-catalog-release",
        "normalizer": {
            "name": "neelagiri-nafiul-overlay",
            "version": "2",
            "code_files_sha256": source_fingerprints(),
        },
        "policy": POLICY,
        "inputs": {
            "base": {
                "snapshot_id": cast("str", bm["snapshot_id"]),
                "identity": base_identity,
                "manifest": _descriptor(bm_raw),
                "catalog": _descriptor(bc_raw),
            },
            "aliases": {
                "snapshot_id": cast("str", am["snapshot_id"]),
                "identity": alias_identity,
                "manifest": _descriptor(am_raw),
                "aliases": _descriptor(aliases_raw),
            },
        },
        "producer": {"package_version": __version__, "python_version": platform.python_version()},
        "counts": counts,
        "files": {"catalog.json": _descriptor(catalog_raw)},
    }


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


def _publish(stage: Path, target: Path) -> None:
    from recsys.contracts.io import publish_no_replace

    publish_no_replace(stage, target)


def compose_release(
    base_dir: str | Path,
    aliases_dir: str | Path,
    alias_data_dir: str | Path,
    output_dir: str | Path,
    *,
    expected_base_identity: str,
    expected_alias_identity: str,
    _base_spec: DatasetSpec | None = None,
    _alias_spec: AliasSpec | None = None,
) -> Path:
    base_dir, aliases_dir, alias_data_dir, output_dir = map(
        Path, (base_dir, aliases_dir, alias_data_dir, output_dir)
    )
    try:
        _paths((base_dir, aliases_dir, alias_data_dir), output_dir)
        loaded = _inputs(
            base_dir,
            aliases_dir,
            alias_data_dir,
            expected_base_identity,
            expected_alias_identity,
            _base_spec,
            _alias_spec,
        )
        catalog, counts = _overlay(loaded[3], loaded[7], loaded[0].items)
        catalog_raw = canonical_bytes(catalog)
        manifest = _expected(
            expected_base_identity, expected_alias_identity, loaded, catalog_raw, counts
        )
        stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
        os.chmod(stage, 0o700)
        try:
            _write(stage / "catalog.json", catalog_raw)
            _write(stage / "release-manifest.json", canonical_bytes(manifest))
            _fsync(stage)
            _publish(stage, output_dir)
            return output_dir / "release-manifest.json"
        finally:
            if stage.exists():
                shutil.rmtree(stage)
    except (
        AliasError,
        ContractError,
        OSError,
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
    ) as exc:
        raise ReleaseError("release_integrity_error") from exc


def _verified_release(
    release_dir: str | Path,
    *,
    base_dir: str | Path,
    aliases_dir: str | Path,
    alias_data_dir: str | Path,
    expected_identity: str,
    _base_spec: DatasetSpec | None = None,
    _alias_spec: AliasSpec | None = None,
) -> VerifiedRelease:
    release_dir, base_dir, aliases_dir, alias_data_dir = map(
        Path, (release_dir, base_dir, aliases_dir, alias_data_dir)
    )
    try:
        _identity(expected_identity)
        _paths((release_dir, base_dir, aliases_dir, alias_data_dir))
        if {path.name for path in release_dir.iterdir()} != {
            "catalog.json",
            "release-manifest.json",
        }:
            raise ReleaseError("release_integrity_error")
        manifest, manifest_raw = _read(release_dir / "release-manifest.json")
        if "sha256:" + _sha(manifest_raw) != expected_identity:
            raise ReleaseError("release_integrity_error")
        inputs_raw = manifest.get("inputs")
        if not isinstance(inputs_raw, dict):
            raise ReleaseError("release_integrity_error")
        base_raw = inputs_raw.get("base")
        alias_raw = inputs_raw.get("aliases")
        if not isinstance(base_raw, dict) or not isinstance(alias_raw, dict):
            raise ReleaseError("release_integrity_error")
        base_identity = base_raw.get("identity")
        alias_identity = alias_raw.get("identity")
        if not isinstance(base_identity, str) or not isinstance(alias_identity, str):
            raise ReleaseError("release_integrity_error")
        loaded = _inputs(
            base_dir,
            aliases_dir,
            alias_data_dir,
            base_identity,
            alias_identity,
            _base_spec,
            _alias_spec,
        )
        catalog, counts = _overlay(loaded[3], loaded[7], loaded[0].items)
        expected_raw = canonical_bytes(catalog)
        actual, actual_raw = _read(release_dir / "catalog.json")
        if actual_raw != expected_raw or not _equal(actual, catalog):
            raise ReleaseError("release_integrity_error")
        expected = _expected(base_identity, alias_identity, loaded, expected_raw, counts)
        # The old producer retains its original shape and trusted identity.
        observed_normalizer = manifest.get("normalizer")
        if (
            not isinstance(observed_normalizer, dict)
            or set(observed_normalizer) != {"name", "version", "code_files_sha256"}
            or observed_normalizer.get("name") != expected["normalizer"]["name"]
        ):
            raise ReleaseError("release_integrity_error")
        if observed_normalizer["version"] == "1":
            names = {"release.py"}
            if observed_normalizer["code_files_sha256"] != {
                "release.py": LEGACY_V1_SOURCE_SHA256["release.py"]
            }:
                raise ReleaseError("release_integrity_error")
        elif observed_normalizer["version"] == "2":
            names = set(SOURCE_FILES)
        else:
            raise ReleaseError("release_integrity_error")
        hashes = observed_normalizer["code_files_sha256"]
        if (
            not isinstance(hashes, dict)
            or set(hashes) != names
            or any(
                not isinstance(value, str) or HEX.fullmatch(value) is None
                for value in hashes.values()
            )
        ):
            raise ReleaseError("release_integrity_error")
        historical_producer = manifest.get("producer")
        if (
            not isinstance(historical_producer, dict)
            or set(historical_producer) != {"package_version", "python_version"}
            or any(
                not isinstance(value, str)
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.+_-]{0,79}", value)
                for value in historical_producer.values()
            )
        ):
            raise ReleaseError("release_integrity_error")
        if (
            observed_normalizer["version"] == "1"
            and historical_producer["package_version"] != LEGACY_V1_PACKAGE_VERSION
        ):
            raise ReleaseError("release_integrity_error")
        expected["normalizer"] = cast("ReleaseNormalizer", observed_normalizer)
        expected["producer"] = cast("ReleaseProducer", historical_producer)
        if not _equal(manifest, expected):
            raise ReleaseError("release_integrity_error")
        summary = ReleaseSummary(
            expected_identity,
            base_identity,
            alias_identity,
            _sha(actual_raw),
            len(actual_raw),
            counts["catalog_count"],
            counts["enrichment_items"],
            counts["catalog_aliases_changed_items"],
            counts["first_alias_changed_items"],
            counts["alias_strings_before"],
            counts["alias_strings_after"],
        )
        return VerifiedRelease(
            summary,
            cast("ReleaseManifest", manifest),  # Equal to the fully checked expected manifest.
            manifest_raw,
            catalog,
            actual_raw,
            loaded[1],
            loaded[5],
        )
    except (
        AliasError,
        ContractError,
        OSError,
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
    ) as exc:
        raise ReleaseError("release_integrity_error") from exc


def validate_release(
    release_dir: str | Path,
    *,
    base_dir: str | Path,
    aliases_dir: str | Path,
    alias_data_dir: str | Path,
    expected_identity: str,
    _base_spec: DatasetSpec | None = None,
    _alias_spec: AliasSpec | None = None,
) -> ReleaseSummary:
    return _verified_release(
        release_dir,
        base_dir=base_dir,
        aliases_dir=aliases_dir,
        alias_data_dir=alias_data_dir,
        expected_identity=expected_identity,
        _base_spec=_base_spec,
        _alias_spec=_alias_spec,
    ).summary
