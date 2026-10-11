"""Compose and verify an immutable research catalog with pinned title aliases."""

import hashlib
import os
import platform
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from . import __version__
from .aliases import AliasError, AliasSpec, _load_spec as _load_alias_spec, validate_aliases
from .bundle import ContractError, canonical_bytes, check_catalog, parse_canonical
from .dataset import DatasetSpec, _load_spec as _load_base_spec

POLICY = "ordered-alias-overlay-v1"
IDENTITY = re.compile(r"sha256:[0-9a-f]{64}\Z")
HEX = re.compile(r"[0-9a-f]{64}\Z")


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
    manifest: dict
    manifest_raw: bytes
    catalog: dict
    catalog_raw: bytes
    base_manifest: dict
    alias_manifest: dict


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _equal(left, right) -> bool:
    return canonical_bytes(left) == canonical_bytes(right)


def _read(path: Path) -> tuple[dict, bytes]:
    if path.is_symlink() or not path.is_file():
        raise ReleaseError("release_integrity_error")
    raw = path.read_bytes()
    return parse_canonical(raw, path.name), raw


def _descriptor(raw: bytes) -> dict:
    return {"bytes": len(raw), "sha256": _sha(raw)}


def _valid_descriptor(value, raw: bytes) -> bool:
    return (isinstance(value, dict) and set(value) == {"bytes", "sha256"}
            and type(value["bytes"]) is int and value["bytes"] >= 0
            and isinstance(value["sha256"], str) and HEX.fullmatch(value["sha256"])
            and _equal(value, _descriptor(raw)))


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
        if (not output.parent.is_dir() or output.parent.is_symlink()
                or output.exists() or output.is_symlink()):
            raise ReleaseError("release_integrity_error")
        target = output.resolve(strict=False)
        for source in inputs:
            origin = source.resolve()
            if target == origin or target.is_relative_to(origin) or origin.is_relative_to(target):
                raise ReleaseError("release_integrity_error")


def _inputs(base_dir: Path, aliases_dir: Path, alias_data_dir: Path,
            base_identity: str, alias_identity: str,
            base_spec: DatasetSpec | None, alias_spec: AliasSpec | None):
    """One upstream full validation, followed by retained, hash-bound consumed bytes."""
    _identity(base_identity)
    _identity(alias_identity)
    _paths((base_dir, aliases_dir, alias_data_dir))
    summary = validate_aliases(aliases_dir, data_dir=alias_data_dir, base_dir=base_dir,
                               expected_base_identity=base_identity,
                               expected_identity=alias_identity,
                               _spec=alias_spec, _base_spec=base_spec)
    base_manifest, base_raw = _read(base_dir / "dataset-manifest.json")
    alias_manifest, alias_raw = _read(aliases_dir / "alias-manifest.json")
    if "sha256:" + _sha(base_raw) != base_identity or "sha256:" + _sha(alias_raw) != alias_identity:
        raise ReleaseError("release_integrity_error")
    base_catalog, base_catalog_raw = _read(base_dir / "catalog.json")
    aliases, aliases_raw = _read(aliases_dir / "aliases.json")
    _, base_registry, _, base_registry_sha, _ = _load_base_spec(base_spec)
    _, alias_registry, _, alias_registry_sha, _ = _load_alias_spec(alias_spec)
    if (not _equal(base_manifest.get("provenance"), base_registry["provenance"])
            or not _equal(base_manifest.get("sources"), base_registry["sources"])
            or not _equal(base_manifest.get("registry_sha256"), base_registry_sha)
            or not _equal(alias_manifest.get("provenance"), alias_registry["provenance"])
            or not _equal(alias_manifest.get("sources"), alias_registry["sources"])
            or not _equal(alias_manifest.get("registry_sha256"), alias_registry_sha)):
        raise ReleaseError("release_integrity_error")
    check_catalog(base_catalog)
    base_block = alias_manifest.get("base")
    expected_base = {"snapshot_id": base_manifest["snapshot_id"], "identity": base_identity,
                     "catalog": base_manifest["files"]["catalog.json"],
                     "catalog_items": len(base_catalog["anime"])}
    if (not _valid_descriptor(base_manifest["files"]["catalog.json"], base_catalog_raw)
            or not _valid_descriptor(alias_manifest["files"]["aliases.json"], aliases_raw)
            or not isinstance(base_block, dict) or not _equal(base_block, expected_base)
            or type(base_block.get("catalog_items")) is not int
            or not isinstance(aliases, dict) or type(aliases.get("schema_version")) is not int
            or aliases["schema_version"] != 1 or set(aliases) != {"schema_version", "aliases"}
            or not isinstance(aliases["aliases"], dict)
            or len(aliases["aliases"]) != summary.items):
        raise ReleaseError("release_integrity_error")
    return summary, base_manifest, base_raw, base_catalog, base_catalog_raw, alias_manifest, alias_raw, aliases, aliases_raw


def _overlay(base: dict, aliases: dict, enrichment_items: int) -> tuple[dict, dict]:
    anime = {}
    counts = {"catalog_count": len(base["anime"]), "enrichment_items": enrichment_items,
              "catalog_aliases_changed_items": 0, "first_alias_changed_items": 0,
              "alias_strings_before": 0, "alias_strings_after": 0}
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
        counts["first_alias_changed_items"] += (combined[0] if combined else None) != (previous[0] if previous else None)
        anime[key] = {**entry, "aliases": combined}
    catalog = {"schema_version": 1, "anime": anime}
    check_catalog(catalog)
    return catalog, counts


def _expected(base_identity, alias_identity, loaded, catalog_raw, counts):
    (_, bm, bm_raw, _, bc_raw, am, am_raw, _, aliases_raw) = loaded
    return {
        "schema_version": 1, "kind": "arb-catalog-release",
        "normalizer": {"name": "neelagiri-nafiul-overlay", "version": "1",
                       "code_files_sha256": {"release.py": _sha(Path(__file__).read_bytes())}},
        "policy": POLICY,
        "inputs": {
            "base": {"snapshot_id": bm["snapshot_id"], "identity": base_identity,
                     "manifest": _descriptor(bm_raw), "catalog": _descriptor(bc_raw)},
            "aliases": {"snapshot_id": am["snapshot_id"], "identity": alias_identity,
                        "manifest": _descriptor(am_raw), "aliases": _descriptor(aliases_raw)}},
        "producer": {"package_version": __version__, "python_version": platform.python_version()},
        "counts": counts, "files": {"catalog.json": _descriptor(catalog_raw)},
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
    from .dataset import _publish_no_replace
    _publish_no_replace(stage, target)


def compose_release(base_dir, aliases_dir, alias_data_dir, output_dir, *,
                    expected_base_identity, expected_alias_identity,
                    _base_spec=None, _alias_spec=None) -> Path:
    base_dir, aliases_dir, alias_data_dir, output_dir = map(
        Path, (base_dir, aliases_dir, alias_data_dir, output_dir))
    try:
        _paths((base_dir, aliases_dir, alias_data_dir), output_dir)
        loaded = _inputs(base_dir, aliases_dir, alias_data_dir, expected_base_identity,
                         expected_alias_identity, _base_spec, _alias_spec)
        catalog, counts = _overlay(loaded[3], loaded[7], loaded[0].items)
        catalog_raw = canonical_bytes(catalog)
        manifest = _expected(expected_base_identity, expected_alias_identity, loaded, catalog_raw, counts)
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
    except (AliasError, ContractError, OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        raise ReleaseError("release_integrity_error") from exc


def _verified_release(release_dir, *, base_dir, aliases_dir, alias_data_dir,
                      expected_identity, _base_spec=None, _alias_spec=None) -> VerifiedRelease:
    release_dir, base_dir, aliases_dir, alias_data_dir = map(
        Path, (release_dir, base_dir, aliases_dir, alias_data_dir))
    try:
        _identity(expected_identity)
        _paths((release_dir, base_dir, aliases_dir, alias_data_dir))
        if {path.name for path in release_dir.iterdir()} != {"catalog.json", "release-manifest.json"}:
            raise ReleaseError("release_integrity_error")
        manifest, manifest_raw = _read(release_dir / "release-manifest.json")
        if "sha256:" + _sha(manifest_raw) != expected_identity:
            raise ReleaseError("release_integrity_error")
        base_identity = manifest["inputs"]["base"]["identity"]
        alias_identity = manifest["inputs"]["aliases"]["identity"]
        loaded = _inputs(base_dir, aliases_dir, alias_data_dir, base_identity,
                         alias_identity, _base_spec, _alias_spec)
        catalog, counts = _overlay(loaded[3], loaded[7], loaded[0].items)
        expected_raw = canonical_bytes(catalog)
        actual, actual_raw = _read(release_dir / "catalog.json")
        if actual_raw != expected_raw or not _equal(actual, catalog):
            raise ReleaseError("release_integrity_error")
        expected = _expected(base_identity, alias_identity, loaded, expected_raw, counts)
        # Historical producing code remains valid when all trusted inputs and output bytes agree.
        historical = manifest.get("normalizer")
        if (not isinstance(historical, dict) or set(historical) != {"name", "version", "code_files_sha256"}
                or historical.get("name") != expected["normalizer"]["name"]
                or historical.get("version") != "1"
                or not isinstance(historical.get("code_files_sha256"), dict)
                or set(historical["code_files_sha256"]) != {"release.py"}
                or not isinstance(historical["code_files_sha256"]["release.py"], str)
                or not HEX.fullmatch(historical["code_files_sha256"]["release.py"])):
            raise ReleaseError("release_integrity_error")
        historical_producer = manifest.get("producer")
        if (not isinstance(historical_producer, dict)
                or set(historical_producer) != {"package_version", "python_version"}
                or any(not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.+_-]{0,79}", value)
                       for value in historical_producer.values())):
            raise ReleaseError("release_integrity_error")
        expected["normalizer"] = historical
        expected["producer"] = historical_producer
        if not _equal(manifest, expected):
            raise ReleaseError("release_integrity_error")
        summary = ReleaseSummary(
            expected_identity, base_identity, alias_identity, _sha(actual_raw), len(actual_raw),
            counts["catalog_count"], counts["enrichment_items"],
            counts["catalog_aliases_changed_items"], counts["first_alias_changed_items"],
            counts["alias_strings_before"], counts["alias_strings_after"])
        return VerifiedRelease(summary, manifest, manifest_raw, catalog, actual_raw, loaded[1], loaded[5])
    except (AliasError, ContractError, OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        raise ReleaseError("release_integrity_error") from exc


def validate_release(release_dir, *, base_dir, aliases_dir, alias_data_dir,
                     expected_identity, _base_spec=None, _alias_spec=None) -> ReleaseSummary:
    return _verified_release(release_dir, base_dir=base_dir, aliases_dir=aliases_dir,
                             alias_data_dir=alias_data_dir, expected_identity=expected_identity,
                             _base_spec=_base_spec, _alias_spec=_alias_spec).summary
