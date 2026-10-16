"""Build and validate a frozen, research-only Neelagiri 2025 dataset."""

import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import zlib
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import date
from importlib import metadata as package_metadata
from pathlib import Path
from typing import TypedDict, cast

from recsys.contracts.bundle import MAX_ID, Catalog, canonical_bytes, check_catalog, parse_canonical
from recsys.contracts.io import (
    ArtifactIOError,
    fsync_directory,
    iter_single_member_gzip,
    publish_no_replace,
)
from recsys.contracts.provenance import (
    LEGACY_V1_PACKAGE_VERSION,
    LEGACY_V1_SOURCE_SHA256,
    SOURCE_FILES,
    source_fingerprints,
)
from recsys.contracts.resources import resource_bytes, resource_path
from recsys.data.interactions import (
    HEADER,
    SCHEMA,
    STATUSES,
    InteractionResult,
    LossReasons,
    build_interactions,
)
from recsys.data.metadata import MetadataFact, MetadataFacts, MetadataResult, normalize_metadata
from recsys.data.sources import SourceDescriptor, SourceError, SourceRegistry, validate_registry

SNAPSHOT = "neelagiri-2025-v1"
REGISTRY_RESOURCE = "source_registry_neelagiri_2025_v1.json"
EXPECTATIONS_RESOURCE = "neelagiri_2025_v1_expectations.json"
OUTPUTS = ("catalog.json", "item-facts.json", "interactions.tsv.gz", "dataset-qa.json")
ALL_OUTPUTS = (*OUTPUTS, "dataset-manifest.json")
LEGACY_CODE_FILES = ("dataset.py", "dataset_metadata.py", "dataset_interactions.py")
HEX64 = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
USER_KEY = re.compile(r"u1_[0-9a-f]{64}\Z", re.ASCII)
POLICIES = {
    "metadata_policy_version": "neelagiri-metadata-v1",
    "interactions_schema": SCHEMA,
    "user_key_scheme": "sha256-arb-user-v1-snapshot-namespace",
    "duplicate_policy": "exact-explicit-triplets-deduplicated",
    "conflict_policy": "whole-pair-quarantine-v1",
    "status_policy": "conflict-to-unknown-v1",
    "positive_threshold": 8,
    "positive_cohort_min_items": 15,
    "recent_cutoff": "2025-10-31",
    "serialization": "canonical-json-utf8-lf-and-sorted-tsv-v1",
    "gzip": "single-member-filename-empty-mtime-zero-level-6",
}


class DatasetError(Exception):
    """Safe, fixed-code public failure for dataset operations."""


class DatasetExpectations(TypedDict):
    schema_version: int
    snapshot_id: str
    source_sha256: dict[str, str]
    counts: dict[str, int]
    distributions: dict[str, dict[str, int]]


class DatasetQa(TypedDict):
    schema_version: int
    counts: dict[str, int]
    distributions: dict[str, dict[str, int]]
    loss_reasons: LossReasons
    metadata_diagnostics: dict[str, int]


class DerivedQa(TypedDict):
    counts: dict[str, int]
    distributions: dict[str, dict[str, int]]


@dataclass(frozen=True)
class DatasetSpec:
    snapshot_id: str
    registry_bytes: bytes
    expectations_resource: str
    expectations_bytes: bytes


@dataclass(frozen=True)
class DatasetSummary:
    identity: str
    catalog_items: int
    interactions: int
    users: int
    positive_users_ge_15: int


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _load_spec(
    spec: DatasetSpec | None,
) -> tuple[DatasetSpec, SourceRegistry, DatasetExpectations, str, str]:
    if spec is None:
        spec = DatasetSpec(
            SNAPSHOT,
            resource_bytes("registries", REGISTRY_RESOURCE),
            resource_path("expectations", EXPECTATIONS_RESOURCE),
            resource_bytes("expectations", EXPECTATIONS_RESOURCE),
        )
    if not isinstance(spec, DatasetSpec) or not isinstance(spec.snapshot_id, str):
        raise DatasetError("dataset_integrity_error")
    try:
        registry = json.loads(spec.registry_bytes)
        validate_registry(registry)
        expected = parse_canonical(spec.expectations_bytes, "expectations")
    except (ValueError, TypeError, UnicodeError, OSError, SourceError) as exc:
        raise DatasetError("dataset_integrity_error") from exc
    if (
        not isinstance(expected, dict)
        or set(expected)
        != {"schema_version", "snapshot_id", "source_sha256", "counts", "distributions"}
        or type(expected["schema_version"]) is not int
        or expected["schema_version"] != 1
        or expected["snapshot_id"] != spec.snapshot_id
        or registry.get("provenance", {}).get("dataset", {}).get("snapshot_id") != spec.snapshot_id
        or spec.expectations_resource
        not in (EXPECTATIONS_RESOURCE, resource_path("expectations", EXPECTATIONS_RESOURCE))
        or len(registry["sources"]) != 2
        or {source["id"] for source in registry["sources"]}
        != {"neelagiri_details_csv", "neelagiri_ratings_csv"}
        or {source["filename"] for source in registry["sources"]} != {"details.csv", "ratings.csv"}
        or expected["source_sha256"]
        != {source["filename"]: source["sha256"] for source in registry["sources"]}
        or not isinstance(expected["counts"], dict)
        or set(expected["counts"]) != _COUNT_KEYS
        or any(type(value) is not int or value < 0 for value in expected["counts"].values())
        or not isinstance(expected["distributions"], dict)
        or set(expected["distributions"]) != _DISTRIBUTION_KEYS
        or any(
            not isinstance(hist, dict) or any(type(v) is not int or v < 0 for v in hist.values())
            for hist in expected["distributions"].values()
        )
    ):
        raise DatasetError("dataset_integrity_error")
    return (
        spec,
        cast("SourceRegistry", registry),
        cast("DatasetExpectations", expected),
        _sha(spec.registry_bytes),
        _sha(spec.expectations_bytes),
    )


_COUNT_KEYS = {
    "metadata_rows",
    "catalog_items",
    "raw_rows",
    "blank_user_rows",
    "zero_or_missing_rows",
    "valid_explicit_rows",
    "raw_explicit_pairs",
    "repeated_explicit_triples",
    "raw_mapped_pairs",
    "unmapped_pairs",
    "conflicting_score_pairs",
    "mixed_unset_pairs",
    "accepted_pairs",
    "accepted_users",
    "accepted_rated_mal_ids",
    "positive_users_ge_15",
    "completed_only_pairs",
    "completed_only_users",
    "completed_only_users_ge_15_positive",
    "recent_2024_plus",
    "recent_2025_plus",
    "rated_2025_not_yet_aired",
    "rated_2026_not_yet_aired",
    "invalid_score_rows",
    "invalid_mal_id_rows",
    "invalid_state_pairs",
    "metadata_duplicate_mal_ids",
    "metadata_invalid_mal_id",
    "metadata_missing_title",
    "metadata_synopsis_boilerplate",
    "metadata_first_release_seasonal_year_disagreement",
    "metadata_missing_synopsis",
}
_DISTRIBUTION_KEYS = {
    "raw_status_histogram",
    "accepted_status_histogram",
    "accepted_score_histogram",
    "rated_items_per_user_bins",
    "positive_items_per_user_bins",
}


def _check_input_manifest(data_dir: Path, registry: SourceRegistry, registry_sha: str) -> None:
    path = data_dir / "source-manifest.json"
    if data_dir.is_symlink() or not data_dir.is_dir() or path.is_symlink() or not path.is_file():
        raise DatasetError("input_hash_mismatch")
    wanted = {
        "schema_version": 1,
        "registry_sha256": registry_sha,
        "provenance": registry["provenance"],
        "verified": [
            {key: source[key] for key in ("filename", "size", "sha256")}
            for source in registry["sources"]
        ],
    }
    try:
        actual = json.loads(path.read_bytes())
    except (OSError, ValueError, UnicodeError) as exc:
        raise DatasetError("input_hash_mismatch") from exc
    if actual != wanted or type(actual.get("schema_version")) is not int:
        raise DatasetError("input_hash_mismatch")


def _copy_verified(source: Path, destination: Path, descriptor: SourceDescriptor) -> None:
    if source.is_symlink() or not source.is_file():
        raise DatasetError("input_hash_mismatch")
    h = hashlib.sha256()
    size = 0
    try:
        with source.open("rb") as inp, destination.open("xb") as out:
            os.chmod(destination, 0o600)
            while chunk := inp.read(1024 * 1024):
                size += len(chunk)
                if size > descriptor["size"]:
                    raise DatasetError("input_hash_mismatch")
                h.update(chunk)
                out.write(chunk)
            out.flush()
            os.fsync(out.fileno())
    except OSError as exc:
        raise DatasetError("input_hash_mismatch") from exc
    if size != descriptor["size"] or h.hexdigest() != descriptor["sha256"]:
        raise DatasetError("input_hash_mismatch")


def _write_json(path: Path, value: object) -> bytes:
    raw = canonical_bytes(value)
    with path.open("xb") as stream:
        os.chmod(path, 0o600)
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return raw


def _recent(facts: dict[str, MetadataFact], rated_ids: frozenset[int] | set[int]) -> dict[str, int]:
    counters: Counter[str] = Counter()
    cutoff = cast("str", POLICIES["recent_cutoff"])  # Frozen policy field is a date string.
    for number in rated_ids:
        fact = facts[str(number)]
        aired = fact["airing_status"] in ("finished_airing", "currently_airing")
        release = fact["first_release_date"]
        if aired and release and "2024-01-01" <= release <= cutoff:
            counters["recent_2024_plus"] += 1
        if aired and release and "2025-01-01" <= release <= cutoff:
            counters["recent_2025_plus"] += 1
        if fact["airing_status"] == "not_yet_aired" and release:
            if release.startswith("2025-"):
                counters["rated_2025_not_yet_aired"] += 1
            elif release.startswith("2026-"):
                counters["rated_2026_not_yet_aired"] += 1
    return {
        key: counters[key]
        for key in (
            "recent_2024_plus",
            "recent_2025_plus",
            "rated_2025_not_yet_aired",
            "rated_2026_not_yet_aired",
        )
    }


def _qa(metadata_result: MetadataResult, interaction: InteractionResult) -> DatasetQa:
    meta = metadata_result.qa
    counts = dict(interaction.qa["counts"])
    counts.update(
        {
            "metadata_rows": meta.get("metadata_rows", 0),
            "catalog_items": meta.get("catalog_items", 0),
            "metadata_duplicate_mal_ids": meta.get("duplicate_mal_ids", 0),
            "metadata_invalid_mal_id": meta.get("invalid_mal_id", 0),
            "metadata_missing_title": meta.get("missing_title", 0),
            "metadata_synopsis_boilerplate": meta.get("synopsis_boilerplate", 0),
            "metadata_first_release_seasonal_year_disagreement": meta.get(
                "first_release_seasonal_year_disagreement", 0
            ),
            "metadata_missing_synopsis": meta.get("missing_synopsis", 0),
        }
    )
    counts.update(_recent(metadata_result.facts["anime"], interaction.rated_ids))
    return {
        "schema_version": 1,
        "counts": counts,
        "distributions": interaction.qa["distributions"],
        "loss_reasons": interaction.qa["loss_reasons"],
        "metadata_diagnostics": meta,
    }


def _compare_expectations(qa: DatasetQa, expected: DatasetExpectations) -> None:
    if any(qa["counts"].get(key) != value for key, value in expected["counts"].items()) or any(
        qa["distributions"].get(key) != value for key, value in expected["distributions"].items()
    ):
        raise DatasetError("qa_mismatch")


def normalize_dataset(
    snapshot: str,
    data_dir: Path,
    output_dir: Path,
    *,
    scratch_dir: Path,
    memory_limit_mib: int = 2048,
    threads: int = 2,
    spill_limit_mib: int = 32768,
    _spec: DatasetSpec | None = None,
) -> Path:
    """Publish the five artifact files only after full-source and aggregate checks."""
    if _spec is None and snapshot != SNAPSHOT:
        raise DatasetError("dataset_integrity_error")
    spec, registry, expected, registry_sha, expectations_sha = _load_spec(_spec)
    if snapshot != spec.snapshot_id:
        raise DatasetError("dataset_integrity_error")
    if any(
        type(value) is not int or value < 1
        for value in (memory_limit_mib, threads, spill_limit_mib)
    ):
        raise DatasetError("resource_limit")
    data_dir, output_dir, scratch_dir = map(Path, (data_dir, output_dir, scratch_dir))
    if output_dir.exists() or output_dir.is_symlink() or not output_dir.parent.is_dir():
        raise DatasetError("dataset_integrity_error")
    _check_input_manifest(data_dir, registry, registry_sha)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    own_scratch = Path(tempfile.mkdtemp(prefix=".dataset-", dir=scratch_dir))
    os.chmod(own_scratch, 0o700)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    os.chmod(stage, 0o700)
    published = False
    try:
        copied = {}
        for descriptor in registry["sources"]:
            name = descriptor["filename"]
            target = own_scratch / name
            _copy_verified(data_dir / name, target, descriptor)
            copied[name] = target
        try:
            meta = normalize_metadata(copied["details.csv"])
        except ValueError as exc:
            raise DatasetError("metadata_schema_invalid") from exc
        _write_json(stage / "catalog.json", meta.catalog)
        _write_json(stage / "item-facts.json", meta.facts)
        try:
            interaction = build_interactions(
                copied["ratings.csv"],
                meta,
                stage / "interactions.tsv.gz",
                own_scratch / "duckdb",
                {
                    "memory_limit_mib": memory_limit_mib,
                    "threads": threads,
                    "spill_limit_mib": spill_limit_mib,
                },
                spec.snapshot_id,
            )
        except ValueError as exc:
            code = str(exc)
            if code not in {
                "ratings_csv_invalid",
                "resource_limit",
                "user_key_collision",
                "dataset_integrity_error",
            }:
                code = "ratings_csv_invalid"
            raise DatasetError(code) from exc
        qa = _qa(meta, interaction)
        _compare_expectations(qa, expected)
        _write_json(stage / "dataset-qa.json", qa)
        try:
            version = package_metadata.version("anime-recommendation-recsys")
        except package_metadata.PackageNotFoundError:
            from recsys import __version__

            version = __version__
        import duckdb

        normalizer = {
            "name": "neelagiri-dataset",
            "version": "2",
            "code_files_sha256": source_fingerprints(),
            "expectations": {
                "resource": resource_path("expectations", EXPECTATIONS_RESOURCE),
                "sha256": expectations_sha,
            },
        }
        files = {}
        for name in OUTPUTS:
            path = stage / name
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            files[name] = {"bytes": path.stat().st_size, "sha256": digest}
        files["interactions.tsv.gz"].update(
            {
                "rows": interaction.rows,
                "uncompressed_bytes": interaction.uncompressed_bytes,
                "canonical_sha256": interaction.canonical_sha256,
            }
        )
        manifest = {
            "schema_version": 1,
            "kind": "arb-research-dataset",
            "snapshot_id": spec.snapshot_id,
            "registry_sha256": registry_sha,
            "provenance": registry["provenance"],
            "sources": registry["sources"],
            "normalizer": normalizer,
            "parameters": POLICIES,
            "producer": {
                "package_version": version,
                "python_version": ".".join(str(x) for x in sys.version_info[:3]),
                "duckdb_version": duckdb.__version__,
                "zlib_runtime_version": zlib.ZLIB_RUNTIME_VERSION,
            },
            "files": files,
        }
        _write_json(stage / "dataset-manifest.json", manifest)
        fsync_directory(stage)
        # Check the just-written content before immutable publication.
        validate_dataset(stage, _spec=spec)
        publish_no_replace(stage, output_dir)
        published = True
        return output_dir / "dataset-manifest.json"
    except DatasetError:
        raise
    except OSError as exc:
        raise DatasetError("resource_limit") from exc
    finally:
        if not published:
            shutil.rmtree(stage, ignore_errors=True)
        shutil.rmtree(own_scratch, ignore_errors=True)


def _read_json_checked(path: Path) -> tuple[dict[str, object], bytes]:
    if path.is_symlink() or not path.is_file():
        raise DatasetError("dataset_integrity_error")
    try:
        raw = path.read_bytes()
        value = parse_canonical(raw, path.name)
    except (OSError, ValueError, UnicodeError) as exc:
        raise DatasetError("dataset_integrity_error") from exc
    if not isinstance(value, dict):
        raise DatasetError("dataset_integrity_error")
    return cast("dict[str, object]", value), raw  # Top-level object checked above.


def _stream_lines(path: Path) -> Iterator[bytes]:
    try:
        yield from iter_single_member_gzip(path)
    except ArtifactIOError as exc:
        raise DatasetError("dataset_integrity_error") from exc


def _validate_interactions(
    path: Path, facts: dict[str, MetadataFact], manifest_entry: dict[str, object]
) -> tuple[DerivedQa, set[int]]:
    canonical = hashlib.sha256()
    uncompressed_bytes = 0
    rows = 0
    users = 0
    positive_cohort = completed_users = completed_positive_cohort = 0
    completed_pairs = 0
    score_hist: Counter[str] = Counter()
    status_hist: Counter[str] = Counter()
    rated_bins: Counter[str] = Counter()
    positive_bins: Counter[str] = Counter()
    rated_ids: set[int] = set()
    last_key = None
    last_mal = 0
    rated = positive = completed = completed_positive = 0
    try:
        lines = iter(_stream_lines(path))
        header = next(lines, None)
        if header != HEADER:
            raise DatasetError("dataset_integrity_error")
        canonical.update(header)
        uncompressed_bytes += len(header)
        for line in lines:
            canonical.update(line)
            uncompressed_bytes += len(line)
            parts = line.rstrip(b"\n").split(b"\t")
            if len(parts) != 4:
                raise DatasetError("dataset_integrity_error")
            try:
                key, mal_text, rating_text, status = (part.decode("ascii") for part in parts)
            except UnicodeError as exc:
                raise DatasetError("dataset_integrity_error") from exc
            if (
                not USER_KEY.fullmatch(key)
                or not re.fullmatch(r"[1-9][0-9]*", mal_text)
                or not re.fullmatch(r"(?:10|[1-9])", rating_text)
                or status not in STATUSES
                or len(mal_text) > 10
            ):
                raise DatasetError("dataset_integrity_error")
            mal = int(mal_text)
            rating = int(rating_text)
            if mal > MAX_ID or str(mal) != mal_text or mal_text not in facts:
                raise DatasetError("dataset_integrity_error")
            if key != last_key:
                if last_key is not None:
                    rated_bins[str(min((rated // 5) * 5, 100))] += 1
                    positive_bins[str(min((positive // 5) * 5, 100))] += 1
                    positive_cohort += positive >= 15
                    completed_users += completed > 0
                    completed_positive_cohort += completed_positive >= 15
                if last_key is not None and key <= last_key:
                    raise DatasetError("dataset_integrity_error")
                users += 1
                rated = positive = completed = completed_positive = 0
                last_mal = 0
                last_key = key
            if mal <= last_mal:
                raise DatasetError("dataset_integrity_error")
            last_mal = mal
            rated += 1
            positive += rating >= 8
            completed += status == "completed"
            completed_positive += status == "completed" and rating >= 8
            completed_pairs += status == "completed"
            score_hist[str(rating)] += 1
            status_hist[status] += 1
            rated_ids.add(mal)
            rows += 1
        if last_key is not None:
            rated_bins[str(min((rated // 5) * 5, 100))] += 1
            positive_bins[str(min((positive // 5) * 5, 100))] += 1
            positive_cohort += positive >= 15
            completed_users += completed > 0
            completed_positive_cohort += completed_positive >= 15
    except (OSError, UnicodeError, zlib.error) as exc:
        raise DatasetError("dataset_integrity_error") from exc
    if (
        rows != manifest_entry.get("rows")
        or uncompressed_bytes != manifest_entry.get("uncompressed_bytes")
        or canonical.hexdigest() != manifest_entry.get("canonical_sha256")
    ):
        raise DatasetError("dataset_integrity_error")
    derived: DerivedQa = {
        "counts": {
            "accepted_pairs": rows,
            "accepted_users": users,
            "accepted_rated_mal_ids": len(rated_ids),
            "positive_users_ge_15": positive_cohort,
            "completed_only_pairs": completed_pairs,
            "completed_only_users": completed_users,
            "completed_only_users_ge_15_positive": completed_positive_cohort,
        },
        "distributions": {
            "accepted_status_histogram": dict(status_hist),
            "accepted_score_histogram": dict(score_hist),
            "rated_items_per_user_bins": dict(rated_bins),
            "positive_items_per_user_bins": dict(positive_bins),
        },
    }
    return derived, rated_ids


def validate_dataset(
    dataset_dir: Path, *, expected_identity: str | None = None, _spec: DatasetSpec | None = None
) -> DatasetSummary:
    """Stream-check a frozen dataset; old supported producer versions remain valid."""
    spec, registry, expected, registry_sha, expectations_sha = _load_spec(_spec)
    dataset_dir = Path(dataset_dir)
    if dataset_dir.is_symlink() or not dataset_dir.is_dir():
        raise DatasetError("dataset_integrity_error")
    try:
        if {p.name for p in dataset_dir.iterdir()} != set(ALL_OUTPUTS):
            raise DatasetError("dataset_integrity_error")
        manifest, manifest_raw = _read_json_checked(dataset_dir / "dataset-manifest.json")
        identity = "sha256:" + _sha(manifest_raw)
        if expected_identity is not None and identity != expected_identity:
            raise DatasetError("dataset_integrity_error")
        if (
            set(manifest)
            != {
                "schema_version",
                "kind",
                "snapshot_id",
                "registry_sha256",
                "provenance",
                "sources",
                "normalizer",
                "parameters",
                "producer",
                "files",
            }
            or type(manifest["schema_version"]) is not int
            or manifest["schema_version"] != 1
            or manifest["kind"] != "arb-research-dataset"
            or manifest["snapshot_id"] != spec.snapshot_id
            or manifest["registry_sha256"] != registry_sha
            or manifest["provenance"] != registry["provenance"]
            or manifest["sources"] != registry["sources"]
            or manifest["parameters"] != POLICIES
            or set(cast("Iterable[str]", manifest["files"])) != set(OUTPUTS)
        ):
            raise DatasetError("dataset_integrity_error")
        manifest_files = cast("dict[str, object]", manifest["files"])  # Checked output set.
        normalizer = manifest["normalizer"]
        if (
            not isinstance(normalizer, dict)
            or set(normalizer) != {"name", "version", "code_files_sha256", "expectations"}
            or normalizer["name"] != "neelagiri-dataset"
        ):
            raise DatasetError("dataset_integrity_error")
        if normalizer["version"] == "1":
            names = set(LEGACY_CODE_FILES)
            expected_resource = spec.expectations_resource
            if expected_resource != EXPECTATIONS_RESOURCE:
                expected_resource = EXPECTATIONS_RESOURCE
            if normalizer["code_files_sha256"] != {
                name: LEGACY_V1_SOURCE_SHA256[name] for name in LEGACY_CODE_FILES
            }:
                raise DatasetError("dataset_integrity_error")
        elif normalizer["version"] == "2":
            names = set(SOURCE_FILES)
            expected_resource = resource_path("expectations", EXPECTATIONS_RESOURCE)
        else:
            raise DatasetError("dataset_integrity_error")
        if (
            normalizer["expectations"]
            != {"resource": expected_resource, "sha256": expectations_sha}
            or not isinstance(normalizer["code_files_sha256"], dict)
            or set(normalizer["code_files_sha256"]) != names
            or any(
                not isinstance(value, str) or HEX64.fullmatch(value) is None
                for value in normalizer["code_files_sha256"].values()
            )
        ):
            raise DatasetError("dataset_integrity_error")
        producer = manifest["producer"]
        if (
            not isinstance(producer, dict)
            or set(producer)
            != {"package_version", "python_version", "duckdb_version", "zlib_runtime_version"}
            or any(
                not isinstance(value, str)
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.+_-]{0,79}", value, re.ASCII)
                for value in producer.values()
            )
        ):
            raise DatasetError("dataset_integrity_error")
        if (
            normalizer["version"] == "1"
            and producer["package_version"] != LEGACY_V1_PACKAGE_VERSION
        ):
            raise DatasetError("dataset_integrity_error")
        for name in OUTPUTS:
            path = dataset_dir / name
            item = manifest_files[name]
            fields = {"bytes", "sha256"}
            if name == "interactions.tsv.gz":
                fields |= {"rows", "uncompressed_bytes", "canonical_sha256"}
            if (
                path.is_symlink()
                or not path.is_file()
                or not isinstance(item, dict)
                or set(item) != fields
                or type(item["bytes"]) is not int
                or item["bytes"] < 0
                or not isinstance(item["sha256"], str)
                or not HEX64.fullmatch(item["sha256"])
            ):
                raise DatasetError("dataset_integrity_error")
            if name == "interactions.tsv.gz" and (
                type(item["rows"]) is not int
                or item["rows"] < 0
                or type(item["uncompressed_bytes"]) is not int
                or item["uncompressed_bytes"] < 0
                or not isinstance(item["canonical_sha256"], str)
                or not HEX64.fullmatch(item["canonical_sha256"])
            ):
                raise DatasetError("dataset_integrity_error")
            h = hashlib.sha256()
            size = 0
            with path.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    size += len(chunk)
                    h.update(chunk)
            if size != item["bytes"] or h.hexdigest() != item["sha256"]:
                raise DatasetError("dataset_integrity_error")
        catalog, _ = _read_json_checked(dataset_dir / "catalog.json")
        facts, _ = _read_json_checked(dataset_dir / "item-facts.json")
        qa, _ = _read_json_checked(dataset_dir / "dataset-qa.json")
        try:
            catalog_keys = check_catalog(catalog)
        except ValueError as exc:
            raise DatasetError("dataset_integrity_error") from exc
        catalog_typed = cast("Catalog", catalog)  # Full catalog validation above.
        if (
            set(facts) != {"schema_version", "anime"}
            or type(facts["schema_version"]) is not int
            or facts["schema_version"] != 1
            or not isinstance(facts["anime"], dict)
            or set(facts["anime"]) != catalog_keys
        ):
            raise DatasetError("dataset_integrity_error")
        for key, fact in facts["anime"].items():
            if not isinstance(fact, dict) or set(fact) != {
                "first_release_date",
                "airing_status",
                "seasonal_year",
            }:
                raise DatasetError("dataset_integrity_error")
            release = fact["first_release_date"]
            if release is not None:
                try:
                    if (
                        not isinstance(release, str)
                        or date.fromisoformat(release).isoformat() != release
                    ):
                        raise DatasetError("dataset_integrity_error")
                except ValueError as exc:
                    raise DatasetError("dataset_integrity_error") from exc
            if (
                fact["airing_status"]
                not in ("finished_airing", "currently_airing", "not_yet_aired", "unknown")
                or fact["seasonal_year"] is not None
                and (
                    type(fact["seasonal_year"]) is not int or not 1 <= fact["seasonal_year"] <= 9999
                )
                or catalog_typed["anime"][key]["year"] != (int(release[:4]) if release else None)
            ):
                raise DatasetError("dataset_integrity_error")
        if (
            set(qa)
            != {"schema_version", "counts", "distributions", "loss_reasons", "metadata_diagnostics"}
            or type(qa["schema_version"]) is not int
            or qa["schema_version"] != 1
            or not isinstance(qa["counts"], dict)
            or not isinstance(qa["distributions"], dict)
        ):
            raise DatasetError("dataset_integrity_error")
        loss = qa["loss_reasons"]
        if (
            not isinstance(loss, dict)
            or set(loss) != {"primary", "overlap_flags"}
            or not isinstance(loss["primary"], dict)
            or not isinstance(loss["overlap_flags"], dict)
            or not set(loss["primary"])
            <= {
                "missing_metadata",
                "explicit_score_conflict",
                "mixed_unset",
                "invalid_state",
                "accepted",
            }
            or any(type(v) is not int or v < 0 for v in loss["primary"].values())
            or sum(loss["primary"].values()) != qa["counts"].get("raw_explicit_pairs")
            or loss["primary"].get("accepted", 0) != qa["counts"].get("accepted_pairs")
            or loss["overlap_flags"]
            != {
                "missing_metadata": qa["counts"].get("unmapped_pairs"),
                "explicit_score_conflict": qa["counts"].get("conflicting_score_pairs"),
                "mixed_unset": qa["counts"].get("mixed_unset_pairs"),
                "invalid_state": qa["counts"].get("invalid_state_pairs"),
            }
            or not isinstance(qa["metadata_diagnostics"], dict)
            or qa["metadata_diagnostics"].get("metadata_rows") != qa["counts"].get("metadata_rows")
            or qa["metadata_diagnostics"].get("catalog_items") != qa["counts"].get("catalog_items")
        ):
            raise DatasetError("dataset_integrity_error")
        facts_typed = cast("MetadataFacts", facts)  # All keys and per-record fields checked.
        qa_typed = cast("DatasetQa", qa)  # QA shape and loss aggregates checked above.
        derived, rated_ids = _validate_interactions(
            dataset_dir / "interactions.tsv.gz",
            facts_typed["anime"],
            cast("dict[str, object]", manifest_files["interactions.tsv.gz"]),
        )
        derived["counts"].update(_recent(facts_typed["anime"], rated_ids))
        if any(qa_typed["counts"].get(k) != v for k, v in derived["counts"].items()) or any(
            qa_typed["distributions"].get(k) != v for k, v in derived["distributions"].items()
        ):
            raise DatasetError("dataset_integrity_error")
        _compare_expectations(qa_typed, expected)
        return DatasetSummary(
            identity,
            len(catalog_keys),
            derived["counts"]["accepted_pairs"],
            derived["counts"]["accepted_users"],
            derived["counts"]["positive_users_ge_15"],
        )
    except DatasetError:
        raise
    except (OSError, TypeError, KeyError, ValueError, OverflowError, AttributeError) as exc:
        raise DatasetError("dataset_integrity_error") from exc
