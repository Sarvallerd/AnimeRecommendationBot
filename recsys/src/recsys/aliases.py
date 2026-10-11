"""Pin and verify title aliases from Nafiul metadata against a frozen base dataset."""

import csv
import ctypes
import hashlib
import io
import json
import os
import re
import shutil
import sys
import tempfile
import unicodedata
from collections import Counter
from dataclasses import dataclass
from importlib import metadata as package_metadata, resources
from pathlib import Path
from urllib.parse import urlsplit

from .bundle import MAX_ID, canonical_bytes, parse_canonical
from .dataset import DatasetError, DatasetSpec, validate_dataset
from .sources import SourceError, validate_registry

SNAPSHOT = "nafiul-2026-v1"
REGISTRY_RESOURCE = "source_registry_nafiul_2026_v1.json"
EXPECTATIONS_RESOURCE = "nafiul_2026_v1_alias_expectations.json"
OUTPUTS = ("aliases.json", "alias-qa.json")
ALL_OUTPUTS = (*OUTPUTS, "alias-manifest.json")
POLICY = "nafiul-base-aliases-v1"
HEX64 = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
IDENTITY = re.compile(r"sha256:[0-9a-f]{64}\Z", re.ASCII)
ID = re.compile(r"[1-9][0-9]*\Z", re.ASCII)
URL_PATH = re.compile(r"/anime/([1-9][0-9]*)(?:/[^/]+)?\Z", re.ASCII)
SENTINELS = frozenset(("unknown", "none", "null", "nan", "n/a"))
CSV_COLUMNS = (
    "mal_id", "title", "title_english", "title_japanese", "title_synonyms", "mal_url",
    "image_url", "type", "source", "episodes", "status", "airing", "aired_from",
    "aired_to", "aired_string", "duration", "content_rating", "score", "scored_by",
    "rank", "popularity", "members", "favorites", "season", "year", "broadcast_day",
    "broadcast_time", "studios", "producers", "licensors", "genres", "themes",
    "demographics", "opening_themes", "ending_themes", "synopsis", "background",
    "approved", "trailer_url",
)
COUNT_KEYS = frozenset((
    "source_rows", "distinct_valid_ids", "invalid_id_rows", "duplicate_id_groups",
    "duplicate_id_rows", "duplicate_overlap_groups", "duplicate_overlap_rows",
    "raw_overlap_rows", "unique_overlap", "unique_outside", "eligible_overlap",
    "base_missing_source", "base_without_unambiguous_match", "base_without_eligible_match",
    "invalid_url_rows", "invalid_url_overlap_rows", "emitted_items", "emitted_strings",
    "emitted_english", "emitted_main", "two_alias_items", "english_only_items",
    "main_only_items", "eligible_without_aliases", "base_without_aliases",
    "accepted_english_eq_base_primary", "accepted_english_eq_source_main",
    "accepted_english_eq_base_alias", "redundant_main", "main_eq_base_primary",
    "main_eq_base_alias", "main_eq_added_english", "emitted_english_not_latin_eligible",
    "first_alias_not_latin_eligible", "ignored_nonempty_synonyms_source",
    "ignored_nonempty_synonyms_overlap", "ignored_nonempty_synonyms_eligible",
))
FIELD_KEYS = frozenset(("accepted", "empty", "sentinel", "trim_changed", "nfc_changed",
                        "rejected_Cc", "rejected_Cf", "rejected_Cs", "rejected_UFFFD"))
FIELD_SCOPES = ("all_source", "eligible_overlap")
FIELD_NAMES = ("title_english", "title")


class AliasError(Exception):
    """A fixed-code failure that never includes source text or private paths."""


@dataclass(frozen=True)
class AliasSpec:
    snapshot_id: str
    registry_bytes: bytes
    expectations_resource: str
    expectations_bytes: bytes


@dataclass(frozen=True)
class AliasSummary:
    identity: str
    base_identity: str
    items: int
    aliases: int
    english: int
    main: int


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _same_json(left: object, right: object) -> bool:
    """Compare frozen JSON values without Python's bool/int/float equality coercion."""
    try:
        return canonical_bytes(left) == canonical_bytes(right)
    except (TypeError, ValueError, OverflowError):
        return False


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _json_unique(raw: bytes) -> dict:
    def reject_constant(_value: str):
        raise ValueError("nonfinite JSON constant")

    value = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object,
                       parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise ValueError("expected JSON object")
    return value


def _fields_shape(fields: object) -> bool:
    return (isinstance(fields, dict) and set(fields) == set(FIELD_SCOPES)
            and all(isinstance(fields[scope], dict)
                    and set(fields[scope]) == set(FIELD_NAMES)
                    and all(isinstance(values, dict) and set(values) == FIELD_KEYS
                            and all(type(value) is int and value >= 0 for value in values.values())
                            for values in fields[scope].values())
                    for scope in FIELD_SCOPES))


def _load_spec(spec: AliasSpec | None) -> tuple[AliasSpec, dict, dict, str, str]:
    if spec is None:
        package = resources.files("recsys")
        spec = AliasSpec(SNAPSHOT, package.joinpath(REGISTRY_RESOURCE).read_bytes(),
                         EXPECTATIONS_RESOURCE, package.joinpath(EXPECTATIONS_RESOURCE).read_bytes())
    if not isinstance(spec, AliasSpec) or not isinstance(spec.snapshot_id, str):
        raise AliasError("alias_integrity_error")
    try:
        registry = _json_unique(spec.registry_bytes)
        validate_registry(registry)
        expectations = parse_canonical(spec.expectations_bytes, "alias_expectations")
    except (ValueError, TypeError, OSError, UnicodeError, SourceError) as exc:
        raise AliasError("alias_integrity_error") from exc
    dataset_provenance = registry.get("provenance", {}).get("dataset")
    if (type(registry.get("schema_version")) is not int
            or not isinstance(dataset_provenance, dict)
            or type(dataset_provenance.get("version")) is not int
            or dataset_provenance["version"] < 1
            or not isinstance(expectations, dict)
            or set(expectations) != {"schema_version", "snapshot_id", "source_sha256",
                                     "base_identity", "counts", "fields"}
            or type(expectations["schema_version"]) is not int
            or expectations["schema_version"] != 1
            or expectations["snapshot_id"] != spec.snapshot_id
            or not isinstance(expectations["base_identity"], str)
            or not IDENTITY.fullmatch(expectations["base_identity"])
            or spec.expectations_resource != EXPECTATIONS_RESOURCE
            or dataset_provenance.get("snapshot_id") != spec.snapshot_id
            or len(registry["sources"]) != 1
            or registry["sources"][0]["id"] != "nafiul_anime_csv"
            or registry["sources"][0]["filename"] != "anime_info.csv"
            or expectations["source_sha256"] != registry["sources"][0]["sha256"]
            or not isinstance(expectations["counts"], dict)
            or set(expectations["counts"]) != COUNT_KEYS
            or any(type(value) is not int or value < 0
                   for value in expectations["counts"].values())
            or not _fields_shape(expectations["fields"])):
        raise AliasError("alias_integrity_error")
    return spec, registry, expectations, _sha(spec.registry_bytes), _sha(spec.expectations_bytes)


def _read_source(data_dir: Path, registry: dict, registry_sha: str) -> bytes:
    descriptor = registry["sources"][0]
    manifest_path = data_dir / "source-manifest.json"
    source_path = data_dir / descriptor["filename"]
    if (data_dir.is_symlink() or not data_dir.is_dir()
            or manifest_path.is_symlink() or not manifest_path.is_file()
            or source_path.is_symlink() or not source_path.is_file()):
        raise AliasError("input_hash_mismatch")
    expected = {"schema_version": 1, "registry_sha256": registry_sha,
                "provenance": registry["provenance"],
                "verified": [{key: descriptor[key] for key in ("filename", "size", "sha256")}]}
    try:
        manifest = _json_unique(manifest_path.read_bytes())
        raw = source_path.read_bytes()
    except (OSError, ValueError, UnicodeError) as exc:
        raise AliasError("input_hash_mismatch") from exc
    if (not isinstance(manifest, dict) or set(manifest) != set(expected)
            or type(manifest.get("schema_version")) is not int
            or not _same_json(manifest, expected)
            or len(raw) != descriptor["size"] or _sha(raw) != descriptor["sha256"]):
        raise AliasError("input_hash_mismatch")
    return raw


def _base_catalog(base_dir: Path, base_identity: str, base_spec: DatasetSpec | None) -> tuple[dict, dict]:
    try:
        validate_dataset(base_dir, expected_identity=base_identity, _spec=base_spec)
        manifest_path = base_dir / "dataset-manifest.json"
        catalog_path = base_dir / "catalog.json"
        if manifest_path.is_symlink() or catalog_path.is_symlink():
            raise AliasError("base_integrity_error")
        manifest_raw = manifest_path.read_bytes()
        if "sha256:" + _sha(manifest_raw) != base_identity:
            raise AliasError("base_integrity_error")
        manifest = parse_canonical(manifest_raw, "base_manifest")
        catalog_raw = catalog_path.read_bytes()
        descriptor = manifest["files"]["catalog.json"]
        if len(catalog_raw) != descriptor["bytes"] or _sha(catalog_raw) != descriptor["sha256"]:
            raise AliasError("base_integrity_error")
        catalog = parse_canonical(catalog_raw, "base_catalog")
        return catalog["anime"], descriptor
    except (DatasetError, OSError, ValueError, TypeError, KeyError) as exc:
        raise AliasError("base_integrity_error") from exc


def _id(raw: str) -> str | None:
    if len(raw) > 10 or not ID.fullmatch(raw):
        return None
    return raw if int(raw) <= MAX_ID else None


def _valid_url(raw: str, mal_id: str) -> bool:
    try:
        url = urlsplit(raw)
        match = URL_PATH.fullmatch(url.path)
        return (url.scheme == "https" and url.netloc == "myanimelist.net"
                and url.username is None and url.password is None and url.port is None
                and "?" not in raw and "#" not in raw and match is not None
                and match.group(1) == mal_id)
    except ValueError:
        return False


def _field(raw: str, counts: Counter) -> str | None:
    stripped = raw.strip()
    normalized = unicodedata.normalize("NFC", stripped)
    counts["trim_changed"] += raw != stripped
    counts["nfc_changed"] += normalized != stripped
    if not normalized:
        counts["empty"] += 1
        return None
    if normalized.casefold() in SENTINELS:
        counts["sentinel"] += 1
        return None
    problems = {"rejected_UFFFD": "\ufffd" in normalized}
    for category in ("Cc", "Cf", "Cs"):
        problems["rejected_" + category] = any(
            unicodedata.category(character) == category for character in normalized)
    for key, present in problems.items():
        counts[key] += present
    if any(problems.values()):
        return None
    counts["accepted"] += 1
    return normalized


def _latin_eligible(value: str) -> bool:
    ascii_letter = False
    for character in unicodedata.normalize("NFKD", value):
        if character.isalpha():
            if not character.isascii() or not character.isalpha():
                return False
            ascii_letter = True
    return ascii_letter


def _rows(raw: bytes) -> list[tuple[str | None, str, str, str, str]]:
    try:
        source = io.StringIO(raw.decode("utf-8"), newline="")
        reader = csv.reader(source, strict=True)
        if tuple(next(reader)) != CSV_COLUMNS:
            raise AliasError("alias_csv_invalid")
        indexes = [CSV_COLUMNS.index(name) for name in
                   ("mal_id", "title", "title_english", "mal_url", "title_synonyms")]
        records = []
        for row in reader:
            if len(row) != len(CSV_COLUMNS):
                raise AliasError("alias_csv_invalid")
            mal_raw, title, english, url, synonyms = (row[index] for index in indexes)
            records.append((_id(mal_raw), title, english, url, synonyms))
        return records
    except (UnicodeError, csv.Error, StopIteration, OverflowError) as exc:
        raise AliasError("alias_csv_invalid") from exc


def _derive(raw: bytes, base: dict) -> tuple[dict, dict]:
    records = _rows(raw)
    ids = Counter(item[0] for item in records if item[0] is not None)
    base_ids = set(base)
    duplicate = {key for key, count in ids.items() if count > 1}
    overlapping_duplicate = duplicate & base_ids
    unique = set(ids) - duplicate
    overlap = unique & base_ids
    counts = Counter({
        "source_rows": len(records), "distinct_valid_ids": len(ids),
        "invalid_id_rows": sum(item[0] is None for item in records),
        "duplicate_id_groups": len(duplicate),
        "duplicate_id_rows": sum(ids[key] for key in duplicate),
        "duplicate_overlap_groups": len(overlapping_duplicate),
        "duplicate_overlap_rows": sum(ids[key] for key in overlapping_duplicate),
        "raw_overlap_rows": sum(item[0] in base_ids for item in records),
        "unique_overlap": len(overlap), "unique_outside": len(unique - base_ids),
        "base_missing_source": len(base_ids - set(ids)),
        "base_without_unambiguous_match": len(base_ids - overlap),
    })
    fields = {scope: {name: Counter() for name in FIELD_NAMES} for scope in FIELD_SCOPES}
    aliases = {}
    for mal_id, main_raw, english_raw, url, synonyms in records:
        _field(english_raw, fields["all_source"]["title_english"])
        _field(main_raw, fields["all_source"]["title"])
        counts["ignored_nonempty_synonyms_source"] += bool(synonyms.strip())
        if mal_id is None:
            continue
        in_base = mal_id in base_ids
        if in_base:
            counts["ignored_nonempty_synonyms_overlap"] += bool(synonyms.strip())
        valid_url = _valid_url(url, mal_id)
        if not valid_url:
            counts["invalid_url_rows"] += 1
            counts["invalid_url_overlap_rows"] += in_base
        if mal_id not in overlap or not valid_url:
            continue
        counts["eligible_overlap"] += 1
        counts["ignored_nonempty_synonyms_eligible"] += bool(synonyms.strip())
        english = _field(english_raw, fields["eligible_overlap"]["title_english"])
        main = _field(main_raw, fields["eligible_overlap"]["title"])
        base_entry = base[mal_id]
        base_title = base_entry["title"]
        base_aliases = base_entry["aliases"]
        if english:
            counts["accepted_english_eq_base_primary"] += english == base_title
            counts["accepted_english_eq_source_main"] += english == main
            counts["accepted_english_eq_base_alias"] += english in base_aliases
        if main:
            eq_primary = main == base_title
            eq_alias = main in base_aliases
            eq_english = main == english
            counts["main_eq_base_primary"] += eq_primary
            counts["main_eq_base_alias"] += eq_alias
            counts["main_eq_added_english"] += eq_english
            counts["redundant_main"] += eq_primary or eq_alias or eq_english
        entries = []
        if english:
            entries.append({"source_field": "title_english", "text": english})
            counts["emitted_english"] += 1
            counts["emitted_english_not_latin_eligible"] += not _latin_eligible(english)
        if main and main != base_title and main not in base_aliases and main != english:
            entries.append({"source_field": "title", "text": main})
            counts["emitted_main"] += 1
        if entries:
            aliases[mal_id] = entries
            counts["first_alias_not_latin_eligible"] += not _latin_eligible(entries[0]["text"])
            counts["two_alias_items"] += len(entries) == 2
            counts["english_only_items"] += len(entries) == 1 and english is not None
            counts["main_only_items"] += len(entries) == 1 and english is None
    counts["emitted_items"] = len(aliases)
    counts["emitted_strings"] = counts["emitted_english"] + counts["emitted_main"]
    counts["eligible_without_aliases"] = counts["eligible_overlap"] - len(aliases)
    counts["base_without_aliases"] = len(base_ids) - len(aliases)
    counts["base_without_eligible_match"] = len(base_ids) - counts["eligible_overlap"]
    qa = {"schema_version": 1,
          "counts": {key: counts[key] for key in sorted(COUNT_KEYS)},
          "fields": {scope: {name: {key: fields[scope][name][key] for key in sorted(FIELD_KEYS)}
                             for name in FIELD_NAMES} for scope in FIELD_SCOPES}}
    return {"schema_version": 1, "aliases": aliases}, qa


def _check_expectations(qa: dict, expectations: dict) -> None:
    if qa["counts"] != expectations["counts"] or qa["fields"] != expectations["fields"]:
        raise AliasError("alias_qa_mismatch")


def _write(path: Path, value: dict) -> bytes:
    raw = canonical_bytes(value)
    with path.open("xb") as stream:
        os.chmod(path, 0o600)
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return raw


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish(stage: Path, target: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.renameat2(-100, os.fsencode(stage), -100, os.fsencode(target), 1) != 0:
        raise OSError(ctypes.get_errno(), "atomic alias publication failed")
    _fsync_dir(target.parent)


def _read_json(path: Path) -> tuple[dict, bytes]:
    if path.is_symlink() or not path.is_file():
        raise AliasError("alias_integrity_error")
    try:
        raw = path.read_bytes()
        parsed = parse_canonical(raw, path.name)
    except (OSError, ValueError, UnicodeError) as exc:
        raise AliasError("alias_integrity_error") from exc
    if not isinstance(parsed, dict):
        raise AliasError("alias_integrity_error")
    return parsed, raw


def _validate_output(stage: Path, manifest: dict, alias_bytes: bytes, qa_bytes: bytes) -> None:
    if {path.name for path in stage.iterdir()} != set(ALL_OUTPUTS):
        raise AliasError("alias_integrity_error")
    for name, expected_raw in (("aliases.json", alias_bytes), ("alias-qa.json", qa_bytes)):
        _, raw = _read_json(stage / name)
        if raw != expected_raw or manifest["files"][name] != {"bytes": len(raw), "sha256": _sha(raw)}:
            raise AliasError("alias_integrity_error")
    parsed, raw = _read_json(stage / "alias-manifest.json")
    if parsed != manifest or raw != canonical_bytes(manifest):
        raise AliasError("alias_integrity_error")


def _safe_paths(data_dir: Path, base_dir: Path, output_dir: Path) -> None:
    if (not output_dir.parent.is_dir() or output_dir.parent.is_symlink()
            or output_dir.exists() or output_dir.is_symlink()):
        raise AliasError("alias_integrity_error")
    resolved_output = output_dir.resolve(strict=False)
    for source in (data_dir, base_dir):
        resolved_source = source.resolve(strict=False)
        if (resolved_output == resolved_source or resolved_output.is_relative_to(resolved_source)
                or resolved_source.is_relative_to(resolved_output)):
            raise AliasError("alias_integrity_error")


def _prepare(data_dir: Path, base_dir: Path, expected_base_identity: str,
             spec: AliasSpec | None, base_spec: DatasetSpec | None):
    if not isinstance(expected_base_identity, str) or not IDENTITY.fullmatch(expected_base_identity):
        raise AliasError("alias_integrity_error")
    loaded_spec, registry, expectations, registry_sha, expectations_sha = _load_spec(spec)
    if expectations["base_identity"] != expected_base_identity:
        raise AliasError("base_integrity_error")
    base, base_descriptor = _base_catalog(base_dir, expected_base_identity, base_spec)
    raw = _read_source(data_dir, registry, registry_sha)
    aliases, qa = _derive(raw, base)
    _check_expectations(qa, expectations)
    return (loaded_spec, registry, registry_sha, expectations_sha, base_descriptor,
            len(base), aliases, qa)


def normalize_aliases(data_dir: Path, base_dir: Path, output_dir: Path, *, snapshot: str,
                      expected_base_identity: str, _spec: AliasSpec | None = None,
                      _base_spec: DatasetSpec | None = None) -> Path:
    """Publish three deterministic alias files after both frozen inputs pass validation."""
    if _spec is None and snapshot != SNAPSHOT:
        raise AliasError("alias_integrity_error")
    if _spec is not None and snapshot != _spec.snapshot_id:
        raise AliasError("alias_integrity_error")
    data_dir, base_dir, output_dir = map(Path, (data_dir, base_dir, output_dir))
    _safe_paths(data_dir, base_dir, output_dir)
    spec, registry, registry_sha, expectations_sha, base_descriptor, base_count, aliases, qa = _prepare(
        data_dir, base_dir, expected_base_identity, _spec, _base_spec)
    if snapshot != spec.snapshot_id:
        raise AliasError("alias_integrity_error")
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    os.chmod(stage, 0o700)
    published = False
    try:
        alias_raw = _write(stage / "aliases.json", aliases)
        qa_raw = _write(stage / "alias-qa.json", qa)
        try:
            package_version = package_metadata.version("anime-recommendation-recsys")
        except package_metadata.PackageNotFoundError:
            from . import __version__
            package_version = __version__
        manifest = {
            "schema_version": 1, "kind": "arb-title-aliases", "snapshot_id": spec.snapshot_id,
            "base": {"snapshot_id": "neelagiri-2025-v1", "identity": expected_base_identity,
                     "catalog": base_descriptor, "catalog_items": base_count},
            "registry_sha256": registry_sha, "provenance": registry["provenance"],
            "sources": registry["sources"],
            "normalizer": {"name": "nafiul-title-aliases", "version": "1", "policy": POLICY,
                           "code_files_sha256": {"aliases.py": _sha(
                               resources.files("recsys").joinpath("aliases.py").read_bytes())},
                           "expectations": {"resource": spec.expectations_resource,
                                            "sha256": expectations_sha}},
            "producer": {"package_version": package_version,
                         "python_version": ".".join(str(x) for x in sys.version_info[:3]),
                         "unicode_version": unicodedata.unidata_version},
            "files": {name: {"bytes": len(raw), "sha256": _sha(raw)}
                      for name, raw in (("aliases.json", alias_raw), ("alias-qa.json", qa_raw))},
        }
        _write(stage / "alias-manifest.json", manifest)
        _fsync_dir(stage)
        _validate_output(stage, manifest, alias_raw, qa_raw)
        _publish(stage, output_dir)
        published = True
        return output_dir / "alias-manifest.json"
    except AliasError:
        raise
    except OSError as exc:
        raise AliasError("alias_integrity_error") from exc
    finally:
        if not published:
            shutil.rmtree(stage, ignore_errors=True)


def validate_aliases(aliases_dir: Path, *, data_dir: Path, base_dir: Path,
                     expected_base_identity: str, expected_identity: str | None = None,
                     _spec: AliasSpec | None = None,
                     _base_spec: DatasetSpec | None = None) -> AliasSummary:
    """Rebuild alias decisions from the verified metadata and base catalog."""
    aliases_dir, data_dir, base_dir = map(Path, (aliases_dir, data_dir, base_dir))
    if (aliases_dir.is_symlink() or not aliases_dir.is_dir()
            or aliases_dir.resolve() in {data_dir.resolve(), base_dir.resolve()}):
        raise AliasError("alias_integrity_error")
    try:
        if {path.name for path in aliases_dir.iterdir()} != set(ALL_OUTPUTS):
            raise AliasError("alias_integrity_error")
        manifest, manifest_raw = _read_json(aliases_dir / "alias-manifest.json")
        identity = "sha256:" + _sha(manifest_raw)
        if expected_identity is not None and identity != expected_identity:
            raise AliasError("alias_integrity_error")
        spec, registry, registry_sha, expectations_sha, base_descriptor, base_count, aliases, qa = _prepare(
            data_dir, base_dir, expected_base_identity, _spec, _base_spec)
        expected_manifest = {
            "schema_version": 1, "kind": "arb-title-aliases", "snapshot_id": spec.snapshot_id,
            "base": {"snapshot_id": "neelagiri-2025-v1", "identity": expected_base_identity,
                     "catalog": base_descriptor, "catalog_items": base_count},
            "registry_sha256": registry_sha, "provenance": registry["provenance"],
            "sources": registry["sources"],
        }
        if (not isinstance(manifest, dict)
                or type(manifest.get("schema_version")) is not int
                or any(not _same_json(manifest.get(key), value)
                       for key, value in expected_manifest.items())
                or set(manifest) != set(expected_manifest) | {"normalizer", "producer", "files"}):
            raise AliasError("alias_integrity_error")
        normalizer = manifest["normalizer"]
        if (not isinstance(normalizer, dict)
                or set(normalizer) != {"name", "version", "policy", "code_files_sha256", "expectations"}
                or normalizer["name"] != "nafiul-title-aliases" or normalizer["version"] != "1"
                or normalizer["policy"] != POLICY
                or normalizer["expectations"] != {"resource": spec.expectations_resource,
                                                   "sha256": expectations_sha}
                or not isinstance(normalizer["code_files_sha256"], dict)
                or set(normalizer["code_files_sha256"]) != {"aliases.py"}
                or any(not isinstance(value, str) or not HEX64.fullmatch(value)
                       for value in normalizer["code_files_sha256"].values())):
            raise AliasError("alias_integrity_error")
        producer = manifest["producer"]
        if (not isinstance(producer, dict)
                or set(producer) != {"package_version", "python_version", "unicode_version"}
                or any(not isinstance(value, str) or not re.fullmatch(
                    r"[A-Za-z0-9][A-Za-z0-9.+_-]{0,79}", value, re.ASCII)
                    for value in producer.values())):
            raise AliasError("alias_integrity_error")
        if not isinstance(manifest["files"], dict) or set(manifest["files"]) != set(OUTPUTS):
            raise AliasError("alias_integrity_error")
        for name, expected_value in (("aliases.json", aliases), ("alias-qa.json", qa)):
            value, raw = _read_json(aliases_dir / name)
            descriptor = manifest["files"][name]
            if (not isinstance(descriptor, dict) or set(descriptor) != {"bytes", "sha256"}
                    or type(descriptor["bytes"]) is not int or descriptor["bytes"] < 0
                    or not isinstance(descriptor["sha256"], str)
                    or not HEX64.fullmatch(descriptor["sha256"])
                    or descriptor != {"bytes": len(raw), "sha256": _sha(raw)}
                    or raw != canonical_bytes(expected_value) or value != expected_value):
                raise AliasError("alias_integrity_error")
        return AliasSummary(identity, expected_base_identity, qa["counts"]["emitted_items"],
                            qa["counts"]["emitted_strings"], qa["counts"]["emitted_english"],
                            qa["counts"]["emitted_main"])
    except AliasError:
        raise
    except (OSError, TypeError, KeyError, ValueError, OverflowError, AttributeError) as exc:
        raise AliasError("alias_integrity_error") from exc
