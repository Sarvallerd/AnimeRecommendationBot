"""Atomic publication and integrity verification of framework-owned weights.

This layer checks the container and provenance. Native tensor decoding belongs to
the framework adapter; successful verification is not proof that a model was fit.
"""

from __future__ import annotations

import errno
import hashlib
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict, cast

from recsys import __version__
from recsys.contracts.bundle import canonical_bytes, parse_canonical
from recsys.contracts.provenance import SOURCE_LAYOUT, source_fingerprints
from recsys.research.artifacts import (
    FileDescriptor,
    PreparationProducer,
    StreamDescriptor,
    cleanup,
    open_stage,
    producer,
    publish,
    safe_path,
    separate,
    write_json,
)
from recsys.research.models.types import ScoringInputs
from recsys.research.protocol import ResearchError
from recsys.research.training.spec import TrainingSpec, TrainingSpecWire, parse_training_spec

MANIFEST = "training-manifest.json"
ITEMS = "items.json"
WEIGHTS = "weights.safetensors"
FILES = {MANIFEST, ITEMS, WEIGHTS}
MAX_MANIFEST = 1024 * 1024
MAX_ITEMS = 8 * 1024 * 1024
MAX_WEIGHTS = 512 * 1024 * 1024
BLOCK = 1024 * 1024
_IDENTITY = re.compile(r"sha256:[0-9a-f]{64}\Z")
_REVISION = re.compile(r"[0-9a-f]{40}\Z")
_FORBIDDEN_WEIGHTS_EXTENSIONS = frozenset({"pt", "pth", "pkl", "pickle", "zip", "npz"})


class TrainingData(TypedDict):
    prepared_identity: str
    protocol_identity: str
    fit_identity: str
    catalog_descriptor: FileDescriptor
    train_interactions: StreamDescriptor
    train_stats: FileDescriptor


class TrainingManifest(TypedDict):
    schema_version: int
    kind: str
    config: TrainingSpecWire
    config_identity: str
    data: TrainingData
    data_identity: str
    producer: PreparationProducer
    source_identity: str
    framework_version: str
    files: dict[str, FileDescriptor]


@dataclass(frozen=True)
class VerifiedTrainingArtifact:
    identity: str
    manifest: TrainingManifest
    spec: TrainingSpec
    mal_ids: tuple[int, ...]
    supported_mal_ids: tuple[int, ...]
    weights_path: Path


def _fail(code: str = "training_artifact_invalid") -> None:
    raise ResearchError(code)


def _identity(value: object) -> None:
    if type(value) is not str or _IDENTITY.fullmatch(value) is None:
        _fail()


def _hash(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _canonical(value: object) -> bytes:
    try:
        return canonical_bytes(value)
    except (TypeError, ValueError, UnicodeError, OverflowError, RecursionError) as exc:
        raise ResearchError("training_artifact_invalid") from exc


def _read_file(path: Path, limit: int, *, minimum: int = 0) -> bytes:
    """Check both the path and opened inode, then bound the actual read."""
    try:
        safe_path(path)
        if not path.is_file():
            _fail("research_unsafe_path")
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            opened = os.fstat(fd)
            if not stat.S_ISREG(opened.st_mode):
                _fail("research_unsafe_path")
            if not minimum <= opened.st_size <= limit:
                _fail("training_resource_limit")
            with os.fdopen(fd, "rb", closefd=False) as stream:
                raw = stream.read(limit + 1)
            if not minimum <= len(raw) <= limit:
                _fail("training_resource_limit")
            if len(raw) != opened.st_size:
                _fail()
            return raw
        finally:
            os.close(fd)
    except OSError as exc:
        raise ResearchError("training_artifact_invalid") from exc


def _read_json(path: Path, limit: int) -> tuple[object, bytes]:
    raw = _read_file(path, limit)
    try:
        return parse_canonical(raw, path.name), raw
    except (TypeError, ValueError, UnicodeError, OverflowError, RecursionError) as exc:
        raise ResearchError("training_artifact_invalid") from exc


def _descriptor(value: object, *, stream: bool = False, limit: int = MAX_WEIGHTS) -> None:
    fields = {"bytes", "sha256"}
    if stream:
        fields |= {"rows", "uncompressed_bytes", "canonical_sha256"}
    if type(value) is not dict or set(value) != fields:
        _fail()
    record = cast("dict[str, object]", value)
    if type(record["bytes"]) is not int or not 0 <= record["bytes"] <= limit:
        _fail()
    for name in ("sha256", "canonical_sha256"):
        entry = record.get(name)
        if name in record and (
            type(entry) is not str or re.fullmatch(r"[0-9a-f]{64}", entry) is None
        ):
            _fail()
    for name in ("rows", "uncompressed_bytes"):
        entry = record.get(name)
        if name in record and (type(entry) is not int or entry < 0 or entry > 64 * 1024**3):
            _fail()


def _data(inputs: ScoringInputs) -> TrainingData:
    return {
        "prepared_identity": inputs.prepared_identity,
        "protocol_identity": inputs.protocol_identity,
        "fit_identity": inputs.fit_identity,
        "catalog_descriptor": inputs.catalog_descriptor,
        "train_interactions": inputs.train_interactions_descriptor,
        "train_stats": inputs.train_stats_descriptor,
    }


def _check_data(value: object) -> TrainingData:
    fields = set(TrainingData.__annotations__)
    if type(value) is not dict or set(value) != fields:
        _fail()
    data = cast("TrainingData", value)
    for name in ("prepared_identity", "protocol_identity", "fit_identity"):
        _identity(data[name])
    _descriptor(data["catalog_descriptor"], limit=64 * 1024**3)
    _descriptor(data["train_interactions"], stream=True, limit=64 * 1024**3)
    _descriptor(data["train_stats"], limit=64 * 1024**3)
    return data


def _check_producer(value: object) -> PreparationProducer:
    fields = set(PreparationProducer.__annotations__)
    if type(value) is not dict or set(value) != fields:
        _fail()
    record = cast("PreparationProducer", value)
    if (
        type(record["code_revision"]) is not str
        or _REVISION.fullmatch(record["code_revision"]) is None
    ):
        _fail()
    for name in ("python_version", "numpy_version"):
        text = record[name]
        if (
            type(text) is not str
            or not 1 <= len(text) <= 128
            or not text.isascii()
            or not text.isprintable()
        ):
            _fail()
    fingerprints = record["code_files_sha256"]
    if type(fingerprints) is not dict or fingerprints != source_fingerprints():
        _fail("training_source_mismatch")
    if record["source_layout"] != SOURCE_LAYOUT or record["package_version"] != __version__:
        _fail("training_source_mismatch")
    return record


def _check_items(value: object) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if type(value) is not dict or set(value) != {"schema_version", "mal_ids", "supported"}:
        _fail()
    record = cast("dict[str, object]", value)
    ids, flags = record["mal_ids"], record["supported"]
    if type(record["schema_version"]) is not int or record["schema_version"] != 1:
        _fail()
    if (
        type(ids) is not list
        or type(flags) is not list
        or len(ids) != len(flags)
        or len(ids) > 100000
    ):
        _fail("training_resource_limit")
    previous = 0
    supported: list[int] = []
    for item, flag in zip(cast("list[object]", ids), cast("list[object]", flags)):
        if type(item) is not int or not previous < item <= 2_147_483_647 or type(flag) is not bool:
            _fail()
        previous = cast("int", item)
        if flag:
            supported.append(previous)
    return tuple(cast("list[int]", ids)), tuple(supported)


def _validate_version(value: object) -> str:
    if (
        type(value) is not str
        or not 1 <= len(value) <= 128
        or not value.isascii()
        or not value.isprintable()
    ):
        _fail()
    return cast("str", value)


def _check_manifest(value: object) -> tuple[TrainingManifest, TrainingSpec]:
    if type(value) is not dict or set(value) != set(TrainingManifest.__annotations__):
        _fail()
    record = cast("TrainingManifest", value)
    if (
        type(record["schema_version"]) is not int
        or record["schema_version"] != 1
        or record["kind"] != "arb-training-artifact"
    ):
        _fail()
    spec = parse_training_spec(record["config"])
    if spec.family != "seed-embedding":
        _fail("training_adapter_unsupported")
    for observed, component in (
        (record["config_identity"], record["config"]),
        (record["data_identity"], record["data"]),
        (record["source_identity"], record["producer"]),
    ):
        _identity(observed)
        if observed != _hash(_canonical(component)):
            _fail()
    _check_data(record["data"])
    _check_producer(record["producer"])
    _validate_version(record["framework_version"])
    files = record["files"]
    if type(files) is not dict or set(files) != {ITEMS, WEIGHTS}:
        _fail()
    _descriptor(files[ITEMS], limit=MAX_ITEMS)
    _descriptor(files[WEIGHTS], limit=MAX_WEIGHTS)
    if files[WEIGHTS]["bytes"] < 1:
        _fail()
    return record, spec


def _check_file(raw: bytes, descriptor: FileDescriptor) -> None:
    if descriptor != {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}:
        _fail()


def _hash_file(path: Path, limit: int) -> FileDescriptor:
    """Hash large weights in bounded blocks without materializing them in memory."""
    try:
        safe_path(path)
        if not path.is_file():
            _fail("research_unsafe_path")
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            opened = os.fstat(fd)
            if not stat.S_ISREG(opened.st_mode):
                _fail("research_unsafe_path")
            if not 1 <= opened.st_size <= limit:
                _fail("training_resource_limit")
            digest = hashlib.sha256()
            size = 0
            with os.fdopen(fd, "rb", closefd=False) as stream:
                while block := stream.read(BLOCK):
                    size += len(block)
                    if size > limit:
                        _fail("training_resource_limit")
                    digest.update(block)
            if size != opened.st_size:
                _fail()
            return {"bytes": size, "sha256": digest.hexdigest()}
        finally:
            os.close(fd)
    except OSError as exc:
        raise ResearchError("training_artifact_invalid") from exc


def verify_training_artifact(
    model_dir: str | Path, *, expected_identity: str, inputs: ScoringInputs | None = None
) -> VerifiedTrainingArtifact:
    """Verify exact bytes and declared provenance, without decoding model tensors."""
    if type(expected_identity) is not str or _IDENTITY.fullmatch(expected_identity) is None:
        _fail("training_identity_mismatch")
    try:
        root = safe_path(model_dir, directory=True)
        if {child.name for child in root.iterdir()} != FILES:
            _fail()
        manifest_value, manifest_raw = _read_json(root / MANIFEST, MAX_MANIFEST)
        if _hash(manifest_raw) != expected_identity:
            _fail("training_identity_mismatch")
        manifest, spec = _check_manifest(manifest_value)
        items_value, items_raw = _read_json(root / ITEMS, MAX_ITEMS)
        _check_file(items_raw, manifest["files"][ITEMS])
        mal_ids, supported = _check_items(items_value)
        if _hash_file(root / WEIGHTS, MAX_WEIGHTS) != manifest["files"][WEIGHTS]:
            _fail()
        if inputs is not None:
            if manifest["data"] != _data(inputs) or mal_ids != inputs.mal_ids:
                _fail("training_data_mismatch")
            positives = {
                int(item)
                for item, stats in inputs.train_stats["items"].items()
                if stats["positive_users"] > 0
            }
            if not set(supported) <= positives:
                _fail("training_data_mismatch")
        return VerifiedTrainingArtifact(
            expected_identity, manifest, spec, mal_ids, supported, root / WEIGHTS
        )
    except OSError as exc:
        raise ResearchError("training_artifact_invalid") from exc


def publish_training_artifact(
    output_dir: str | Path,
    *,
    spec: TrainingSpec,
    inputs: ScoringInputs,
    weights_path: Path,
    supported_mal_ids: tuple[int, ...],
    code_revision: str,
    framework_version: str,
) -> Path:
    """Stage three files, verify them, and publish a new directory atomically."""
    parsed = parse_training_spec(spec.wire())
    if parsed.family != "seed-embedding":
        _fail("training_adapter_unsupported")
    _validate_version(framework_version)
    if type(code_revision) is not str or _REVISION.fullmatch(code_revision) is None:
        _fail()
    target = Path(output_dir)
    source = Path(weights_path)
    if source.suffix != ".safetensors" or any(
        part.lower() in _FORBIDDEN_WEIGHTS_EXTENSIONS for part in source.name.split(".")[1:-1]
    ):
        _fail()
    try:
        safe_path(target.parent, directory=True)
        safe_path(source, directory=False)
        separate(target, source)
        if target.exists() or target.is_symlink():
            _fail("training_output_exists")
        if not 1 <= source.stat().st_size <= MAX_WEIGHTS:
            _fail("training_resource_limit")
    except OSError as exc:
        raise ResearchError("research_unsafe_path") from exc
    if len(inputs.mal_ids) > 100000:
        _fail("training_resource_limit")
    if (
        any(type(item) is not int for item in supported_mal_ids)
        or tuple(sorted(set(supported_mal_ids))) != supported_mal_ids
        or not set(supported_mal_ids) <= set(inputs.mal_ids)
    ):
        _fail("training_data_mismatch")
    positives = {
        int(item)
        for item, stats in inputs.train_stats["items"].items()
        if stats["positive_users"] > 0
    }
    if not set(supported_mal_ids) <= positives:
        _fail("training_data_mismatch")
    stage: Path | None = None
    try:
        stage = open_stage(target)
        digest = hashlib.sha256()
        size = 0
        source_fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with (
            os.fdopen(source_fd, "rb") as inp,
            open(
                stage / WEIGHTS, "xb", opener=lambda name, flags: os.open(name, flags, 0o600)
            ) as out,
        ):
            if not stat.S_ISREG(os.fstat(inp.fileno()).st_mode):
                _fail("research_unsafe_path")
            while block := inp.read(BLOCK):
                size += len(block)
                if size > MAX_WEIGHTS:
                    _fail("training_resource_limit")
                out.write(block)
                digest.update(block)
            if size < 1:
                _fail()
            out.flush()
            os.fsync(out.fileno())
        supported_set = set(supported_mal_ids)
        items = {
            "schema_version": 1,
            "mal_ids": list(inputs.mal_ids),
            "supported": [item in supported_set for item in inputs.mal_ids],
        }
        _check_items(items)
        items_descriptor = write_json(stage / ITEMS, items)
        if items_descriptor["bytes"] > MAX_ITEMS:
            _fail("training_resource_limit")
        data = _data(inputs)
        author = producer(code_revision)
        manifest: TrainingManifest = {
            "schema_version": 1,
            "kind": "arb-training-artifact",
            "config": parsed.wire(),
            "config_identity": _hash(_canonical(parsed.wire())),
            "data": data,
            "data_identity": _hash(_canonical(data)),
            "producer": author,
            "source_identity": _hash(_canonical(author)),
            "framework_version": framework_version,
            "files": {
                ITEMS: items_descriptor,
                WEIGHTS: {"bytes": size, "sha256": digest.hexdigest()},
            },
        }
        manifest_descriptor = write_json(stage / MANIFEST, manifest)
        if manifest_descriptor["bytes"] > MAX_MANIFEST:
            _fail("training_resource_limit")
        identity = "sha256:" + manifest_descriptor["sha256"]
        verify_training_artifact(stage, expected_identity=identity, inputs=inputs)
        publish(stage, target)
        stage = None
        return target / MANIFEST
    except BaseException as exc:
        if stage is not None:
            cleanup(stage)
        if isinstance(exc, OSError):
            if exc.errno == errno.EEXIST:
                raise ResearchError("training_output_exists") from exc
            raise ResearchError("training_artifact_invalid") from exc
        raise
