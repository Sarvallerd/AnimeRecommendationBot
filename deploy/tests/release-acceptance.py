#!/usr/bin/env python3
"""Private, opt-in full-input offline release compatibility and read-only verification."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import re
import stat
import subprocess
import sys
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, TypedDict, cast

if TYPE_CHECKING:
    from deploy.tests.acceptance_types import (
        AcceptanceProfile,
        BundleSummaryView,
        ContainerRecord,
        CurrentAcceptanceProfile,
        EvidenceModule,
        JsonObject,
        PathDescriptor,
        ProfileLock,
        ProfileModule,
        SavedDescriptor,
        SnapshotRecord,
    )

ROOT = Path(__file__).resolve().parents[2]
NEW = "neelagiri-2025-nafiul-2026-v1"
LABELS = ("old", "new-A", "new-B", "new-A-return", "old-restore")
CHECKS = (
    "reference",
    "deterministic_ab",
    "python_rust",
    "semantic_expectations",
    "input_immutability",
    "database_history",
    "original_settings",
    "restored_old_prepare",
)
REASON = {
    "PASS",
    "INVALID_ARGUMENT",
    "UNSAFE_PATH",
    "UNSAFE_SETTINGS",
    "SOURCE_CHANGED",
    "PROFILE_MISMATCH",
    "INPUT_CHANGED",
    "BUILD_FAILED",
    "BUILD_MISMATCH",
    "RUNTIME_UNAVAILABLE",
    "BOT_RUNNING",
    "DATABASE_UNHEALTHY",
    "DATABASE_HISTORY",
    "PREPARE_FAILED",
    "PREPARE_TIMEOUT",
    "PHASE_MISMATCH",
    "RESTORE_FAILED",
    "RESTORE_PENDING",
    "EVIDENCE_INVALID",
    "TRANSPORT_TIMEOUT",
}


def load(name: str, filename: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load acceptance script {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


p_module = load("arb033_profiles", "acceptance-profiles.py")
for export in ("load_profile", "check_profile_bundle", "canonical", "digest"):
    if not callable(getattr(p_module, export, None)):
        raise ImportError(f"acceptance profiles missing {export}")
p = cast("ProfileModule", p_module)
e_module = load("arb033_evidence", "acceptance-evidence.py")
for export in (
    "canonical",
    "utc",
    "read_settings",
    "container",
    "snapshot",
    "validate_database",
):
    if not callable(getattr(e_module, export, None)):
        raise ImportError(f"acceptance evidence missing {export}")
e = cast("EvidenceModule", e_module)


class ReleaseError(ValueError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class ReferenceCheckoutRequired(ReleaseError):
    """Current source cannot reproduce the accepted v1 producer bytes."""


class Selection(TypedDict):
    path: Path
    identity: str
    count: int


class Retained(TypedDict):
    input_files: dict[str, PathDescriptor]
    input_paths: list[Path]
    old_files: dict[str, PathDescriptor]


class PhaseEntry(TypedDict):
    label: str
    bundle_identity: str
    settings: SavedDescriptor
    snapshot: SavedDescriptor


class SwitchState(TypedDict):
    begun: bool


class RecordedBot(TypedDict):
    running: bool


class RecordedPostgres(TypedDict):
    id: str
    image_id: str
    database_volume: str
    started_at: str
    running: bool


class RecordedPrepare(TypedDict):
    id: str
    image_id: str
    status: str
    exit_code: int
    user: str
    read_only_root: bool
    restart_count: int
    artifact_mount: dict[str, object]
    bundle_id: str
    bundle_count: int
    started_at: str


class RecordedRuntime(TypedDict):
    bot: RecordedBot | None
    prepare: RecordedPrepare | None
    postgres: RecordedPostgres | None


class RecordedSnapshot(TypedDict):
    schema_version: int
    kind: str
    snapshot_sql_sha256: str
    source_commit: str
    captured_at: str
    deployment: dict[str, str]
    runtime: RecordedRuntime
    database: JsonObject | None


class SourceRecord(TypedDict):
    commit: str
    tool_sha256: str
    compose_sha256: str
    dockerfile_sha256: dict[str, str]


class OriginalRecord(TypedDict):
    settings_sha256: str
    settings_uid: int
    settings_gid: int
    settings_mode: int
    bundle_identity: str
    bundle_path: str
    bundle_file_sha256: dict[str, str]


class InputsRecord(TypedDict):
    base_identity: str
    alias_identity: str
    catalog_release_identity: str
    embeddings_sha256: str


class BuildRecord(TypedDict):
    identity: str
    path: str
    count: int
    report_sha256: str
    file_sha256: dict[str, str]
    producer: JsonObject
    block_size: int
    report: SavedDescriptor


class RuntimeReport(TypedDict):
    project: str
    builder_image_id: str
    bot_image_id: str
    postgres_image_id: str
    database_volume: str
    baseline_snapshot_sha256: str
    prepare_sequence: list[PhaseEntry]


class PreparationReport(TypedDict):
    schema_version: int
    kind: str
    scope: str
    status: str
    reason: str
    profile: JsonObject
    source: SourceRecord
    original: OriginalRecord
    inputs: InputsRecord
    builds: dict[str, BuildRecord]
    runtime: RuntimeReport
    checks: dict[str, str]
    live_acceptance: str
    production_history_preservation: str
    live_verifier: str
    saved_evidence: dict[str, SavedDescriptor]


def need(condition: bool, code: str) -> None:
    if not condition:
        raise ReleaseError(code)


def record(value: object, names: tuple[str, ...], code: str) -> dict[str, object]:
    need(type(value) is dict and set(value) == set(names), code)
    return cast("dict[str, object]", value)


def mapping(value: object, code: str) -> dict[str, object]:
    need(type(value) is dict and all(type(key) is str for key in value), code)
    return cast("dict[str, object]", value)


def required(value: object, names: tuple[str, ...], code: str) -> dict[str, object]:
    data = mapping(value, code)
    need(all(name in data for name in names), code)
    return data


def text_value(value: object, code: str) -> str:
    if type(value) is not str:
        raise ReleaseError(code)
    return value


def integer_value(value: object, code: str) -> int:
    if type(value) is not int:
        raise ReleaseError(code)
    return value


def items(value: object, code: str) -> list[object]:
    need(type(value) is list, code)
    return cast("list[object]", value)


def string_map(value: object, code: str) -> dict[str, str]:
    content = mapping(value, code)
    for item in content.values():
        text_value(item, code)
    return cast("dict[str, str]", content)


def saved_shape(value: object, code: str) -> SavedDescriptor:
    data = record(value, ("file", "sha256"), code)
    text_value(data["file"], code)
    text_value(data["sha256"], code)
    return cast("SavedDescriptor", data)


def path_shape(value: object, code: str) -> PathDescriptor:
    data = record(value, ("path", "sha256", "bytes"), code)
    text_value(data["path"], code)
    text_value(data["sha256"], code)
    integer_value(data["bytes"], code)
    return cast("PathDescriptor", data)


def path_map(value: object, code: str) -> dict[str, PathDescriptor]:
    data = mapping(value, code)
    for item in data.values():
        path_shape(item, code)
    return cast("dict[str, PathDescriptor]", data)


def snapshot_shape(value: JsonObject, code: str) -> RecordedSnapshot:
    data = required(
        value,
        (
            "schema_version",
            "kind",
            "snapshot_sql_sha256",
            "source_commit",
            "captured_at",
            "deployment",
            "runtime",
            "database",
        ),
        code,
    )
    integer_value(data["schema_version"], code)
    for name in ("kind", "snapshot_sql_sha256", "source_commit", "captured_at"):
        text_value(data[name], code)
    deployment = required(
        data["deployment"], ("project", "compose_sha256", "settings_sha256"), code
    )
    for name in ("project", "compose_sha256", "settings_sha256"):
        text_value(deployment[name], code)
    runtime = required(data["runtime"], ("bot", "prepare", "postgres"), code)
    bot = runtime["bot"]
    if bot is not None:
        bot_data = required(bot, ("running",), code)
        need(type(bot_data["running"]) is bool, code)
    pg = runtime["postgres"]
    if pg is not None:
        pg_data = required(pg, ("id", "image_id", "database_volume", "started_at", "running"), code)
        for name in ("id", "image_id", "database_volume", "started_at"):
            text_value(pg_data[name], code)
        need(type(pg_data["running"]) is bool, code)
    prepare = runtime["prepare"]
    if prepare is not None:
        prepare_data = required(
            prepare,
            (
                "id",
                "image_id",
                "status",
                "exit_code",
                "user",
                "read_only_root",
                "restart_count",
                "artifact_mount",
                "bundle_id",
                "bundle_count",
                "started_at",
            ),
            code,
        )
        for name in ("id", "image_id", "status", "user", "bundle_id", "started_at"):
            text_value(prepare_data[name], code)
        for name in ("exit_code", "restart_count", "bundle_count"):
            integer_value(prepare_data[name], code)
        need(type(prepare_data["read_only_root"]) is bool, code)
        mount = required(prepare_data["artifact_mount"], ("source", "rw"), code)
        text_value(mount["source"], code)
        need(type(mount["rw"]) is bool, code)
    if data["database"] is not None:
        mapping(data["database"], code)
    return cast("RecordedSnapshot", value)


def manifest_sources(value: JsonObject, code: str) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for item in items(value.get("sources"), code):
        source = required(item, ("name", "sha256", "metadata"), code)
        name = text_value(source["name"], code)
        text_value(source["sha256"], code)
        mapping(source["metadata"], code)
        result[name] = source
    return result


def report_shape(value: JsonObject) -> PreparationReport:
    """Check complete persisted wire types without rebuilding or dropping any field."""
    code = "EVIDENCE_INVALID"
    report = record(
        value,
        (
            "schema_version",
            "kind",
            "scope",
            "status",
            "reason",
            "profile",
            "source",
            "original",
            "inputs",
            "builds",
            "runtime",
            "checks",
            "live_acceptance",
            "production_history_preservation",
            "live_verifier",
            "saved_evidence",
        ),
        code,
    )
    integer_value(report["schema_version"], code)
    for name in (
        "kind",
        "scope",
        "status",
        "reason",
        "live_acceptance",
        "production_history_preservation",
        "live_verifier",
    ):
        text_value(report[name], code)
    profile = mapping(report["profile"], code)
    for name in ("id", "sha256", "release_id", "release_lock_sha256"):
        text_value(profile[name], code)
    source = record(
        report["source"],
        ("commit", "tool_sha256", "compose_sha256", "dockerfile_sha256"),
        code,
    )
    for name in ("commit", "tool_sha256", "compose_sha256"):
        text_value(source[name], code)
    string_map(source["dockerfile_sha256"], code)
    original = mapping(report["original"], code)
    for name in ("settings_sha256", "bundle_identity", "bundle_path"):
        text_value(original[name], code)
    for name in ("settings_uid", "settings_gid", "settings_mode"):
        integer_value(original[name], code)
    string_map(original["bundle_file_sha256"], code)
    inputs = mapping(report["inputs"], code)
    for name in (
        "base_identity",
        "alias_identity",
        "catalog_release_identity",
        "embeddings_sha256",
    ):
        text_value(inputs[name], code)
    builds = mapping(report["builds"], code)
    need(set(builds) == {"A", "B"}, code)
    for entry in builds.values():
        build = record(
            entry,
            (
                "identity",
                "path",
                "count",
                "report_sha256",
                "file_sha256",
                "producer",
                "block_size",
                "report",
            ),
            code,
        )
        for name in ("identity", "path", "report_sha256"):
            text_value(build[name], code)
        for name in ("count", "block_size"):
            integer_value(build[name], code)
        string_map(build["file_sha256"], code)
        mapping(build["producer"], code)
        saved_shape(build["report"], code)
    runtime = record(
        report["runtime"],
        (
            "project",
            "builder_image_id",
            "bot_image_id",
            "postgres_image_id",
            "database_volume",
            "baseline_snapshot_sha256",
            "prepare_sequence",
        ),
        code,
    )
    for name in (
        "project",
        "builder_image_id",
        "bot_image_id",
        "postgres_image_id",
        "database_volume",
        "baseline_snapshot_sha256",
    ):
        text_value(runtime[name], code)
    for entry in items(runtime["prepare_sequence"], code):
        phase = record(entry, ("label", "bundle_identity", "settings", "snapshot"), code)
        text_value(phase["label"], code)
        text_value(phase["bundle_identity"], code)
        saved_shape(phase["settings"], code)
        saved_shape(phase["snapshot"], code)
    string_map(report["checks"], code)
    records = mapping(report["saved_evidence"], code)
    for item in records.values():
        saved_shape(item, code)
    return cast("PreparationReport", value)


def current_profile(profile: AcceptanceProfile, code: str) -> CurrentAcceptanceProfile:
    """Narrow a pinned release profile only after checking its complete nested shape."""
    need(
        profile.id == NEW
        and profile.release_id == NEW
        and profile.policy == "release-lock-v1"
        and profile.sha256 == p.TRUSTED_NEW_PROFILE_SHA
        and profile.lock_sha256 == p.TRUSTED_LOCK_SHA,
        code,
    )
    lock = record(
        profile.lock,
        ("schema_version", "kind", "release_id", "inputs", "reference", "semantic_expectations"),
        code,
    )
    need(
        integer_value(lock["schema_version"], code) == 1
        and lock["kind"] == "arb-release-lock"
        and lock["release_id"] == NEW,
        code,
    )
    inputs = record(
        lock["inputs"], ("algorithm", "aliases", "base", "catalog_release", "embeddings"), code
    )
    for source_name in ("base", "aliases"):
        source = record(inputs[source_name], ("identity", "snapshot_id"), code)
        text_value(source["identity"], code)
        text_value(source["snapshot_id"], code)
    release = record(
        inputs["catalog_release"], ("catalog_count", "catalog_sha256", "identity"), code
    )
    integer_value(release["catalog_count"], code)
    for name in ("catalog_sha256", "identity"):
        text_value(release[name], code)
    embeddings = record(
        inputs["embeddings"],
        ("registry_sha256", "snapshot_id", "text_sha256", "text_size"),
        code,
    )
    integer_value(embeddings["text_size"], code)
    for name in ("registry_sha256", "snapshot_id", "text_sha256"):
        text_value(embeddings[name], code)
    algorithm = record(
        inputs["algorithm"],
        ("max_neighbors", "name", "parameters", "similarity", "version"),
        code,
    )
    integer_value(algorithm["max_neighbors"], code)
    for name in ("name", "similarity", "version"):
        text_value(algorithm[name], code)
    need(type(algorithm["parameters"]) is dict, code)
    reference = record(
        lock["reference"],
        (
            "block_size",
            "build_report_sha256",
            "bundle_identity",
            "catalog_sha256",
            "neighbors_sha256",
            "producer",
        ),
        code,
    )
    integer_value(reference["block_size"], code)
    for name in ("build_report_sha256", "bundle_identity", "catalog_sha256", "neighbors_sha256"):
        text_value(reference[name], code)
    producer = record(
        reference["producer"],
        (
            "code_files_sha256",
            "numpy_version",
            "package_version",
            "python_version",
            "source_commit",
        ),
        code,
    )
    for name in ("numpy_version", "package_version", "python_version", "source_commit"):
        text_value(producer[name], code)
    code_files = producer["code_files_sha256"]
    need(
        type(code_files) is dict
        and all(type(name) is str and type(digest) is str for name, digest in code_files.items()),
        code,
    )
    semantic = record(
        lock["semantic_expectations"],
        (
            "catalog_count",
            "neighbor_count_histogram",
            "smoke_seed",
            "empty_seed",
            "vector_status_counts",
            "zero_vector_ids",
        ),
        code,
    )
    integer_value(semantic["catalog_count"], code)
    for name in ("neighbor_count_histogram", "vector_status_counts"):
        counts = semantic[name]
        need(
            type(counts) is dict
            and all(type(key) is str and type(count) is int for key, count in counts.items()),
            code,
        )
    smoke = record(semantic["smoke_seed"], ("mal_id", "query", "neighbor_ids"), code)
    integer_value(smoke["mal_id"], code)
    text_value(smoke["query"], code)
    need(
        type(smoke["neighbor_ids"]) is list
        and all(type(item) is int for item in smoke["neighbor_ids"]),
        code,
    )
    if semantic["empty_seed"] is not None:
        empty = record(semantic["empty_seed"], ("mal_id", "query"), code)
        integer_value(empty["mal_id"], code)
        text_value(empty["query"], code)
    need(
        type(semantic["zero_vector_ids"]) is list
        and all(type(item) is int for item in semantic["zero_vector_ids"])
        and profile.inputs == inputs
        and profile.reference == reference
        and profile.expectations == semantic,
        code,
    )
    return cast("CurrentAcceptanceProfile", profile)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def guarded(path: str | Path, *, directory: bool = False) -> Path:
    path = Path(path)
    need(
        path.is_absolute()
        and ".." not in path.parts
        and all(not part.is_symlink() for part in (path, *path.parents)),
        "UNSAFE_PATH",
    )
    need(path.is_dir() if directory else path.is_file(), "UNSAFE_PATH")
    return path


def primary_root(checkout: Path = ROOT) -> Path:
    result = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={checkout}",
            "rev-parse",
            "--path-format=absolute",
            "--git-common-dir",
        ],
        cwd=checkout,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    need(result.returncode == 0, "SOURCE_CHANGED")
    common = guarded(Path(result.stdout.strip()), directory=True)
    need(common.name == ".git" and common.parent.is_dir(), "SOURCE_CHANGED")
    return common.parent


def private_roots(checkout: Path = ROOT) -> tuple[Path, Path]:
    primary = primary_root(checkout)
    return (primary / ".arb", primary / ".tools" / "arb033-acceptance")


def project_lock_dir(checkout: Path = ROOT) -> Path:
    return primary_root(checkout) / ".arb" / "acceptance-locks"


def private_file(path: Path, raw: bytes) -> SavedDescriptor:
    need(
        not path.exists()
        and not path.is_symlink()
        and path.parent.is_dir()
        and stat.S_IMODE(path.parent.stat().st_mode) & 0o077 == 0,
        "UNSAFE_PATH",
    )
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return {"file": path.name, "sha256": sha(raw)}


def save_json(path: Path, value: object) -> SavedDescriptor:
    return private_file(path, e.canonical(value))


def saved(evidence: Path, descriptor: object) -> bytes:
    need(
        type(descriptor) is dict
        and set(descriptor) == {"file", "sha256"}
        and type(descriptor["file"]) is str
        and type(descriptor["sha256"]) is str
        and Path(descriptor["file"]).name == descriptor["file"]
        and re.fullmatch("[0-9a-f]{64}", descriptor["sha256"]) is not None,
        "EVIDENCE_INVALID",
    )
    descriptor = cast("SavedDescriptor", descriptor)
    path = guarded(evidence / descriptor["file"])
    raw = path.read_bytes()
    need(sha(raw) == descriptor["sha256"], "EVIDENCE_INVALID")
    return raw


def saved_json(evidence: Path, descriptor: object) -> JsonObject:
    return p.canonical(saved(evidence, descriptor))


def descriptor(path: str | Path) -> PathDescriptor:
    path = guarded(path)
    return {"path": str(path), "sha256": file_sha(path), "bytes": path.stat().st_size}


def check_descriptor(item: object) -> None:
    need(
        type(item) is dict
        and set(item) == {"path", "sha256", "bytes"}
        and type(item["path"]) is str
        and type(item["sha256"]) is str
        and re.fullmatch("[0-9a-f]{64}", item["sha256"]) is not None
        and type(item["bytes"]) is int
        and item["bytes"] >= 0,
        "EVIDENCE_INVALID",
    )
    item = cast("PathDescriptor", item)
    path = guarded(Path(item["path"]))
    need(path.stat().st_size == item["bytes"] and file_sha(path) == item["sha256"], "INPUT_CHANGED")


def files_under(paths: Iterable[Path]) -> dict[str, PathDescriptor]:
    result: dict[str, PathDescriptor] = {}
    for base in paths:
        base = guarded(base, directory=True)
        for path in sorted(base.rglob("*")):
            need(not path.is_symlink(), "UNSAFE_PATH")
            if path.is_file():
                result[str(path)] = descriptor(path)
    return result


def source_head() -> str:
    root = subprocess.run(
        ["git", "-c", f"safe.directory={ROOT}", "rev-parse", "--show-toplevel"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    need(root.returncode == 0 and root.stdout.strip() == str(ROOT), "SOURCE_CHANGED")
    result = subprocess.run(
        ["git", "-c", f"safe.directory={ROOT}", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    need(
        result.returncode == 0 and re.fullmatch("[0-9a-f]{40}", result.stdout.strip()) is not None,
        "SOURCE_CHANGED",
    )
    return result.stdout.strip()


def clean(head: str) -> None:
    need(source_head() == head, "SOURCE_CHANGED")
    result = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={ROOT}",
            "status",
            "--porcelain",
            "--untracked-files=normal",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    need(result.returncode == 0 and not result.stdout, "SOURCE_CHANGED")


def env_clean() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if not k.startswith(("ARB_", "COMPOSE_"))}


def command(
    argv: Sequence[str], *, timeout: int = 300, input_text: str | None = None, both: bool = False
) -> str:
    try:
        result = subprocess.run(
            argv,
            cwd=ROOT,
            env=env_clean(),
            input=input_text,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ReleaseError("TRANSPORT_TIMEOUT") from exc
    need(result.returncode == 0, "BUILD_FAILED" if timeout > 300 else "RUNTIME_UNAVAILABLE")
    return result.stdout + result.stderr if both else result.stdout


class Transport:
    def __init__(self, project: str, settings: Path) -> None:
        self.project = project
        self.settings = settings

    def compose(self, *args: str, timeout: int = 300, input_text: str | None = None) -> str:
        return command(
            [
                "docker",
                "compose",
                "--env-file",
                str(self.settings),
                "-p",
                self.project,
                "-f",
                str(ROOT / "compose.yaml"),
                *args,
            ],
            timeout=timeout,
            input_text=input_text,
        )

    def docker(self, *args: str) -> str:
        return command(["docker", *args], both=bool(args and args[0] == "logs"))

    def evidence(self) -> None:
        e.compose = lambda project, settings, *args, input_text=None: self.compose(
            *args, input_text=input_text
        )
        e.docker = lambda *args: self.docker(*args)


def postgres_health(t: Transport) -> ContainerRecord:
    pg = e.container(t.project, t.settings, "postgres")
    if pg is None:
        raise ReleaseError("DATABASE_UNHEALTHY")
    need(pg["running"], "DATABASE_UNHEALTHY")
    status = t.docker("inspect", "--format", "{{.State.Health.Status}}", pg["id"]).strip()
    need(status == "healthy" and bool(pg["database_volume"]), "DATABASE_UNHEALTHY")
    return pg


def quiescent(t: Transport, *, require_prepare_idle: bool = True) -> ContainerRecord:
    bot = e.container(t.project, t.settings, "bot")
    need(
        bot is None or not bot["running"],
        "BOT_RUNNING",
    )
    prepare = e.container(t.project, t.settings, "prepare")
    need(not require_prepare_idle or prepare is None or not prepare["running"], "PREPARE_FAILED")
    return postgres_health(t)


def validated_database(db: object) -> JsonObject:
    e.validate_database(db)
    return cast("JsonObject", db)


def nonempty_history(db: object) -> None:
    db = validated_database(db)
    need(
        any(
            db[name]
            for name in (
                "requests",
                "delivered",
                "anime_events",
                "recommendation_ratings",
                "feedback",
                "legacy_requests",
                "legacy_feedback",
            )
        ),
        "DATABASE_HISTORY",
    )


def change_selection(original: bytes, bundle: Path) -> bytes:
    lines = original.decode().splitlines(keepends=True)
    need(sum(line.startswith("ARB_BUNDLE_DIR=") for line in lines) == 1, "UNSAFE_SETTINGS")
    return "".join(
        "ARB_BUNDLE_DIR=" + str(bundle) + "\n" if line.startswith("ARB_BUNDLE_DIR=") else line
        for line in lines
    ).encode()


def setup_paths(data: Path, lock: ProfileLock) -> dict[str, Path]:
    base = data / "datasets" / lock["inputs"]["base"]["identity"].replace(":", "-")
    alias = data / "aliases" / lock["inputs"]["aliases"]["identity"].replace(":", "-")
    release = data / "releases" / lock["inputs"]["catalog_release"]["identity"].replace(":", "-")
    alias_raw = data / "raw" / "nafiul-2026-v1"
    glove_dir = data / "raw" / "glove-6b-300d-v1"
    glove = glove_dir / "glove.6B.300d.txt"
    for path in (base, alias, release, alias_raw, glove_dir):
        guarded(path, directory=True)
    guarded(glove)
    return {
        "base": base,
        "aliases": alias,
        "release": release,
        "alias_raw": alias_raw,
        "glove": glove,
    }


def builder_args(paths: dict[str, Path], lock: ProfileLock) -> list[str]:
    return [
        "--release-dir",
        "/data/" + str(paths["release"].relative_to(paths["data"])),
        "--release-identity",
        lock["inputs"]["catalog_release"]["identity"],
        "--base-dir",
        "/data/" + str(paths["base"].relative_to(paths["data"])),
        "--aliases-dir",
        "/data/" + str(paths["aliases"].relative_to(paths["data"])),
        "--alias-data-dir",
        "/data/" + str(paths["alias_raw"].relative_to(paths["data"])),
    ]


def probe_code() -> str:
    return """import hashlib,importlib.resources as r,json,platform,numpy,recsys
names=('build.py','bundle.py','export.py','release.py','release_build.py')
files={name:hashlib.sha256(r.files('recsys').joinpath(name).read_bytes()).hexdigest() for name in names}
lock=hashlib.sha256(r.files('recsys').joinpath('release_neelagiri_2025_nafiul_2026_v1.json').read_bytes()).hexdigest()
registry=hashlib.sha256(r.files('recsys').joinpath('source_registry_glove_6b_300d_v1.json').read_bytes()).hexdigest()
print(json.dumps({'python_version':platform.python_version(),'numpy_version':numpy.__version__,'package_version':recsys.__version__,'code_files_sha256':files,'lock_sha256':lock,'registry_sha256':registry},sort_keys=True))"""


def check_probe(probe: object, profile: CurrentAcceptanceProfile) -> None:
    ref = profile.reference["producer"]
    probe = record(
        probe,
        (
            "python_version",
            "numpy_version",
            "package_version",
            "code_files_sha256",
            "lock_sha256",
            "registry_sha256",
        ),
        "BUILD_MISMATCH",
    )
    need(
        all(
            probe[key] == ref[key] for key in ("python_version", "numpy_version", "package_version")
        )
        and probe["code_files_sha256"] == ref["code_files_sha256"]
        and probe["lock_sha256"] == profile.lock_sha256
        and probe["registry_sha256"] == profile.inputs["embeddings"]["registry_sha256"],
        "BUILD_MISMATCH",
    )


def check_report(
    report: JsonObject,
    report_raw: bytes,
    summary: BundleSummaryView,
    profile: CurrentAcceptanceProfile,
    block: int,
) -> str:
    inputs = mapping(report.get("inputs"), "BUILD_MISMATCH")
    glove = mapping(inputs.get("glove.6B.300d.txt"), "BUILD_MISMATCH")
    need(
        report.get("schema_version") == 2
        and report.get("kind") == "arb-release-build"
        and report.get("catalog_release_identity") == profile.inputs["catalog_release"]["identity"]
        and glove.get("sha256") == profile.inputs["embeddings"]["text_sha256"]
        and report.get("files")
        == {
            "catalog.json": {"sha256": summary.hashes["catalog.json"]},
            "neighbors.json": {"sha256": summary.hashes["neighbors.json"]},
        },
        "BUILD_MISMATCH",
    )
    ref = profile.reference["producer"]
    producer = mapping(report.get("producer"), "BUILD_MISMATCH")
    need(
        all(
            producer.get(key) == ref[key]
            for key in ("python_version", "numpy_version", "package_version")
        )
        and producer.get("code_files")
        == {name: ref["code_files_sha256"][name] for name in ("build.py", "release_build.py")},
        "BUILD_MISMATCH",
    )
    execution = mapping(report.get("execution"), "BUILD_MISMATCH")
    need(
        type(execution.get("requested_block_rows")) is int
        and execution["requested_block_rows"] == block
        and execution.get("effective_block_rows") == block,
        "BUILD_MISMATCH",
    )
    semantic = profile.expectations
    coverage = mapping(report.get("coverage"), "BUILD_MISMATCH")
    need(
        coverage.get("neighbor_count_histogram") == semantic["neighbor_count_histogram"]
        and coverage.get("vector_status_counts") == semantic["vector_status_counts"]
        and coverage.get("zero_vector_ids") == semantic["zero_vector_ids"],
        "BUILD_MISMATCH",
    )
    details = {
        key: mapping(item, "BUILD_MISMATCH")
        for key, item in mapping(report.get("anime"), "BUILD_MISMATCH").items()
    }
    need(
        type(details) is dict
        and set(details) == set(summary.neighbors)
        and all(
            type(item) is dict
            and item.get("vector_status") in ("no_tokens", "oov_only", "zero_vector", "ready")
            for item in details.values()
        )
        and {
            name: sum(item["vector_status"] == name for item in details.values())
            for name in ("ready", "oov_only", "no_tokens", "zero_vector")
        }
        == semantic["vector_status_counts"]
        and sorted(
            int(key) for key, item in details.items() if item.get("vector_status") != "ready"
        )
        == semantic["zero_vector_ids"]
        and all(
            details[key]["vector_status"] == "ready" or not summary.neighbors[key]
            for key in details
        )
        and all(details[key]["neighbor_count"] == len(summary.neighbors[key]) for key in details),
        "BUILD_MISMATCH",
    )
    return sha(report_raw)


def bundle_files(bundle: Path) -> dict[str, PathDescriptor]:
    return {
        name: descriptor(bundle / name)
        for name in ("catalog.json", "neighbors.json", "manifest.json")
    }


def check_retained(retained: Retained, old_path: Path) -> None:
    need(
        retained["input_files"] == files_under(retained["input_paths"])
        and retained["old_files"] == bundle_files(old_path),
        "INPUT_CHANGED",
    )


def phase_time(previous_capture: str, started_at: str, captured_at: str) -> None:
    need(e.utc(previous_capture) < e.utc(started_at) <= e.utc(captured_at), "PHASE_MISMATCH")


def snapshot_phase(
    t: Transport,
    evidence: Path,
    label: str,
    settings: Path,
    baseline: SnapshotRecord,
    expected: Selection,
    previous_id: str | None,
    previous_capture: str,
    bot_image: str,
    pg0: ContainerRecord,
    switch_state: SwitchState,
) -> tuple[PhaseEntry, str, str]:
    t.settings = settings
    quiescent(t)
    switch_state["begun"] = True
    t.compose("up", "-d", "--no-deps", "--force-recreate", "prepare")
    current = None
    for _ in range(120):
        current = e.container(t.project, settings, "prepare")
        if current is not None and current["status"] == "exited":
            break
        time.sleep(2)
    if current is None or current["status"] != "exited":
        raise ReleaseError("PREPARE_TIMEOUT")
    need(
        current["exit_code"] == 0
        and current["id"] != previous_id
        and current["image_id"] == bot_image
        and current["user"] == "10001:10001"
        and current["read_only_root"] is True
        and current["restart_count"] == 0
        and current["artifact_mount"] == {"source": str(expected["path"]), "rw": False}
        and current["bundle_id"] == expected["identity"]
        and current["bundle_count"] == expected["count"],
        "PHASE_MISMATCH",
    )
    pg = postgres_health(t)
    need(
        all(pg[key] == pg0[key] for key in ("id", "image_id", "database_volume", "started_at")),
        "DATABASE_HISTORY",
    )
    shot = e.snapshot(t.project, settings)
    phase_time(previous_capture, current["started_at"], shot["captured_at"])
    bot = shot["runtime"]["bot"]
    need(bot is None or not bot["running"], "BOT_RUNNING")
    need(shot["database"] == baseline["database"], "DATABASE_HISTORY")
    e.validate_database(shot["database"])
    evidence_file = save_json(evidence / f"snapshot-{label}.json", shot)
    return (
        {
            "label": label,
            "bundle_identity": expected["identity"],
            "settings": {"file": settings.name, "sha256": file_sha(settings)},
            "snapshot": evidence_file,
        },
        current["id"],
        shot["captured_at"],
    )


def run_prepare_sequence(
    t: Transport,
    evidence: Path,
    settings: Path,
    original_state: tuple[str, int, int, int],
    head: str,
    baseline: SnapshotRecord,
    selected: dict[str, Selection],
    settings_by_label: dict[str, Path],
    bot_image: str,
    pg0: ContainerRecord,
    retained: Retained,
    old_path: Path,
    switch_state: SwitchState,
) -> list[PhaseEntry]:
    sequence: list[PhaseEntry] = []
    previous = e.container(t.project, settings, "prepare")
    previous_id = previous["id"] if previous else None
    previous_capture = baseline["captured_at"]
    for label in LABELS:
        selection = selected[
            "old" if label.startswith("old") else "new-B" if label == "new-B" else "new-A"
        ]
        need(
            (
                sha(settings.read_bytes()),
                settings.stat().st_uid,
                settings.stat().st_gid,
                stat.S_IMODE(settings.stat().st_mode),
            )
            == original_state,
            "UNSAFE_SETTINGS",
        )
        clean(head)
        check_retained(retained, old_path)
        entry, previous_id, previous_capture = snapshot_phase(
            t,
            evidence,
            label,
            settings_by_label[label],
            baseline,
            selection,
            previous_id,
            previous_capture,
            bot_image,
            pg0,
            switch_state,
        )
        sequence.append(entry)
    return sequence


def restore_old(
    t: Transport,
    settings: Path,
    original_state: tuple[str, int, int, int],
    old_settings: Path,
    old_path: Path,
    old: BundleSummaryView,
    pg0: ContainerRecord,
    retained: Retained,
) -> str:
    try:
        t.settings = settings
        need(
            (
                sha(settings.read_bytes()),
                settings.stat().st_uid,
                settings.stat().st_gid,
                stat.S_IMODE(settings.stat().st_mode),
            )
            == original_state,
            "RESTORE_PENDING",
        )
        pg = quiescent(t)
        need(
            all(pg[key] == pg0[key] for key in ("id", "image_id", "database_volume", "started_at")),
            "RESTORE_PENDING",
        )
        check_retained(retained, old_path)
    except Exception:
        return "PENDING"
    try:
        t.settings = old_settings
        t.compose("up", "-d", "--no-deps", "--force-recreate", "prepare")
        restored = None
        for _ in range(120):
            restored = e.container(t.project, old_settings, "prepare")
            if restored is not None and restored["status"] == "exited":
                break
            time.sleep(2)
        need(
            restored is not None
            and restored["exit_code"] == 0
            and restored["bundle_id"] == old.identity
            and restored["artifact_mount"] == {"source": str(old_path), "rw": False},
            "RESTORE_FAILED",
        )
        return "PASS"
    except Exception:
        return "FAIL"


def restore_after_failure(
    switch_state: SwitchState,
    t: Transport,
    settings: Path,
    original_state: tuple[str, int, int, int],
    old_settings: Path | None,
    old_path: Path,
    old: BundleSummaryView,
    pg0: ContainerRecord | None,
    retained: Retained,
) -> str:
    if not switch_state["begun"] or old_settings is None or pg0 is None:
        return "PENDING"
    return restore_old(t, settings, original_state, old_settings, old_path, old, pg0, retained)


def verify_record(evidence: Path) -> PreparationReport:
    evidence = guarded(evidence, directory=True)
    raw_report = saved_json(
        evidence,
        {
            "file": "preparation-report.json",
            "sha256": file_sha(evidence / "preparation-report.json"),
        },
    )
    need(
        set(raw_report)
        == {
            "schema_version",
            "kind",
            "scope",
            "status",
            "reason",
            "profile",
            "source",
            "original",
            "inputs",
            "builds",
            "runtime",
            "checks",
            "live_acceptance",
            "production_history_preservation",
            "live_verifier",
            "saved_evidence",
        },
        "EVIDENCE_INVALID",
    )
    need(
        type(raw_report["schema_version"]) is int
        and raw_report["schema_version"] == 2
        and raw_report["kind"] == "arb-offline-release-acceptance"
        and raw_report["scope"] == "full-inputs-offline-compatibility"
        and raw_report["status"] == "PASS"
        and raw_report["reason"] == "PASS"
        and raw_report["live_acceptance"] == "PENDING"
        and raw_report["production_history_preservation"] == "PENDING"
        and raw_report["live_verifier"] == "REQUIRES_DATA06_V2",
        "EVIDENCE_INVALID",
    )
    report = report_shape(raw_report)
    records = report["saved_evidence"]
    need(
        set(records)
        == {
            "profile",
            "old_profile",
            "lock",
            "original_settings",
            "probe",
            "baseline",
            "input_files",
            "input_layout",
            "bundle_files",
            "snapshot_sql",
            "tool",
            "compose",
            "builder_dockerfile",
            "bot_dockerfile",
        },
        "EVIDENCE_INVALID",
    )
    need(
        report["source"]
        == {
            "commit": report["source"].get("commit"),
            "tool_sha256": sha(saved(evidence, records["tool"])),
            "compose_sha256": sha(saved(evidence, records["compose"])),
            "dockerfile_sha256": {
                "builder": sha(saved(evidence, records["builder_dockerfile"])),
                "bot": sha(saved(evidence, records["bot_dockerfile"])),
            },
        }
        and re.fullmatch("[0-9a-f]{40}", report["source"]["commit"]) is not None,
        "EVIDENCE_INVALID",
    )
    profile_raw = saved(evidence, records["profile"])
    lock_raw = saved(evidence, records["lock"])
    raw_profile = p.load_profile(NEW, _profile_bytes=profile_raw, _lock_bytes=lock_raw)
    need(
        raw_profile.sha256 == p.TRUSTED_NEW_PROFILE_SHA
        and raw_profile.lock_sha256 == p.TRUSTED_LOCK_SHA,
        "EVIDENCE_INVALID",
    )
    profile = current_profile(raw_profile, "EVIDENCE_INVALID")
    need(
        report["profile"]
        == {
            "id": profile.id,
            "sha256": profile.sha256,
            "release_id": NEW,
            "release_lock_sha256": profile.lock_sha256,
        },
        "EVIDENCE_INVALID",
    )
    original_raw = saved(evidence, records["original_settings"])
    original = report["original"]
    need(
        sha(original_raw) == original["settings_sha256"]
        and type(original["settings_uid"]) is int
        and type(original["settings_gid"]) is int
        and type(original["settings_mode"]) is int
        and (evidence / records["original_settings"]["file"]).stat().st_uid
        == original["settings_uid"]
        and (evidence / records["original_settings"]["file"]).stat().st_gid
        == original["settings_gid"]
        and stat.S_IMODE((evidence / records["original_settings"]["file"]).stat().st_mode)
        == original["settings_mode"],
        "EVIDENCE_INVALID",
    )
    original_value = e.read_settings(evidence / records["original_settings"]["file"])
    old_path = Path(original_value["ARB_BUNDLE_DIR"])
    need(str(old_path) == original["bundle_path"], "EVIDENCE_INVALID")
    old_profile = p.load_profile(
        p.LEGACY_RELEASE, _profile_bytes=saved(evidence, records["old_profile"])
    )
    need(old_profile.sha256 == p.TRUSTED_LEGACY_PROFILE_SHA, "EVIDENCE_INVALID")
    old_summary = p.check_profile_bundle(
        old_path, old_profile, expected_identity=original["bundle_identity"]
    )
    need(old_summary.hashes == original["bundle_file_sha256"], "EVIDENCE_INVALID")
    probe = saved_json(evidence, records["probe"])
    # Historical verification uses the saved probe and lock; it never compares today's HEAD or installed image.
    ref = profile.reference["producer"]
    need(
        all(
            probe.get(key) == ref[key]
            for key in ("python_version", "numpy_version", "package_version")
        )
        and probe.get("code_files_sha256") == ref["code_files_sha256"]
        and probe.get("lock_sha256") == profile.lock_sha256
        and probe.get("registry_sha256") == profile.inputs["embeddings"]["registry_sha256"],
        "EVIDENCE_INVALID",
    )
    input_files_raw = saved_json(evidence, records["input_files"])
    layout_raw = saved_json(evidence, records["input_layout"])
    need(
        set(layout_raw) == {"base", "aliases", "release", "alias_raw", "glove"}, "EVIDENCE_INVALID"
    )
    layout = string_map(layout_raw, "EVIDENCE_INVALID")
    need(
        input_files_raw
        == files_under(
            [Path(layout[key]) for key in ("base", "aliases", "release", "alias_raw")]
            + [Path(layout["glove"]).parent]
        ),
        "INPUT_CHANGED",
    )
    input_files = path_map(input_files_raw, "EVIDENCE_INVALID")
    for key, filename, identity in (
        ("base", "dataset-manifest.json", profile.inputs["base"]["identity"]),
        ("aliases", "alias-manifest.json", profile.inputs["aliases"]["identity"]),
        ("release", "release-manifest.json", profile.inputs["catalog_release"]["identity"]),
    ):
        target = Path(layout[key]) / filename
        need(
            str(target) in input_files and "sha256:" + file_sha(target) == identity,
            "EVIDENCE_INVALID",
        )
    need(
        str(Path(layout["release"]) / "catalog.json") in input_files
        and file_sha(Path(layout["release"]) / "catalog.json")
        == profile.inputs["catalog_release"]["catalog_sha256"]
        and str(Path(layout["glove"])) in input_files
        and file_sha(Path(layout["glove"])) == profile.inputs["embeddings"]["text_sha256"]
        and Path(layout["glove"]).stat().st_size == profile.inputs["embeddings"]["text_size"],
        "EVIDENCE_INVALID",
    )
    need(type(input_files) is dict and bool(input_files), "EVIDENCE_INVALID")
    for path, item in input_files.items():
        need(path == item["path"], "EVIDENCE_INVALID")
        check_descriptor(item)
    for key in ("base_identity", "alias_identity", "catalog_release_identity", "embeddings_sha256"):
        need(key in report["inputs"], "EVIDENCE_INVALID")
    need(
        report["inputs"]["base_identity"] == profile.inputs["base"]["identity"]
        and report["inputs"]["alias_identity"] == profile.inputs["aliases"]["identity"]
        and report["inputs"]["catalog_release_identity"]
        == profile.inputs["catalog_release"]["identity"]
        and report["inputs"]["embeddings_sha256"] == profile.inputs["embeddings"]["text_sha256"],
        "EVIDENCE_INVALID",
    )
    bundle_records_raw = saved_json(evidence, records["bundle_files"])
    need(set(bundle_records_raw) == {"old", "A", "B"}, "EVIDENCE_INVALID")
    bundle_records: dict[str, dict[str, PathDescriptor]] = {}
    for label, raw_files in bundle_records_raw.items():
        files = path_map(raw_files, "EVIDENCE_INVALID")
        bundle_records[label] = files
        need(set(files) == {"catalog.json", "neighbors.json", "manifest.json"}, "EVIDENCE_INVALID")
        for item in files.values():
            check_descriptor(item)
    need(
        {name: item["sha256"] for name, item in bundle_records["old"].items()}
        == old_summary.hashes,
        "EVIDENCE_INVALID",
    )
    need(
        all(item["path"] == str(old_path / name) for name, item in bundle_records["old"].items()),
        "EVIDENCE_INVALID",
    )
    builds = report["builds"]
    need(set(builds) == {"A", "B"}, "EVIDENCE_INVALID")
    reports: dict[str, JsonObject] = {}
    bundles: dict[str, BundleSummaryView] = {}
    for label, block in (("A", 256), ("B", 128)):
        entry = builds[label]
        need(
            set(entry)
            == {
                "identity",
                "path",
                "count",
                "report_sha256",
                "file_sha256",
                "producer",
                "block_size",
                "report",
            },
            "EVIDENCE_INVALID",
        )
        summary = p.check_profile_bundle(
            Path(entry["path"]), profile, expected_identity=entry["identity"]
        )
        need(
            summary.count == entry["count"]
            and summary.hashes == entry["file_sha256"]
            and entry["block_size"] == block
            and {name: item["sha256"] for name, item in bundle_records[label].items()}
            == summary.hashes
            and all(
                item["path"] == str(Path(entry["path"]) / name)
                for name, item in bundle_records[label].items()
            ),
            "EVIDENCE_INVALID",
        )
        raw = saved(evidence, entry["report"])
        reports[label] = p.canonical(raw)
        need(
            check_report(reports[label], raw, summary, profile, block) == entry["report_sha256"]
            and entry["producer"] == reports[label]["producer"],
            "EVIDENCE_INVALID",
        )
        manifest = p.canonical((Path(entry["path"]) / "manifest.json").read_bytes())
        sources = manifest_sources(manifest, "EVIDENCE_INVALID")
        report_metadata = mapping(sources["build_report"]["metadata"], "EVIDENCE_INVALID")
        exporter_metadata = mapping(sources["bundle_exporter"]["metadata"], "EVIDENCE_INVALID")
        need(
            sources["build_report"]["sha256"] == entry["report_sha256"]
            and report_metadata["producer"] == reports[label]["producer"]
            and sources["bundle_exporter"]["sha256"]
            == profile.reference["producer"]["code_files_sha256"]["release_build.py"]
            and exporter_metadata["export_sha256"]
            == profile.reference["producer"]["code_files_sha256"]["export.py"]
            and exporter_metadata["bundle_sha256"]
            == profile.reference["producer"]["code_files_sha256"]["bundle.py"],
            "EVIDENCE_INVALID",
        )
        bundles[label] = summary
    need(
        builds["A"]["identity"] == profile.reference["bundle_identity"]
        and builds["A"]["report_sha256"] == profile.reference["build_report_sha256"]
        and bundles["A"].hashes["catalog.json"] == profile.reference["catalog_sha256"]
        and bundles["A"].hashes["neighbors.json"] == profile.reference["neighbors_sha256"]
        and builds["A"]["identity"] != builds["B"]["identity"]
        and bundles["A"].hashes["catalog.json"] == bundles["B"].hashes["catalog.json"]
        and bundles["A"].hashes["neighbors.json"] == bundles["B"].hashes["neighbors.json"],
        "EVIDENCE_INVALID",
    )
    a, b = copy.deepcopy(reports["A"]), copy.deepcopy(reports["B"])
    del a["execution"]
    del b["execution"]
    need(a == b, "EVIDENCE_INVALID")
    baseline_raw = saved_json(evidence, records["baseline"])
    nonempty_history(baseline_raw["database"])
    baseline = snapshot_shape(baseline_raw, "EVIDENCE_INVALID")
    runtime = report["runtime"]
    need(
        set(runtime)
        == {
            "project",
            "builder_image_id",
            "bot_image_id",
            "postgres_image_id",
            "database_volume",
            "baseline_snapshot_sha256",
            "prepare_sequence",
        },
        "EVIDENCE_INVALID",
    )
    need(
        baseline["schema_version"] == 1
        and baseline["kind"] == "full"
        and baseline["snapshot_sql_sha256"] == sha(saved(evidence, records["snapshot_sql"]))
        and baseline["source_commit"] == report["source"]["commit"]
        and baseline["deployment"]
        == {
            "project": runtime["project"],
            "compose_sha256": report["source"]["compose_sha256"],
            "settings_sha256": original["settings_sha256"],
        }
        and records["baseline"]["sha256"] == runtime["baseline_snapshot_sha256"],
        "EVIDENCE_INVALID",
    )
    pg0 = baseline["runtime"]["postgres"]
    need(
        pg0 is not None
        and pg0["running"]
        and pg0["image_id"] == runtime["postgres_image_id"]
        and pg0["database_volume"] == runtime["database_volume"]
        and (baseline["runtime"]["bot"] is None or not baseline["runtime"]["bot"]["running"]),
        "EVIDENCE_INVALID",
    )
    if pg0 is None:
        raise ReleaseError("EVIDENCE_INVALID")
    sequence = runtime["prepare_sequence"]
    need(
        type(sequence) is list
        and len(sequence) == 5
        and [item["label"] for item in sequence] == list(LABELS),
        "EVIDENCE_INVALID",
    )
    last_id = None
    previous_capture = baseline["captured_at"]
    for phase_entry in sequence:
        need(
            set(phase_entry) == {"label", "bundle_identity", "settings", "snapshot"},
            "EVIDENCE_INVALID",
        )
        label = phase_entry["label"]
        expected = (
            old_summary if label.startswith("old") else bundles["B" if label == "new-B" else "A"]
        )
        expected_path = (
            old_path
            if label.startswith("old")
            else Path(builds["B" if label == "new-B" else "A"]["path"])
        )
        need(phase_entry["bundle_identity"] == expected.identity, "EVIDENCE_INVALID")
        phase_raw = saved(evidence, phase_entry["settings"])
        need(phase_raw == change_selection(original_raw, expected_path), "EVIDENCE_INVALID")
        e.read_settings(evidence / phase_entry["settings"]["file"])
        shot = snapshot_shape(saved_json(evidence, phase_entry["snapshot"]), "EVIDENCE_INVALID")
        need(
            shot["schema_version"] == 1
            and shot["kind"] == "full"
            and shot["source_commit"] == report["source"]["commit"]
            and shot["snapshot_sql_sha256"] == baseline["snapshot_sql_sha256"]
            and shot["deployment"]
            == {
                "project": runtime["project"],
                "compose_sha256": report["source"]["compose_sha256"],
                "settings_sha256": phase_entry["settings"]["sha256"],
            }
            and shot["database"] == baseline["database"],
            "EVIDENCE_INVALID",
        )
        e.validate_database(shot["database"])
        bot = shot["runtime"]["bot"]
        pg = shot["runtime"]["postgres"]
        prepare = shot["runtime"]["prepare"]
        if prepare is None:
            raise ReleaseError("EVIDENCE_INVALID")
        phase_time(previous_capture, prepare["started_at"], shot["captured_at"])
        need(
            (bot is None or not bot["running"])
            and pg is not None
            and pg["running"]
            and all(
                pg[key] == pg0[key] for key in ("id", "image_id", "database_volume", "started_at")
            )
            and prepare is not None
            and prepare["id"] != last_id
            and prepare["image_id"] == runtime["bot_image_id"]
            and prepare["status"] == "exited"
            and prepare["exit_code"] == 0
            and prepare["user"] == "10001:10001"
            and prepare["read_only_root"] is True
            and prepare["restart_count"] == 0
            and prepare["artifact_mount"] == {"source": str(expected_path), "rw": False}
            and prepare["bundle_id"] == expected.identity
            and prepare["bundle_count"] == expected.count,
            "EVIDENCE_INVALID",
        )
        last_id = prepare["id"]
        previous_capture = shot["captured_at"]
    need(report["checks"] == {key: "PASS" for key in CHECKS}, "EVIDENCE_INVALID")
    return report


def require_v1_reference_checkout() -> CurrentAcceptanceProfile:
    """Reject a fresh v1 build from moved 0.2 source before any state mutation."""
    profile = current_profile(p.load_profile(NEW), "BUILD_MISMATCH")
    expected = profile.reference["producer"]["code_files_sha256"]
    source = ROOT / "src/recsys"
    if any(
        not (source / name).is_file() or file_sha(source / name) != digest
        for name, digest in expected.items()
    ):
        raise ReferenceCheckoutRequired("BUILD_MISMATCH")
    if not (source / "source_registry_glove_6b_300d_v1.json").is_file() or (
        file_sha(source / "source_registry_glove_6b_300d_v1.json")
        != profile.inputs["embeddings"]["registry_sha256"]
    ):
        raise ReferenceCheckoutRequired("BUILD_MISMATCH")
    return profile


def run(args: argparse.Namespace) -> dict[str, object]:
    # The accepted production v1 procedure runs only from the pinned 1441573f
    # checkout with Python 3.12.14. This gate is read-only and precedes locks,
    # evidence creation, Docker, and dataset inspection.
    require_v1_reference_checkout()
    import fcntl

    head = source_head()
    clean(head)
    data = guarded(args.data_dir, directory=True)
    settings = guarded(args.settings)
    evidence = Path(args.evidence_dir)
    allowed_private_roots = private_roots()
    need(
        evidence.is_absolute()
        and ".." not in evidence.parts
        and not evidence.exists()
        and not evidence.is_symlink()
        and evidence.parent.is_dir()
        and all(not item.is_symlink() for item in (evidence.parent, *evidence.parent.parents))
        and stat.S_IMODE(evidence.parent.stat().st_mode) == 0o700
        and ".worktrees" not in evidence.parts
        and ".worktrees" not in data.parts
        and not evidence.is_relative_to(data)
        and not data.is_relative_to(evidence),
        "UNSAFE_PATH",
    )
    need(
        any(
            all(item.is_relative_to(private_root) for item in (data, evidence, settings))
            for private_root in allowed_private_roots
        ),
        "UNSAFE_PATH",
    )
    need(
        e.PROJECT.fullmatch(args.project) is not None
        and e.IDENTITY.fullmatch(args.old_identity) is not None,
        "INVALID_ARGUMENT",
    )
    values = e.read_settings(settings)
    need(
        values["ARB_DATA_DIR"] == str(data)
        and settings.stat().st_uid == os.getuid()
        and data.stat().st_uid == os.getuid()
        and int(values["ARB_UID"]) == os.getuid()
        and int(values["ARB_GID"]) == os.getgid()
        and stat.S_IMODE(settings.stat().st_mode) & 0o077 == 0,
        "UNSAFE_SETTINGS",
    )
    for key in ("ARB_BOT_ENV", "ARB_POSTGRES_ENV"):
        secret = guarded(Path(values[key]))
        need(
            secret.stat().st_uid == os.getuid()
            and stat.S_IMODE(secret.stat().st_mode) & 0o077 == 0,
            "UNSAFE_SETTINGS",
        )
    old_path = guarded(Path(values["ARB_BUNDLE_DIR"]), directory=True)
    need(old_path.is_relative_to(data / "bundles"), "UNSAFE_SETTINGS")
    profile = current_profile(p.load_profile(NEW), "PROFILE_MISMATCH")
    old_profile = p.load_profile(p.LEGACY_RELEASE)
    old = p.check_profile_bundle(old_path, old_profile, expected_identity=args.old_identity)
    need(
        old.identity != profile.reference["bundle_identity"]
        and old.hashes["catalog.json"] != profile.reference["catalog_sha256"],
        "PROFILE_MISMATCH",
    )
    paths = setup_paths(data, profile.lock)
    paths["data"] = data
    glove = paths["glove"]
    need(
        file_sha(glove) == profile.inputs["embeddings"]["text_sha256"]
        and glove.stat().st_size == profile.inputs["embeddings"]["text_size"],
        "INPUT_CHANGED",
    )
    lock_dir = project_lock_dir()
    guarded(lock_dir.parent, directory=True)
    lock_dir.mkdir(mode=0o700, exist_ok=True)
    lock_dir_stat = guarded(lock_dir, directory=True).stat()
    need(
        stat.S_IMODE(lock_dir_stat.st_mode) == 0o700 and lock_dir_stat.st_uid == os.getuid(),
        "UNSAFE_PATH",
    )
    lock_path = lock_dir / (args.project + ".lock")
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
    try:
        lock_stat = os.fstat(lock_fd)
        need(
            stat.S_ISREG(lock_stat.st_mode)
            and lock_stat.st_uid == os.getuid()
            and stat.S_IMODE(lock_stat.st_mode) == 0o600,
            "UNSAFE_PATH",
        )
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ReleaseError("RUNTIME_UNAVAILABLE") from exc
        return _run_locked(
            args, head, data, settings, evidence, values, old_path, old, profile, paths
        )
    finally:
        os.close(lock_fd)


def _run_locked(
    args: argparse.Namespace,
    head: str,
    data: Path,
    settings: Path,
    evidence: Path,
    values: dict[str, str],
    old_path: Path,
    old: BundleSummaryView,
    profile: CurrentAcceptanceProfile,
    paths: dict[str, Path],
) -> dict[str, object]:
    original_raw = settings.read_bytes()
    original_stat = settings.stat()
    original_state = (
        sha(original_raw),
        original_stat.st_uid,
        original_stat.st_gid,
        stat.S_IMODE(original_stat.st_mode),
    )
    evidence.mkdir(mode=0o700)
    saved_records = {}
    saved_records["profile"] = private_file(
        evidence / "profile.json", p.regular_bytes(p.PROFILE_DIR / p.PROFILE_FILES[NEW])
    )
    saved_records["old_profile"] = private_file(
        evidence / "old-profile.json",
        p.regular_bytes(p.PROFILE_DIR / p.PROFILE_FILES[p.LEGACY_RELEASE]),
    )
    saved_records["lock"] = private_file(
        evidence / "release-lock.json",
        p.regular_bytes(ROOT / "src/recsys/resources/releases" / p.LOCK_RESOURCE),
    )
    saved_records["original_settings"] = private_file(
        evidence / "original-settings.env", original_raw
    )
    (evidence / "original-settings.env").chmod(original_state[3])
    for key, path in (
        ("tool", Path(__file__)),
        ("compose", ROOT / "compose.yaml"),
        ("builder_dockerfile", ROOT / "deploy/Dockerfile.builder"),
        ("bot_dockerfile", ROOT / "deploy/Dockerfile.bot"),
        ("snapshot_sql", e.SQL),
    ):
        saved_records[key] = private_file(evidence / (key + ".source"), guarded(path).read_bytes())
    input_paths = [paths[key] for key in ("base", "aliases", "release", "alias_raw")]
    input_paths.append(paths["glove"].parent)
    before_files = files_under(input_paths)
    saved_records["input_files"] = save_json(evidence / "input-files.json", before_files)
    saved_records["input_layout"] = save_json(
        evidence / "input-layout.json",
        {key: str(paths[key]) for key in ("base", "aliases", "release", "alias_raw", "glove")},
    )
    old_files = bundle_files(old_path)
    retained: Retained = {
        "input_files": before_files,
        "input_paths": input_paths,
        "old_files": old_files,
    }
    t = Transport(args.project, settings)
    t.evidence()
    switch_state: SwitchState = {"begun": False}
    baseline = None
    old_settings = None
    pg0: ContainerRecord | None = None
    try:
        quiescent(t)
        baseline = e.snapshot(args.project, settings)
        nonempty_history(baseline["database"])
        pg0 = baseline["runtime"]["postgres"]
        if pg0 is None:
            raise ReleaseError("DATABASE_UNHEALTHY")
        saved_records["baseline"] = save_json(evidence / "snapshot-baseline.json", baseline)
        t.compose("--profile", "tools", "build", "builder", "bot", timeout=7200)
        clean(head)
        builder_image = t.docker(
            "image", "inspect", "--format", "{{.Id}}", args.project + "-builder:local"
        ).strip()
        bot_image = t.docker(
            "image", "inspect", "--format", "{{.Id}}", args.project + "-bot:local"
        ).strip()
        need(bool(builder_image and bot_image), "BUILD_FAILED")
        probe_text = t.compose(
            "--profile",
            "tools",
            "run",
            "--rm",
            "--no-deps",
            "--entrypoint",
            "python",
            "builder",
            "-c",
            probe_code(),
            timeout=600,
        )
        try:
            probe = json.loads(probe_text.strip())
        except ValueError as exc:
            raise ReleaseError("BUILD_MISMATCH") from exc
        check_probe(probe, profile)
        saved_records["probe"] = save_json(evidence / "producer-probe.json", probe)
        common = builder_args(paths, profile.lock)
        t.compose(
            "--profile",
            "tools",
            "run",
            "--rm",
            "--no-deps",
            "builder",
            "release",
            "validate",
            "/data/" + str(paths["release"].relative_to(data)),
            *common[2:],
            timeout=7200,
        )
        clean(head)
        work = data / "work" / ("arb033-" + evidence.name)
        work.parent.mkdir(mode=0o700, exist_ok=True)
        need(
            guarded(work.parent, directory=True).stat().st_uid == os.getuid() and not work.exists(),
            "UNSAFE_PATH",
        )
        work.mkdir(mode=0o700)
        store = work / "export-store"
        store.mkdir(mode=0o700)
        builds: dict[str, BuildRecord] = {}
        summaries: dict[str, BundleSummaryView] = {}
        report_values: dict[str, JsonObject] = {}
        for label, block in (("A", 256), ("B", 128)):
            output = work / ("build-" + label)
            build_out = "/data/" + str(output.relative_to(data))
            glove_in = "/data/" + str(paths["glove"].relative_to(data))
            t.compose(
                "--profile",
                "tools",
                "run",
                "--rm",
                "--no-deps",
                "builder",
                "release",
                "build",
                *common,
                "--glove",
                glove_in,
                "--block-size",
                str(block),
                "--output-dir",
                build_out,
                timeout=7200,
            )
            clean(head)
            report_path = output / "build-report.json"
            report_raw = guarded(report_path).read_bytes()
            report_value = p.canonical(report_raw)
            t.compose(
                "--profile",
                "tools",
                "run",
                "--rm",
                "--no-deps",
                "builder",
                "release",
                "export",
                *common,
                "--neighbors",
                build_out + "/neighbors.json",
                "--build-report",
                build_out + "/build-report.json",
                "--output-dir",
                "/data/" + str(store.relative_to(data)),
                timeout=7200,
            )
            manifest_candidates = [item for item in store.iterdir() if item.is_dir()]
            need(len(manifest_candidates) == 1, "BUILD_MISMATCH")
            candidate = manifest_candidates[0]
            identity = "sha256:" + candidate.name.removeprefix("sha256-")
            summary = p.check_profile_bundle(candidate, profile, expected_identity=identity)
            final = data / "bundles" / candidate.name
            need(not final.exists() and not final.is_symlink(), "BUILD_MISMATCH")
            candidate.rename(final)
            final.chmod(0o755)
            for name in ("catalog.json", "neighbors.json", "manifest.json"):
                (final / name).chmod(0o644)
            summary = p.check_profile_bundle(final, profile, expected_identity=identity)
            report_hash = check_report(report_value, report_raw, summary, profile, block)
            report_copy = private_file(evidence / ("build-report-" + label + ".json"), report_raw)
            builds[label] = {
                "identity": identity,
                "path": str(final),
                "count": summary.count,
                "report_sha256": report_hash,
                "file_sha256": summary.hashes,
                # canonical() has already validated the recursive JSON value.
                "producer": cast("JsonObject", mapping(report_value["producer"], "BUILD_MISMATCH")),
                "block_size": block,
                "report": report_copy,
            }
            summaries[label] = summary
            report_values[label] = report_value
        a, b = copy.deepcopy(report_values["A"]), copy.deepcopy(report_values["B"])
        del a["execution"]
        del b["execution"]
        need(
            a == b
            and builds["A"]["identity"] != builds["B"]["identity"]
            and summaries["A"].hashes["catalog.json"] == summaries["B"].hashes["catalog.json"]
            and summaries["A"].hashes["neighbors.json"] == summaries["B"].hashes["neighbors.json"],
            "BUILD_MISMATCH",
        )
        ref = profile.reference
        need(
            builds["A"]["identity"] == ref["bundle_identity"]
            and builds["A"]["report_sha256"] == ref["build_report_sha256"]
            and summaries["A"].hashes["catalog.json"] == ref["catalog_sha256"]
            and summaries["A"].hashes["neighbors.json"] == ref["neighbors_sha256"],
            "BUILD_MISMATCH",
        )
        check_retained(retained, old_path)
        settings_by_label = {}
        selected: dict[str, Selection] = {
            "old": {"path": old_path, "identity": old.identity, "count": old.count},
            "new-A": {
                "path": Path(builds["A"]["path"]),
                "identity": builds["A"]["identity"],
                "count": builds["A"]["count"],
            },
            "new-B": {
                "path": Path(builds["B"]["path"]),
                "identity": builds["B"]["identity"],
                "count": builds["B"]["count"],
            },
        }
        for label in LABELS:
            selection = selected[
                "old" if label.startswith("old") else "new-B" if label == "new-B" else "new-A"
            ]
            settings_by_label[label] = evidence / ("settings-" + label + ".env")
            private_file(
                settings_by_label[label], change_selection(original_raw, selection["path"])
            )
        old_settings = settings_by_label["old-restore"]
        for label, selection in (("A", selected["new-A"]), ("B", selected["new-B"])):
            t.settings = settings_by_label["new-" + label]
            quiescent(t)
            rust = t.compose(
                "run",
                "--rm",
                "--no-deps",
                "--entrypoint",
                "/usr/local/bin/check_bundle",
                "prepare",
                "/artifacts",
                timeout=600,
            )
            need(
                rust.strip().splitlines()[-1]
                == selection["identity"] + " records=" + str(selection["count"]),
                "BUILD_MISMATCH",
            )
        t.settings = settings
        sequence = run_prepare_sequence(
            t,
            evidence,
            settings,
            original_state,
            head,
            baseline,
            selected,
            settings_by_label,
            bot_image,
            pg0,
            retained,
            old_path,
            switch_state,
        )
        t.settings = settings
        need(
            (
                sha(settings.read_bytes()),
                settings.stat().st_uid,
                settings.stat().st_gid,
                stat.S_IMODE(settings.stat().st_mode),
            )
            == original_state
            and before_files == files_under(input_paths)
            and old_files == bundle_files(old_path),
            "INPUT_CHANGED",
        )
        quiescent(t)
        saved_records["bundle_files"] = save_json(
            evidence / "bundle-files.json",
            {
                "old": old_files,
                "A": bundle_files(Path(builds["A"]["path"])),
                "B": bundle_files(Path(builds["B"]["path"])),
            },
        )
        report = {
            "schema_version": 2,
            "kind": "arb-offline-release-acceptance",
            "scope": "full-inputs-offline-compatibility",
            "status": "PASS",
            "reason": "PASS",
            "profile": {
                "id": profile.id,
                "sha256": profile.sha256,
                "release_id": NEW,
                "release_lock_sha256": profile.lock_sha256,
            },
            "source": {
                "commit": head,
                "tool_sha256": file_sha(Path(__file__)),
                "compose_sha256": file_sha(ROOT / "compose.yaml"),
                "dockerfile_sha256": {
                    "builder": file_sha(ROOT / "deploy/Dockerfile.builder"),
                    "bot": file_sha(ROOT / "deploy/Dockerfile.bot"),
                },
            },
            "original": {
                "settings_sha256": original_state[0],
                "settings_uid": original_state[1],
                "settings_gid": original_state[2],
                "settings_mode": original_state[3],
                "bundle_identity": old.identity,
                "bundle_path": str(old_path),
                "bundle_file_sha256": old.hashes,
            },
            "inputs": {
                "base_identity": profile.inputs["base"]["identity"],
                "alias_identity": profile.inputs["aliases"]["identity"],
                "catalog_release_identity": profile.inputs["catalog_release"]["identity"],
                "embeddings_sha256": profile.inputs["embeddings"]["text_sha256"],
            },
            "builds": builds,
            "runtime": {
                "project": args.project,
                "builder_image_id": builder_image,
                "bot_image_id": bot_image,
                "postgres_image_id": pg0["image_id"],
                "database_volume": pg0["database_volume"],
                "baseline_snapshot_sha256": saved_records["baseline"]["sha256"],
                "prepare_sequence": sequence,
            },
            "checks": {key: "PASS" for key in CHECKS},
            "live_acceptance": "PENDING",
            "production_history_preservation": "PENDING",
            "live_verifier": "REQUIRES_DATA06_V2",
            "saved_evidence": saved_records,
        }
        save_json(evidence / "preparation-report.json", report)
        verify_record(evidence)
        clean(head)
        return report
    except BaseException as exc:
        reason = (
            exc.code
            if isinstance(exc, (ReleaseError, p.ProfileError, e.EvidenceError))
            else "BUILD_FAILED"
        )
        restoration = restore_after_failure(
            switch_state, t, settings, original_state, old_settings, old_path, old, pg0, retained
        )
        try:
            save_json(
                evidence / "failure-status.json",
                {
                    "schema_version": 1,
                    "status": "FAIL",
                    "reason": reason if reason in REASON else "BUILD_FAILED",
                    "restoration": restoration,
                    "live_acceptance": "PENDING",
                },
            )
        except Exception:
            pass
        raise ReleaseError(reason if reason in REASON else "BUILD_FAILED") from exc


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="command", required=True)
    go = commands.add_parser("run")
    go.add_argument("--project", required=True)
    go.add_argument("--settings", type=Path, required=True)
    go.add_argument("--data-dir", type=Path, required=True)
    go.add_argument("--evidence-dir", type=Path, required=True)
    go.add_argument("--old-identity", required=True)
    go.add_argument("--offline", action="store_true", required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--evidence-dir", type=Path, required=True)
    return cli


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "run":
            run(args)
        else:
            verify_record(args.evidence_dir)
    except (
        ReleaseError,
        p.ProfileError,
        e.EvidenceError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        IndexError,
        subprocess.SubprocessError,
    ) as exc:
        code = (
            exc.code
            if isinstance(exc, (ReleaseError, p.ProfileError, e.EvidenceError))
            else "EVIDENCE_INVALID"
        )
        print(
            "FAIL "
            + (
                code
                if code in REASON or code in ("UNKNOWN_RELEASE", "LOCK_SHA_MISMATCH")
                else "EVIDENCE_INVALID"
            ),
            file=sys.stderr,
        )
        if isinstance(exc, ReferenceCheckoutRequired):
            print(
                "Use pinned v1 source checkout 1441573fbbbffc4693c7e1100aaae4458d9e88d4 with Python 3.12.14 to reconstruct the accepted release.",
                file=sys.stderr,
            )
        return 1
    print("PASS full-inputs-offline-compatibility; live PENDING; production history PENDING")
    return 0


if __name__ == "__main__":
    sys.exit(main())
