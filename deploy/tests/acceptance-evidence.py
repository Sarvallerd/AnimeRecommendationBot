#!/usr/bin/env python3
"""Private, read-only evidence for the ARB-020 live acceptance."""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import importlib.util
import json
import operator
import os
import re
import stat
import subprocess
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, NotRequired, TypedDict, cast

if TYPE_CHECKING:
    from deploy.tests.acceptance_types import (
        ContainerRecord,
        DeploymentRecord,
        JsonObject,
        JsonValue,
        LegacyBundle,
        LiveEvidenceMain,
        PathBundleContract,
        SnapshotRecord,
    )

    from recsys.contracts.bundle import Neighbors


type HistoricalRow = dict[str, object]


class HistoricalDatabaseRecord(TypedDict):
    migrations: list[HistoricalRow]
    users: list[HistoricalRow]
    requests: list[HistoricalRow]
    delivered: list[HistoricalRow]
    anime_events: list[HistoricalRow]
    anime_current: list[HistoricalRow]
    recommendation_ratings: list[HistoricalRow]
    feedback: list[HistoricalRow]
    legacy_users: list[HistoricalRow]
    legacy_requests: list[HistoricalRow]
    legacy_feedback: list[HistoricalRow]


class HistoricalFullRuntime(TypedDict):
    bot: ContainerRecord
    prepare: ContainerRecord | None
    postgres: ContainerRecord


class HistoricalFullSnapshot(TypedDict):
    schema_version: int
    kind: str
    snapshot_sql_sha256: str
    source_commit: str
    captured_at: str
    deployment: DeploymentRecord
    runtime: HistoricalFullRuntime
    database: HistoricalDatabaseRecord


class HistoricalDelta(TypedDict):
    requests: list[HistoricalRow]
    delivered: list[HistoricalRow]
    anime_events: list[HistoricalRow]
    anime_current: list[HistoricalRow]
    recommendation_ratings: list[HistoricalRow]
    feedback: list[HistoricalRow]
    users: list[HistoricalRow]


class HistoricalBundleReference(TypedDict):
    identity: str
    path: str


class HistoricalBuildReport(TypedDict):
    source_commit: str
    project: str
    compose_sha256: str
    settings_sha256: str
    database_volume: str | None
    bundles: dict[str, HistoricalBundleReference]
    schema_version: NotRequired[int]
    kind: NotRequired[str]
    profile: NotRequired[JsonObject]


class HistoricalActor(TypedDict):
    tg_id: int
    chat_id: int
    anchor_request_id: int


class HistoricalAnchor(TypedDict):
    before_sha256: str
    after_sha256: str
    build_report_sha256: str


class HistoricalContext(TypedDict):
    schema_version: int
    project: str
    database_volume: str | None
    actor: HistoricalActor
    bundles: dict[str, HistoricalBundleReference]
    anchor: HistoricalAnchor


class HistoricalTransitionResult(TypedDict):
    schema_version: int
    case: str
    phase: str | None
    status: str
    reason: str
    counts: dict[str, int]
    bundle_id: str | None
    context_sha256: NotRequired[str]
    input_sha256: NotRequired[dict[str, str]]


class HistoricalSummary(TypedDict):
    schema_version: int
    live_acceptance: str
    machine: dict[str, int]
    human: dict[str, int]
    context_sha256: str


class ObservedScenario(TypedDict):
    id: str
    status: str
    observed_at: JsonValue
    observation: str


ROOT = Path(__file__).resolve().parents[2]
SQL = Path(__file__).with_name("acceptance-snapshot.sql")
HEX = re.compile(r"[0-9a-f]{64}\Z")
COMMIT = re.compile(r"[0-9a-f]{40}\Z")
PROJECT = re.compile(r"[a-z][a-z0-9_-]{2,63}\Z")
IDENTITY = re.compile(r"sha256:[0-9a-f]{64}\Z")
LOG_IDENTITY = re.compile(
    r"Loaded recommendation bundle (sha256:[0-9a-f]{64}) \(([0-9]+) records\)"
)
TABLES = (
    "migrations",
    "users",
    "requests",
    "delivered",
    "anime_events",
    "anime_current",
    "recommendation_ratings",
    "feedback",
    "legacy_users",
    "legacy_requests",
    "legacy_feedback",
)
KEYS = {
    "migrations": ("version",),
    "users": ("tg_id",),
    "requests": ("id",),
    "delivered": ("id",),
    "anime_events": ("id",),
    "anime_current": ("tg_id", "mal_id"),
    "recommendation_ratings": ("position_id",),
    "feedback": ("id",),
}
IMMUTABLE = ("migrations", "delivered", "anime_events", "recommendation_ratings", "feedback")
LEGACY = ("legacy_users", "legacy_requests", "legacy_feedback")
CASES = (
    "recommendations",
    "recommendation-scores",
    "recommendation-repeat",
    "empty",
    "anime-score-1",
    "anime-score-10",
    "anime-repeat",
    "feedback-unicode",
    "navigation-search",
    "navigation-choice",
    "navigation-anime-score",
    "navigation-feedback",
    "restart",
    "db-outage",
    "feedback-retry",
    "feedback-retry-repeat",
    "update",
    "rollback",
)
NO_EFFECT = (
    "recommendation-repeat",
    "anime-repeat",
    "feedback-retry-repeat",
    "navigation-search",
    "navigation-choice",
    "navigation-anime-score",
    "navigation-feedback",
    "restart",
    "db-outage",
)
PHASES = (
    "setup",
    "action",
    "repeat",
    "after-restart",
    "during-outage",
    "after-recovery",
    "after-update",
    "after-rollback",
)
UI_IDS = (
    "navigation",
    "selection-cards",
    "score-ranges",
    "confirmations",
    "empty-notice",
    "consumed-controls",
    "stale-controls",
    "restart",
    "outage-failure",
    "retry-success",
    "update",
    "rollback",
)
CANARY_PHRASE = "длинный отзыв без обрезки; "
CANARY_COMMON = (
    "Русский ё, 日本語, 漢字, 🎌, e\u0301; кавычки \"' и обратная \\ черта; "
    "процент %, подчёркивание _, <угол> & амперсанд.\n" + CANARY_PHRASE * 12 + "\nКонец."
)
CANARIES = {name: f"ARB020 feedback/{name} v1\n{CANARY_COMMON}" for name in ("unicode", "retry")}


class EvidenceError(ValueError):
    """A fixed-code failure whose private values must not reach the terminal."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def require(condition: bool, code: str) -> None:
    if not condition:
        raise EvidenceError(code)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(value: object) -> bytes:
    return (
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        + "\n"
    ).encode()


def read_json(path: Path) -> JsonObject:
    require(not path.is_symlink() and path.is_file(), "UNSAFE_INPUT")
    value = json.loads(path.read_bytes())
    require(isinstance(value, dict), "INVALID_JSON")
    return cast("JsonObject", value)


def private_parent(path: Path) -> None:
    parent = path.parent
    require(parent.is_dir() and not parent.is_symlink(), "PRIVATE_DIRECTORY_REQUIRED")
    require(stat.S_IMODE(parent.stat().st_mode) & 0o077 == 0, "PRIVATE_DIRECTORY_REQUIRED")


def write_new(path: Path, value: object) -> str:
    private_parent(path)
    require(not path.exists() and not path.is_symlink(), "OUTPUT_EXISTS")
    raw = canonical(value)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return sha(raw)


def read_settings(path: Path) -> dict[str, str]:
    require(not path.is_symlink() and path.is_file(), "UNSAFE_SETTINGS")
    allowed = {
        "ARB_DATA_DIR",
        "ARB_BUNDLE_DIR",
        "ARB_UID",
        "ARB_GID",
        "ARB_LOG",
        "ARB_BOT_ENV",
        "ARB_POSTGRES_ENV",
    }
    settings: dict[str, str] = {}
    for line in path.read_text().splitlines():
        require("\r" not in line and not any(ord(char) < 32 for char in line), "UNSAFE_SETTINGS")
        require("=" in line and not line.startswith("#"), "UNSAFE_SETTINGS")
        key, value = line.split("=", 1)
        require(
            key in allowed
            and key not in settings
            and bool(value)
            and not re.search(r"\$|`|\\", value),
            "UNSAFE_SETTINGS",
        )
        settings[key] = value
    require(set(settings) == allowed and settings["ARB_LOG"] == "info", "UNSAFE_SETTINGS")
    for key in ("ARB_DATA_DIR", "ARB_BUNDLE_DIR", "ARB_BOT_ENV", "ARB_POSTGRES_ENV"):
        require(Path(settings[key]).is_absolute(), "UNSAFE_SETTINGS")
    for key in ("ARB_UID", "ARB_GID"):
        require(settings[key].isdigit() and int(settings[key]) > 0, "UNSAFE_SETTINGS")
    return settings


def compose(project: str, settings: Path, *args: str, input_text: str | None = None) -> str:
    require(PROJECT.fullmatch(project) is not None, "INVALID_PROJECT")
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("ARB_") and not key.startswith("COMPOSE_")
    }
    command = [
        "docker",
        "compose",
        "--env-file",
        str(settings),
        "-p",
        project,
        "-f",
        str(ROOT / "compose.yaml"),
        *args,
    ]
    result = subprocess.run(
        command, cwd=ROOT, env=env, input=input_text, capture_output=True, text=True, check=False
    )
    require(result.returncode == 0, "COMPOSE_FAILED")
    return result.stdout


def docker(*args: str) -> str:
    result = subprocess.run(["docker", *args], capture_output=True, text=True, check=False)
    require(result.returncode == 0, "DOCKER_FAILED")
    return result.stdout + result.stderr if args and args[0] == "logs" else result.stdout


def container(project: str, settings: Path, service: str) -> ContainerRecord | None:
    identifiers = compose(project, settings, "ps", "-a", "-q", service).split()
    require(len(identifiers) <= 1, "AMBIGUOUS_CONTAINER")
    if not identifiers:
        return None
    data = json.loads(docker("inspect", identifiers[0]))
    require(len(data) == 1 and data[0]["Id"] == identifiers[0], "INVALID_CONTAINER")
    item = data[0]
    labels = item["Config"].get("Labels") or {}
    require(
        labels.get("com.docker.compose.project") == project
        and labels.get("com.docker.compose.service") == service,
        "WRONG_CONTAINER",
    )
    state = item["State"]
    mount = next((m for m in item["Mounts"] if m["Destination"] == "/artifacts"), None)
    volume = next(
        (
            m["Name"]
            for m in item["Mounts"]
            if m["Destination"] == "/var/lib/postgresql/data" and m["Type"] == "volume"
        ),
        None,
    )
    identity = None
    count = None
    if service in ("bot", "prepare") and state["StartedAt"] not in ("", "0001-01-01T00:00:00Z"):
        logs = docker("logs", "--since", state["StartedAt"], item["Id"])
        matches = LOG_IDENTITY.findall(logs)
        require(len(matches) <= 1, "AMBIGUOUS_STARTUP_IDENTITY")
        if matches:
            identity, count_text = matches[0]
            count = int(count_text)
    return {
        "id": item["Id"],
        "image_id": item["Image"],
        "project": project,
        "service": service,
        "user": item["Config"].get("User", ""),
        "status": state["Status"],
        "running": state["Running"],
        "exit_code": state["ExitCode"],
        "started_at": state["StartedAt"],
        "restart_count": item["RestartCount"],
        "read_only_root": item["HostConfig"]["ReadonlyRootfs"],
        "artifact_mount": None if mount is None else {"source": mount["Source"], "rw": mount["RW"]},
        "database_volume": volume,
        "bundle_id": identity,
        "bundle_count": count,
    }


def snapshot(project: str, settings: Path, runtime_only: bool = False) -> SnapshotRecord:
    read_settings(settings)
    postgres = container(project, settings, "postgres")
    prepare = container(project, settings, "prepare")
    bot = container(project, settings, "bot")
    database = None
    if not runtime_only:
        require(postgres is not None and postgres["running"], "DATABASE_UNAVAILABLE")
        output = compose(
            project,
            settings,
            "exec",
            "-T",
            "postgres",
            "psql",
            "-X",
            "-q",
            "-A",
            "-t",
            "-v",
            "ON_ERROR_STOP=1",
            "-U",
            "anime_bot",
            "-d",
            "anime_bot",
            input_text=SQL.read_text(),
        )
        lines = [line for line in output.splitlines() if line.strip()]
        require(len(lines) == 1, "INVALID_DATABASE_SNAPSHOT")
        database = json.loads(lines[0])
        require(
            isinstance(database, dict) and set(database) == set(TABLES), "INVALID_DATABASE_SNAPSHOT"
        )
    source_commit = subprocess.run(
        ["git", "-c", f"safe.directory={ROOT}", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    require(COMMIT.fullmatch(source_commit) is not None, "INVALID_SOURCE_COMMIT")
    return {
        "schema_version": 1,
        "kind": "runtime_only" if runtime_only else "full",
        "snapshot_sql_sha256": sha(SQL.read_bytes()),
        "source_commit": source_commit,
        "captured_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "deployment": {
            "project": project,
            "compose_sha256": sha((ROOT / "compose.yaml").read_bytes()),
            "settings_sha256": sha(settings.read_bytes()),
        },
        "runtime": {"bot": bot, "prepare": prepare, "postgres": postgres},
        "database": database,
    }


def utc(value: object) -> dt.datetime:
    try:
        if not isinstance(value, str):
            raise AttributeError("timestamp has no replace method")
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, AttributeError) as exc:
        raise EvidenceError("INVALID_TIMESTAMP") from exc
    require(parsed.tzinfo is not None, "INVALID_TIMESTAMP")
    return parsed.astimezone(dt.timezone.utc)


def integer(value: object, low: int | None = None, high: int | None = None) -> bool:
    return type(value) is int and (low is None or value >= low) and (high is None or value <= high)


type HistoricalComparable = int | float | str | list[object]


def historical_greater(left: object, right: object) -> bool:
    if not isinstance(left, (int, float, str, list)) or not isinstance(
        right, (int, float, str, list)
    ):
        raise TypeError("historical values are not comparable")
    return bool(
        operator.gt(cast("HistoricalComparable", left), cast("HistoricalComparable", right))
    )


def historical_max(values: list[object]) -> object:
    if not values:
        raise ValueError("max() arg is an empty sequence")
    largest = values[0]
    for value in values[1:]:
        if historical_greater(value, largest):
            largest = value
    return largest


def historical_action_parts(value: object) -> list[str]:
    if not isinstance(value, str):
        raise AttributeError(f"'{type(value).__name__}' object has no attribute 'split'")
    return value.split(":")


def index_rows[T: Mapping[str, object]](
    rows: list[T], fields: tuple[str, ...]
) -> dict[tuple[object, ...], T]:
    require(isinstance(rows, list), "INVALID_ROWS")
    result: dict[tuple[object, ...], T] = {}
    for row in rows:
        require(isinstance(row, dict) and all(field in row for field in fields), "INVALID_ROWS")
        key = tuple(row[field] for field in fields)
        require(key not in result, "DUPLICATE_KEY")
        result[key] = row
    return result


def validate_database(db: object, bundle: LegacyBundle | None = None) -> None:
    require(isinstance(db, dict) and set(db) == set(TABLES), "INVALID_DATABASE_SNAPSHOT")
    row_tables = cast("dict[str, list[dict[str, object]]]", db)
    maps = {table: index_rows(row_tables[table], fields) for table, fields in KEYS.items()}
    require(set(maps["migrations"]) == {(1,)}, "MIGRATION_MISMATCH")
    for table in LEGACY:
        require(
            isinstance(row_tables[table], list)
            and all(isinstance(row, dict) for row in row_tables[table]),
            "INVALID_ROWS",
        )
    db = cast("HistoricalDatabaseRecord", db)
    users = maps["users"]
    requests = maps["requests"]
    delivered = maps["delivered"]
    events = maps["anime_events"]
    positions = {(row["request_id"], row["rank"]) for row in db["delivered"]}
    require(len(positions) == len(delivered), "DUPLICATE_RANK")
    require(
        len({(r["tg_id"], r["action_key"]) for r in db["requests"]}) == len(requests),
        "DUPLICATE_ACTION",
    )
    require(
        len({(r["tg_id"], r["action_key"]) for r in db["feedback"]}) == len(db["feedback"]),
        "DUPLICATE_ACTION",
    )
    for request_row in db["requests"]:
        require(
            integer(request_row["id"], 1)
            and integer(request_row["tg_id"], 1)
            and (request_row["tg_id"],) in users
            and isinstance(request_row["action_key"], str)
            and bool(request_row["action_key"])
            and isinstance(request_row["raw_query"], str),
            "INVALID_REQUEST",
        )
        selected = request_row["seed_mal_id"] is not None
        require(
            (
                not selected
                and request_row["bundle_id"] is None
                and request_row["resolved_at"] is None
            )
            or (
                selected
                and integer(request_row["seed_mal_id"], 1)
                and isinstance(request_row["bundle_id"], str)
                and IDENTITY.fullmatch(request_row["bundle_id"]) is not None
                and request_row["resolved_at"] is not None
            ),
            "INVALID_REQUEST",
        )
    for delivery_row in db["delivered"]:
        request = requests.get((delivery_row["request_id"],))
        require(
            request is not None
            and request["tg_id"] == delivery_row["tg_id"]
            and integer(delivery_row["rank"], 1, 5)
            and integer(delivery_row["mal_id"], 1)
            and integer(delivery_row["chat_id"])
            and delivery_row["chat_id"] != 0
            and integer(delivery_row["message_id"], 1)
            and delivery_row["mal_id"] != request["seed_mal_id"],
            "INVALID_DELIVERY",
        )
        if request is None:
            raise EvidenceError("INVALID_DELIVERY")
        if bundle is not None and request["bundle_id"] == bundle["id"]:
            neighbors = bundle["neighbors"][str(request["seed_mal_id"])]
            require(
                cast("int", delivery_row["rank"]) <= len(neighbors)
                and neighbors[cast("int", delivery_row["rank"]) - 1]["mal_id"]
                == delivery_row["mal_id"],
                "DELIVERY_ORDER",
            )
    for event_row in db["anime_events"]:
        request = requests.get((event_row["request_id"],))
        require(
            request is not None
            and request["tg_id"] == event_row["tg_id"]
            and request["seed_mal_id"] == event_row["mal_id"]
            and integer(event_row["score"], 1, 10),
            "INVALID_ANIME_EVENT",
        )
    require(
        len({r["request_id"] for r in db["anime_events"]}) == len(events), "DUPLICATE_ANIME_EVENT"
    )
    for current_row in db["anime_current"]:
        matching = [
            event
            for event in db["anime_events"]
            if (event["tg_id"], event["mal_id"]) == (current_row["tg_id"], current_row["mal_id"])
        ]
        require(
            bool(matching)
            and current_row["current_event_id"]
            == historical_max([event["id"] for event in matching]),
            "CURRENT_POINTER",
        )
    require(
        set(maps["anime_current"]) == {(r["tg_id"], r["mal_id"]) for r in db["anime_events"]},
        "CURRENT_POINTER",
    )
    for rating_row in db["recommendation_ratings"]:
        position = delivered.get((rating_row["position_id"],))
        require(
            position is not None
            and position["tg_id"] == rating_row["tg_id"]
            and integer(rating_row["score"], 0, 5),
            "INVALID_RECOMMENDATION_SCORE",
        )
    for feedback_row in db["feedback"]:
        require(
            integer(feedback_row["tg_id"], 1)
            and (feedback_row["tg_id"],) in users
            and isinstance(feedback_row["body"], str)
            and bool(feedback_row["body"]),
            "INVALID_FEEDBACK",
        )


def load_bundle(path: Path) -> LegacyBundle:
    module_path = ROOT / "contracts" / "check_bundle.py"
    spec = importlib.util.spec_from_file_location("arb020_bundle", module_path)
    if spec is None or spec.loader is None:
        raise EvidenceError("BUNDLE_CONTENT")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "check_bundle", None)) or not callable(
        getattr(module, "read_canonical", None)
    ):
        raise EvidenceError("BUNDLE_CONTENT")
    contract = cast("PathBundleContract", module)
    identity, count = contract.check_bundle(path)
    neighbors = cast("Neighbors", contract.read_canonical(path / "neighbors.json")[0])["neighbors"]
    require(count == 17562, "CATALOG_COUNT")
    require(
        [item["mal_id"] for item in neighbors["1"]] == [400, 709, 4088, 329, 1303]
        and neighbors["40089"] == [],
        "BUNDLE_CONTENT",
    )
    require(
        sum(len(items) == 5 for items in neighbors.values()) == 17561
        and sum(len(items) == 0 for items in neighbors.values()) == 1,
        "BUNDLE_CONTENT",
    )
    return {"id": identity, "count": count, "neighbors": neighbors}


def context_from_anchor(
    before: HistoricalFullSnapshot,
    after: HistoricalFullSnapshot,
    report: HistoricalBuildReport,
    before_hash: str,
    after_hash: str,
    report_hash: str,
) -> HistoricalContext:
    profile_block = report.get("profile")
    require(
        not (
            report.get("schema_version") == 2
            or report.get("kind") == "arb-offline-release-acceptance"
            or (
                isinstance(profile_block, dict)
                and profile_block.get("release_id") == "neelagiri-2025-nafiul-2026-v1"
            )
        ),
        "UNSUPPORTED_LIVE_PROFILE",
    )
    require(before["kind"] == after["kind"] == "full", "FULL_SNAPSHOT_REQUIRED")
    require(
        before["deployment"] == after["deployment"]
        and before["source_commit"] == after["source_commit"],
        "DEPLOYMENT_CHANGED",
    )
    require(utc(before["captured_at"]) < utc(after["captured_at"]), "TIME_ORDER")
    require(
        report.get("source_commit") == after["source_commit"]
        and report.get("project") == after["deployment"]["project"]
        and report.get("compose_sha256") == after["deployment"]["compose_sha256"]
        and report.get("settings_sha256") == after["deployment"]["settings_sha256"]
        and report.get("database_volume") == after["runtime"]["postgres"]["database_volume"],
        "BUILD_REPORT_CONTEXT",
    )
    a = report["bundles"]["A"]
    b = report["bundles"]["B"]
    require(
        a["identity"] != b["identity"]
        and after["runtime"]["bot"] is not None
        and after["runtime"]["bot"]["bundle_id"] == a["identity"],
        "ANCHOR_BUNDLE",
    )
    old = index_rows(before["database"]["requests"], ("id",))
    fresh = [
        row
        for row in after["database"]["requests"]
        if (row["id"],) not in old
        and row["raw_query"] == "Cowboy Bebop"
        and row["seed_mal_id"] == 1
        and row["bundle_id"] == a["identity"]
    ]
    require(len(fresh) == 1, "ANCHOR_AMBIGUOUS")
    validate_database(before["database"])
    validate_database(after["database"])
    request = fresh[0]
    require(
        integer(request["tg_id"], 1)
        and isinstance(request["action_key"], str)
        and request["action_key"].startswith("msg:")
        and request["action_key"].count(":") == 2,
        "ANCHOR_ACTION",
    )
    chat = int(cast("str", request["action_key"]).split(":")[1])
    require(chat == request["tg_id"], "ANCHOR_CHAT")
    selected_bundle = load_bundle(Path(a["path"]))
    require(selected_bundle["id"] == a["identity"], "ANCHOR_BUNDLE")
    positions = [r for r in after["database"]["delivered"] if r["request_id"] == request["id"]]
    ordered = sorted(positions, key=lambda row: cast("int", row["rank"]))
    expected_ids = [row["mal_id"] for row in selected_bundle["neighbors"]["1"]]
    require(
        len(ordered) == 5
        and [r["rank"] for r in ordered] == [1, 2, 3, 4, 5]
        and [r["mal_id"] for r in ordered] == expected_ids
        and all(
            r["tg_id"] == chat and r["chat_id"] == chat and integer(r["message_id"], 1)
            for r in ordered
        ),
        "ANCHOR_DELIVERIES",
    )
    require(
        not any(
            r["position_id"] in {p["id"] for p in positions}
            for r in after["database"]["recommendation_ratings"]
        ),
        "ANCHOR_SCORED",
    )
    return {
        "schema_version": 1,
        "project": after["deployment"]["project"],
        "database_volume": after["runtime"]["postgres"]["database_volume"],
        "actor": {
            "tg_id": cast("int", request["tg_id"]),
            "chat_id": chat,
            "anchor_request_id": cast("int", request["id"]),
        },
        "bundles": report["bundles"],
        "anchor": {
            "before_sha256": before_hash,
            "after_sha256": after_hash,
            "build_report_sha256": report_hash,
        },
    }


def new_rows(
    before: HistoricalDatabaseRecord, after: HistoricalDatabaseRecord, table: str
) -> list[dict[str, object]]:
    prior_tables = cast("dict[str, list[dict[str, object]]]", before)
    current_tables = cast("dict[str, list[dict[str, object]]]", after)
    old = index_rows(prior_tables[table], KEYS[table])
    return [
        row for row in current_tables[table] if tuple(row[key] for key in KEYS[table]) not in old
    ]


def require_bot_bundle(
    snapshot: HistoricalFullSnapshot, selected: HistoricalBundleReference
) -> ContainerRecord:
    bot = snapshot["runtime"]["bot"]
    require(
        bot is not None
        and bot["running"]
        and bot["bundle_id"] == selected["identity"]
        and bot["bundle_count"] == 17562
        and bot["user"] == "10001:10001"
        and bot["read_only_root"]
        and bot["artifact_mount"] is not None
        and not bot["artifact_mount"]["rw"]
        and Path(bot["artifact_mount"]["source"]).resolve() == Path(selected["path"]).resolve(),
        "RUNTIME_BUNDLE",
    )
    if bot is None:
        raise EvidenceError("RUNTIME_BUNDLE")
    return bot


def verify_transition(
    case: str,
    context: HistoricalContext,
    before: HistoricalFullSnapshot,
    after: HistoricalFullSnapshot,
    bundle: LegacyBundle,
    during: SnapshotRecord | None = None,
    phase: str | None = None,
) -> HistoricalTransitionResult:
    require(case in CASES and context.get("schema_version") == 1, "INVALID_CASE")
    require(before.get("kind") == after.get("kind") == "full", "FULL_SNAPSHOT_REQUIRED")
    require(
        before["deployment"]["project"] == after["deployment"]["project"] == context["project"]
        and before["deployment"]["compose_sha256"] == after["deployment"]["compose_sha256"]
        and before["snapshot_sql_sha256"] == after["snapshot_sql_sha256"] == sha(SQL.read_bytes())
        and before["source_commit"] == after["source_commit"],
        "DEPLOYMENT_CHANGED",
    )
    require(utc(before["captured_at"]) < utc(after["captured_at"]), "TIME_ORDER")
    old_volume = before["runtime"]["postgres"]["database_volume"]
    require(
        bool(old_volume)
        and old_volume
        == after["runtime"]["postgres"]["database_volume"]
        == context["database_volume"],
        "DATABASE_VOLUME_CHANGED",
    )
    actor = context["actor"]["tg_id"]
    require(integer(actor, 1) and context["actor"]["chat_id"] == actor, "INVALID_ACTOR")
    require(
        context["bundles"]["A"]["identity"] != context["bundles"]["B"]["identity"],
        "INVALID_CONTEXT",
    )
    expected = context["bundles"]["B" if case == "update" else "A"]["identity"]
    require(bundle["id"] == expected, "WRONG_BUNDLE")
    validate_database(before["database"], bundle)
    validate_database(after["database"], bundle)
    old, current = before["database"], after["database"]
    old_tables = cast("dict[str, list[dict[str, object]]]", old)
    current_tables = cast("dict[str, list[dict[str, object]]]", current)
    for table in IMMUTABLE:
        existing = index_rows(current_tables[table], KEYS[table])
        for key, historical_row in index_rows(old_tables[table], KEYS[table]).items():
            require(existing.get(key) == historical_row, "OLD_HISTORY_CHANGED")
    for table in LEGACY:
        require(
            collections.Counter(canonical(row) for row in old_tables[table])
            == collections.Counter(canonical(row) for row in current_tables[table]),
            "LEGACY_CHANGED",
        )
    previous_requests = index_rows(old["requests"], ("id",))
    pending = []
    for current_request in current["requests"]:
        earlier = previous_requests.get((current_request["id"],))
        if earlier is None:
            continue
        immutable = ("id", "tg_id", "action_key", "raw_query", "created_at")
        row_fields = cast("Mapping[str, object]", current_request)
        earlier_fields = cast("Mapping[str, object]", earlier)
        require(
            all(row_fields[field] == earlier_fields[field] for field in immutable),
            "OLD_REQUEST_CHANGED",
        )
        if current_request != earlier:
            require(
                earlier["seed_mal_id"] is None and current_request["seed_mal_id"] is not None,
                "UNRELATED_RESOLUTION",
            )
            pending.append(current_request)
    for new_user in new_rows(old, current, "users"):
        require(new_user["tg_id"] == actor, "UNRELATED_WRITE")
    old_users = index_rows(old["users"], ("tg_id",))
    for current_user in current["users"]:
        old_user = old_users.get((current_user["tg_id"],))
        if old_user is not None:
            require(
                current_user["created_at"] == old_user["created_at"]
                and (
                    current_user == old_user
                    or current_user["tg_id"] == actor
                    and all(
                        cast("Mapping[str, object]", current_user)[key]
                        == cast("Mapping[str, object]", old_user)[key]
                        for key in current_user
                        if key
                        not in (
                            "language_code",
                            "first_name",
                            "last_name",
                            "username",
                            "updated_at",
                        )
                    )
                ),
                "USER_HISTORY_CHANGED",
            )
    old_pointers = index_rows(old["anime_current"], ("tg_id", "mal_id"))
    new_events = new_rows(old, current, "anime_events")
    current_pointers = index_rows(current["anime_current"], ("tg_id", "mal_id"))
    for key, old_pointer in old_pointers.items():
        later = current_pointers.get(key)
        require(
            later is not None and later["created_at"] == old_pointer["created_at"],
            "CURRENT_POINTER",
        )
        if not any((e["tg_id"], e["mal_id"]) == key for e in new_events):
            require(later == old_pointer, "CURRENT_POINTER")
        else:
            if later is None:
                raise EvidenceError("CURRENT_POINTER")
            require(
                historical_greater(later["current_event_id"], old_pointer["current_event_id"])
                and utc(later["updated_at"]) >= utc(old_pointer["updated_at"]),
                "CURRENT_POINTER",
            )
    delta = cast(
        "HistoricalDelta",
        {table: new_rows(old, current, table) for table in KEYS if table != "migrations"},
    )
    require(
        all(
            row.get("tg_id") == actor
            for table in (
                "requests",
                "delivered",
                "anime_events",
                "recommendation_ratings",
                "feedback",
            )
            for row in delta[table]
        ),
        "UNRELATED_WRITE",
    )
    require(len(pending) <= 1, "UNRELATED_RESOLUTION")
    require(len(delta["requests"]) + len(pending) <= 1, "AMBIGUOUS_REQUEST")
    for changed_request in delta["requests"] + pending:
        parts = cast("str", changed_request["action_key"]).split(":")
        require(
            len(parts) == 3
            and parts[0] == "msg"
            and parts[1] == str(actor)
            and parts[2].isdigit()
            and int(parts[2]) > 0,
            "INVALID_ACTION_KEY",
        )
    for changed_feedback in delta["feedback"]:
        parts = historical_action_parts(changed_feedback["action_key"])
        require(
            len(parts) == 3
            and parts[0] == "msg"
            and parts[1] == str(actor)
            and parts[2].isdigit()
            and int(parts[2]) > 0,
            "INVALID_ACTION_KEY",
        )
    request = delta["requests"] + pending
    changed = {
        table: len(delta[table])
        for table in ("requests", "delivered", "anime_events", "recommendation_ratings", "feedback")
    }
    changed["resolved_pending"] = len(pending)
    selected_request = request[0] if request else None
    if case in (
        "recommendations",
        "empty",
        "anime-score-1",
        "anime-score-10",
        "update",
        "rollback",
    ):
        require(
            selected_request is not None
            and selected_request["bundle_id"] == expected
            and selected_request["tg_id"] == actor,
            "EXPECTED_REQUEST_MISSING",
        )
        if selected_request is None:
            raise EvidenceError("EXPECTED_REQUEST_MISSING")
        seed = 40089 if case == "empty" else 1
        query = "Pittanko!! Nekozakana" if case == "empty" else "Cowboy Bebop"
        require(
            selected_request["seed_mal_id"] == seed and selected_request["raw_query"] == query,
            "WRONG_SELECTION",
        )
        if case in ("recommendations", "empty", "update", "rollback"):
            ordered = sorted(delta["delivered"], key=lambda row: cast("int", row["rank"]))
            targets = [item["mal_id"] for item in bundle["neighbors"][str(seed)]]
            require(
                len(ordered) == len(targets)
                and [r["rank"] for r in ordered] == list(range(1, len(targets) + 1))
                and [r["mal_id"] for r in ordered] == targets
                and all(
                    r["request_id"] == selected_request["id"]
                    and r["chat_id"] == context["actor"]["chat_id"]
                    and integer(r["message_id"], 1)
                    for r in ordered
                ),
                "DELIVERY_ORDER",
            )
        if case in ("anime-score-1", "anime-score-10"):
            value = 1 if case.endswith("-1") else 10
            require(
                len(new_events) == 1
                and new_events[0]["request_id"] == selected_request["id"]
                and new_events[0]["score"] == value,
                "ANIME_SCORE_MISSING",
            )
    else:
        require(not request, "UNEXPECTED_REQUEST")
    if case == "recommendation-scores":
        anchor = context["actor"]["anchor_request_id"]
        positions = {r["rank"]: r["id"] for r in current["delivered"] if r["request_id"] == anchor}
        require(
            len(delta["recommendation_ratings"]) == 2
            and {(r["position_id"], r["score"]) for r in delta["recommendation_ratings"]}
            == {(positions.get(1), 0), (positions.get(2), 5)},
            "RECOMMENDATION_SCORE_MISSING",
        )
    if case in ("feedback-unicode", "feedback-retry"):
        name = "unicode" if case.endswith("unicode") else "retry"
        require(
            len(delta["feedback"]) == 1
            and delta["feedback"][0]["body"] == CANARIES[name]
            and delta["feedback"][0]["tg_id"] == actor,
            "FEEDBACK_BODY",
        )
    allowed = {
        "recommendations": (1, 5, 0, 0, 0),
        "empty": (1, 0, 0, 0, 0),
        "anime-score-1": (1, 0, 1, 0, 0),
        "anime-score-10": (1, 0, 1, 0, 0),
        "recommendation-scores": (0, 0, 0, 2, 0),
        "feedback-unicode": (0, 0, 0, 0, 1),
        "feedback-retry": (0, 0, 0, 0, 1),
        "update": (1, 5, 0, 0, 0),
        "rollback": (1, 5, 0, 0, 0),
    }.get(case, (0, 0, 0, 0, 0))
    actual = tuple(
        changed[table] + (changed["resolved_pending"] if table == "requests" else 0)
        for table in ("requests", "delivered", "anime_events", "recommendation_ratings", "feedback")
    )
    require(actual == allowed, "UNEXPECTED_EFFECTS")
    if case in NO_EFFECT:
        require(
            all(old_tables[table] == current_tables[table] for table in TABLES if table != "users"),
            "UNEXPECTED_EFFECTS",
        )
    if case in ("restart", "db-outage", "feedback-retry", "feedback-retry-repeat"):
        first = before["runtime"]["bot"]
        last = after["runtime"]["bot"]
        require(
            first is not None and last is not None and first["id"] == last["id"],
            "BOT_PROCESS_CHANGED",
        )
        if case == "restart":
            require(
                utc(first["started_at"]) < utc(last["started_at"])
                and last["bundle_id"] == expected,
                "RESTART_NOT_OBSERVED",
            )
        else:
            require(
                first["started_at"] == last["started_at"]
                and first["restart_count"] == last["restart_count"],
                "BOT_PROCESS_CHANGED",
            )
    if case == "db-outage":
        require(
            during is not None
            and during["kind"] == "runtime_only"
            and utc(before["captured_at"]) < utc(during["captured_at"]) < utc(after["captured_at"])
            and during["runtime"]["postgres"] is not None
            and not during["runtime"]["postgres"]["running"]
            and during["runtime"]["bot"] is not None
            and during["runtime"]["bot"]["id"] == before["runtime"]["bot"]["id"]
            and during["runtime"]["bot"]["started_at"] == before["runtime"]["bot"]["started_at"],
            "OUTAGE_NOT_OBSERVED",
        )
    selected = context["bundles"]["B" if case == "update" else "A"]
    bot = require_bot_bundle(after, selected)
    if case in ("update", "rollback"):
        previous = context["bundles"]["A" if case == "update" else "B"]
        old_bot = require_bot_bundle(before, previous)
        require(
            old_bot["id"] != bot["id"] and utc(old_bot["started_at"]) < utc(bot["started_at"]),
            "BOT_NOT_RECREATED",
        )
        prepare = after["runtime"]["prepare"]
        previous_prepare = before["runtime"]["prepare"]
        require(
            prepare is not None
            and prepare["id"] != bot["id"]
            and (previous_prepare is None or prepare["id"] != previous_prepare["id"])
            and prepare["status"] == "exited"
            and not prepare["running"]
            and prepare["exit_code"] == 0
            and prepare["bundle_id"] == selected["identity"]
            and prepare["bundle_count"] == 17562
            and prepare["user"] == "10001:10001"
            and prepare["read_only_root"]
            and prepare["artifact_mount"] is not None
            and not prepare["artifact_mount"]["rw"]
            and Path(prepare["artifact_mount"]["source"]).resolve()
            == Path(selected["path"]).resolve()
            and prepare["image_id"] == bot["image_id"]
            and utc(old_bot["started_at"]) < utc(prepare["started_at"]) <= utc(bot["started_at"])
            and after["runtime"]["postgres"]["running"],
            "PREPARE_GATE",
        )
    return {
        "schema_version": 1,
        "case": case,
        "phase": phase,
        "status": "PASS",
        "reason": "OK",
        "counts": changed,
        "bundle_id": bundle["id"],
    }


def summarize(
    context: HistoricalContext, results_dir: Path, observations: JsonObject
) -> HistoricalSummary:
    require(results_dir.is_dir() and not results_dir.is_symlink(), "RESULTS_DIR")
    context_hash = sha(canonical(context))
    result_keys = set()
    passed = 0
    bad = False
    bundle_cache = {}
    for path in results_dir.glob("*.json"):
        item = read_json(path)
        require(
            item.get("schema_version") == 1
            and item.get("case") in CASES
            and item.get("context_sha256") == context_hash,
            "INVALID_RESULT",
        )
        key = cast("str", item["case"])
        require(key not in result_keys, "DUPLICATE_RESULT")
        result_keys.add(key)
        hashes = item.get("input_sha256", {})
        require(
            isinstance(hashes, dict)
            and set(hashes) in ({"before", "after"}, {"before", "after", "during"}),
            "INVALID_RESULT",
        )
        hashes = cast("JsonObject", hashes)
        cached_inputs = {}
        for name, source_hash in hashes.items():
            require(
                isinstance(source_hash, str) and HEX.fullmatch(source_hash) is not None,
                "INVALID_RESULT",
            )
            cached = results_dir / "inputs" / f"{source_hash}.json"
            require(
                cached.is_file()
                and not cached.is_symlink()
                and sha(cached.read_bytes()) == source_hash,
                "INPUT_HASH_MISMATCH",
            )
            cached_inputs[name] = read_json(cached)
        bundle_name = "B" if key == "update" else "A"
        if item.get("status") == "PASS":
            if bundle_name not in bundle_cache:
                bundle_cache[bundle_name] = load_bundle(
                    Path(context["bundles"][bundle_name]["path"])
                )
            phase = item.get("phase")
            require(phase is None or isinstance(phase, str), "INVALID_RESULT")
            expected = verify_transition(
                key,
                context,
                cast("HistoricalFullSnapshot", cached_inputs["before"]),
                cast("HistoricalFullSnapshot", cached_inputs["after"]),
                bundle_cache[bundle_name],
                cast("SnapshotRecord | None", cached_inputs.get("during")),
                cast("str | None", phase),
            )
            require(
                all(
                    item.get(field) == expected[field]
                    for field in (
                        "schema_version",
                        "case",
                        "phase",
                        "status",
                        "reason",
                        "counts",
                        "bundle_id",
                    )
                ),
                "RESULT_MISMATCH",
            )
            passed += 1
        else:
            require(
                item.get("status") == "FAIL"
                and isinstance(item.get("reason"), str)
                and re.fullmatch(r"[A-Z_]+", cast("str", item["reason"])) is not None,
                "INVALID_RESULT",
            )
            bad = True
    require(
        observations.get("schema_version") == 1
        and observations.get("context_sha256") == context_hash,
        "INVALID_OBSERVATIONS",
    )
    seen = set()
    ui_bad = False
    ui_pending = False
    scenarios_raw = observations.get("scenarios", [])
    if not isinstance(scenarios_raw, (list, dict, str)):
        raise TypeError("scenarios is not iterable")
    scenarios = cast("Iterable[JsonValue]", scenarios_raw)
    checked_scenarios: list[ObservedScenario] = []
    for scenario_raw in scenarios:
        require(
            isinstance(scenario_raw, dict)
            and set(scenario_raw) == {"id", "status", "observed_at", "observation"}
            and scenario_raw["id"] in UI_IDS
            and scenario_raw["id"] not in seen
            and scenario_raw["status"] in ("PENDING", "PASS", "FAIL")
            and isinstance(scenario_raw["observation"], str)
            and len(scenario_raw["observation"]) <= 240,
            "INVALID_OBSERVATIONS",
        )
        scenario = cast("ObservedScenario", scenario_raw)
        checked_scenarios.append(scenario)
        seen.add(scenario["id"])
        if scenario["status"] == "PASS":
            observed_at = scenario["observed_at"]
            if not isinstance(observed_at, str):
                raise EvidenceError("INVALID_TIMESTAMP")
            utc(observed_at)
            require(scenario["observation"].strip() != "", "INVALID_OBSERVATIONS")
        ui_bad |= scenario["status"] == "FAIL"
        ui_pending |= scenario["status"] == "PENDING"
    require(seen == set(UI_IDS), "INVALID_OBSERVATIONS")
    status = (
        "FAIL"
        if bad or ui_bad
        else "PENDING"
        if result_keys != set(CASES) or ui_pending
        else "PASS"
    )
    return {
        "schema_version": 1,
        "live_acceptance": status,
        "machine": {"passed": passed, "required": len(CASES)},
        "human": {
            "passed": sum(scenario["status"] == "PASS" for scenario in checked_scenarios),
            "required": len(UI_IDS),
        },
        "context_sha256": context_hash,
    }


def parser() -> argparse.ArgumentParser:
    top = argparse.ArgumentParser(description=__doc__)
    commands = top.add_subparsers(dest="command", required=True)
    snap = commands.add_parser("snapshot")
    snap.add_argument("--project", required=True)
    snap.add_argument("--settings", type=Path, required=True)
    snap.add_argument("--output", type=Path, required=True)
    snap.add_argument("--runtime-only", action="store_true")
    bind = commands.add_parser("bind-actor")
    for arg in ("before", "after", "build-report", "output"):
        bind.add_argument("--" + arg, type=Path, required=True)
    canary = commands.add_parser("canary")
    canary.add_argument("--name", choices=tuple(CANARIES), required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--case", choices=CASES, required=True)
    for arg in ("context", "before", "after", "bundle", "output"):
        verify.add_argument("--" + arg, type=Path, required=True)
    verify.add_argument("--during", type=Path)
    verify.add_argument("--phase", choices=PHASES)
    summary = commands.add_parser("summarize")
    for arg in ("context", "results-dir", "observations", "output"):
        summary.add_argument("--" + arg, type=Path, required=True)
    return top


def main(argv: list[str] | None = None) -> int:
    requested = sys.argv[1:] if argv is None else argv
    if requested and requested[0] == "live-v2":
        module_path = Path(__file__).with_name("live_evidence_v2.py")
        spec = importlib.util.spec_from_file_location("arb034_live_v2", module_path)
        if spec is None or spec.loader is None:
            raise EvidenceError("INVALID_CASE")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not callable(getattr(module, "main", None)):
            raise EvidenceError("INVALID_CASE")
        return cast("LiveEvidenceMain", module).main(requested[1:])
    args = parser().parse_args(argv)
    if args.command == "canary":
        print(CANARIES[args.name], end="")
        return 0
    try:
        if args.command == "snapshot":
            write_new(args.output, snapshot(args.project, args.settings, args.runtime_only))
        elif args.command == "bind-actor":
            before = cast("HistoricalFullSnapshot", read_json(args.before))
            after = cast("HistoricalFullSnapshot", read_json(args.after))
            report = cast("HistoricalBuildReport", read_json(args.build_report))
            bound_context = context_from_anchor(
                before,
                after,
                report,
                sha(args.before.read_bytes()),
                sha(args.after.read_bytes()),
                sha(args.build_report.read_bytes()),
            )
            write_new(args.output, bound_context)
        elif args.command == "verify":
            context = cast("HistoricalContext", read_json(args.context))
            before = cast("HistoricalFullSnapshot", read_json(args.before))
            after = cast("HistoricalFullSnapshot", read_json(args.after))
            during = cast("SnapshotRecord", read_json(args.during)) if args.during else None
            inputs = {"before": args.before, "after": args.after}
            if args.during:
                inputs["during"] = args.during
            input_hashes = {key: sha(path.read_bytes()) for key, path in inputs.items()}
            inputs_dir = args.output.parent / "inputs"
            require(
                inputs_dir.is_dir()
                and not inputs_dir.is_symlink()
                and stat.S_IMODE(inputs_dir.stat().st_mode) & 0o077 == 0,
                "PRIVATE_DIRECTORY_REQUIRED",
            )
            for key, path in inputs.items():
                target = inputs_dir / f"{input_hashes[key]}.json"
                if target.exists():
                    require(
                        not target.is_symlink() and sha(target.read_bytes()) == input_hashes[key],
                        "INPUT_HASH_MISMATCH",
                    )
                else:
                    write_new(target, read_json(path))
                    require(sha(target.read_bytes()) == input_hashes[key], "INPUT_HASH_MISMATCH")
            try:
                bundle = load_bundle(args.bundle)
                result = verify_transition(
                    args.case, context, before, after, bundle, during, args.phase
                )
            except (EvidenceError, OSError, ValueError, KeyError, TypeError, IndexError) as exc:
                code = exc.code if isinstance(exc, EvidenceError) else "INVALID_EVIDENCE"
                result = {
                    "schema_version": 1,
                    "case": args.case,
                    "phase": args.phase,
                    "status": "FAIL",
                    "reason": code,
                    "counts": {},
                    "bundle_id": None,
                }
            result["context_sha256"] = sha(args.context.read_bytes())
            result["input_sha256"] = input_hashes
            write_new(args.output, result)
            if result["status"] == "FAIL":
                print(f"FAIL {result['reason']}", file=sys.stderr)
                return 1
        elif args.command == "summarize":
            summary_value = summarize(
                cast("HistoricalContext", read_json(args.context)),
                args.results_dir,
                read_json(args.observations),
            )
            write_new(args.output, summary_value)
            print(summary_value["live_acceptance"])
            return 0 if summary_value["live_acceptance"] == "PASS" else 1
        print("PASS")
        return 0
    except (
        EvidenceError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        IndexError,
        subprocess.SubprocessError,
    ) as exc:
        code = exc.code if isinstance(exc, EvidenceError) else "INVALID_EVIDENCE"
        print(f"FAIL {code}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
