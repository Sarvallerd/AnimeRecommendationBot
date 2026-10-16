"""Small synthetic ARB-034 verifier checks; no full inputs or Docker."""

import copy
import datetime as dt
import importlib.util
import os
import tempfile
import unittest
from collections.abc import Mapping
from pathlib import Path
from types import EllipsisType, SimpleNamespace
from typing import Callable, NotRequired, TypedDict, cast
from unittest import mock

SCRIPT = Path(__file__).with_name("live_verifier_v2.py")
spec = importlib.util.spec_from_file_location("arb034_test_verifier", SCRIPT)
if spec is None or spec.loader is None:
    raise ImportError(f"Cannot load live verifier from {SCRIPT}")
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)
A = "sha256:" + "a" * 64
B = "sha256:" + "b" * 64
STAMP = dt.datetime(2026, 10, 12, tzinfo=dt.timezone.utc)


class ArtifactMountFixture(TypedDict):
    source: str
    rw: bool


class BotFixture(TypedDict):
    id: str
    image_id: str
    project: str
    service: str
    user: str
    status: str
    running: bool
    exit_code: int
    started_at: str
    restart_count: int
    read_only_root: bool
    artifact_mount: ArtifactMountFixture
    database_volume: None
    bundle_id: str
    bundle_count: int


type DatabaseFixture = dict[str, list[dict[str, int | str]]]


class DeploymentFixture(TypedDict):
    project: str
    compose_sha256: str
    settings_sha256: str


class RuntimeFixture(TypedDict):
    bot: BotFixture | None
    prepare: BotFixture | None
    postgres: dict[str, str | bool]


class StateFixture(TypedDict):
    schema_version: int
    kind: str
    captured_at: str
    source_commit: str
    snapshot_sql_sha256: str
    deployment: DeploymentFixture
    runtime: RuntimeFixture
    database: DatabaseFixture


class PollersFixture(TypedDict):
    matching: int
    scope_complete: bool
    scope: str
    matching_container_ids_sha256: list[str]


class SettingsFixture(TypedDict):
    sha256: str
    uid: int
    gid: int
    mode: int


class GuardsFixture(TypedDict):
    stable: bool
    daemon_id: str
    settings: SettingsFixture
    credential_sha256: dict[str, str]
    postgres_health: str
    postgres_identity: dict[str, str | None]
    pollers: PollersFixture


class ShotFixture(TypedDict):
    schema_version: int
    kind: str
    state: StateFixture
    guards: GuardsFixture


class PlanSourceFixture(TypedDict):
    commit: str
    compose_sha256: str
    snapshot_sql_sha256: str
    verifier_sha256: NotRequired[str]
    helper_sha256: NotRequired[str]
    dispatch_sha256: NotRequired[str]


class PlanSettingsFixture(TypedDict):
    uid: int
    gid: int
    mode: int
    credential_sha256: dict[str, str]
    original_copy: NotRequired[str]
    target_copy: NotRequired[str]


class PlanExpectationsFixture(TypedDict):
    smoke_seed: dict[str, str | int | list[int]]
    empty_seed: dict[str, str | int] | None


class PlanBundleFixture(TypedDict):
    identity: str
    path: str
    count: int
    image_id: str
    expectations: PlanExpectationsFixture
    histogram: dict[str, int]


class AliasProbeFixture(TypedDict):
    query: str
    mal_id: int


class DescriptionProbeFixture(TypedDict):
    seed_mal_id: int
    query: str
    neighbor_mal_id: int


class UiProbeFixture(TypedDict):
    schema_version: int
    kind: str
    alias_search: AliasProbeFixture
    description_edit: DescriptionProbeFixture


class OfflineFixture(TypedDict):
    report_sha256: str
    evidence_dir: NotRequired[str]
    project: NotRequired[str]
    database_volume: NotRequired[str]


class ProfileFixture(TypedDict):
    id: str
    sha256: str
    lock_sha256: str
    release_id: NotRequired[str]


class PlanFixture(TypedDict):
    schema_version: int
    kind: str
    service: dict[str, str]
    source: PlanSourceFixture
    settings: PlanSettingsFixture
    expected_settings_sha256: dict[str, str]
    bundles: dict[str, PlanBundleFixture]
    ui_probes: UiProbeFixture
    offline: OfflineFixture
    profile: ProfileFixture
    baseline: NotRequired[ShotFixture]
    baseline_sha256: NotRequired[str]


def time(minute: int | float) -> str:
    return (STAMP + dt.timedelta(minutes=minute)).isoformat()


def bot(identity: str, role: str, identifier: str, started: int | float) -> BotFixture:
    return {
        "id": identifier,
        "image_id": "image-" + role,
        "project": "real-project",
        "service": "bot",
        "user": "10001:10001",
        "status": "running",
        "running": True,
        "exit_code": 0,
        "started_at": time(started),
        "restart_count": 0,
        "read_only_root": True,
        "artifact_mount": {"source": "/tmp/" + role, "rw": False},
        "database_volume": None,
        "bundle_id": identity,
        "bundle_count": 2 if role == "original" else 3,
    }


def database() -> DatabaseFixture:
    return {
        "migrations": [{"version": 1, "applied_at": time(0)}],
        "users": [{"tg_id": 42, "created_at": time(0), "updated_at": time(0)}],
        "requests": [
            {
                "id": 1,
                "tg_id": 42,
                "action_key": "msg:42:1",
                "raw_query": "Old",
                "seed_mal_id": 1,
                "bundle_id": A,
                "created_at": time(0),
                "resolved_at": time(0),
            }
        ],
        "delivered": [
            {
                "id": 1,
                "request_id": 1,
                "tg_id": 42,
                "rank": 1,
                "mal_id": 4,
                "chat_id": 42,
                "message_id": 10,
                "delivered_at": time(0),
            }
        ],
        "anime_events": [],
        "anime_current": [],
        "recommendation_ratings": [],
        "feedback": [],
        "legacy_users": [{"old": 1}],
        "legacy_requests": [],
        "legacy_feedback": [],
    }


def plan() -> PlanFixture:
    return {
        "schema_version": 2,
        "kind": "arb-live-plan",
        "service": {
            "project": "real-project",
            "daemon_id": "daemon",
            "postgres_id": "pg",
            "postgres_image_id": "pg-image",
            "database_volume": "real-volume",
        },
        "source": {"commit": "f" * 40, "compose_sha256": "c" * 64, "snapshot_sql_sha256": "d" * 64},
        "settings": {
            "uid": 1000,
            "gid": 1000,
            "mode": 0o600,
            "credential_sha256": {"ARB_BOT_ENV": "e" * 64, "ARB_POSTGRES_ENV": "f" * 64},
        },
        "expected_settings_sha256": {"original": "1" * 64, "target": "2" * 64},
        "bundles": {
            "original": {
                "identity": A,
                "path": "/tmp/original",
                "count": 2,
                "image_id": "image-original",
                "expectations": {
                    "smoke_seed": {"mal_id": 1, "query": "Old", "neighbor_ids": [4]},
                    "empty_seed": None,
                },
                "histogram": {"0": 0},
            },
            "target": {
                "identity": B,
                "path": "/tmp/target",
                "count": 3,
                "image_id": "image-target",
                "expectations": {
                    "smoke_seed": {"mal_id": 8, "query": "New", "neighbor_ids": [9, 10]},
                    "empty_seed": None,
                },
                "histogram": {"0": 0},
            },
        },
        "ui_probes": {
            "schema_version": 1,
            "kind": "arb-live-ui-probes",
            "alias_search": {"query": "Alias", "mal_id": 8},
            "description_edit": {"seed_mal_id": 8, "query": "New", "neighbor_mal_id": 9},
        },
        "offline": {"report_sha256": "3" * 64},
        "profile": {"id": "new", "sha256": "4" * 64, "lock_sha256": "5" * 64},
    }


def shot(
    minute: int | float,
    role: str = "original",
    bot_item: BotFixture | None | EllipsisType = None,
    prepare: BotFixture | None = None,
    settings: str | None = None,
    db: DatabaseFixture | None = None,
    matching: int | None = None,
) -> ShotFixture:
    settings = settings or ("1" * 64 if role == "original" else "2" * 64)
    if bot_item is ...:
        bot_item = bot(
            A if role == "original" else B,
            role,
            "old-bot" if role == "original" else "new-bot",
            0 if role == "original" else 2.5,
        )
    if matching is None:
        matching = 1 if bot_item is not None else 0
    return {
        "schema_version": 2,
        "kind": "arb-live-capture",
        "state": {
            "schema_version": 1,
            "kind": "full",
            "captured_at": time(minute),
            "source_commit": "f" * 40,
            "snapshot_sql_sha256": "d" * 64,
            "deployment": {
                "project": "real-project",
                "compose_sha256": "c" * 64,
                "settings_sha256": settings,
            },
            "runtime": {
                "bot": bot_item,
                "prepare": prepare,
                "postgres": {
                    "id": "pg",
                    "image_id": "pg-image",
                    "database_volume": "real-volume",
                    "running": True,
                },
            },
            "database": copy.deepcopy(db or database()),
        },
        "guards": {
            "stable": True,
            "daemon_id": "daemon",
            "settings": {"sha256": settings, "uid": 1000, "gid": 1000, "mode": 0o600},
            "credential_sha256": {"ARB_BOT_ENV": "e" * 64, "ARB_POSTGRES_ENV": "f" * 64},
            "postgres_health": "healthy",
            "postgres_identity": {
                "id": "pg",
                "image_id": "pg-image",
                "database_volume": "real-volume",
                "started_at": None,
            },
            "pollers": {
                "matching": matching,
                "scope_complete": True,
                "scope": "docker-daemon-and-known-local-processes",
                "matching_container_ids_sha256": (
                    [v.sha(bot_item["id"].encode())] if matching and bot_item is not None else []
                ),
            },
        },
    }


def sequence() -> tuple[ShotFixture, ShotFixture, ShotFixture, ShotFixture]:
    before = shot(0, bot_item=bot(A, "original", "old-bot", -1))
    stopped = shot(1, bot_item=None, matching=0)
    prepared = shot(2, role="target", bot_item=None, matching=0)
    prep = bot(B, "target", "prepare-new", 1.5)
    prep["service"] = "prepare"
    prep["status"] = "exited"
    prep["running"] = False
    prepared["state"]["runtime"]["prepare"] = prep
    started = shot(
        3, role="target", bot_item=bot(B, "target", "new-bot", 2.5), prepare=copy.deepcopy(prep)
    )
    return before, stopped, prepared, started


class LiveV2Tests(unittest.TestCase):
    def fail_code(self, code: str, fn: Callable[[], object]) -> None:
        with self.assertRaises(v.Error) as caught:
            fn()
        self.assertEqual(caught.exception.code, code)

    def test_operation_recomputes_distinct_bundle_transition(self) -> None:
        values = sequence()
        result = v.verify_operation(plan(), "initial-update", *values)
        self.assertEqual(result["counts"], [1, 0, 0, 1])
        self.assertEqual(result["to"], "target")
        changed = copy.deepcopy(values)
        changed_prepare = changed[2]["state"]["runtime"]["prepare"]
        assert changed_prepare is not None
        changed_prepare["exit_code"] = 1
        self.fail_code(
            "PREPARE_GATE", lambda: v.verify_operation(plan(), "initial-update", *changed)
        )
        changed = copy.deepcopy(values)
        changed[1]["guards"]["pollers"]["matching"] = 2
        self.fail_code(
            "SECOND_POLLER", lambda: v.verify_operation(plan(), "initial-update", *changed)
        )
        changed = copy.deepcopy(values)
        changed[1]["guards"]["pollers"]["scope_complete"] = False
        self.fail_code(
            "POLLER_SCOPE_PENDING", lambda: v.verify_operation(plan(), "initial-update", *changed)
        )
        changed = copy.deepcopy(values)
        changed[2]["guards"]["postgres_identity"]["id"] = "replacement"
        self.fail_code(
            "DATABASE_VOLUME_CHANGED",
            lambda: v.verify_operation(plan(), "initial-update", *changed),
        )
        changed = copy.deepcopy(values)
        changed[3]["state"]["database"]["migrations"][0]["applied_at"] = time(1)
        self.fail_code(
            "DATABASE_HISTORY_CHANGED",
            lambda: v.verify_operation(plan(), "initial-update", *changed),
        )
        changed = copy.deepcopy(values)
        changed_bot = changed[3]["state"]["runtime"]["bot"]
        assert changed_bot is not None
        changed_bot["started_at"] = time(1.5)
        self.fail_code(
            "BOT_NOT_RECREATED", lambda: v.verify_operation(plan(), "initial-update", *changed)
        )

    def test_operation_rejects_consistently_malformed_database_rows(self) -> None:
        mutations: tuple[tuple[str, str, int | str | None], ...] = (
            ("users", "created_at", None),
            ("requests", "created_at", None),
            ("users", "created_at", 7),
            ("requests", "created_at", 7),
            ("delivered", "id", True),
            ("delivered", "id", "position-id"),
        )
        for table, field, replacement in mutations:
            with self.subTest(table=table, field=field, replacement=replacement):
                captures = copy.deepcopy(sequence())
                for capture in captures:
                    row = capture["state"]["database"][table][0]
                    if replacement is None:
                        row.pop(field)
                    else:
                        row[field] = replacement
                self.fail_code(
                    "INVALID_DATABASE_SNAPSHOT",
                    lambda: v.verify_operation(plan(), "initial-update", *captures),
                )

    def test_poller_inventory_identifies_the_selected_bot(self) -> None:
        value = plan()
        running = shot(4, "target", bot_item=bot(B, "target", "accepted-bot", 2.5))
        self.assertIs(v._valid_capture(running, value, "target"), running["state"])
        for hashes in ([v.sha(b"different-container")], []):
            changed = copy.deepcopy(running)
            changed["guards"]["pollers"]["matching_container_ids_sha256"] = hashes
            self.fail_code("POLLER_IDENTITY", lambda: v._valid_capture(changed, value, "target"))
        zero = shot(1, bot_item=None, matching=0)
        self.assertIs(v._valid_capture(zero, value, "original"), zero["state"])
        zero["guards"]["pollers"]["matching_container_ids_sha256"] = [v.sha(b"unrelated-container")]
        self.fail_code("POLLER_IDENTITY", lambda: v._valid_capture(zero, value, "original"))
        values = list(sequence())
        values[3]["guards"]["pollers"]["matching_container_ids_sha256"] = [
            v.sha(b"different-container")
        ]
        self.fail_code(
            "POLLER_IDENTITY", lambda: v.verify_operation(value, "initial-update", *values)
        )
        final = shot(4, "target", bot_item=bot(B, "target", "accepted-bot", 2.5))
        final["guards"]["pollers"]["matching_container_ids_sha256"] = []
        self.fail_code("POLLER_IDENTITY", lambda: v.summarize_evidence(value, {}, {}, final))

    def test_real_flow_actor_order_and_private_chat(self) -> None:
        before, stopped, prepared, started = sequence()
        operation = v.verify_operation(plan(), "initial-update", before, stopped, prepared, started)
        after = copy.deepcopy(started)
        after["state"]["captured_at"] = time(4)
        after["state"]["database"]["requests"].append(
            {
                "id": 2,
                "tg_id": 42,
                "action_key": "msg:42:2",
                "raw_query": "New",
                "seed_mal_id": 8,
                "bundle_id": B,
                "created_at": time(4),
                "resolved_at": time(4),
            }
        )
        for rank, target in enumerate((9, 10), 1):
            after["state"]["database"]["delivered"].append(
                {
                    "id": rank + 1,
                    "request_id": 2,
                    "tg_id": 42,
                    "rank": rank,
                    "mal_id": target,
                    "chat_id": 42,
                    "message_id": rank + 10,
                    "delivered_at": time(4),
                }
            )
        bundles = {
            "original": {"id": A, "neighbors": {"1": [{"mal_id": 4}]}},
            "target": {"id": B, "neighbors": {"8": [{"mal_id": 9}, {"mal_id": 10}]}},
        }
        with mock.patch.object(
            v, "_bundle", side_effect=lambda _plan, role: (bundles[role], None, None)
        ):
            context = v.bind_context(plan(), operation, started, after)
            self.assertEqual(context["actor"]["target_anchor_request_id"], 2)
            stopped_anchor = copy.deepcopy(started)
            stopped_bot = stopped_anchor["state"]["runtime"]["bot"]
            assert stopped_bot is not None
            stopped_bot["running"] = False
            self.fail_code(
                "BOT_NOT_RUNNING", lambda: v.bind_context(plan(), operation, stopped_anchor, after)
            )
            unpolled_anchor = copy.deepcopy(after)
            unpolled_anchor["guards"]["pollers"]["matching"] = 0
            unpolled_anchor["guards"]["pollers"]["matching_container_ids_sha256"] = []
            self.fail_code(
                "POLLER_COUNT", lambda: v.bind_context(plan(), operation, started, unpolled_anchor)
            )
            mismatched_anchor = copy.deepcopy(after)
            mismatched_anchor["guards"]["pollers"]["matching_container_ids_sha256"] = [
                v.sha(b"different-container")
            ]
            self.fail_code(
                "POLLER_IDENTITY",
                lambda: v.bind_context(plan(), operation, started, mismatched_anchor),
            )
            result = v.verify_case(
                plan(), context, "initial-update", before, after, operation=operation
            )
            self.assertEqual(result["counts"]["delivered"], 2)
            forged_operation = copy.deepcopy(operation)
            forged_operation["started_capture_time"] = None
            self.fail_code(
                "OPERATION_REQUIRED",
                lambda: v.verify_case(
                    plan(), context, "initial-update", before, after, operation=forged_operation
                ),
            )
            wrong = copy.deepcopy(after)
            wrong["state"]["database"]["delivered"][-1]["mal_id"] = 9
            self.fail_code(
                "DELIVERY_ORDER", lambda: v.bind_context(plan(), operation, started, wrong)
            )
            wrong = copy.deepcopy(after)
            wrong["state"]["database"]["requests"][-1]["action_key"] = "msg:55:2"
            self.fail_code(
                "INVALID_ACTION_KEY", lambda: v.bind_context(plan(), operation, started, wrong)
            )
            wrong = copy.deepcopy(after)
            wrong["state"]["database"]["requests"].append(
                copy.deepcopy(wrong["state"]["database"]["requests"][-1])
            )
            wrong["state"]["database"]["requests"][-1]["id"] = 3
            wrong["state"]["database"]["requests"][-1]["action_key"] = "msg:42:3"
            self.fail_code(
                "FLOW_AMBIGUOUS", lambda: v.bind_context(plan(), operation, started, wrong)
            )
            empty_before = copy.deepcopy(after)
            empty_after = copy.deepcopy(after)
            empty_after["state"]["captured_at"] = time(5)
            self.assertEqual(
                v.verify_case(plan(), context, "empty", empty_before, empty_after)["status"],
                "NOT_APPLICABLE",
            )
            description = v.verify_case(
                plan(), context, "description-edit", empty_before, empty_after
            )
            self.assertEqual(description["counts"]["requests"], 0)
            alias_after = copy.deepcopy(after)
            alias_after["state"]["captured_at"] = time(5)
            alias_after["state"]["database"]["requests"].append(
                {
                    "id": 3,
                    "tg_id": 42,
                    "action_key": "msg:42:3",
                    "raw_query": "Alias",
                    "seed_mal_id": 8,
                    "bundle_id": B,
                    "created_at": time(5),
                    "resolved_at": time(5),
                }
            )
            for rank, target in enumerate((9, 10), 1):
                alias_after["state"]["database"]["delivered"].append(
                    {
                        "id": rank + 3,
                        "request_id": 3,
                        "tg_id": 42,
                        "rank": rank,
                        "mal_id": target,
                        "chat_id": 42,
                        "message_id": rank + 20,
                        "delivered_at": time(5),
                    }
                )
            self.assertEqual(
                v.verify_case(plan(), context, "alias-search", after, alias_after)["status"], "PASS"
            )
            wrong_alias = copy.deepcopy(alias_after)
            wrong_alias["state"]["database"]["requests"][-1]["raw_query"] = "New"
            self.fail_code(
                "WRONG_SELECTION",
                lambda: v.verify_case(plan(), context, "alias-search", after, wrong_alias),
            )
            false_plan = plan()
            false_plan["bundles"]["target"]["histogram"]["0"] = 1
            false_context = copy.deepcopy(context)
            false_context["plan_sha256"] = v.digest(false_plan)
            self.fail_code(
                "FALSE_EMPTY_NA",
                lambda: v.verify_case(
                    false_plan, false_context, "empty", empty_before, empty_after
                ),
            )

    def test_case_windows_require_running_bot_and_one_poller(self) -> None:
        value = plan()
        context = {
            "schema_version": 2,
            "kind": "arb-live-context",
            "plan_sha256": v.digest(value),
            "actor": {"tg_id": 42, "chat_id": 42},
        }
        live_bot = bot(B, "target", "target-bot", 2.5)
        before = shot(4, "target", bot_item=live_bot)
        after = shot(6, "target", bot_item=live_bot)
        during = copy.deepcopy(before)
        during["state"]["kind"] = "runtime_only"
        during["state"]["captured_at"] = time(5)
        during["state"]["database"] = None  # type: ignore[typeddict-item]  # runtime-only negative fixture
        during["state"]["runtime"]["postgres"]["running"] = False
        during["guards"]["postgres_health"] = "unhealthy"
        with mock.patch.object(v, "_bundle", return_value=({"id": B, "neighbors": {}}, None, None)):
            self.assertEqual(
                v.verify_case(value, context, "navigation-search", before, after)["status"], "PASS"
            )
            self.assertEqual(
                v.verify_case(value, context, "db-outage", before, after, during=during)["status"],
                "PASS",
            )
            for endpoint in ("before", "after"):
                for field, code in (
                    ("running", "BOT_NOT_RUNNING"),
                    ("matching", "POLLER_COUNT"),
                ):
                    changed = {"before": copy.deepcopy(before), "after": copy.deepcopy(after)}
                    if field == "running":
                        changed_bot = changed[endpoint]["state"]["runtime"]["bot"]
                        assert changed_bot is not None
                        changed_bot["running"] = False
                    else:
                        changed[endpoint]["guards"]["pollers"]["matching"] = 0
                        changed[endpoint]["guards"]["pollers"]["matching_container_ids_sha256"] = []
                    self.fail_code(
                        code, lambda: v.verify_case(value, context, "navigation-search", **changed)
                    )
                wrong_identity = {"before": copy.deepcopy(before), "after": copy.deepcopy(after)}
                wrong_identity[endpoint]["guards"]["pollers"]["matching_container_ids_sha256"] = [
                    v.sha(b"other-container")
                ]
                self.fail_code(
                    "POLLER_IDENTITY",
                    lambda: v.verify_case(value, context, "navigation-search", **wrong_identity),
                )
            for field, code in (
                ("running", "BOT_NOT_RUNNING"),
                ("matching", "POLLER_COUNT"),
            ):
                during_changed = copy.deepcopy(during)
                if field == "running":
                    changed_bot = during_changed["state"]["runtime"]["bot"]
                    assert changed_bot is not None
                    changed_bot["running"] = False
                else:
                    during_changed["guards"]["pollers"]["matching"] = 0
                    during_changed["guards"]["pollers"]["matching_container_ids_sha256"] = []
                self.fail_code(
                    code,
                    lambda: v.verify_case(
                        value, context, "db-outage", before, after, during=during_changed
                    ),
                )
            local_only = copy.deepcopy(during)
            local_only["guards"]["pollers"]["matching_container_ids_sha256"] = []
            self.fail_code(
                "POLLER_IDENTITY",
                lambda: v.verify_case(
                    value, context, "db-outage", before, after, during=local_only
                ),
            )

    def test_description_coordinates_are_actor_owned_private_delivery(self) -> None:
        value = plan()
        context = {
            "schema_version": 2,
            "kind": "arb-live-context",
            "plan_sha256": v.digest(value),
            "actor": {"tg_id": 42, "chat_id": 42},
        }
        db = database()
        db["users"].append({"tg_id": 55, "created_at": time(0), "updated_at": time(0)})
        db["requests"].append(
            {
                "id": 2,
                "tg_id": 55,
                "action_key": "msg:55:2",
                "raw_query": "New",
                "seed_mal_id": 8,
                "bundle_id": B,
                "created_at": time(1),
                "resolved_at": time(1),
            }
        )
        db["delivered"].append(
            {
                "id": 2,
                "request_id": 2,
                "tg_id": 55,
                "rank": 1,
                "mal_id": 9,
                "chat_id": 55,
                "message_id": 11,
                "delivered_at": time(1),
            }
        )
        live_bot = bot(B, "target", "target-bot", 2.5)

        def window(rows: DatabaseFixture) -> tuple[ShotFixture, ShotFixture]:
            return (
                shot(4, "target", bot_item=live_bot, db=rows),
                shot(5, "target", bot_item=live_bot, db=rows),
            )

        with mock.patch.object(
            v, "_bundle", return_value=({"id": B, "neighbors": {"8": [{"mal_id": 9}]}}, None, None)
        ):
            before, after = window(db)
            self.fail_code(
                "DESCRIPTION_COORDINATES",
                lambda: v.verify_case(value, context, "description-edit", before, after),
            )
            own = copy.deepcopy(db)
            own["requests"][-1].update(tg_id=42, action_key="msg:42:2")
            own["delivered"][-1].update(tg_id=42, chat_id=42)
            before, after = window(own)
            self.assertEqual(
                v.verify_case(value, context, "description-edit", before, after)["status"], "PASS"
            )
            public_chat = copy.deepcopy(own)
            public_chat["delivered"][-1]["chat_id"] = 55
            before, after = window(public_chat)
            self.fail_code(
                "DESCRIPTION_COORDINATES",
                lambda: v.verify_case(value, context, "description-edit", before, after),
            )
            foreign_delivery = copy.deepcopy(db)
            foreign_delivery["requests"].append(
                {
                    "id": 3,
                    "tg_id": 42,
                    "action_key": "msg:42:3",
                    "raw_query": "New",
                    "seed_mal_id": 8,
                    "bundle_id": B,
                    "created_at": time(1),
                    "resolved_at": time(1),
                }
            )
            before, after = window(foreign_delivery)
            self.fail_code(
                "DESCRIPTION_COORDINATES",
                lambda: v.verify_case(value, context, "description-edit", before, after),
            )

    def test_history_mutation_and_missing_ui_never_pass(self) -> None:
        old = database()
        later = copy.deepcopy(old)
        later["legacy_users"][0]["old"] = 2
        self.fail_code("LEGACY_CHANGED", lambda: v._preserved(old, later, 42))
        later = copy.deepcopy(old)
        later["requests"][0]["created_at"] = time(1)
        self.fail_code("OLD_REQUEST_CHANGED", lambda: v._preserved(old, later, 42))
        self.assertEqual(len(v.CASES), 21)
        final = shot(4, "target", bot_item=bot(B, "target", "new-bot", 2.5))
        value = v.summarize_evidence(plan(), {}, {}, final)
        self.assertEqual(value["live_acceptance"], "PENDING")
        self.assertEqual(value["production_history_preservation"], "PENDING")

    def test_plan_binds_reference_a_and_separate_historical_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            report_path = parent / "preparation-report.json"
            report_path.write_bytes(b"{}\n")
            original = parent / "original.env"
            target = parent / "target.env"
            prefix = (
                b"ARB_DATA_DIR=/tmp/data\nARB_UID=1000\nARB_GID=1000\n"
                b"ARB_LOG=info\nARB_BOT_ENV=/tmp/bot.env\n"
                b"ARB_POSTGRES_ENV=/tmp/postgres.env\n"
            )
            original.write_bytes(prefix + b"ARB_BUNDLE_DIR=/tmp/original\n")
            target.write_bytes(prefix + b"ARB_BUNDLE_DIR=/tmp/target\n")
            original.chmod(0o600)
            target.chmod(0o600)
            value = plan()
            value["profile"] = {
                "id": v.p.NEW_RELEASE,
                "release_id": v.p.NEW_RELEASE,
                "sha256": v.p.TRUSTED_NEW_PROFILE_SHA,
                "lock_sha256": v.p.TRUSTED_LOCK_SHA,
            }
            value["offline"]["evidence_dir"] = str(parent)
            value["offline"]["project"] = "offline-project"
            value["offline"]["database_volume"] = "offline-volume"
            value["offline"]["report_sha256"] = v.sha(report_path.read_bytes())
            value["source"]["verifier_sha256"] = v.sha(SCRIPT.read_bytes())
            value["source"]["helper_sha256"] = v.sha(
                SCRIPT.with_name("live_evidence_v2.py").read_bytes()
            )
            value["source"]["dispatch_sha256"] = v.sha(
                SCRIPT.with_name("acceptance-evidence.py").read_bytes()
            )
            value["source"]["compose_sha256"] = v.sha((v.e.ROOT / "compose.yaml").read_bytes())
            value["source"]["snapshot_sql_sha256"] = v.sha(v.e.SQL.read_bytes())
            value["settings"]["uid"] = os.getuid()
            value["settings"]["gid"] = os.getgid()
            value["settings"]["mode"] = 0o600
            value["settings"]["original_copy"] = str(original)
            value["settings"]["target_copy"] = str(target)
            value["expected_settings_sha256"] = {
                "original": v.sha(original.read_bytes()),
                "target": v.sha(target.read_bytes()),
            }
            base = shot(0, bot_item=bot(A, "original", "old-bot", -1))
            base["guards"]["settings"]["uid"] = os.getuid()
            base["guards"]["settings"]["gid"] = os.getgid()
            base["guards"]["settings"]["sha256"] = value["expected_settings_sha256"]["original"]
            base["state"]["deployment"]["settings_sha256"] = value["expected_settings_sha256"][
                "original"
            ]
            base["state"]["snapshot_sql_sha256"] = value["source"]["snapshot_sql_sha256"]
            base["state"]["deployment"]["compose_sha256"] = value["source"]["compose_sha256"]
            value["baseline"] = base
            value["baseline_sha256"] = v.digest(base)
            old_summary = SimpleNamespace(
                identity=A, hashes={"catalog.json": "1"}, count=2, histogram={"0": 0}
            )
            target_summary = SimpleNamespace(
                identity=B,
                hashes={"catalog.json": "2", "neighbors.json": "3"},
                count=3,
                histogram={"0": 0},
            )
            old_profile = SimpleNamespace(release_id=v.p.LEGACY_RELEASE)
            target_profile = SimpleNamespace(
                release_id=v.p.NEW_RELEASE,
                reference={"bundle_identity": B, "catalog_sha256": "2", "neighbors_sha256": "3"},
            )
            report = {
                "source": {"commit": "a" * 40},
                "runtime": {
                    "project": "offline-project",
                    "database_volume": "offline-volume",
                    "bot_image_id": "image-target",
                },
                "profile": {
                    "sha256": v.p.TRUSTED_NEW_PROFILE_SHA,
                    "release_lock_sha256": v.p.TRUSTED_LOCK_SHA,
                },
                "original": {"bundle_identity": A, "bundle_file_sha256": old_summary.hashes},
                "builds": {"A": {"identity": B, "file_sha256": target_summary.hashes}},
            }

            def synthetic_bundle(
                _plan: PlanFixture, role: str
            ) -> tuple[dict[str, object], SimpleNamespace, SimpleNamespace]:
                summary = old_summary if role == "original" else target_summary
                profile = old_profile if role == "original" else target_profile
                return {"id": summary.identity, "neighbors": {}}, summary, profile

            with (
                mock.patch.object(v, "_bundle", side_effect=synthetic_bundle),
                mock.patch.object(v, "_check_probes"),
                mock.patch.object(v.r, "verify_record", return_value=report),
            ):
                self.assertIs(v.validate_plan(value), value)
                self.assertNotEqual(report["source"]["commit"], value["source"]["commit"])
                alternate = copy.deepcopy(value)
                alternate["baseline"]["guards"]["pollers"]["matching_container_ids_sha256"] = [
                    v.sha(b"other-container")
                ]
                alternate["baseline_sha256"] = v.digest(alternate["baseline"])
                self.fail_code("POLLER_IDENTITY", lambda: v.validate_plan(alternate))
                alternate = copy.deepcopy(value)
                alternate["bundles"]["target"]["identity"] = "sha256:" + "c" * 64
                self.fail_code("REFERENCE_MISMATCH", lambda: v.validate_plan(alternate))
                alternate = copy.deepcopy(value)
                alternate["profile"]["lock_sha256"] = "0" * 64
                self.fail_code("PROFILE_MISMATCH", lambda: v.validate_plan(alternate))
                alternate = copy.deepcopy(value)
                alternate["offline"]["report_sha256"] = "0" * 64
                self.fail_code("OFFLINE_CHANGED", lambda: v.validate_plan(alternate))

    def test_three_window_history_and_final_runtime(self) -> None:
        initial = sequence()
        first_target = bot(B, "target", "new-bot", 2.5)
        rollback_before = shot(5, "target", bot_item=first_target)
        rollback_stopped = shot(6, "target", bot_item=None, matching=0)
        rollback_prepared = shot(7, "original", bot_item=None, matching=0)
        old_prepare = bot(A, "original", "old-prepare-2", 6.5)
        old_prepare["service"] = "prepare"
        old_prepare["status"] = "exited"
        old_prepare["running"] = False
        rollback_prepared["state"]["runtime"]["prepare"] = old_prepare
        second_old = bot(A, "original", "old-bot-2", 7.5)
        rollback_started = shot(
            8, "original", bot_item=second_old, prepare=copy.deepcopy(old_prepare)
        )
        final_before = shot(9, "original", bot_item=second_old)
        final_stopped = shot(10, "original", bot_item=None, matching=0)
        final_prepared = shot(11, "target", bot_item=None, matching=0)
        target_prepare = bot(B, "target", "target-prepare-3", 10.5)
        target_prepare["service"] = "prepare"
        target_prepare["status"] = "exited"
        target_prepare["running"] = False
        final_prepared["state"]["runtime"]["prepare"] = target_prepare
        final_bot = bot(B, "target", "new-bot-3", 11.5)
        final_started = shot(
            12, "target", bot_item=final_bot, prepare=copy.deepcopy(target_prepare)
        )
        inputs = {
            "initial-update": dict(zip(("before", "stopped", "prepared", "started"), initial)),
            "rollback": dict(
                zip(
                    ("before", "stopped", "prepared", "started"),
                    (rollback_before, rollback_stopped, rollback_prepared, rollback_started),
                )
            ),
            "final-update": dict(
                zip(
                    ("before", "stopped", "prepared", "started"),
                    (final_before, final_stopped, final_prepared, final_started),
                )
            ),
        }
        value = plan()
        value["baseline"] = initial[0]
        ops = {name: v.verify_operation(value, name, **items) for name, items in inputs.items()}
        final = shot(13, "target", bot_item=final_bot, prepare=copy.deepcopy(target_prepare))
        context = {"actor": {"tg_id": 42}}
        summary = v.summarize_evidence(value, ops, {}, final, context, operation_inputs=inputs)
        self.assertEqual(summary["production_history_preservation"], "PASS")
        self.assertEqual(summary["live_acceptance"], "PENDING")
        changed = copy.deepcopy(final)
        changed["state"]["database"]["migrations"][0]["applied_at"] = time(1)
        self.fail_code(
            "OLD_HISTORY_CHANGED",
            lambda: v.summarize_evidence(value, ops, {}, changed, context, operation_inputs=inputs),
        )
        changed_inputs = copy.deepcopy(inputs)
        changed_final_bot = changed_inputs["final-update"]["started"]["state"]["runtime"]["bot"]
        assert changed_final_bot is not None
        changed_final_bot["id"] = "new-bot"
        self.fail_code(
            "OPERATION_CHAIN",
            lambda: v.summarize_evidence(
                value, ops, {}, final, context, operation_inputs=changed_inputs
            ),
        )

    def test_summary_bounds_every_saved_case_window(self) -> None:
        value = plan()
        baseline = shot(0, bot_item=bot(A, "original", "old-bot", -1))
        value["baseline"] = baseline
        anchor = shot(4, "target", bot_item=bot(B, "target", "target-bot", 2.5))
        final = shot(13, "target", bot_item=bot(B, "target", "final-bot", 11.5))
        context = {"actor": {"tg_id": 42}, "anchor_input_sha256": {"after": v.digest(anchor)}}
        results = {"initial-update": {"status": "PASS"}, "navigation-search": {"status": "PASS"}}
        inputs = {
            "initial-update": {"before": baseline, "after": anchor},
            "navigation-search": {
                "before": shot(5, "target", bot_item=bot(B, "target", "target-bot", 2.5)),
                "after": shot(6, "target", bot_item=bot(B, "target", "target-bot", 2.5)),
            },
        }
        operations = {
            "initial-update": {"started": shot(3, "target")},
            "rollback": {"before": shot(8, "target"), "started": shot(10, "original")},
            "final-update": {"before": shot(11, "original"), "started": shot(12, "target")},
        }

        def summarize(
            current_inputs: dict[str, dict[str, ShotFixture]],
            current_operations: dict[str, dict[str, ShotFixture]] = operations,
        ) -> Mapping[str, object]:
            summary = v.summarize_evidence(
                value,
                {},
                results,
                final,
                context,
                operation_inputs=current_operations,
                case_inputs=current_inputs,
            )
            assert isinstance(summary, dict) and isinstance(summary.get("live_acceptance"), str)
            return cast("Mapping[str, object]", summary)

        self.assertEqual(summarize(inputs)["live_acceptance"], "PENDING")
        missing = copy.deepcopy(inputs)
        del missing["navigation-search"]
        self.fail_code("CASE_INPUTS_REQUIRED", lambda: summarize(missing))
        extra = copy.deepcopy(inputs)
        extra["description-edit"] = copy.deepcopy(inputs["navigation-search"])
        self.fail_code("CASE_INPUTS_REQUIRED", lambda: summarize(extra))
        after_final = copy.deepcopy(inputs)
        after_final["navigation-search"]["after"]["state"]["captured_at"] = time(14)
        self.fail_code("CASE_TIME_ORDER", lambda: summarize(after_final))
        during_after_final = copy.deepcopy(inputs)
        during_after_final["navigation-search"]["during"] = shot(14, "target")
        self.fail_code("CASE_TIME_ORDER", lambda: summarize(during_after_final))
        pre_anchor = copy.deepcopy(inputs)
        pre_anchor["navigation-search"]["before"]["state"]["captured_at"] = time(3.5)
        self.fail_code("CASE_TIME_ORDER", lambda: summarize(pre_anchor))
        rollback_phase = copy.deepcopy(inputs)
        rollback_phase["navigation-search"]["before"]["state"]["captured_at"] = time(9)
        rollback_phase["navigation-search"]["after"]["state"]["captured_at"] = time(10)
        self.fail_code("CASE_TIME_ORDER", lambda: summarize(rollback_phase))
        late_anchor = copy.deepcopy(inputs)
        late_anchor["initial-update"]["after"]["state"]["captured_at"] = time(9)
        late_context = copy.deepcopy(context)
        late_context["anchor_input_sha256"]["after"] = v.digest(
            late_anchor["initial-update"]["after"]
        )
        self.fail_code(
            "CASE_TIME_ORDER",
            lambda: v.summarize_evidence(
                value,
                {},
                results,
                final,
                late_context,
                operation_inputs=operations,
                case_inputs=late_anchor,
            ),
        )

    def test_ui_pass_needs_observation_and_verified_case_reference(self) -> None:
        value = plan()
        baseline = shot(0, bot_item=bot(A, "original", "old-bot", -1))
        anchor = shot(3, "target", bot_item=bot(B, "target", "new-bot", 2.5))
        value["baseline"] = baseline
        context = {
            "actor": {"tg_id": 42},
            "plan_sha256": v.digest(value),
            "anchor_input_sha256": {"after": v.digest(anchor)},
        }
        result = {"case": "initial-update", "status": "PASS"}
        cases = {"initial-update": result}
        case_inputs = {"initial-update": {"before": baseline, "after": anchor}}
        final = shot(4, "target", bot_item=bot(B, "target", "new-bot", 2.5))
        entries = [
            {
                "id": identifier,
                "status": "PENDING",
                "observed_at": None,
                "observation": "",
                "case_sha256": [],
            }
            for identifier in v.UI_IDS
        ]
        observations = {
            "schema_version": 2,
            "kind": "arb-live-observations",
            "plan_sha256": v.digest(value),
            "context_sha256": v.digest(context),
            "scenarios": entries,
        }

        def summarize() -> Mapping[str, object]:
            summary = v.summarize_evidence(
                value, {}, cases, final, context, observations, case_inputs=case_inputs
            )
            assert isinstance(summary, dict) and isinstance(summary.get("live_ui_acceptance"), str)
            return cast("Mapping[str, object]", summary)

        summary = summarize()
        self.assertEqual(summary["live_ui_acceptance"], "PENDING")
        entries[0].update(
            status="PASS",
            observation="One target search was observed",
            case_sha256=[v.digest(result)],
        )
        self.fail_code("UNOBSERVED_UI_PASS", summarize)
        entries[0]["observed_at"] = time(3)
        entries[0]["case_sha256"] = ["0" * 64]
        self.fail_code("INVALID_OBSERVATIONS", summarize)
        entries[0]["case_sha256"] = [v.digest(result)]
        self.assertEqual(summarize()["live_acceptance"], "PENDING")
        entries[-1]["id"] = entries[0]["id"]
        self.fail_code("INVALID_OBSERVATIONS", summarize)

    def test_ui_probe_must_be_actual_alias_and_described_neighbor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            (target / "catalog.json").write_bytes(
                v.canonical(
                    {
                        "anime": {
                            "8": {"title": "New", "aliases": ["Alias"], "synopsis": "Seed"},
                            "9": {"title": "Neighbor", "aliases": [], "synopsis": "Description"},
                        }
                    }
                )
            )
            value = plan()
            value["bundles"]["target"]["path"] = str(target)
            bundle = {"neighbors": {"8": [{"mal_id": 9}]}}
            v._check_probes(value, bundle)
            value["ui_probes"]["alias_search"]["query"] = "New"
            self.fail_code("INVALID_UI_PROBES", lambda: v._check_probes(value, bundle))
            value["ui_probes"]["alias_search"]["query"] = "Alias"
            value["ui_probes"]["description_edit"]["neighbor_mal_id"] = 10
            self.fail_code("INVALID_UI_PROBES", lambda: v._check_probes(value, bundle))

    def test_applicable_empty_case_requires_real_request_and_no_delivery(self) -> None:
        value = plan()
        value["bundles"]["target"]["expectations"]["empty_seed"] = {"mal_id": 12, "query": "Empty"}
        value["bundles"]["target"]["histogram"]["0"] = 1
        context = {
            "schema_version": 2,
            "kind": "arb-live-context",
            "plan_sha256": v.digest(value),
            "actor": {"tg_id": 42, "chat_id": 42},
        }
        live_bot = bot(B, "target", "target-bot", 2.5)
        before = shot(4, "target", bot_item=live_bot)
        after = shot(5, "target", bot_item=live_bot)
        after["state"]["database"]["requests"].append(
            {
                "id": 2,
                "tg_id": 42,
                "action_key": "msg:42:2",
                "raw_query": "Empty",
                "seed_mal_id": 12,
                "bundle_id": B,
                "created_at": time(5),
                "resolved_at": time(5),
            }
        )
        with mock.patch.object(
            v, "_bundle", return_value=({"id": B, "neighbors": {"12": []}}, None, None)
        ):
            receipt = v.verify_case(value, context, "empty", before, after)
            self.assertEqual(receipt["status"], "PASS")
            self.assertEqual(receipt["counts"]["delivered"], 0)
            after["state"]["database"]["delivered"].append(
                {
                    "id": 2,
                    "request_id": 2,
                    "tg_id": 42,
                    "rank": 1,
                    "mal_id": 99,
                    "chat_id": 42,
                    "message_id": 12,
                    "delivered_at": time(5),
                }
            )
            self.fail_code(
                "DELIVERY_ORDER", lambda: v.verify_case(value, context, "empty", before, after)
            )


if __name__ == "__main__":
    unittest.main()
