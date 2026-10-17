"""Adversarial, secret-free fixtures for the live evidence verifier."""

from __future__ import annotations

import contextlib
import copy
import datetime as dt
import importlib.util
import io
import json
import pathlib
import stat
import subprocess
import sys
import tempfile
import unittest
from collections.abc import Callable, Iterable
from typing import TypedDict, cast
from unittest import mock

SCRIPT = pathlib.Path(__file__).with_name("acceptance-evidence.py")
SPEC = importlib.util.spec_from_file_location("arb020_evidence", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"Cannot load acceptance evidence from {SCRIPT}")
e = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(e)
A = "sha256:" + "a" * 64
B = "sha256:" + "b" * 64
TIME = "2026-10-03T00:00:00+00:00"
TARGETS = [400, 709, 4088, 329, 1303]


type DatabaseFixture = dict[str, list[dict[str, object]]]


class MountFixture(TypedDict):
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
    artifact_mount: MountFixture
    database_volume: None
    bundle_id: str
    bundle_count: int


class PostgresFixture(TypedDict):
    id: str
    database_volume: str | None
    running: bool


class RuntimeFixture(TypedDict):
    bot: BotFixture
    prepare: BotFixture | None
    postgres: PostgresFixture


class DeploymentFixture(TypedDict):
    project: str
    compose_sha256: str
    settings_sha256: str


class ShotFixture(TypedDict):
    schema_version: int
    kind: str
    snapshot_sql_sha256: str
    source_commit: str
    captured_at: str
    deployment: DeploymentFixture
    runtime: RuntimeFixture
    database: DatabaseFixture


class ActorFixture(TypedDict):
    tg_id: int
    chat_id: int
    anchor_request_id: int


class BundlePathFixture(TypedDict):
    identity: str
    path: str


class ContextFixture(TypedDict):
    schema_version: int
    project: str
    database_volume: str | None
    actor: ActorFixture
    bundles: dict[str, BundlePathFixture]
    anchor: dict[str, object]


class NeighborFixture(TypedDict):
    mal_id: int


class BundleFixture(TypedDict):
    id: str
    count: int
    neighbors: dict[str, list[NeighborFixture]]


class TransitionFixture(TypedDict):
    status: str
    counts: dict[str, int]


class HarnessRuntimeFixture(TypedDict):
    postgres: dict[str, str | bool]
    bot: None
    prepare: dict[str, str | int] | None


class HarnessShotFixture(TypedDict):
    source_commit: str
    database: dict[str, list[dict[str, int]]]
    runtime: HarnessRuntimeFixture


def request(
    identifier: int,
    *,
    actor: int = 42,
    key: str | None = None,
    query: str = "Cowboy Bebop",
    seed: int | None = 1,
    bundle: str | None = A,
) -> dict[str, object]:
    key = key or f"msg:{actor}:{identifier + 99}"
    return {
        "id": identifier,
        "tg_id": actor,
        "action_key": key,
        "raw_query": query,
        "seed_mal_id": seed,
        "bundle_id": bundle,
        "resolved_at": TIME if seed else None,
        "created_at": TIME,
    }


def delivery(
    identifier: int, request_id: int, rank: int, target: int, actor: int = 42
) -> dict[str, object]:
    return {
        "id": identifier,
        "request_id": request_id,
        "tg_id": actor,
        "rank": rank,
        "mal_id": target,
        "chat_id": actor,
        "message_id": identifier + 100,
        "delivered_at": TIME,
    }


def database() -> DatabaseFixture:
    return {
        "migrations": [{"version": 1, "applied_at": TIME}],
        "users": [
            {
                "tg_id": 42,
                "language_code": "ru",
                "first_name": "Тест",
                "last_name": None,
                "username": None,
                "created_at": TIME,
                "updated_at": TIME,
            }
        ],
        "requests": [request(1)],
        "delivered": [delivery(rank, 1, rank, target) for rank, target in enumerate(TARGETS, 1)],
        "anime_events": [],
        "anime_current": [],
        "recommendation_ratings": [],
        "feedback": [],
        "legacy_users": [],
        "legacy_requests": [],
        "legacy_feedback": [],
    }


def runtime(identity: str = A, *, started: str = TIME, running: bool = True) -> BotFixture:
    return {
        "id": "container-1",
        "image_id": "image-1",
        "project": "arb020-test",
        "service": "bot",
        "user": "10001:10001",
        "status": "running" if running else "exited",
        "running": running,
        "exit_code": 0,
        "started_at": started,
        "restart_count": 0,
        "read_only_root": True,
        "artifact_mount": {
            "source": "/tmp/arb020-A" if identity == A else "/tmp/arb020-B",
            "rw": False,
        },
        "database_volume": None,
        "bundle_id": identity,
        "bundle_count": 17562,
    }


def shot(
    db: DatabaseFixture | None = None, minute: int | float = 0, identity: str = A
) -> ShotFixture:
    stamp = (dt.datetime.fromisoformat(TIME) + dt.timedelta(minutes=minute)).isoformat()
    return {
        "schema_version": 1,
        "kind": "full",
        "snapshot_sql_sha256": e.sha(e.SQL.read_bytes()),
        "source_commit": "f" * 40,
        "captured_at": stamp,
        "deployment": {
            "project": "arb020-test",
            "compose_sha256": "c" * 64,
            "settings_sha256": "d" * 64,
        },
        "runtime": {
            "bot": runtime(identity),
            "prepare": None,
            "postgres": {
                "id": "pg-1",
                "database_volume": "arb020-test_postgres_data",
                "running": True,
            },
        },
        "database": copy.deepcopy(db or database()),
    }


def context() -> ContextFixture:
    return {
        "schema_version": 1,
        "project": "arb020-test",
        "database_volume": "arb020-test_postgres_data",
        "actor": {"tg_id": 42, "chat_id": 42, "anchor_request_id": 1},
        "bundles": {
            "A": {"identity": A, "path": "/tmp/arb020-A"},
            "B": {"identity": B, "path": "/tmp/arb020-B"},
        },
        "anchor": {},
    }


def bundle(identity: str = A) -> BundleFixture:
    return {
        "id": identity,
        "count": 17562,
        "neighbors": {"1": [{"mal_id": target} for target in TARGETS], "40089": []},
    }


def check(
    case: str,
    before: ShotFixture,
    after: ShotFixture,
    *,
    identity: str = A,
    during: ShotFixture | None = None,
) -> TransitionFixture:
    result = e.verify_transition(case, context(), before, after, bundle(identity), during)
    assert (
        isinstance(result, dict)
        and isinstance(result.get("status"), str)
        and isinstance(result.get("counts"), dict)
    )
    return cast("TransitionFixture", result)


class EvidenceTests(unittest.TestCase):
    def test_explicit_live_v2_dispatch_preserves_v1_parser(self) -> None:
        with contextlib.redirect_stdout(io.StringIO()) as output:
            with self.assertRaises(SystemExit) as caught:
                e.main(["live-v2", "--help"])
        self.assertEqual(caught.exception.code, 0)
        self.assertIn("verify-operation", output.getvalue())
        self.assertIn("bind-actor", e.parser().format_help())
        self.assertNotIn("live-v2", e.parser().format_help())

    def assert_rejected(self, code: str, callback: Callable[[], object]) -> None:
        with self.assertRaises(e.EvidenceError) as caught:
            callback()
        self.assertEqual(caught.exception.code, code)

    def test_historical_database_keeps_partially_validated_rows(self) -> None:
        minimal = database()
        minimal["users"][0].pop("created_at")
        minimal["requests"][0].pop("created_at")
        minimal["delivered"][0]["id"] = True
        e.validate_database(minimal)
        minimal["delivered"][0]["id"] = "historical-position"
        e.validate_database(minimal)

    def test_historical_string_event_identifiers_remain_accepted(self) -> None:
        historical = database()
        historical["anime_events"].append(
            {"id": "event-1", "request_id": 1, "tg_id": 42, "mal_id": 1, "score": 5}
        )
        historical["anime_current"].append(
            {
                "tg_id": 42,
                "mal_id": 1,
                "current_event_id": "event-1",
                "created_at": TIME,
                "updated_at": TIME,
            }
        )
        e.validate_database(historical)
        self.assertEqual(e.historical_max(["event-1", "event-2"]), "event-2")
        self.assertTrue(e.historical_greater("event-2", "event-1"))
        self.assert_rejected("INVALID_TIMESTAMP", lambda: e.utc(7))
        with self.assertRaises(AttributeError):
            e.historical_action_parts(7)

    def test_new_offline_report_is_rejected_before_actor_binding(self) -> None:
        report = {
            "schema_version": 2,
            "kind": "arb-offline-release-acceptance",
            "profile": {"release_id": "neelagiri-2025-nafiul-2026-v1"},
        }
        self.assert_rejected(
            "UNSUPPORTED_LIVE_PROFILE",
            lambda: e.context_from_anchor({}, {}, report, "a" * 64, "b" * 64, "c" * 64),
        )

    def recommendation_pair(self) -> tuple[ShotFixture, ShotFixture]:
        before = shot(minute=0)
        after = shot(minute=1)
        after["database"]["requests"].append(request(2))
        after["database"]["delivered"].extend(
            delivery(5 + rank, 2, rank, target) for rank, target in enumerate(TARGETS, 1)
        )
        return before, after

    def test_actor_bind_and_recommendation(self) -> None:
        before, after = self.recommendation_pair()
        self.assertEqual(check("recommendations", before, after)["counts"]["delivered"], 5)
        report = {
            "bundles": context()["bundles"],
            "project": "arb020-test",
            "source_commit": "f" * 40,
            "compose_sha256": "c" * 64,
            "settings_sha256": "d" * 64,
            "database_volume": "arb020-test_postgres_data",
        }
        empty = shot({**database(), "requests": [], "delivered": []}, minute=0)
        with mock.patch.object(e, "load_bundle", return_value=bundle()):
            bound = e.context_from_anchor(
                empty, shot(minute=1), report, "a" * 64, "b" * 64, "c" * 64
            )
        self.assertEqual(bound["actor"]["anchor_request_id"], 1)
        wrong = shot(minute=1)
        wrong["database"]["delivered"][0]["mal_id"] = 999
        with mock.patch.object(e, "load_bundle", return_value=bundle()):
            self.assert_rejected(
                "ANCHOR_DELIVERIES",
                lambda: e.context_from_anchor(empty, wrong, report, "a" * 64, "b" * 64, "c" * 64),
            )
        wrong["database"]["delivered"][0]["mal_id"] = TARGETS[1]
        with mock.patch.object(e, "load_bundle", return_value=bundle()):
            self.assert_rejected(
                "ANCHOR_DELIVERIES",
                lambda: e.context_from_anchor(empty, wrong, report, "a" * 64, "b" * 64, "c" * 64),
            )
        after["database"]["requests"][-1]["tg_id"] = 43
        self.assert_rejected("INVALID_REQUEST", lambda: check("recommendations", before, after))

    def test_historical_null_volume_binds_but_transition_rejects(self) -> None:
        empty = shot({**database(), "requests": [], "delivered": []}, minute=0)
        after = shot(minute=1)
        empty["runtime"]["postgres"]["database_volume"] = None
        after["runtime"]["postgres"]["database_volume"] = None
        report = {
            "bundles": context()["bundles"],
            "project": "arb020-test",
            "source_commit": "f" * 40,
            "compose_sha256": "c" * 64,
            "settings_sha256": "d" * 64,
            "database_volume": None,
        }
        with mock.patch.object(e, "load_bundle", return_value=bundle()):
            bound = e.context_from_anchor(empty, after, report, "a" * 64, "b" * 64, "c" * 64)
        self.assertIsNone(bound["database_volume"])
        self.assert_rejected(
            "DATABASE_VOLUME_CHANGED",
            lambda: e.verify_transition("recommendations", bound, empty, after, bundle()),
        )

    def test_wrong_order_rank_seed_bundle_and_coordinates(self) -> None:
        before, after = self.recommendation_pair()
        self.assert_rejected(
            "WRONG_BUNDLE", lambda: check("recommendations", before, after, identity=B)
        )
        altered = copy.deepcopy(after)
        altered["database"]["delivered"][-1]["mal_id"] = 999
        self.assert_rejected("DELIVERY_ORDER", lambda: check("recommendations", before, altered))
        altered = copy.deepcopy(after)
        altered["database"]["delivered"][-1]["rank"] = 1
        self.assert_rejected("DUPLICATE_RANK", lambda: check("recommendations", before, altered))
        altered = copy.deepcopy(after)
        altered["database"]["requests"][-1]["seed_mal_id"] = 40089
        self.assert_rejected("DELIVERY_ORDER", lambda: check("recommendations", before, altered))
        altered = copy.deepcopy(after)
        altered["database"]["delivered"][-1]["chat_id"] = 77
        self.assert_rejected("DELIVERY_ORDER", lambda: check("recommendations", before, altered))

    def test_recommendation_scores_and_duplicate_effects(self) -> None:
        before = shot(minute=0)
        after = shot(minute=1)
        after["database"]["recommendation_ratings"] = [
            {"position_id": 1, "tg_id": 42, "score": 0, "created_at": TIME},
            {"position_id": 2, "tg_id": 42, "score": 5, "created_at": TIME},
        ]
        self.assertEqual(
            check("recommendation-scores", before, after)["counts"]["recommendation_ratings"], 2
        )
        duplicate = copy.deepcopy(after)
        duplicate["database"]["recommendation_ratings"].append(
            copy.deepcopy(duplicate["database"]["recommendation_ratings"][0])
        )
        self.assert_rejected(
            "DUPLICATE_KEY", lambda: check("recommendation-scores", before, duplicate)
        )
        after["database"]["recommendation_ratings"].pop()
        self.assert_rejected(
            "RECOMMENDATION_SCORE_MISSING", lambda: check("recommendation-scores", before, after)
        )

    def test_score_and_pointer_update_preserve_old_history(self) -> None:
        before = shot(minute=0)
        after = shot(minute=1)
        after["database"]["requests"].append(request(2))
        after["database"]["anime_events"].append(
            {"id": 1, "request_id": 2, "tg_id": 42, "mal_id": 1, "score": 1, "created_at": TIME}
        )
        after["database"]["anime_current"].append(
            {
                "tg_id": 42,
                "mal_id": 1,
                "current_event_id": 1,
                "created_at": TIME,
                "updated_at": TIME,
            }
        )
        self.assertEqual(check("anime-score-1", before, after)["counts"]["anime_events"], 1)
        later = copy.deepcopy(after)
        later["captured_at"] = (
            dt.datetime.fromisoformat(TIME) + dt.timedelta(minutes=2)
        ).isoformat()
        later["database"]["requests"].append(request(3))
        later["database"]["anime_events"].append(
            {"id": 2, "request_id": 3, "tg_id": 42, "mal_id": 1, "score": 10, "created_at": TIME}
        )
        later["database"]["anime_current"][0]["current_event_id"] = 2
        later["database"]["anime_current"][0]["updated_at"] = later["captured_at"]
        self.assertEqual(check("anime-score-10", after, later)["counts"]["anime_events"], 1)
        later["database"]["anime_current"][0]["current_event_id"] = 1
        self.assert_rejected("CURRENT_POINTER", lambda: check("anime-score-10", after, later))

    def test_canary_exactness_and_old_rows(self) -> None:
        before = shot(minute=0)
        after = shot(minute=1)
        self.assertGreater(len(e.CANARIES["unicode"]), 255)
        after["database"]["feedback"].append(
            {
                "id": 1,
                "tg_id": 42,
                "action_key": "msg:42:200",
                "body": e.CANARIES["unicode"],
                "created_at": TIME,
            }
        )
        self.assertEqual(check("feedback-unicode", before, after)["counts"]["feedback"], 1)
        for body in (
            e.CANARIES["unicode"][:-1],
            e.CANARIES["unicode"].replace("e\u0301", "é"),
            e.CANARIES["unicode"] + " ",
        ):
            mutated = copy.deepcopy(after)
            mutated["database"]["feedback"][0]["body"] = body
            self.assert_rejected(
                "FEEDBACK_BODY", lambda: check("feedback-unicode", before, mutated)
            )
        prior = copy.deepcopy(before)
        prior["database"]["feedback"].append(
            {"id": 7, "tg_id": 42, "action_key": "msg:42:7", "body": "старый", "created_at": TIME}
        )
        new = copy.deepcopy(after)
        new["database"]["feedback"].append(
            {"id": 7, "tg_id": 42, "action_key": "msg:42:7", "body": "изменён", "created_at": TIME}
        )
        self.assert_rejected("OLD_HISTORY_CHANGED", lambda: check("feedback-unicode", prior, new))
        prior = shot(minute=0)
        prior["database"]["legacy_feedback"] = [{"tg_id": 42, "msg": "legacy"}]
        self.assert_rejected("LEGACY_CHANGED", lambda: check("feedback-unicode", prior, after))
        prior = shot(minute=0)
        prior["database"]["migrations"][0]["applied_at"] = "2026-10-02T00:00:00+00:00"
        self.assert_rejected("OLD_HISTORY_CHANGED", lambda: check("feedback-unicode", prior, after))

    def test_pending_resolution_only_expected_and_user_profile_update(self) -> None:
        before = shot(minute=0)
        pending = request(2, seed=None, bundle=None)
        before["database"]["requests"].append(pending)
        after = copy.deepcopy(before)
        after["captured_at"] = (
            dt.datetime.fromisoformat(TIME) + dt.timedelta(minutes=1)
        ).isoformat()
        after["database"]["requests"][-1].update(
            seed_mal_id=40089, bundle_id=A, resolved_at=TIME, raw_query="Pittanko!! Nekozakana"
        )
        self.assert_rejected("OLD_REQUEST_CHANGED", lambda: check("empty", before, after))
        before["database"]["requests"][-1]["raw_query"] = "Pittanko!! Nekozakana"
        after["database"]["users"][0]["first_name"] = "Новое имя"
        after["database"]["users"][0]["updated_at"] = after["captured_at"]
        self.assertEqual(check("empty", before, after)["counts"]["resolved_pending"], 1)
        after["database"]["requests"][-1]["seed_mal_id"] = 1
        self.assert_rejected("WRONG_SELECTION", lambda: check("empty", before, after))

    def test_restart_outage_and_runtime_only_rejection(self) -> None:
        before = shot(minute=0)
        after = shot(minute=2)
        self.assert_rejected("RESTART_NOT_OBSERVED", lambda: check("restart", before, after))
        after["runtime"]["bot"]["started_at"] = "2026-10-03T00:01:00+00:00"
        self.assertEqual(check("restart", before, after)["status"], "PASS")
        during = shot(minute=1)
        during["kind"] = "runtime_only"
        during["database"] = None  # type: ignore[typeddict-item]  # runtime-only negative fixture
        during["runtime"]["postgres"]["running"] = False
        self.assert_rejected(
            "BOT_PROCESS_CHANGED", lambda: check("db-outage", before, after, during=during)
        )
        after["runtime"]["bot"]["started_at"] = TIME
        self.assertEqual(check("db-outage", before, after, during=during)["status"], "PASS")
        after["kind"] = "runtime_only"
        self.assert_rejected(
            "FULL_SNAPSHOT_REQUIRED", lambda: check("db-outage", before, after, during=during)
        )

    def switched(self, after: ShotFixture, identity: str, suffix: int, minute: int) -> ShotFixture:
        after["runtime"]["bot"]["id"] = f"container-{suffix}"
        after["runtime"]["bot"]["started_at"] = (
            dt.datetime.fromisoformat(TIME) + dt.timedelta(minutes=minute)
        ).isoformat()
        prepare = copy.deepcopy(after["runtime"]["bot"])
        prepare["id"] = f"prepare-{suffix}"
        prepare["service"] = "prepare"
        prepare["status"] = "exited"
        prepare["running"] = False
        prepare["exit_code"] = 0
        prepare["started_at"] = (
            dt.datetime.fromisoformat(TIME) + dt.timedelta(minutes=minute, seconds=-10)
        ).isoformat()
        after["runtime"]["prepare"] = prepare
        return after

    def test_update_rollback_need_new_bundle_requests(self) -> None:
        before = shot(minute=0)
        after = self.switched(shot(minute=1, identity=B), B, 2, 1)
        after["database"]["requests"].append(request(2, bundle=B))
        after["database"]["delivered"].extend(
            delivery(5 + rank, 2, rank, target) for rank, target in enumerate(TARGETS, 1)
        )
        self.assertEqual(check("update", before, after, identity=B)["status"], "PASS")
        old_only = self.switched(shot(minute=1, identity=B), B, 2, 1)
        self.assert_rejected(
            "EXPECTED_REQUEST_MISSING", lambda: check("update", before, old_only, identity=B)
        )
        rollback = self.switched(shot(copy.deepcopy(after["database"]), minute=2), A, 3, 2)
        rollback["database"]["requests"].append(request(3, bundle=A))
        rollback["database"]["delivered"].extend(
            delivery(10 + rank, 3, rank, target) for rank, target in enumerate(TARGETS, 1)
        )
        self.assertEqual(check("rollback", after, rollback)["status"], "PASS")
        for case, left, right, expected_identity in (
            ("update", before, after, B),
            ("rollback", after, rollback, A),
        ):
            same_target = copy.deepcopy(left)
            same_target["runtime"]["bot"]["bundle_id"] = expected_identity
            same_target["runtime"]["bot"]["artifact_mount"]["source"] = (
                "/tmp/arb020-B" if expected_identity == B else "/tmp/arb020-A"
            )
            self.assert_rejected(
                "RUNTIME_BUNDLE",
                lambda: check(case, same_target, right, identity=expected_identity),
            )
            no_recreate = copy.deepcopy(right)
            no_recreate["runtime"]["bot"]["id"] = left["runtime"]["bot"]["id"]
            self.assert_rejected(
                "BOT_NOT_RECREATED",
                lambda: check(case, left, no_recreate, identity=expected_identity),
            )
            failed_prepare = copy.deepcopy(right)
            prepare = failed_prepare["runtime"]["prepare"]
            assert prepare is not None
            prepare["exit_code"] = 1
            self.assert_rejected(
                "PREPARE_GATE",
                lambda: check(case, left, failed_prepare, identity=expected_identity),
            )
            stale_prepare = copy.deepcopy(right)
            prepare = stale_prepare["runtime"]["prepare"]
            assert prepare is not None
            prepare["started_at"] = left["runtime"]["bot"]["started_at"]
            self.assert_rejected(
                "PREPARE_GATE", lambda: check(case, left, stale_prepare, identity=expected_identity)
            )
            wrong_prepare = copy.deepcopy(right)
            prepare = wrong_prepare["runtime"]["prepare"]
            assert prepare is not None
            prepare["bundle_id"] = A if expected_identity == B else B
            self.assert_rejected(
                "PREPARE_GATE", lambda: check(case, left, wrong_prepare, identity=expected_identity)
            )

    def test_missing_events_ranges_and_early_outage_save(self) -> None:
        before = shot(minute=0)
        after = shot(minute=1)
        after["database"]["requests"].append(request(2))
        self.assert_rejected("ANIME_SCORE_MISSING", lambda: check("anime-score-1", before, after))
        after["database"]["anime_events"].append(
            {"id": 1, "request_id": 2, "tg_id": 42, "mal_id": 1, "score": 11, "created_at": TIME}
        )
        after["database"]["anime_current"].append(
            {
                "tg_id": 42,
                "mal_id": 1,
                "current_event_id": 1,
                "created_at": TIME,
                "updated_at": TIME,
            }
        )
        self.assert_rejected("INVALID_ANIME_EVENT", lambda: check("anime-score-1", before, after))
        after = shot(minute=1)
        after["database"]["recommendation_ratings"].append(
            {"position_id": 1, "tg_id": 42, "score": 6, "created_at": TIME}
        )
        self.assert_rejected(
            "INVALID_RECOMMENDATION_SCORE", lambda: check("recommendation-scores", before, after)
        )
        after = shot(minute=2)
        after["database"]["feedback"].append(
            {
                "id": 1,
                "tg_id": 42,
                "action_key": "msg:42:200",
                "body": e.CANARIES["retry"],
                "created_at": TIME,
            }
        )
        during = shot(minute=1)
        during["kind"] = "runtime_only"
        during["database"] = None  # type: ignore[typeddict-item]  # runtime-only negative fixture
        during["runtime"]["postgres"]["running"] = False
        self.assert_rejected(
            "UNEXPECTED_EFFECTS", lambda: check("db-outage", before, after, during=during)
        )

    def test_unrelated_resolution_and_summary_duplicate(self) -> None:
        before = shot(minute=0)
        after = shot(minute=1)
        other = request(2, key="msg:42:301", seed=None, bundle=None)
        before["database"]["requests"].append(other)
        after["database"]["requests"].append(
            {**other, "seed_mal_id": 1, "bundle_id": A, "resolved_at": TIME}
        )
        self.assert_rejected(
            "UNEXPECTED_REQUEST", lambda: check("recommendation-repeat", before, after)
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            results = root / "results"
            results.mkdir(mode=0o700)
            inputs = results / "inputs"
            inputs.mkdir(mode=0o700)
            body = e.canonical({"schema_version": 1})
            digest = e.sha(body)
            (inputs / f"{digest}.json").write_bytes(body)
            item = {
                "schema_version": 1,
                "case": "recommendations",
                "phase": None,
                "status": "FAIL",
                "reason": "INVALID_EVIDENCE",
                "counts": {},
                "bundle_id": None,
                "context_sha256": e.sha(e.canonical(context())),
                "input_sha256": {"before": digest, "after": digest},
            }
            (results / "one.json").write_bytes(e.canonical(item))
            (results / "two.json").write_bytes(e.canonical(item))
            observations = {
                "schema_version": 1,
                "context_sha256": item["context_sha256"],
                "scenarios": [
                    {"id": name, "status": "PENDING", "observed_at": None, "observation": ""}
                    for name in e.UI_IDS
                ],
            }
            self.assert_rejected(
                "DUPLICATE_RESULT", lambda: e.summarize(context(), results, observations)
            )

    def harness_python(self, index: int) -> str:
        lines = SCRIPT.with_name("full-catalog-acceptance.sh").read_text().splitlines()
        starts = [number for number, line in enumerate(lines) if "<<'PY'" in line]
        end = next(
            number for number in range(starts[index] + 1, len(lines)) if lines[number] == "PY"
        )
        return "\n".join(lines[starts[index] + 1 : end])

    def test_optimized_python_still_rejects_provenance_and_runtime_mismatches(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            data = root / "data"
            raw = data / "raw"
            raw.mkdir(parents=True)
            (raw / "source-manifest.json").write_text(json.dumps({"registry_sha256": "wrong"}))

            def optimized(
                index: int, *args: pathlib.Path | str
            ) -> subprocess.CompletedProcess[str]:
                return subprocess.run(
                    [sys.executable, "-O", "-", *(str(item) for item in args)],
                    cwd=SCRIPT.parents[2],
                    input=self.harness_python(index),
                    text=True,
                    capture_output=True,
                )

            invalid_source = optimized(
                1, data, root / "bundle-a", root / "bundle-b", root / "evidence"
            )
            self.assertNotEqual(invalid_source.returncode, 0)
            self.assertIn("ARTIFACT_PROVENANCE", invalid_source.stderr)
            settings = root / "compose.env"
            settings.write_text("ARB_BUNDLE_DIR=/old\n")
            unsafe_select = optimized(2, settings, root / "missing-bundle")
            self.assertNotEqual(unsafe_select.returncode, 0)
            self.assertIn("BUNDLE_SELECTION", unsafe_select.stderr)
            folder = root / "evidence"
            (folder / "snapshots").mkdir(parents=True)

            def snap(
                commit: str = "f" * 40,
                volume: str = "volume",
                database: list[dict[str, int]] | None = None,
            ) -> HarnessShotFixture:
                return {
                    "source_commit": commit,
                    "database": {
                        "migrations": [{"version": 1}],
                        "users": [] if database is None else database,
                    },
                    "runtime": {
                        "postgres": {"database_volume": volume, "running": True},
                        "bot": None,
                        "prepare": None,
                    },
                }

            def write_shots(items: Iterable[HarnessShotFixture]) -> None:
                for name, item in zip(
                    ("prepare-1-A.json", "prepare-2-B.json", "prepare-3-A.json"), items
                ):
                    (folder / "snapshots" / name).write_text(json.dumps(item))

            args = (
                folder,
                data,
                "arb020-test",
                root / "bundle-a",
                root / "bundle-b",
                settings,
                SCRIPT.parents[2],
                "f" * 40,
            )
            write_shots((snap(), snap(commit="e" * 40), snap()))
            wrong_commit = optimized(3, *args)
            self.assertNotEqual(wrong_commit.returncode, 0)
            self.assertIn("PREPARE_INVARIANTS", wrong_commit.stderr)
            write_shots((snap(), snap(database=[{"tg_id": 9}]), snap()))
            wrong_database = optimized(3, *args)
            self.assertNotEqual(wrong_database.returncode, 0)
            self.assertIn("PREPARE_INVARIANTS", wrong_database.stderr)
            write_shots((snap(), snap(volume="other-volume"), snap()))
            wrong_volume = optimized(3, *args)
            self.assertNotEqual(wrong_volume.returncode, 0)
            self.assertIn("PREPARE_INVARIANTS", wrong_volume.stderr)
            bad_prepare = snap()
            bad_prepare["runtime"]["prepare"] = {"status": "exited", "exit_code": 1}
            write_shots((bad_prepare, bad_prepare, bad_prepare))
            failed_gate = optimized(3, *args)
            self.assertNotEqual(failed_gate.returncode, 0)
            self.assertIn("PREPARE_INVARIANTS", failed_gate.stderr)

    def test_harness_source_state_rejects_dirty_and_changed_head(self) -> None:
        lines = SCRIPT.with_name("full-catalog-acceptance.sh").read_text().splitlines()
        start = lines.index("source_state() {")
        end = next(number for number in range(start + 1, len(lines)) if lines[number] == "}")
        function = "\n".join(lines[start : end + 1])
        with tempfile.TemporaryDirectory() as temporary:
            repo = pathlib.Path(temporary)

            def git(*args: str) -> str:
                return subprocess.run(
                    ["git", *args], cwd=repo, capture_output=True, text=True, check=True
                ).stdout.strip()

            git("init", "-q")
            git("config", "user.name", "Test")
            git("config", "user.email", "test@example.invalid")
            source = repo / "source.txt"
            source.write_text("original")
            git("add", "source.txt")
            git("commit", "-qm", "baseline")
            baseline = git("rev-parse", "HEAD")
            script = (
                "set -euo pipefail\nrepo=$1\nstarting_head=$2\nfail() { exit 1; }\n"
                + function
                + "\nsource_state\n"
            )

            def run() -> int:
                return subprocess.run(
                    ["bash", "-c", script, "--", str(repo), baseline],
                    cwd=repo,
                    capture_output=True,
                    text=True,
                ).returncode

            self.assertEqual(run(), 0)
            source.write_text("dirty")
            self.assertNotEqual(run(), 0)
            git("restore", "source.txt")
            extra = repo / "new.txt"
            extra.write_text("untracked")
            self.assertNotEqual(run(), 0)
            extra.unlink()
            source.write_text("changed")
            git("add", "source.txt")
            git("commit", "-qm", "changed")
            self.assertNotEqual(run(), 0)

    def test_embedded_harness_python_compiles(self) -> None:
        lines = SCRIPT.with_name("full-catalog-acceptance.sh").read_text().splitlines()
        starts = [index for index, line in enumerate(lines) if "<<'PY'" in line]
        self.assertEqual(len(starts), 4)
        for start in starts:
            end = next(i for i in range(start + 1, len(lines)) if lines[i] == "PY")
            compile("\n".join(lines[start + 1 : end]), f"harness-{start}", "exec")

    def test_private_output_and_ui_pending(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            private = root / "private"
            private.mkdir(mode=0o700)
            target = private / "one.json"
            e.write_new(target, {"value": 1})
            self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o600)
            self.assert_rejected("OUTPUT_EXISTS", lambda: e.write_new(target, {"value": 2}))
            link = private / "link.json"
            link.symlink_to(target)
            self.assert_rejected("OUTPUT_EXISTS", lambda: e.write_new(link, {"value": 3}))
            public = root / "public"
            public.mkdir(mode=0o755)
            self.assert_rejected(
                "PRIVATE_DIRECTORY_REQUIRED", lambda: e.write_new(public / "x.json", {})
            )
            results = private / "results"
            results.mkdir(mode=0o700)
            observations = {
                "schema_version": 1,
                "context_sha256": e.sha(e.canonical(context())),
                "scenarios": [
                    {"id": name, "status": "PENDING", "observed_at": None, "observation": ""}
                    for name in e.UI_IDS
                ],
            }
            self.assertEqual(
                e.summarize(context(), results, observations)["live_acceptance"], "PENDING"
            )
            observations["scenarios"][0]["status"] = "FAIL"
            self.assertEqual(
                e.summarize(context(), results, observations)["live_acceptance"], "FAIL"
            )

    def test_noniterable_observations_keep_historical_cli_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            private = pathlib.Path(temporary) / "private"
            private.mkdir(mode=0o700)
            results = private / "results"
            results.mkdir(mode=0o700)
            observations = {
                "schema_version": 1,
                "context_sha256": e.sha(e.canonical(context())),
                "scenarios": 1,
            }
            with self.assertRaises(TypeError):
                e.summarize(context(), results, observations)
            context_path = private / "context.json"
            observations_path = private / "observations.json"
            context_path.write_bytes(e.canonical(context()))
            observations_path.write_bytes(e.canonical(observations))
            error_output = io.StringIO()
            with contextlib.redirect_stderr(error_output):
                status = e.main(
                    [
                        "summarize",
                        "--context",
                        str(context_path),
                        "--results-dir",
                        str(results),
                        "--observations",
                        str(observations_path),
                        "--output",
                        str(private / "summary.json"),
                    ]
                )
            self.assertEqual(status, 1)
            self.assertEqual(error_output.getvalue(), "FAIL INVALID_EVIDENCE\n")


if __name__ == "__main__":
    unittest.main()
