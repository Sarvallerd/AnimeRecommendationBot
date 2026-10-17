"""Read-only ARB-034 live acceptance recomputation. Never operates a service."""

from __future__ import annotations

import collections
import importlib.util
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Literal, NotRequired, Protocol, TypedDict, cast, overload

if TYPE_CHECKING:
    from datetime import datetime

    from deploy.tests.acceptance_types import (
        AcceptanceProfile,
        BundleSummaryView,
        ContainerRecord,
        DeploymentRecord,
        EvidenceModule,
        LegacyBundle,
        LiveCaptureRecord,
        ProfileExpectations,
        ProfileModule,
        RuntimeRecord,
        SnapshotRecord,
    )

    from recsys.contracts.bundle import Catalog, Neighbor


class BundlePlan(TypedDict):
    role: str
    identity: str
    path: str
    release_id: str
    file_sha256: dict[str, str]
    count: int
    histogram: dict[str, int]
    expectations: ProfileExpectations
    image_id: str


class OfflinePlan(TypedDict):
    evidence_dir: str
    report_sha256: str
    project: str
    database_volume: str


class ProfilePlan(TypedDict):
    id: str
    release_id: str
    sha256: str
    lock_sha256: str


class SourcePlan(TypedDict):
    commit: str
    compose_sha256: str
    snapshot_sql_sha256: str
    verifier_sha256: str
    helper_sha256: str
    dispatch_sha256: str


class ServicePlan(TypedDict):
    project: str
    daemon_id: str
    postgres_id: str
    postgres_image_id: str
    database_volume: str


class SettingsPlan(TypedDict):
    sha256: str
    uid: int
    gid: int
    mode: int
    credential_sha256: dict[str, str]
    original_copy: str
    target_copy: str


class UiAliasProbe(TypedDict):
    mal_id: int
    query: str


class UiDescriptionProbe(TypedDict):
    seed_mal_id: int
    query: str
    neighbor_mal_id: int


class UiProbes(TypedDict):
    schema_version: int
    kind: str
    alias_search: UiAliasProbe
    description_edit: UiDescriptionProbe


class LivePlan(TypedDict):
    schema_version: int
    kind: str
    offline: OfflinePlan
    profile: ProfilePlan
    source: SourcePlan
    service: ServicePlan
    settings: SettingsPlan
    expected_settings_sha256: dict[str, str]
    bundles: dict[str, BundlePlan]
    ui_probes: UiProbes
    baseline_sha256: str
    baseline: LiveCaptureRecord


class RequestRow(TypedDict):
    id: int
    tg_id: int
    action_key: str
    raw_query: str
    created_at: str
    seed_mal_id: int | None
    bundle_id: str | None


class DeliveredRow(TypedDict):
    id: int
    request_id: int
    rank: int
    mal_id: int
    tg_id: int
    chat_id: int
    message_id: int


class AnimeEventRow(TypedDict):
    id: int
    request_id: int
    tg_id: int
    mal_id: int
    score: int


class CurrentRow(TypedDict):
    tg_id: int
    mal_id: int
    created_at: str
    updated_at: str
    current_event_id: int


class UserRow(TypedDict):
    tg_id: int
    created_at: str


class RatingRow(TypedDict):
    position_id: int
    tg_id: int
    score: int


class FeedbackRow(TypedDict):
    id: int
    tg_id: int
    action_key: str
    body: str


class MigrationRow(TypedDict):
    version: int


class DatabaseRecord(TypedDict):
    migrations: list[MigrationRow]
    users: list[UserRow]
    requests: list[RequestRow]
    delivered: list[DeliveredRow]
    anime_events: list[AnimeEventRow]
    anime_current: list[CurrentRow]
    recommendation_ratings: list[RatingRow]
    feedback: list[FeedbackRow]
    legacy_users: list[dict[str, object]]
    legacy_requests: list[dict[str, object]]
    legacy_feedback: list[dict[str, object]]


class DatabaseDelta(TypedDict):
    requests: list[RequestRow]
    resolved_pending: list[RequestRow]
    delivered: list[DeliveredRow]
    anime_events: list[AnimeEventRow]
    recommendation_ratings: list[RatingRow]
    feedback: list[FeedbackRow]


class ValidatedState(TypedDict):
    schema_version: int
    kind: str
    snapshot_sql_sha256: str
    source_commit: str
    captured_at: str
    deployment: DeploymentRecord
    runtime: RuntimeRecord
    database: DatabaseRecord


OperationReceipt = TypedDict(
    "OperationReceipt",
    {
        "schema_version": int,
        "kind": str,
        "plan_sha256": str,
        "operation": str,
        "from": str,
        "to": str,
        "status": str,
        "reason": str,
        "counts": list[int],
        "started_capture_time": str | None,
        "input_sha256": dict[str, str],
    },
)


class ActorContext(TypedDict):
    tg_id: int
    chat_id: int
    target_anchor_request_id: int


class LiveContext(TypedDict):
    schema_version: int
    kind: str
    plan_sha256: str
    offline_report_sha256: str
    profile: ProfilePlan
    service: ServicePlan
    bundles: dict[str, str]
    actor: ActorContext
    anchor_input_sha256: dict[str, str]
    initial_operation_sha256: str


class CaseReceipt(TypedDict):
    schema_version: int
    kind: str
    plan_sha256: str
    context_sha256: str
    case: str
    status: str
    reason: str
    bundle_role: str
    bundle_identity: str
    counts: dict[str, int]
    input_sha256: dict[str, str]
    operation_sha256: str | None


class OperationInputs(TypedDict):
    before: LiveCaptureRecord
    stopped: LiveCaptureRecord
    prepared: LiveCaptureRecord
    started: LiveCaptureRecord


class CaseInputs(TypedDict):
    before: LiveCaptureRecord
    after: LiveCaptureRecord
    during: NotRequired[LiveCaptureRecord]


class ObservationEntry(TypedDict):
    id: str
    status: str
    observed_at: str | None
    observation: str
    case_sha256: list[str]


class MachineSummary(TypedDict):
    passed: int
    required: int


class SelectedSummary(TypedDict):
    role: str
    identity: str
    captured_at: str


class SummaryRecord(TypedDict):
    schema_version: int
    kind: str
    offline_compatibility: str
    operations: dict[str, str]
    production_history_preservation: str
    machine: MachineSummary
    live_ui_acceptance: str
    live_acceptance: str
    selected: SelectedSummary
    poller_scope: str
    limits: list[str]


class OfflineRuntime(TypedDict):
    project: str
    database_volume: str
    bot_image_id: str


class OfflineProfile(TypedDict):
    sha256: str
    release_lock_sha256: str


class OfflineOriginal(TypedDict):
    bundle_identity: str
    bundle_file_sha256: dict[str, str]


class OfflineBuild(TypedDict):
    identity: str
    file_sha256: dict[str, str]


class OfflineReport(TypedDict):
    runtime: OfflineRuntime
    profile: OfflineProfile
    original: OfflineOriginal
    builds: dict[str, OfflineBuild]


if TYPE_CHECKING:

    class _Evidence(EvidenceModule, Protocol):
        KEYS: dict[str, tuple[str, ...]]
        LEGACY: tuple[str, ...]
        CANARIES: dict[str, str]

        def integer(
            self, value: object, low: int | None = None, high: int | None = None
        ) -> bool: ...

        def index_rows[T: Mapping[str, object]](
            self, rows: list[T], fields: tuple[str, ...]
        ) -> dict[tuple[object, ...], T]: ...

    class _Release(Protocol):
        e: _Evidence
        p: ProfileModule

        def verify_record(self, evidence: Path) -> OfflineReport: ...

        def change_selection(self, original: bytes, bundle: Path) -> bytes: ...


def _load(name: str, filename: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load acceptance module {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _release_module() -> _Release:
    module = _load("arb034_offline", "release-acceptance.py")
    exports = vars(module)
    evidence = exports.get("e")
    profiles = exports.get("p")
    if not isinstance(evidence, ModuleType) or not isinstance(profiles, ModuleType):
        raise ImportError("Cannot load acceptance module release-acceptance.py")
    if any(
        not callable(getattr(evidence, name, None))
        for name in ("require", "sha", "canonical", "utc", "validate_database", "index_rows")
    ) or any(
        not callable(getattr(profiles, name, None))
        for name in ("canonical", "load_profile", "check_profile_bundle")
    ):
        raise ImportError("Cannot load acceptance module release-acceptance.py")
    if any(not callable(exports.get(name)) for name in ("verify_record", "change_selection")):
        raise ImportError("Cannot load acceptance module release-acceptance.py")
    return cast("_Release", module)


r = _release_module()
e, p = r.e, r.p
Error = e.EvidenceError
need = e.require
sha = e.sha
canonical = e.canonical
OPERATIONS = ("initial-update", "rollback", "final-update")
CASES = tuple(item for item in e.CASES if item not in ("recommendations", "update")) + (
    "initial-update",
    "final-update",
    "alias-search",
    "description-edit",
    "catalog-stale",
)
UI_IDS = (
    "actor-confirmation",
    "search-selection",
    "alias-search",
    "card-content-and-order",
    "description-edit",
    "score-ranges",
    "confirmations",
    "empty-notice",
    "consumed-controls",
    "catalog-stale-controls",
    "restart",
    "outage-failure",
    "retry-success",
    "initial-update",
    "rollback",
    "final-update",
    "navigation",
)
UI_CASE = {
    "actor-confirmation": "initial-update",
    "search-selection": "initial-update",
    "alias-search": "alias-search",
    "card-content-and-order": "initial-update",
    "description-edit": "description-edit",
    "score-ranges": "recommendation-scores",
    "confirmations": "feedback-unicode",
    "empty-notice": "empty",
    "consumed-controls": "recommendation-repeat",
    "catalog-stale-controls": "catalog-stale",
    "restart": "restart",
    "outage-failure": "db-outage",
    "retry-success": "feedback-retry",
    "initial-update": "initial-update",
    "rollback": "rollback",
    "final-update": "final-update",
    "navigation": "navigation-search",
}
NO_EFFECT = set(e.NO_EFFECT) | {"description-edit", "catalog-stale"}
FLOW = {
    "initial-update": "target",
    "rollback": "original",
    "final-update": "target",
    "alias-search": "target",
}


_DATABASE_FIELDS: dict[str, dict[str, tuple[type[object], ...]]] = {
    "migrations": {"version": (int,)},
    "users": {"tg_id": (int,), "created_at": (str,)},
    "requests": {
        "id": (int,),
        "tg_id": (int,),
        "action_key": (str,),
        "raw_query": (str,),
        "created_at": (str,),
        "seed_mal_id": (int, type(None)),
        "bundle_id": (str, type(None)),
    },
    "delivered": {
        "id": (int,),
        "request_id": (int,),
        "rank": (int,),
        "mal_id": (int,),
        "tg_id": (int,),
        "chat_id": (int,),
        "message_id": (int,),
    },
    "anime_events": {
        "id": (int,),
        "request_id": (int,),
        "tg_id": (int,),
        "mal_id": (int,),
        "score": (int,),
    },
    "anime_current": {
        "tg_id": (int,),
        "mal_id": (int,),
        "created_at": (str,),
        "updated_at": (str,),
        "current_event_id": (int,),
    },
    "recommendation_ratings": {
        "position_id": (int,),
        "tg_id": (int,),
        "score": (int,),
    },
    "feedback": {
        "id": (int,),
        "tg_id": (int,),
        "action_key": (str,),
        "body": (str,),
    },
}


def _validated_database(value: object) -> DatabaseRecord:
    tables_expected = set(_DATABASE_FIELDS) | {
        "legacy_users",
        "legacy_requests",
        "legacy_feedback",
    }
    need(type(value) is dict and set(value) == tables_expected, "INVALID_DATABASE_SNAPSHOT")
    tables = cast("dict[str, object]", value)
    for table, rows in tables.items():
        need(type(rows) is list, "INVALID_DATABASE_SNAPSHOT")
        for row in cast("list[object]", rows):
            need(type(row) is dict, "INVALID_DATABASE_SNAPSHOT")
            if table in _DATABASE_FIELDS:
                fields = _DATABASE_FIELDS[table]
                entry = cast("dict[str, object]", row)
                need(
                    all(
                        field in entry and type(entry[field]) in accepted
                        for field, accepted in fields.items()
                    ),
                    "INVALID_DATABASE_SNAPSHOT",
                )
    e.validate_database(value)
    return cast("DatabaseRecord", value)


def digest(value: object) -> str:
    return sha(canonical(value))


def vtime(capture: LiveCaptureRecord) -> datetime:
    return e.utc(capture["state"]["captured_at"])


@overload
def _valid_capture(
    item: object, plan: LivePlan, role: str | None = None, *, full: Literal[True] = True
) -> ValidatedState: ...


@overload
def _valid_capture(
    item: object, plan: LivePlan, role: str | None = None, *, full: Literal[False]
) -> SnapshotRecord: ...


def _valid_capture(
    item: object, plan: LivePlan, role: str | None = None, *, full: bool = True
) -> ValidatedState | SnapshotRecord:
    need(
        type(item) is dict
        and item.get("schema_version") == 2
        and item.get("kind") == "arb-live-capture"
        and type(item.get("state")) is dict
        and type(item.get("guards")) is dict,
        "INVALID_CAPTURE",
    )
    item = cast("LiveCaptureRecord", item)
    state, guards = item["state"], item["guards"]
    need(
        state.get("schema_version") == 1
        and state.get("kind") == ("full" if full else "runtime_only")
        and state.get("deployment", {}).get("project") == plan["service"]["project"]
        and state["deployment"]["compose_sha256"] == plan["source"]["compose_sha256"]
        and state["snapshot_sql_sha256"] == plan["source"]["snapshot_sql_sha256"]
        and state["source_commit"] == plan["source"]["commit"],
        "CAPTURE_SOURCE",
    )
    need(
        guards.get("stable") is True
        and guards.get("daemon_id") == plan["service"]["daemon_id"]
        and guards.get("credential_sha256") == plan["settings"]["credential_sha256"]
        and guards.get("settings", {}).get("uid") == plan["settings"]["uid"]
        and guards["settings"]["gid"] == plan["settings"]["gid"]
        and guards["settings"]["mode"] == plan["settings"]["mode"],
        "CAPTURE_GUARDS",
    )
    inventory = guards.get("pollers", {})
    need(
        type(inventory.get("matching")) is int and 0 <= inventory["matching"] <= 1, "SECOND_POLLER"
    )
    ids = inventory.get("matching_container_ids_sha256")
    need(
        type(ids) is list
        and len(ids) <= inventory["matching"]
        and all(type(value) is str and e.HEX.fullmatch(value) for value in ids),
        "POLLER_IDENTITY",
    )
    bot = state["runtime"].get("bot")
    if bot is not None and bot.get("running") is True:
        need(inventory["matching"] == 1, "POLLER_COUNT")
        need(type(bot.get("id")) is str and ids == [sha(bot["id"].encode())], "POLLER_IDENTITY")
    need(
        inventory.get("scope") == "docker-daemon-and-known-local-processes", "POLLER_SCOPE_PENDING"
    )
    need(inventory.get("scope_complete") is True, "POLLER_SCOPE_PENDING")
    expected_hashes = plan["expected_settings_sha256"]
    need(
        guards["settings"]["sha256"] in expected_hashes.values()
        and state["deployment"]["settings_sha256"] == guards["settings"]["sha256"],
        "SETTINGS_CHANGED",
    )
    if role is not None:
        need(guards["settings"]["sha256"] == expected_hashes[role], "SETTINGS_CHANGED")
    pg = state["runtime"].get("postgres")
    need(
        pg is not None
        and pg.get("id") == plan["service"]["postgres_id"]
        and pg.get("image_id") == plan["service"]["postgres_image_id"]
        and pg.get("database_volume") == plan["service"]["database_volume"]
        and guards.get("postgres_identity")
        == {key: pg.get(key) for key in ("id", "image_id", "database_volume", "started_at")},
        "DATABASE_VOLUME_CHANGED",
    )
    assert pg is not None
    if full:
        need(
            pg.get("running") is True and guards.get("postgres_health") == "healthy",
            "DATABASE_UNHEALTHY",
        )
        _validated_database(state["database"])
        return cast("ValidatedState", state)
    return state


def _bundle(plan: LivePlan, role: str) -> tuple[LegacyBundle, BundleSummaryView, AcceptanceProfile]:
    value = plan["bundles"][role]
    profile = p.load_profile(value["release_id"])
    summary = p.check_profile_bundle(
        Path(value["path"]), profile, expected_identity=value["identity"]
    )
    need(
        summary.hashes == value["file_sha256"]
        and summary.count == value["count"]
        and summary.histogram == value["histogram"]
        and value["expectations"] == profile.expectations,
        "BUNDLE_CHANGED",
    )
    return (
        {
            "id": summary.identity,
            "count": summary.count,
            "neighbors": cast("dict[str, list[Neighbor]]", summary.neighbors),
        },
        summary,
        profile,
    )


def _check_probes(plan: LivePlan, target: LegacyBundle) -> None:
    probes = plan["ui_probes"]
    need(
        type(probes) is dict
        and probes.get("schema_version") == 1
        and probes.get("kind") == "arb-live-ui-probes"
        and set(probes) == {"schema_version", "kind", "alias_search", "description_edit"},
        "INVALID_UI_PROBES",
    )
    catalog_value = p.canonical(
        (Path(plan["bundles"]["target"]["path"]) / "catalog.json").read_bytes()
    )
    need(type(catalog_value.get("anime")) is dict, "INVALID_UI_PROBES")
    catalog = cast("Catalog", catalog_value)["anime"]
    alias, description = probes["alias_search"], probes["description_edit"]
    need(
        type(alias) is dict
        and set(alias) == {"mal_id", "query"}
        and type(description) is dict
        and set(description) == {"seed_mal_id", "query", "neighbor_mal_id"},
        "INVALID_UI_PROBES",
    )
    alias_entry = catalog.get(str(alias["mal_id"]))
    seed_entry = catalog.get(str(description["seed_mal_id"]))
    neighbor_entry = catalog.get(str(description["neighbor_mal_id"]))
    need(
        alias_entry is not None and seed_entry is not None and neighbor_entry is not None,
        "INVALID_UI_PROBES",
    )
    assert alias_entry is not None and seed_entry is not None and neighbor_entry is not None
    need(
        alias["query"] in alias_entry.get("aliases", [])
        and alias["query"] != alias_entry.get("title")
        and description["query"] in (seed_entry.get("title"), *seed_entry.get("aliases", []))
        and description["neighbor_mal_id"]
        in [item["mal_id"] for item in target["neighbors"].get(str(description["seed_mal_id"]), [])]
        and bool(neighbor_entry.get("synopsis")),
        "INVALID_UI_PROBES",
    )


def validate_plan(plan: object, *, verify_offline: bool = True) -> LivePlan:
    need(
        type(plan) is dict
        and plan.get("schema_version") == 2
        and plan.get("kind") == "arb-live-plan",
        "INVALID_PLAN",
    )
    assert isinstance(plan, dict)
    need(
        all(
            type(plan.get(key)) is dict
            for key in (
                "bundles",
                "service",
                "offline",
                "profile",
                "source",
                "settings",
                "expected_settings_sha256",
                "ui_probes",
                "baseline",
            )
        ),
        "INVALID_PLAN",
    )
    plan = cast("LivePlan", plan)
    need(
        set(plan["bundles"]) == {"original", "target"}
        and plan["bundles"]["original"]["identity"] != plan["bundles"]["target"]["identity"]
        and plan["service"]["project"] != plan["offline"]["project"]
        and plan["service"]["database_volume"] != plan["offline"]["database_volume"],
        "INVALID_PLAN",
    )
    need(
        plan["profile"]["release_id"] == p.NEW_RELEASE
        and plan["profile"]["id"] == p.NEW_RELEASE
        and plan["profile"]["sha256"] == p.TRUSTED_NEW_PROFILE_SHA
        and plan["profile"]["lock_sha256"] == p.TRUSTED_LOCK_SHA,
        "PROFILE_MISMATCH",
    )
    old, old_summary, old_profile = _bundle(plan, "original")
    target, target_summary, target_profile = _bundle(plan, "target")
    reference = target_profile.reference
    need(reference is not None, "REFERENCE_MISMATCH")
    assert reference is not None
    need(
        old_summary.identity == plan["bundles"]["original"]["identity"]
        and target_summary.identity == plan["bundles"]["target"]["identity"]
        and old_profile.release_id == p.LEGACY_RELEASE
        and target_profile.release_id == p.NEW_RELEASE
        and target_summary.identity == reference["bundle_identity"]
        and target_summary.hashes["catalog.json"] == reference["catalog_sha256"]
        and target_summary.hashes["neighbors.json"] == reference["neighbors_sha256"],
        "REFERENCE_MISMATCH",
    )
    _check_probes(plan, target)
    if verify_offline:
        report_path = Path(plan["offline"]["evidence_dir"]) / "preparation-report.json"
        need(sha(report_path.read_bytes()) == plan["offline"]["report_sha256"], "OFFLINE_CHANGED")
        report = r.verify_record(Path(plan["offline"]["evidence_dir"]))
        need(
            report["runtime"]["project"] == plan["offline"]["project"]
            and report["runtime"]["database_volume"] == plan["offline"]["database_volume"]
            and report["profile"]["sha256"] == plan["profile"]["sha256"]
            and report["profile"]["release_lock_sha256"] == plan["profile"]["lock_sha256"]
            and report["original"]["bundle_identity"] == old_summary.identity
            and report["original"]["bundle_file_sha256"] == old_summary.hashes
            and report["builds"]["A"]["identity"] == target_summary.identity
            and report["builds"]["A"]["file_sha256"] == target_summary.hashes
            and report["runtime"]["bot_image_id"] == plan["bundles"]["target"]["image_id"],
            "OFFLINE_MISMATCH",
        )
    baseline = plan["baseline"]
    state = _valid_capture(baseline, plan, "original")
    need(
        digest(baseline) == plan["baseline_sha256"]
        and baseline["guards"]["pollers"]["matching"] == 1,
        "BASELINE_MISMATCH",
    )
    baseline_bot = _runtime(state["runtime"]["bot"], plan["bundles"]["original"], plan, "bot")
    need(baseline_bot["running"] is True, "BASELINE_MISMATCH")
    need(
        plan["source"]["verifier_sha256"] == sha(Path(__file__).read_bytes())
        and plan["source"]["helper_sha256"]
        == sha(Path(__file__).with_name("live_evidence_v2.py").read_bytes())
        and plan["source"]["dispatch_sha256"]
        == sha(Path(__file__).with_name("acceptance-evidence.py").read_bytes())
        and plan["source"]["compose_sha256"] == sha((e.ROOT / "compose.yaml").read_bytes())
        and plan["source"]["snapshot_sql_sha256"] == sha(e.SQL.read_bytes()),
        "VERIFIER_CHANGED",
    )
    original = Path(plan["settings"]["original_copy"])
    target_copy = Path(plan["settings"]["target_copy"])
    for copy_path in (original, target_copy):
        need(
            copy_path.is_absolute()
            and not copy_path.is_symlink()
            and copy_path.is_file()
            and copy_path.stat().st_uid == plan["settings"]["uid"]
            and copy_path.stat().st_gid == plan["settings"]["gid"]
            and (copy_path.stat().st_mode & 0o777) == plan["settings"]["mode"],
            "SETTINGS_COPY_CHANGED",
        )
    need(
        sha(original.read_bytes()) == plan["expected_settings_sha256"]["original"]
        and sha(target_copy.read_bytes()) == plan["expected_settings_sha256"]["target"]
        and r.change_selection(original.read_bytes(), Path(plan["bundles"]["target"]["path"]))
        == target_copy.read_bytes(),
        "SETTINGS_COPY_CHANGED",
    )
    need(
        e.read_settings(original)["ARB_BUNDLE_DIR"] == plan["bundles"]["original"]["path"]
        and e.read_settings(target_copy)["ARB_BUNDLE_DIR"] == plan["bundles"]["target"]["path"],
        "SETTINGS_COPY_CHANGED",
    )
    return plan


def _runtime(
    item: ContainerRecord | None, selected: BundlePlan, plan: LivePlan, service: str
) -> ContainerRecord:
    need(
        item is not None
        and item.get("service") == service
        and item.get("bundle_id") == selected["identity"]
        and item.get("bundle_count") == selected["count"]
        and item.get("image_id") == selected["image_id"]
        and item.get("user") == "10001:10001"
        and item.get("read_only_root") is True
        and item.get("artifact_mount") == {"source": selected["path"], "rw": False},
        "RUNTIME_BUNDLE",
    )
    assert item is not None
    return item


def _live_bot(
    capture: LiveCaptureRecord, state: SnapshotRecord | ValidatedState, plan: LivePlan, role: str
) -> ContainerRecord:
    bot = _runtime(state["runtime"]["bot"], plan["bundles"][role], plan, "bot")
    need(bot["running"] is True, "BOT_NOT_RUNNING")
    need(capture["guards"]["pollers"]["matching"] == 1, "POLLER_COUNT")
    return bot


def verify_operation(
    plan: LivePlan,
    operation: str,
    before: LiveCaptureRecord,
    stopped: LiveCaptureRecord,
    prepared: LiveCaptureRecord,
    started: LiveCaptureRecord,
) -> OperationReceipt:
    need(operation in OPERATIONS, "INVALID_OPERATION")
    src, dst = (
        ("target", "original")
        if operation == "rollback"
        else (("original", "target") if operation == "initial-update" else ("original", "target"))
    )
    states = [
        _valid_capture(item, plan, src if i < 2 else dst)
        for i, item in enumerate((before, stopped, prepared, started))
    ]
    times = [e.utc(item["captured_at"]) for item in states]
    need(all(a < b for a, b in zip(times, times[1:])), "TIME_ORDER")
    need(
        [item["guards"]["pollers"]["matching"] for item in (before, stopped, prepared, started)]
        == [1, 0, 0, 1],
        "POLLER_COUNT",
    )
    need(
        all(item["database"] == states[0]["database"] for item in states[1:]),
        "DATABASE_HISTORY_CHANGED",
    )
    pg = states[0]["runtime"]["postgres"]
    need(all(item["runtime"]["postgres"] == pg for item in states[1:]), "DATABASE_CHANGED")
    oldbot = _runtime(states[0]["runtime"]["bot"], plan["bundles"][src], plan, "bot")
    need(
        oldbot["running"] is True
        and e.utc(oldbot["started_at"]) <= times[0]
        and (
            states[0]["runtime"]["prepare"] is None
            or states[0]["runtime"]["prepare"]["running"] is False
        )
        and (
            states[1]["runtime"]["prepare"] is None
            or states[1]["runtime"]["prepare"]["running"] is False
        )
        and (
            states[1]["runtime"]["bot"] is None
            or states[1]["runtime"]["bot"]["id"] == oldbot["id"]
            and not states[1]["runtime"]["bot"]["running"]
        )
        and (
            states[2]["runtime"]["bot"] is None
            or states[2]["runtime"]["bot"]["id"] == oldbot["id"]
            and not states[2]["runtime"]["bot"]["running"]
        ),
        "BOT_NOT_STOPPED",
    )
    prepare = _runtime(states[2]["runtime"]["prepare"], plan["bundles"][dst], plan, "prepare")
    need(
        prepare["running"] is False
        and prepare["status"] == "exited"
        and prepare["exit_code"] == 0
        and prepare["id"]
        != (
            states[0]["runtime"]["prepare"]["id"]
            if states[0]["runtime"]["prepare"] is not None
            else None
        )
        and e.utc(states[1]["captured_at"]) < e.utc(prepare["started_at"]) <= times[2],
        "PREPARE_GATE",
    )
    bot = _runtime(states[3]["runtime"]["bot"], plan["bundles"][dst], plan, "bot")
    need(
        bot["running"] is True
        and bot["id"] != oldbot["id"]
        and e.utc(states[2]["captured_at"]) < e.utc(bot["started_at"]) <= times[3]
        and states[3]["runtime"]["prepare"] == prepare,
        "BOT_NOT_RECREATED",
    )
    return {
        "schema_version": 2,
        "kind": "arb-live-operation",
        "plan_sha256": digest(plan),
        "operation": operation,
        "from": src,
        "to": dst,
        "status": "PASS",
        "reason": "OK",
        "counts": [1, 0, 0, 1],
        "started_capture_time": states[3]["captured_at"],
        "input_sha256": {
            key: digest(value)
            for key, value in zip(
                ("before", "stopped", "prepared", "started"), (before, stopped, prepared, started)
            )
        },
    }


def _new[T: Mapping[str, object]](
    before: list[T], after: list[T], fields: tuple[str, ...]
) -> list[T]:
    old = e.index_rows(before, fields)
    now = e.index_rows(after, fields)
    need(all(now.get(key) == value for key, value in old.items()), "OLD_HISTORY_CHANGED")
    return [row for key, row in now.items() if key not in old]


def _actor_key(row: RequestRow | FeedbackRow, actor: int | None) -> None:
    parts = row["action_key"].split(":")
    need(
        len(parts) == 3
        and parts[0] == "msg"
        and parts[1] == str(actor)
        and parts[2].isdigit()
        and int(parts[2]) > 0,
        "INVALID_ACTION_KEY",
    )


def _preserved(
    before: DatabaseRecord,
    after: DatabaseRecord,
    actor: int | None = None,
    *,
    allow_pointer: bool = False,
    allow_user: bool = False,
) -> DatabaseDelta:
    for table in ("legacy_users", "legacy_requests", "legacy_feedback"):
        need(
            collections.Counter(map(canonical, before[table]))
            == collections.Counter(map(canonical, after[table])),
            "LEGACY_CHANGED",
        )
    delta: DatabaseDelta = {
        "delivered": _new(before["delivered"], after["delivered"], e.KEYS["delivered"]),
        "anime_events": _new(before["anime_events"], after["anime_events"], e.KEYS["anime_events"]),
        "recommendation_ratings": _new(
            before["recommendation_ratings"],
            after["recommendation_ratings"],
            e.KEYS["recommendation_ratings"],
        ),
        "feedback": _new(before["feedback"], after["feedback"], e.KEYS["feedback"]),
        "requests": [],
        "resolved_pending": [],
    }
    need(
        _new(before["migrations"], after["migrations"], e.KEYS["migrations"]) == [],
        "MIGRATION_CHANGED",
    )
    old_requests = e.index_rows(before["requests"], ("id",))
    now_requests = e.index_rows(after["requests"], ("id",))
    pending: list[RequestRow] = []
    for key, request_row in old_requests.items():
        later_request = now_requests.get(key)
        need(later_request is not None, "OLD_HISTORY_CHANGED")
        assert later_request is not None
        if later_request != request_row:
            need(
                request_row["seed_mal_id"] is None
                and later_request["seed_mal_id"] is not None
                and all(
                    request_row.get(k) == later_request.get(k)
                    for k in ("id", "tg_id", "action_key", "raw_query", "created_at")
                ),
                "OLD_REQUEST_CHANGED",
            )
            pending.append(later_request)
    delta["requests"] = [row for key, row in now_requests.items() if key not in old_requests]
    delta["resolved_pending"] = pending
    oldusers = e.index_rows(before["users"], ("tg_id",))
    nowusers = e.index_rows(after["users"], ("tg_id",))
    for key, user_row in oldusers.items():
        later_user = nowusers.get(key)
        need(
            later_user is not None
            and later_user["created_at"] == user_row["created_at"]
            and (
                later_user == user_row
                or allow_user
                and key == (actor,)
                and all(
                    later_user.get(field) == user_row.get(field)
                    for field in user_row
                    if field
                    not in ("language_code", "first_name", "last_name", "username", "updated_at")
                )
            ),
            "USER_HISTORY_CHANGED",
        )
    need(all(key == (actor,) for key in nowusers if key not in oldusers), "UNRELATED_WRITE")
    oldptr = e.index_rows(before["anime_current"], ("tg_id", "mal_id"))
    newptr = e.index_rows(after["anime_current"], ("tg_id", "mal_id"))
    events = delta["anime_events"]
    for key, pointer_row in oldptr.items():
        later_pointer = newptr.get(key)
        need(
            later_pointer is not None
            and later_pointer["created_at"] == pointer_row["created_at"]
            and (
                later_pointer == pointer_row
                or allow_pointer
                and key[0] == actor
                and any((event["tg_id"], event["mal_id"]) == key for event in events)
                and later_pointer["current_event_id"] > pointer_row["current_event_id"]
                and e.utc(later_pointer["updated_at"]) >= e.utc(pointer_row["updated_at"])
            ),
            "CURRENT_POINTER",
        )
    need(all(key[0] == actor for key in newptr if key not in oldptr), "UNRELATED_WRITE")
    for rows in (
        delta["requests"],
        delta["delivered"],
        delta["anime_events"],
        delta["recommendation_ratings"],
        delta["feedback"],
    ):
        need(all(row["tg_id"] == actor for row in rows), "UNRELATED_WRITE")
    need(all(row["tg_id"] == actor for row in pending), "UNRELATED_WRITE")
    for row in delta["requests"] + pending:
        _actor_key(row, actor)
    for feedback_row in delta["feedback"]:
        _actor_key(feedback_row, actor)
    return delta


def _flow(
    before: ValidatedState,
    after: ValidatedState,
    role: str,
    plan: LivePlan,
    actor: int | None = None,
    *,
    query: str | None = None,
    seed: int | None = None,
) -> tuple[RequestRow, DatabaseDelta]:
    db = before["database"]
    current = after["database"]
    known, _, _ = _bundle(plan, role)
    for item in (db, current):
        e.validate_database(item)
        e.validate_database(item, known)
    if actor is None:
        earlier = e.index_rows(db["requests"], ("id",))
        fresh = [
            row
            for row in current["requests"]
            if (row["id"],) not in earlier
            or earlier[(row["id"],)]["seed_mal_id"] is None
            and row["seed_mal_id"] is not None
        ]
        need(len(fresh) == 1, "FLOW_AMBIGUOUS")
        actor = fresh[0]["tg_id"]
    delta = _preserved(db, current, actor, allow_user=True)
    candidates = delta["requests"] + delta["resolved_pending"]
    need(len(candidates) == 1, "FLOW_AMBIGUOUS")
    request = candidates[0]
    expected = plan["bundles"][role]["expectations"]["smoke_seed"]
    seed = expected["mal_id"] if seed is None else seed
    query = expected["query"] if query is None else query
    need(
        request["seed_mal_id"] == seed
        and request["raw_query"] == query
        and request["bundle_id"] == plan["bundles"][role]["identity"],
        "WRONG_SELECTION",
    )
    parts = request["action_key"].split(":")
    need(
        len(parts) == 3
        and parts[0] == "msg"
        and parts[1].isdigit()
        and parts[2].isdigit()
        and int(parts[1]) == request["tg_id"]
        and (actor is None or request["tg_id"] == actor),
        "ANCHOR_CHAT",
    )
    expected_ids = [item["mal_id"] for item in known["neighbors"][str(seed)]]
    deliveries = sorted(delta["delivered"], key=lambda row: row["rank"])
    need(
        len(deliveries) == len(expected_ids)
        and [row["rank"] for row in deliveries] == list(range(1, len(expected_ids) + 1))
        and [row["mal_id"] for row in deliveries] == expected_ids
        and all(
            row["request_id"] == request["id"]
            and row["chat_id"] == actor
            and row["tg_id"] == actor
            and e.integer(row["message_id"], 1)
            for row in deliveries
        ),
        "DELIVERY_ORDER",
    )
    need(
        not any(
            row["position_id"] in {d["id"] for d in deliveries}
            for row in current["recommendation_ratings"]
        ),
        "ANCHOR_SCORED",
    )
    need(
        not any(delta[table] for table in ("anime_events", "recommendation_ratings", "feedback")),
        "UNEXPECTED_EFFECTS",
    )
    return request, delta


def bind_context(
    plan: LivePlan,
    initial_operation: OperationReceipt,
    before: LiveCaptureRecord,
    after: LiveCaptureRecord,
) -> LiveContext:
    need(
        initial_operation.get("operation") == "initial-update"
        and initial_operation.get("status") == "PASS"
        and initial_operation.get("plan_sha256") == digest(plan),
        "OPERATION_REQUIRED",
    )
    before_state = _valid_capture(before, plan, "target")
    after_state = _valid_capture(after, plan, "target")
    _live_bot(before, before_state, plan, "target")
    _live_bot(after, after_state, plan, "target")
    need(
        digest(before) == initial_operation["input_sha256"]["started"]
        and e.utc(before_state["captured_at"]) < e.utc(after_state["captured_at"]),
        "TIME_ORDER",
    )
    request, _ = _flow(before_state, after_state, "target", plan)
    return {
        "schema_version": 2,
        "kind": "arb-live-context",
        "plan_sha256": digest(plan),
        "offline_report_sha256": plan["offline"]["report_sha256"],
        "profile": plan["profile"],
        "service": plan["service"],
        "bundles": {name: item["identity"] for name, item in plan["bundles"].items()},
        "actor": {
            "tg_id": request["tg_id"],
            "chat_id": request["tg_id"],
            "target_anchor_request_id": request["id"],
        },
        "anchor_input_sha256": {"before": digest(before), "after": digest(after)},
        "initial_operation_sha256": digest(initial_operation),
    }


def verify_case(
    plan: LivePlan,
    context: LiveContext,
    case: str,
    before: LiveCaptureRecord,
    after: LiveCaptureRecord,
    *,
    during: LiveCaptureRecord | None = None,
    operation: OperationReceipt | None = None,
) -> CaseReceipt:
    need(
        case in CASES
        and context.get("schema_version") == 2
        and context.get("kind") == "arb-live-context"
        and context["plan_sha256"] == digest(plan),
        "INVALID_CONTEXT",
    )
    actor = context["actor"]["tg_id"]
    need(e.integer(actor, 1) and context["actor"]["chat_id"] == actor, "INVALID_ACTOR")
    role = "original" if case == "rollback" else "target"
    prior_role = (
        "target"
        if case == "rollback"
        else "original"
        if case in ("initial-update", "final-update")
        else "target"
    )
    bs = _valid_capture(before, plan, prior_role)
    af = _valid_capture(after, plan, role)
    before_bot = _live_bot(before, bs, plan, prior_role)
    after_bot = _live_bot(after, af, plan, role)
    need(e.utc(bs["captured_at"]) < e.utc(af["captured_at"]), "TIME_ORDER")
    if case in ("initial-update", "rollback", "final-update"):
        need(
            operation is not None
            and operation.get("operation") == case
            and operation.get("status") == "PASS"
            and operation.get("plan_sha256") == digest(plan)
            and operation["input_sha256"]["before"] == digest(before)
            and operation["started_capture_time"] is not None
            and e.utc(af["captured_at"]) > e.utc(operation["started_capture_time"]),
            "OPERATION_REQUIRED",
        )
        request, delta = _flow(bs, af, role, plan, actor)
        if case == "initial-update":
            need(
                request["id"] == context["actor"]["target_anchor_request_id"]
                and digest(after) == context["anchor_input_sha256"]["after"]
                and digest(operation) == context["initial_operation_sha256"],
                "ANCHOR_MISMATCH",
            )
    elif case == "alias-search":
        probe = plan["ui_probes"]["alias_search"]
        _, delta = _flow(bs, af, role, plan, actor, query=probe["query"], seed=probe["mal_id"])
    else:
        if case == "catalog-stale":
            need(
                operation is not None
                and operation.get("operation") == "final-update"
                and operation.get("plan_sha256") == digest(plan)
                and operation["started_capture_time"] is not None
                and e.utc(bs["captured_at"]) >= e.utc(operation["started_capture_time"]),
                "OPERATION_REQUIRED",
            )
        e.validate_database(bs["database"])
        e.validate_database(af["database"])
        bundle, _, _ = _bundle(plan, role)
        for db in (bs["database"], af["database"]):
            e.validate_database(db, bundle)
        delta = _preserved(
            bs["database"],
            af["database"],
            actor,
            allow_pointer=case in ("anime-score-1", "anime-score-10"),
            allow_user=True,
        )
        counts = {
            key: len(delta[key])
            for key in (
                "requests",
                "delivered",
                "anime_events",
                "recommendation_ratings",
                "feedback",
            )
        }
        need(len(delta["resolved_pending"]) <= 1, "AMBIGUOUS_REQUEST")
        if case in NO_EFFECT:
            need(
                all(value == 0 for value in counts.values()) and not delta["resolved_pending"],
                "UNEXPECTED_EFFECTS",
            )
            if case == "description-edit":
                description_probe = plan["ui_probes"]["description_edit"]
                need(
                    any(
                        row["tg_id"] == actor
                        and row["seed_mal_id"] == description_probe["seed_mal_id"]
                        and row["raw_query"] == description_probe["query"]
                        and row["bundle_id"] == plan["bundles"]["target"]["identity"]
                        and any(
                            d["request_id"] == row["id"]
                            and d["tg_id"] == actor
                            and d["chat_id"] == actor
                            and d["mal_id"] == description_probe["neighbor_mal_id"]
                            for d in bs["database"]["delivered"]
                        )
                        for row in bs["database"]["requests"]
                    ),
                    "DESCRIPTION_COORDINATES",
                )
            if case == "catalog-stale":
                need(
                    any(
                        row["tg_id"] == actor
                        and row["bundle_id"] == plan["bundles"]["original"]["identity"]
                        and any(
                            d["request_id"] == row["id"]
                            and d["tg_id"] == actor
                            and d["chat_id"] == actor
                            for d in bs["database"]["delivered"]
                        )
                        for row in bs["database"]["requests"]
                    ),
                    "ORIGINAL_CONTROL_MISSING",
                )
        elif case == "empty":
            empty = plan["bundles"]["target"]["expectations"]["empty_seed"]
            if empty is None:
                need(plan["bundles"]["target"]["histogram"]["0"] == 0, "FALSE_EMPTY_NA")
                return _case_result(
                    plan,
                    context,
                    case,
                    role,
                    before,
                    after,
                    during,
                    operation,
                    counts,
                    "NOT_APPLICABLE",
                )
            _flow(bs, af, role, plan, actor, query=empty["query"], seed=empty["mal_id"])
        elif case in ("anime-score-1", "anime-score-10"):
            requests = delta["requests"] + delta["resolved_pending"]
            smoke = plan["bundles"]["target"]["expectations"]["smoke_seed"]
            need(
                len(requests) == 1
                and len(delta["anime_events"]) == 1
                and requests[0]["seed_mal_id"] == smoke["mal_id"]
                and requests[0]["raw_query"] == smoke["query"]
                and requests[0]["bundle_id"] == plan["bundles"]["target"]["identity"]
                and delta["anime_events"][0]["request_id"] == requests[0]["id"]
                and delta["anime_events"][0]["score"] == (1 if case.endswith("1") else 10)
                and not delta["delivered"]
                and not delta["feedback"]
                and not delta["recommendation_ratings"],
                "ANIME_SCORE_MISSING",
            )
        elif case == "recommendation-scores":
            positions = {
                row["rank"]: row["id"]
                for row in af["database"]["delivered"]
                if row["request_id"] == context["actor"]["target_anchor_request_id"]
            }
            need(
                {(row["position_id"], row["score"]) for row in delta["recommendation_ratings"]}
                == {(positions[1], 0), (positions[2], 5)}
                and len(delta["recommendation_ratings"]) == 2
                and not any(
                    delta[key]
                    for key in (
                        "requests",
                        "delivered",
                        "anime_events",
                        "feedback",
                        "resolved_pending",
                    )
                ),
                "RECOMMENDATION_SCORE_MISSING",
            )
        elif case in ("feedback-unicode", "feedback-retry"):
            name = "unicode" if case == "feedback-unicode" else "retry"
            need(
                len(delta["feedback"]) == 1
                and delta["feedback"][0]["body"] == e.CANARIES[name]
                and not any(
                    delta[key]
                    for key in (
                        "requests",
                        "delivered",
                        "anime_events",
                        "recommendation_ratings",
                        "resolved_pending",
                    )
                ),
                "FEEDBACK_BODY",
            )
        else:
            need(all(value == 0 for value in counts.values()), "UNEXPECTED_EFFECTS")
    if case == "restart":
        need(
            before_bot["id"] == after_bot["id"]
            and e.utc(before_bot["started_at"]) < e.utc(after_bot["started_at"]),
            "RESTART_NOT_OBSERVED",
        )
    if case in NO_EFFECT and case not in ("restart", "db-outage"):
        need(
            before_bot["id"] == after_bot["id"]
            and before_bot["started_at"] == after_bot["started_at"],
            "BOT_PROCESS_CHANGED",
        )
    if case in ("db-outage", "feedback-retry", "feedback-retry-repeat"):
        need(
            before_bot["id"] == after_bot["id"]
            and before_bot["started_at"] == after_bot["started_at"],
            "BOT_PROCESS_CHANGED",
        )
    if case == "db-outage":
        need(during is not None, "OUTAGE_NOT_OBSERVED")
        assert during is not None
        ds = _valid_capture(during, plan, "target", full=False)
        during_bot = _live_bot(during, ds, plan, "target")
        during_pg = ds["runtime"]["postgres"]
        assert during_pg is not None
        need(
            e.utc(bs["captured_at"]) < e.utc(ds["captured_at"]) < e.utc(af["captured_at"])
            and during_pg["running"] is False
            and during_bot["id"] == before_bot["id"]
            and during_bot["started_at"] == before_bot["started_at"],
            "OUTAGE_NOT_OBSERVED",
        )
    counts = {
        key: len(delta[key])
        for key in ("requests", "delivered", "anime_events", "recommendation_ratings", "feedback")
    }
    counts["resolved_pending"] = len(delta["resolved_pending"])
    return _case_result(plan, context, case, role, before, after, during, operation, counts, "PASS")


def _case_result(
    plan: LivePlan,
    context: LiveContext,
    case: str,
    role: str,
    before: LiveCaptureRecord,
    after: LiveCaptureRecord,
    during: LiveCaptureRecord | None,
    operation: OperationReceipt | None,
    counts: dict[str, int],
    status: str,
) -> CaseReceipt:
    inputs = {"before": digest(before), "after": digest(after)}
    if during is not None:
        inputs["during"] = digest(during)
    return {
        "schema_version": 2,
        "kind": "arb-live-case",
        "plan_sha256": digest(plan),
        "context_sha256": digest(context),
        "case": case,
        "status": status,
        "reason": "OK" if status == "PASS" else "EMPTY_NOT_APPLICABLE",
        "bundle_role": role,
        "bundle_identity": plan["bundles"][role]["identity"],
        "counts": counts,
        "input_sha256": inputs,
        "operation_sha256": digest(operation) if operation is not None else None,
    }


def _full_database(capture: LiveCaptureRecord) -> DatabaseRecord:
    return _validated_database(capture["state"]["database"])


def _required_container(
    capture: LiveCaptureRecord, service: Literal["bot", "prepare"]
) -> ContainerRecord:
    container = capture["state"]["runtime"][service]
    need(container is not None, "OPERATION_CHAIN")
    assert container is not None
    return container


def summarize_evidence(
    plan: LivePlan,
    operations: dict[str, OperationReceipt],
    results: dict[str, CaseReceipt],
    final: LiveCaptureRecord,
    context: LiveContext | None = None,
    observations: object | None = None,
    operation_inputs: dict[str, OperationInputs] | None = None,
    case_inputs: dict[str, CaseInputs] | None = None,
) -> SummaryRecord:
    need(set(operations) <= set(OPERATIONS) and set(results) <= set(CASES), "INVALID_SUMMARY")
    final_state = _valid_capture(final, plan, "target")
    _runtime(final_state["runtime"]["bot"], plan["bundles"]["target"], plan, "bot")
    final_ok = (
        final_state["runtime"]["bot"] is not None
        and final_state["runtime"]["bot"]["running"] is True
        and final["guards"]["pollers"]["matching"] == 1
    )
    operation_status = {key: value["status"] for key, value in operations.items()}
    machine_status = {key: value["status"] for key, value in results.items()}
    if results or case_inputs is not None:
        need(type(case_inputs) is dict and set(case_inputs) == set(results), "CASE_INPUTS_REQUIRED")
    if results:
        need(
            context is not None
            and case_inputs is not None
            and "initial-update" in case_inputs
            and type(plan.get("baseline")) is dict,
            "CASE_INPUTS_REQUIRED",
        )
        assert context is not None and case_inputs is not None
        anchor = case_inputs["initial-update"]["after"]
        need(digest(anchor) == context["anchor_input_sha256"]["after"], "ANCHOR_MISMATCH")
        baseline_time, anchor_time, final_time = (
            vtime(item) for item in (plan["baseline"], anchor, final)
        )
        need(baseline_time < anchor_time <= final_time, "CASE_TIME_ORDER")
        for case, inputs in case_inputs.items():
            need(
                type(inputs) is dict
                and {"before", "after"} <= set(inputs)
                and set(inputs) <= {"before", "after", "during"},
                "CASE_INPUTS_REQUIRED",
            )
            before_time, after_time = vtime(inputs["before"]), vtime(inputs["after"])
            lower = baseline_time if case == "initial-update" else anchor_time
            need(lower <= before_time < after_time <= final_time, "CASE_TIME_ORDER")
            if "during" in inputs:
                need(before_time < vtime(inputs["during"]) < after_time, "CASE_TIME_ORDER")
        if operation_inputs is not None and set(operation_inputs) == set(OPERATIONS):
            initial, rollback, last = (operation_inputs[name] for name in OPERATIONS)
            rollback_before, rollback_started = (
                vtime(rollback[name]) for name in ("before", "started")
            )
            final_before, final_started = (vtime(last[name]) for name in ("before", "started"))
            need(vtime(initial["started"]) < anchor_time <= rollback_before, "CASE_TIME_ORDER")
            for case, inputs in case_inputs.items():
                before_time, after_time = vtime(inputs["before"]), vtime(inputs["after"])
                if case == "initial-update":
                    valid = after_time <= rollback_before
                elif case == "rollback":
                    valid = (
                        rollback_before <= before_time
                        and rollback_started < after_time <= final_before
                    )
                elif case == "final-update":
                    valid = final_before <= before_time and final_started < after_time
                else:
                    valid = (
                        anchor_time <= before_time
                        and after_time <= rollback_before
                        or final_started <= before_time
                        and after_time <= final_time
                    )
                    if case == "catalog-stale":
                        valid = final_started <= before_time and after_time <= final_time
                need(valid, "CASE_TIME_ORDER")
    history = "PENDING"
    if (
        len(operations) == 3
        and context is not None
        and all(value == "PASS" for value in operation_status.values())
    ):
        actor = context["actor"]["tg_id"]
        need(
            operation_inputs is not None and set(operation_inputs) == set(OPERATIONS),
            "OPERATION_INPUTS_REQUIRED",
        )
        assert operation_inputs is not None
        initial, rollback, last = (operation_inputs[name] for name in OPERATIONS)
        started_ids = [
            _required_container(item["started"], "bot")["id"] for item in (initial, rollback, last)
        ]
        prepare_ids = [
            _required_container(item["prepared"], "prepare")["id"]
            for item in (initial, rollback, last)
        ]
        need(
            len(set(started_ids)) == 3
            and len(set(prepare_ids)) == 3
            and _required_container(initial["started"], "bot")["id"]
            == _required_container(rollback["before"], "bot")["id"]
            and _required_container(rollback["started"], "bot")["id"]
            == _required_container(last["before"], "bot")["id"],
            "OPERATION_CHAIN",
        )
        need(
            vtime(plan["baseline"]) <= vtime(initial["before"])
            and vtime(initial["started"])
            < vtime(rollback["before"])
            < vtime(rollback["started"])
            < vtime(last["before"])
            < vtime(last["started"])
            < vtime(final),
            "TIME_ORDER",
        )
        for earlier, later in (
            (plan["baseline"], initial["before"]),
            (initial["started"], rollback["before"]),
            (rollback["started"], last["before"]),
            (last["started"], final),
        ):
            _preserved(
                _full_database(earlier),
                _full_database(later),
                actor,
                allow_pointer=True,
                allow_user=True,
            )
        need(
            final_state["runtime"]["bot"] == _required_container(last["started"], "bot"),
            "FINAL_RUNTIME_CHANGED",
        )
        _preserved(
            _full_database(plan["baseline"]),
            final_state["database"],
            actor,
            allow_pointer=True,
            allow_user=True,
        )
        history = "PASS"
    catalog_stale = results.get("catalog-stale")
    if catalog_stale is not None and catalog_stale["status"] == "PASS":
        need(
            context is not None
            and case_inputs is not None
            and {"rollback", "catalog-stale"} <= set(case_inputs),
            "CASE_INPUTS_REQUIRED",
        )
        assert context is not None and case_inputs is not None
        rollback_before_db = _full_database(case_inputs["rollback"]["before"])
        rollback_after = _full_database(case_inputs["rollback"]["after"])
        stale_before = _full_database(case_inputs["catalog-stale"]["before"])
        actor = context["actor"]["tg_id"]
        old_keys = {row["id"] for row in rollback_before_db["requests"]}
        fresh = [
            row
            for row in rollback_after["requests"]
            if row["id"] not in old_keys
            and row["tg_id"] == actor
            and row["bundle_id"] == plan["bundles"]["original"]["identity"]
        ]
        need(
            len(fresh) == 1
            and any(
                row["request_id"] == fresh[0]["id"]
                and row["tg_id"] == actor
                and row["chat_id"] == actor
                for row in stale_before["delivered"]
            ),
            "ROLLBACK_CONTROL_MISSING",
        )
    ui = "PENDING"
    if observations is not None:
        need(
            context is not None
            and type(observations) is dict
            and observations.get("schema_version") == 2
            and observations.get("kind") == "arb-live-observations"
            and observations.get("plan_sha256") == digest(plan)
            and observations.get("context_sha256") == digest(context),
            "INVALID_OBSERVATIONS",
        )
        assert context is not None and type(observations) is dict
        raw_observations = cast("dict[str, object]", observations)
        entries = raw_observations.get("scenarios")
        need(
            type(entries) is list
            and len(entries) == len(UI_IDS)
            and all(type(item) is dict and type(item.get("id")) is str for item in entries)
            and {item["id"] for item in entries} == set(UI_IDS),
            "INVALID_OBSERVATIONS",
        )
        assert type(entries) is list
        refs = {digest(value) for value in results.values()}
        checked_entries: list[ObservationEntry] = []
        for raw_entry in entries:
            need(
                type(raw_entry) is dict
                and set(raw_entry) == {"id", "status", "observed_at", "observation", "case_sha256"}
                and raw_entry["status"] in ("PENDING", "PASS", "FAIL", "NOT_APPLICABLE")
                and type(raw_entry["observation"]) is str
                and len(raw_entry["observation"]) <= 240
                and (raw_entry["observed_at"] is None or type(raw_entry["observed_at"]) is str)
                and type(raw_entry["case_sha256"]) is list
                and all(type(ref) is str for ref in raw_entry["case_sha256"])
                and len(set(raw_entry["case_sha256"])) == len(raw_entry["case_sha256"])
                and all(ref in refs for ref in raw_entry["case_sha256"]),
                "INVALID_OBSERVATIONS",
            )
            entry = cast("ObservationEntry", raw_entry)
            checked_entries.append(entry)
            expected = results.get(UI_CASE[entry["id"]])
            if entry["status"] == "PASS":
                need(
                    expected is not None
                    and expected["status"] == "PASS"
                    and digest(expected) in entry["case_sha256"]
                    and entry["observed_at"] is not None
                    and e.utc(entry["observed_at"]) <= e.utc(final_state["captured_at"])
                    and bool(entry["observation"]),
                    "UNOBSERVED_UI_PASS",
                )
            if entry["status"] == "NOT_APPLICABLE":
                need(
                    entry["id"] == "empty-notice"
                    and machine_status.get("empty") == "NOT_APPLICABLE"
                    and entry["observed_at"] is None,
                    "FALSE_UI_NA",
                )
            if entry["status"] == "PENDING":
                need(entry["observed_at"] is None, "INVALID_OBSERVATIONS")
        if any(entry["status"] == "FAIL" for entry in checked_entries):
            ui = "FAIL"
        elif all(entry["status"] in ("PASS", "NOT_APPLICABLE") for entry in checked_entries):
            ui = "PASS"
    if (
        any(value == "FAIL" for value in (*operation_status.values(), *machine_status.values()))
        or ui == "FAIL"
    ):
        live = "FAIL"
    elif (
        context is not None
        and len(results) == len(CASES)
        and len(operations) == len(OPERATIONS)
        and final_ok
        and history == "PASS"
        and ui == "PASS"
        and all(value in ("PASS", "NOT_APPLICABLE") for value in machine_status.values())
    ):
        live = "PASS"
    else:
        live = "PENDING"
    return {
        "schema_version": 2,
        "kind": "arb-live-summary",
        "offline_compatibility": "PASS",
        "operations": operation_status,
        "production_history_preservation": history,
        "machine": {
            "passed": sum(value == "PASS" for value in machine_status.values()),
            "required": len(CASES),
        },
        "live_ui_acceptance": ui,
        "live_acceptance": live,
        "selected": {
            "role": "target",
            "identity": plan["bundles"]["target"]["identity"],
            "captured_at": final_state["captured_at"],
        },
        "poller_scope": final["guards"]["pollers"]["scope"],
        "limits": [
            "Local inventory cannot exclude an unknown remote poller.",
            "Telegram send and PostgreSQL write are not exactly once.",
        ],
    }
