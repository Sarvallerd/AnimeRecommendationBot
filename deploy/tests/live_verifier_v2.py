"""Read-only ARB-034 live acceptance recomputation. Never operates a service."""

from __future__ import annotations

import collections
import importlib.util
from pathlib import Path


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


r = _load("arb034_offline", "release-acceptance.py")
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


def digest(value):
    return sha(canonical(value))


def vtime(capture):
    return e.utc(capture["state"]["captured_at"])


def _valid_capture(item, plan, role=None, *, full=True):
    need(
        type(item) is dict
        and item.get("schema_version") == 2
        and item.get("kind") == "arb-live-capture"
        and type(item.get("state")) is dict
        and type(item.get("guards")) is dict,
        "INVALID_CAPTURE",
    )
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
    if full:
        need(
            pg.get("running") is True and guards.get("postgres_health") == "healthy",
            "DATABASE_UNHEALTHY",
        )
        e.validate_database(state["database"])
    return state


def _bundle(plan, role):
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
    return {"id": summary.identity, "neighbors": summary.neighbors}, summary, profile


def _check_probes(plan, target):
    probes = plan["ui_probes"]
    need(
        type(probes) is dict
        and probes.get("schema_version") == 1
        and probes.get("kind") == "arb-live-ui-probes"
        and set(probes) == {"schema_version", "kind", "alias_search", "description_edit"},
        "INVALID_UI_PROBES",
    )
    catalog = p.canonical((Path(plan["bundles"]["target"]["path"]) / "catalog.json").read_bytes())[
        "anime"
    ]
    alias, description = probes["alias_search"], probes["description_edit"]
    need(
        type(alias) is dict
        and set(alias) == {"mal_id", "query"}
        and type(description) is dict
        and set(description) == {"seed_mal_id", "query", "neighbor_mal_id"},
        "INVALID_UI_PROBES",
    )
    alias_entry = catalog.get(str(alias["mal_id"]), {})
    seed_entry = catalog.get(str(description["seed_mal_id"]), {})
    neighbor_entry = catalog.get(str(description["neighbor_mal_id"]), {})
    need(
        alias["query"] in alias_entry.get("aliases", [])
        and alias["query"] != alias_entry.get("title")
        and description["query"] in (seed_entry.get("title"), *seed_entry.get("aliases", []))
        and description["neighbor_mal_id"]
        in [item["mal_id"] for item in target["neighbors"].get(str(description["seed_mal_id"]), [])]
        and bool(neighbor_entry.get("synopsis")),
        "INVALID_UI_PROBES",
    )


def validate_plan(plan, *, verify_offline=True):
    need(
        type(plan) is dict
        and plan.get("schema_version") == 2
        and plan.get("kind") == "arb-live-plan",
        "INVALID_PLAN",
    )
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
    need(
        old_summary.identity == plan["bundles"]["original"]["identity"]
        and target_summary.identity == plan["bundles"]["target"]["identity"]
        and old_profile.release_id == p.LEGACY_RELEASE
        and target_profile.release_id == p.NEW_RELEASE
        and target_summary.identity == target_profile.reference["bundle_identity"]
        and target_summary.hashes["catalog.json"] == target_profile.reference["catalog_sha256"]
        and target_summary.hashes["neighbors.json"] == target_profile.reference["neighbors_sha256"],
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
    _runtime(state["runtime"]["bot"], plan["bundles"]["original"], plan, "bot")
    need(state["runtime"]["bot"]["running"] is True, "BASELINE_MISMATCH")
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


def _runtime(item, selected, plan, service):
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
    return item


def _live_bot(capture, state, plan, role):
    bot = _runtime(state["runtime"]["bot"], plan["bundles"][role], plan, "bot")
    need(bot["running"] is True, "BOT_NOT_RUNNING")
    need(capture["guards"]["pollers"]["matching"] == 1, "POLLER_COUNT")
    return bot


def verify_operation(plan, operation, before, stopped, prepared, started):
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
        and prepare["id"] != (states[0]["runtime"]["prepare"] or {}).get("id")
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


def _new(before, after, table):
    old = e.index_rows(before[table], e.KEYS[table])
    now = e.index_rows(after[table], e.KEYS[table])
    need(all(now.get(key) == value for key, value in old.items()), "OLD_HISTORY_CHANGED")
    return [row for key, row in now.items() if key not in old]


def _actor_key(row, actor):
    parts = row["action_key"].split(":")
    need(
        len(parts) == 3
        and parts[0] == "msg"
        and parts[1] == str(actor)
        and parts[2].isdigit()
        and int(parts[2]) > 0,
        "INVALID_ACTION_KEY",
    )


def _preserved(before, after, actor=None, *, allow_pointer=False, allow_user=False):
    for table in e.LEGACY:
        need(
            collections.Counter(map(canonical, before[table]))
            == collections.Counter(map(canonical, after[table])),
            "LEGACY_CHANGED",
        )
    delta = {
        table: _new(before, after, table)
        for table in e.KEYS
        if table not in ("migrations", "anime_current", "users", "requests")
    }
    need(_new(before, after, "migrations") == [], "MIGRATION_CHANGED")
    old_requests = e.index_rows(before["requests"], ("id",))
    now_requests = e.index_rows(after["requests"], ("id",))
    pending = []
    for key, row in old_requests.items():
        later = now_requests.get(key)
        need(later is not None, "OLD_HISTORY_CHANGED")
        if later != row:
            need(
                row["seed_mal_id"] is None
                and later["seed_mal_id"] is not None
                and all(
                    row.get(k) == later.get(k)
                    for k in ("id", "tg_id", "action_key", "raw_query", "created_at")
                ),
                "OLD_REQUEST_CHANGED",
            )
            pending.append(later)
    delta["requests"] = [row for key, row in now_requests.items() if key not in old_requests]
    delta["resolved_pending"] = pending
    oldusers = e.index_rows(before["users"], ("tg_id",))
    nowusers = e.index_rows(after["users"], ("tg_id",))
    for key, row in oldusers.items():
        later = nowusers.get(key)
        need(
            later is not None
            and later["created_at"] == row["created_at"]
            and (
                later == row
                or allow_user
                and key == (actor,)
                and all(
                    later.get(field) == row.get(field)
                    for field in row
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
    for key, row in oldptr.items():
        later = newptr.get(key)
        need(
            later is not None
            and later["created_at"] == row["created_at"]
            and (
                later == row
                or allow_pointer
                and key[0] == actor
                and any((event["tg_id"], event["mal_id"]) == key for event in events)
                and later["current_event_id"] > row["current_event_id"]
                and e.utc(later["updated_at"]) >= e.utc(row["updated_at"])
            ),
            "CURRENT_POINTER",
        )
    need(all(key[0] == actor for key in newptr if key not in oldptr), "UNRELATED_WRITE")
    for table in ("requests", "delivered", "anime_events", "recommendation_ratings", "feedback"):
        need(all(row["tg_id"] == actor for row in delta[table]), "UNRELATED_WRITE")
    need(all(row["tg_id"] == actor for row in pending), "UNRELATED_WRITE")
    for row in delta["requests"] + pending + delta["feedback"]:
        _actor_key(row, actor)
    return delta


def _flow(before, after, role, plan, actor=None, *, query=None, seed=None):
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


def bind_context(plan, initial_operation, before, after):
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


def verify_case(plan, context, case, before, after, *, during=None, operation=None):
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
    _live_bot(before, bs, plan, prior_role)
    _live_bot(after, af, plan, role)
    need(e.utc(bs["captured_at"]) < e.utc(af["captured_at"]), "TIME_ORDER")
    if case in ("initial-update", "rollback", "final-update"):
        need(
            operation is not None
            and operation.get("operation") == case
            and operation.get("status") == "PASS"
            and operation.get("plan_sha256") == digest(plan)
            and operation["input_sha256"]["before"] == digest(before)
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
                probe = plan["ui_probes"]["description_edit"]
                need(
                    any(
                        row["tg_id"] == actor
                        and row["seed_mal_id"] == probe["seed_mal_id"]
                        and row["raw_query"] == probe["query"]
                        and row["bundle_id"] == plan["bundles"]["target"]["identity"]
                        and any(
                            d["request_id"] == row["id"]
                            and d["tg_id"] == actor
                            and d["chat_id"] == actor
                            and d["mal_id"] == probe["neighbor_mal_id"]
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
            bs["runtime"]["bot"]["id"] == af["runtime"]["bot"]["id"]
            and e.utc(bs["runtime"]["bot"]["started_at"])
            < e.utc(af["runtime"]["bot"]["started_at"]),
            "RESTART_NOT_OBSERVED",
        )
    if case in NO_EFFECT and case not in ("restart", "db-outage"):
        need(
            bs["runtime"]["bot"]["id"] == af["runtime"]["bot"]["id"]
            and bs["runtime"]["bot"]["started_at"] == af["runtime"]["bot"]["started_at"],
            "BOT_PROCESS_CHANGED",
        )
    if case in ("db-outage", "feedback-retry", "feedback-retry-repeat"):
        need(
            bs["runtime"]["bot"]["id"] == af["runtime"]["bot"]["id"]
            and bs["runtime"]["bot"]["started_at"] == af["runtime"]["bot"]["started_at"],
            "BOT_PROCESS_CHANGED",
        )
    if case == "db-outage":
        need(during is not None, "OUTAGE_NOT_OBSERVED")
        ds = _valid_capture(during, plan, "target", full=False)
        _live_bot(during, ds, plan, "target")
        need(
            e.utc(bs["captured_at"]) < e.utc(ds["captured_at"]) < e.utc(af["captured_at"])
            and ds["runtime"]["postgres"]["running"] is False
            and ds["runtime"]["bot"]["id"] == bs["runtime"]["bot"]["id"]
            and ds["runtime"]["bot"]["started_at"] == bs["runtime"]["bot"]["started_at"],
            "OUTAGE_NOT_OBSERVED",
        )
    counts = {
        key: len(delta[key])
        for key in ("requests", "delivered", "anime_events", "recommendation_ratings", "feedback")
    }
    counts["resolved_pending"] = len(delta["resolved_pending"])
    return _case_result(plan, context, case, role, before, after, during, operation, counts, "PASS")


def _case_result(plan, context, case, role, before, after, during, operation, counts, status):
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


def summarize_evidence(
    plan,
    operations,
    results,
    final,
    context=None,
    observations=None,
    operation_inputs=None,
    case_inputs=None,
):
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
            and "initial-update" in case_inputs
            and type(plan.get("baseline")) is dict,
            "CASE_INPUTS_REQUIRED",
        )
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
        initial, rollback, last = (operation_inputs[name] for name in OPERATIONS)
        started_ids = [
            item["started"]["state"]["runtime"]["bot"]["id"] for item in (initial, rollback, last)
        ]
        prepare_ids = [
            item["prepared"]["state"]["runtime"]["prepare"]["id"]
            for item in (initial, rollback, last)
        ]
        need(
            len(set(started_ids)) == 3
            and len(set(prepare_ids)) == 3
            and initial["started"]["state"]["runtime"]["bot"]["id"]
            == rollback["before"]["state"]["runtime"]["bot"]["id"]
            and rollback["started"]["state"]["runtime"]["bot"]["id"]
            == last["before"]["state"]["runtime"]["bot"]["id"],
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
                earlier["state"]["database"],
                later["state"]["database"],
                actor,
                allow_pointer=True,
                allow_user=True,
            )
        need(
            final_state["runtime"]["bot"] == last["started"]["state"]["runtime"]["bot"],
            "FINAL_RUNTIME_CHANGED",
        )
        _preserved(
            plan["baseline"]["state"]["database"],
            final_state["database"],
            actor,
            allow_pointer=True,
            allow_user=True,
        )
        history = "PASS"
    if results.get("catalog-stale", {}).get("status") == "PASS":
        need(
            case_inputs is not None and {"rollback", "catalog-stale"} <= set(case_inputs),
            "CASE_INPUTS_REQUIRED",
        )
        rollback_before = case_inputs["rollback"]["before"]["state"]["database"]
        rollback_after = case_inputs["rollback"]["after"]["state"]["database"]
        stale_before = case_inputs["catalog-stale"]["before"]["state"]["database"]
        actor = context["actor"]["tg_id"]
        old_keys = {row["id"] for row in rollback_before["requests"]}
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
            and observations.get("schema_version") == 2
            and observations.get("kind") == "arb-live-observations"
            and observations.get("plan_sha256") == digest(plan)
            and observations.get("context_sha256") == digest(context),
            "INVALID_OBSERVATIONS",
        )
        entries = observations.get("scenarios")
        need(
            type(entries) is list
            and len(entries) == len(UI_IDS)
            and {item["id"] for item in entries} == set(UI_IDS),
            "INVALID_OBSERVATIONS",
        )
        refs = {digest(value) for value in results.values()}
        for entry in entries:
            need(
                set(entry) == {"id", "status", "observed_at", "observation", "case_sha256"}
                and entry["status"] in ("PENDING", "PASS", "FAIL", "NOT_APPLICABLE")
                and type(entry["observation"]) is str
                and len(entry["observation"]) <= 240
                and type(entry["case_sha256"]) is list
                and len(set(entry["case_sha256"])) == len(entry["case_sha256"])
                and all(ref in refs for ref in entry["case_sha256"]),
                "INVALID_OBSERVATIONS",
            )
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
        if any(entry["status"] == "FAIL" for entry in entries):
            ui = "FAIL"
        elif all(entry["status"] in ("PASS", "NOT_APPLICABLE") for entry in entries):
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
