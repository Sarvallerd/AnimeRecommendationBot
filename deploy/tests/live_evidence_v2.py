#!/usr/bin/env python3
"""Private ARB-034 live evidence capture and dispatch. No service transitions."""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import os
import re
import stat
import subprocess
import sys
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from deploy.tests.acceptance_types import (
        AcceptanceProfile,
        BaselineCaptureView,
        BundleSummaryView,
        CatalogEntryView,
        CurrentAcceptanceProfile,
        JsonObject,
        LiveCaptureRecord,
        LiveGuardRecord,
        LiveVerifierModule,
        PollerInventoryRecord,
        PostgresIdentityRecord,
        ProfileExpectations,
        SettingsGuardRecord,
        UiProbesView,
    )
    from deploy.tests.live_verifier_v2 import (
        BundlePlan,
        CaseInputs,
        CaseReceipt,
        LiveContext,
        LivePlan,
        OperationInputs,
        OperationReceipt,
    )


def _load(name: str, filename: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load acceptance module {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v_module = _load("arb034_verifier", "live_verifier_v2.py")
if any(getattr(v_module, name, None) is None for name in ("e", "p", "r")) or any(
    not callable(getattr(v_module, name, None))
    for name in (
        "digest",
        "validate_plan",
        "_valid_capture",
        "verify_operation",
        "bind_context",
        "verify_case",
        "summarize_evidence",
    )
):
    raise ImportError("Live verifier dependencies unavailable")
v = cast("LiveVerifierModule", v_module)
loaded_e = v.e
if (
    not isinstance(getattr(loaded_e, "PROJECT", None), re.Pattern)
    or not isinstance(getattr(loaded_e, "SQL", None), Path)
    or not isinstance(getattr(loaded_e, "ROOT", None), Path)
    or not isinstance(getattr(loaded_e, "EvidenceError", None), type)
    or not issubclass(loaded_e.EvidenceError, Exception)
    or any(
        not callable(getattr(loaded_e, name, None))
        for name in (
            "require",
            "sha",
            "canonical",
            "read_settings",
            "container",
            "snapshot",
            "read_json",
            "write_new",
            "private_parent",
            "compose",
            "docker",
        )
    )
):
    raise ImportError("Live evidence dependency unavailable")
e = loaded_e
p, r = v.p, v.r
need, Error = e.require, e.EvidenceError
ROOT = Path(__file__).resolve().parents[2]


def command(
    argv: Sequence[str],
    *,
    timeout: int | float = 90,
    input_text: str | None = None,
    both: bool = False,
    env: Mapping[str, str] | None = None,
) -> str:
    try:
        result = subprocess.run(
            argv,
            cwd=ROOT,
            input=input_text,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        raise Error("TRANSPORT_TIMEOUT") from exc
    need(result.returncode == 0, "TRANSPORT_FAILED")
    return result.stdout + result.stderr if both else result.stdout


def _compose(project: str, settings: Path, *args: str, input_text: str | None = None) -> str:
    need(e.PROJECT.fullmatch(project) is not None, "INVALID_PROJECT")
    clean = {
        key: value for key, value in os.environ.items() if not key.startswith(("ARB_", "COMPOSE_"))
    }
    return command(
        [
            "docker",
            "compose",
            "--env-file",
            str(settings),
            "-p",
            project,
            "-f",
            str(ROOT / "compose.yaml"),
            *args,
        ],
        input_text=input_text,
        env=clean,
    )


def _docker(*args: str) -> str:
    return command(["docker", *args], both=bool(args and args[0] == "logs"))


@contextlib.contextmanager
def _bounded_transport() -> Iterator[None]:
    old_compose, old_docker = e.compose, e.docker
    e.compose, e.docker = _compose, _docker
    try:
        yield
    finally:
        e.compose, e.docker = old_compose, old_docker


def _regular_sha(path: Path) -> str:
    need(path.is_absolute() and not path.is_symlink() and path.is_file(), "UNSAFE_PRIVATE_FILE")
    need(stat.S_IMODE(path.stat().st_mode) & 0o077 == 0, "UNSAFE_PRIVATE_FILE")
    return e.sha(path.read_bytes())


def _settings_guard(path: Path) -> SettingsGuardRecord:
    values = e.read_settings(path)
    mode = path.stat()
    need(stat.S_IMODE(mode.st_mode) & 0o077 == 0, "UNSAFE_SETTINGS")
    credentials = {
        key: _regular_sha(Path(values[key])) for key in ("ARB_BOT_ENV", "ARB_POSTGRES_ENV")
    }
    return {
        "settings": {
            "sha256": e.sha(path.read_bytes()),
            "uid": mode.st_uid,
            "gid": mode.st_gid,
            "mode": stat.S_IMODE(mode.st_mode),
        },
        "credential_sha256": credentials,
    }


def _token(path: Path) -> str:
    raw = path.read_text()
    matches = [
        line.split("=", 1)[1] for line in raw.splitlines() if line.startswith("TELOXIDE_TOKEN=")
    ]
    need(len(matches) == 1 and bool(matches[0]), "TOKEN_UNAVAILABLE")
    return matches[0]


def _inventory(token: str) -> PollerInventoryRecord:
    identifiers = _docker("ps", "-q").split()
    matches = []
    docker_pids = set()
    for identifier in identifiers:
        data = json.loads(_docker("inspect", identifier))
        need(type(data) is list and len(data) == 1, "INVENTORY_INCOMPLETE")
        item = data[0]
        env = item.get("Config", {}).get("Env") or []
        if "TELOXIDE_TOKEN=" + token in env:
            matches.append(item["Id"])
            docker_pids.add(item.get("State", {}).get("Pid"))
    # A known local bot process whose environment cannot be inspected makes scope incomplete.
    local = 0
    proc = Path("/proc")
    complete = proc.is_dir()
    if proc.is_dir():
        for entry in proc.iterdir():
            if not entry.name.isdigit() or int(entry.name) in docker_pids:
                continue
            try:
                cmd = (entry / "cmdline").read_bytes()
            except (OSError, PermissionError):
                complete = False
                continue
            executable = cmd.split(b"\0", 1)[0]
            if executable not in (b"/usr/local/bin/bot", b"bot") and not executable.endswith(
                (b"/target/release/bot", b"/target/debug/bot")
            ):
                continue
            try:
                cgroup = (entry / "cgroup").read_bytes()
                if any(identifier[:12].encode() in cgroup for identifier in matches):
                    continue
                env = (entry / "environ").read_bytes()
                if b"TELOXIDE_TOKEN=" + token.encode() + b"\0" in env:
                    local += 1
            except (OSError, PermissionError):
                complete = False
    need(len(matches) + local <= 1, "SECOND_POLLER")
    return {
        "matching": len(matches) + local,
        "scope_complete": complete,
        "scope": "docker-daemon-and-known-local-processes",
        "matching_container_ids_sha256": [e.sha(item.encode()) for item in matches],
    }


def _guards(project: str, settings: Path) -> LiveGuardRecord:
    values = e.read_settings(settings)
    settings_guard = _settings_guard(settings)
    daemon_id = _docker("info", "--format", "{{.ID}}").strip()
    need(bool(daemon_id), "DAEMON_UNAVAILABLE")
    pollers = _inventory(_token(Path(values["ARB_BOT_ENV"])))
    pg = e.container(project, settings, "postgres")
    postgres_identity: PostgresIdentityRecord | None
    if pg is None:
        postgres_identity = None
    else:
        postgres_identity = {
            "id": pg["id"],
            "image_id": pg["image_id"],
            "database_volume": pg["database_volume"],
            "started_at": pg["started_at"],
        }
    postgres_health = (
        "absent"
        if pg is None
        else _docker("inspect", "--format", "{{.State.Health.Status}}", pg["id"]).strip()
    )
    return {
        **settings_guard,
        "daemon_id": daemon_id,
        "pollers": pollers,
        "postgres_identity": postgres_identity,
        "postgres_health": postgres_health,
    }


def capture(project: str, settings: Path, *, runtime_only: bool = False) -> LiveCaptureRecord:
    need(settings.is_absolute() and settings.is_file(), "UNSAFE_SETTINGS")
    with _bounded_transport():
        before = _guards(project, settings)
        state = e.snapshot(project, settings, runtime_only)
        after = _guards(project, settings)
    need(before == after, "UNSTABLE_CAPTURE")
    after["stable"] = True
    return {"schema_version": 2, "kind": "arb-live-capture", "state": state, "guards": after}


def _read(path: Path) -> JsonObject:
    need(path.is_absolute(), "ABSOLUTE_PATH_REQUIRED")
    e.private_parent(path)
    need(stat.S_IMODE(path.stat().st_mode) & 0o077 == 0, "UNSAFE_PRIVATE_FILE")
    return e.read_json(path)


def _save(path: Path, value: object) -> str:
    need(path.is_absolute(), "ABSOLUTE_PATH_REQUIRED")
    return e.write_new(path, value)


def _bytes_new(path: Path, raw: bytes, original: Path) -> None:
    e.private_parent(path)
    need(not path.exists() and not path.is_symlink(), "OUTPUT_EXISTS")
    fd = os.open(
        path,
        os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW,
        stat.S_IMODE(original.stat().st_mode),
    )
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.chown(path, original.stat().st_uid, original.stat().st_gid)


def _catalog(path: Path) -> dict[str, CatalogEntryView]:
    raw = p.canonical((path / "catalog.json").read_bytes())
    anime = raw.get("anime") if isinstance(raw, dict) else None
    need(
        isinstance(anime, dict)
        and all(
            isinstance(mal_id, str)
            and isinstance(entry, dict)
            and ("title" not in entry or isinstance(entry["title"], str))
            and ("synopsis" not in entry or isinstance(entry["synopsis"], str))
            and (
                "aliases" not in entry
                or isinstance(entry["aliases"], list)
                and all(isinstance(alias, str) for alias in entry["aliases"])
            )
            for mal_id, entry in anime.items()
        ),
        "INVALID_UI_PROBES",
    )
    return cast("dict[str, CatalogEntryView]", anime)


def _checked_baseline(raw: JsonObject) -> BaselineCaptureView:
    state = raw.get("state")
    if raw.get("kind") != "arb-live-capture" or not isinstance(state, dict):
        raise Error("BASELINE_MISMATCH")
    runtime = state.get("runtime")
    guards = raw.get("guards")
    if state.get("kind") != "full" or not isinstance(runtime, dict) or not isinstance(guards, dict):
        raise Error("BASELINE_MISMATCH")
    bot, postgres = runtime.get("bot"), runtime.get("postgres")
    pollers = guards.get("pollers")
    if (
        not isinstance(state.get("source_commit"), str)
        or not isinstance(bot, dict)
        or not isinstance(postgres, dict)
        or not isinstance(pollers, dict)
    ):
        raise Error("BASELINE_MISMATCH")
    mount = bot.get("artifact_mount")
    matching_ids = pollers.get("matching_container_ids_sha256")
    if not (
        isinstance(bot.get("id"), str)
        and isinstance(bot.get("image_id"), str)
        and (bot.get("bundle_id") is None or isinstance(bot.get("bundle_id"), str))
        and (bot.get("bundle_count") is None or type(bot.get("bundle_count")) is int)
        and (
            mount is None
            or isinstance(mount, dict)
            and isinstance(mount.get("source"), str)
            and type(mount.get("rw")) is bool
        )
        and isinstance(postgres.get("id"), str)
        and isinstance(postgres.get("image_id"), str)
        and type(postgres.get("running")) is bool
        and (
            postgres.get("database_volume") is None
            or isinstance(postgres.get("database_volume"), str)
        )
        and isinstance(guards.get("daemon_id"), str)
        and isinstance(guards.get("postgres_health"), str)
        and type(pollers.get("matching")) is int
        and type(pollers.get("scope_complete")) is bool
        and isinstance(matching_ids, list)
        and all(isinstance(value, str) for value in matching_ids)
    ):
        raise Error("BASELINE_MISMATCH")
    return cast("BaselineCaptureView", raw)


def _checked_ui_probes(raw: JsonObject) -> UiProbesView:
    if not (
        raw.get("schema_version") == 1
        and raw.get("kind") == "arb-live-ui-probes"
        and set(raw) == {"schema_version", "kind", "alias_search", "description_edit"}
    ):
        raise Error("INVALID_UI_PROBES")
    alias, description = raw["alias_search"], raw["description_edit"]
    if not (
        isinstance(alias, dict)
        and set(alias) == {"mal_id", "query"}
        and type(alias.get("mal_id")) is int
        and isinstance(alias.get("query"), str)
        and isinstance(description, dict)
        and set(description) == {"seed_mal_id", "query", "neighbor_mal_id"}
        and type(description.get("seed_mal_id")) is int
        and isinstance(description.get("query"), str)
        and type(description.get("neighbor_mal_id")) is int
    ):
        raise Error("INVALID_UI_PROBES")
    return cast("UiProbesView", raw)


def make_plan(
    project: str,
    settings: Path,
    offline_evidence: Path,
    baseline_path: Path,
    target_bundle: Path,
    ui_probes_path: Path,
    output: Path,
) -> LivePlan:
    need(
        all(
            path.is_absolute()
            for path in (
                settings,
                offline_evidence,
                baseline_path,
                target_bundle,
                ui_probes_path,
                output,
            )
        ),
        "ABSOLUTE_PATH_REQUIRED",
    )
    report = r.verify_record(offline_evidence)
    report_sha = _regular_sha(offline_evidence / "preparation-report.json")
    baseline = _checked_baseline(_read(baseline_path))
    original_settings = e.read_settings(settings)
    original_path = Path(original_settings["ARB_BUNDLE_DIR"])
    old_profile = p.load_profile(p.LEGACY_RELEASE)
    target_profile = cast("CurrentAcceptanceProfile", p.load_profile(p.NEW_RELEASE))
    old = p.check_profile_bundle(
        original_path, old_profile, expected_identity=report["original"]["bundle_identity"]
    )
    target = p.check_profile_bundle(
        target_bundle, target_profile, expected_identity=report["builds"]["A"]["identity"]
    )
    need(
        old.hashes == report["original"]["bundle_file_sha256"]
        and target.hashes == report["builds"]["A"]["file_sha256"]
        and target.identity == target_profile.reference["bundle_identity"]
        and project != report["runtime"]["project"],
        "OFFLINE_MISMATCH",
    )
    probes = _checked_ui_probes(_read(ui_probes_path))
    catalog = _catalog(target_bundle)
    alias = probes["alias_search"]
    description = probes["description_edit"]
    entry = catalog.get(str(alias["mal_id"]), {})
    need(
        alias["query"] in entry.get("aliases", [])
        and alias["query"] != entry.get("title")
        and str(description["seed_mal_id"]) in catalog
        and description["neighbor_mal_id"]
        in [item["mal_id"] for item in target.neighbors[str(description["seed_mal_id"])]]
        and bool(catalog[str(description["neighbor_mal_id"])].get("synopsis")),
        "INVALID_UI_PROBES",
    )
    verifier_path = getattr(v_module, "__file__", None)
    if not isinstance(verifier_path, str):
        raise Error("VERIFIER_CHANGED")
    source = {
        "commit": baseline["state"]["source_commit"],
        "compose_sha256": e.sha((ROOT / "compose.yaml").read_bytes()),
        "snapshot_sql_sha256": e.sha(e.SQL.read_bytes()),
        "verifier_sha256": e.sha(Path(verifier_path).read_bytes()),
        "helper_sha256": e.sha(Path(__file__).read_bytes()),
        "dispatch_sha256": e.sha(Path(__file__).with_name("acceptance-evidence.py").read_bytes()),
    }
    guard = _settings_guard(settings)
    selected = baseline["state"]["runtime"]["bot"]
    pg = baseline["state"]["runtime"]["postgres"]
    need(
        selected is not None
        and pg is not None
        and pg["running"]
        and baseline["guards"]["postgres_health"] == "healthy"
        and baseline["guards"]["pollers"]["matching"] == 1
        and baseline["guards"]["pollers"].get("matching_container_ids_sha256")
        == [e.sha(selected["id"].encode())]
        and baseline["guards"]["pollers"]["scope_complete"] is True
        and selected["bundle_id"] == old.identity
        and selected["bundle_count"] == old.count
        and selected["artifact_mount"] == {"source": str(original_path), "rw": False},
        "BASELINE_MISMATCH",
    )
    if selected is None or pg is None:
        raise Error("BASELINE_MISMATCH")
    target_raw = r.change_selection(settings.read_bytes(), target_bundle)
    need(isinstance(target_raw, bytes), "OFFLINE_MISMATCH")
    old_copy = output.parent / "original-settings.env"
    target_copy = output.parent / "target-settings.env"
    need(all(not path.exists() for path in (old_copy, target_copy, output)), "OUTPUT_EXISTS")

    def bundle(
        role: str,
        summary: BundleSummaryView,
        profile: AcceptanceProfile | CurrentAcceptanceProfile,
        path: Path,
        image: str,
    ) -> BundlePlan:
        return {
            "role": role,
            "identity": summary.identity,
            "path": str(path),
            "release_id": profile.release_id,
            "file_sha256": summary.hashes,
            "count": summary.count,
            "histogram": summary.histogram,
            "expectations": cast("ProfileExpectations", profile.expectations),
            "image_id": image,
        }

    plan = {
        "schema_version": 2,
        "kind": "arb-live-plan",
        "offline": {
            "evidence_dir": str(offline_evidence),
            "report_sha256": report_sha,
            "project": report["runtime"]["project"],
            "database_volume": report["runtime"]["database_volume"],
        },
        "profile": {
            "id": target_profile.id,
            "release_id": target_profile.release_id,
            "sha256": target_profile.sha256,
            "lock_sha256": target_profile.lock_sha256,
        },
        "source": source,
        "service": {
            "project": project,
            "daemon_id": baseline["guards"]["daemon_id"],
            "postgres_id": pg["id"],
            "postgres_image_id": pg["image_id"],
            "database_volume": pg["database_volume"],
        },
        "settings": {
            **guard["settings"],
            "credential_sha256": guard["credential_sha256"],
            "original_copy": str(old_copy),
            "target_copy": str(target_copy),
        },
        "expected_settings_sha256": {
            "original": e.sha(settings.read_bytes()),
            "target": e.sha(target_raw),
        },
        "bundles": {
            "original": bundle("original", old, old_profile, original_path, selected["image_id"]),
            "target": bundle(
                "target", target, target_profile, target_bundle, report["runtime"]["bot_image_id"]
            ),
        },
        "ui_probes": probes,
        "baseline_sha256": v.digest(baseline),
        "baseline": baseline,
    }
    _bytes_new(old_copy, settings.read_bytes(), settings)
    _bytes_new(target_copy, target_raw, settings)
    checked_plan = v.validate_plan(plan, verify_offline=False)
    _save(output, plan)
    return checked_plan


def _cache(directory: Path, value: object) -> str:
    digest = v.digest(value)
    if not isinstance(digest, str):
        raise Error("INPUT_HASH_MISMATCH")
    path = directory / (digest + ".json")
    if path.exists():
        need(not path.is_symlink() and _read(path) == value, "INPUT_HASH_MISMATCH")
    else:
        _save(path, value)
    return digest


def _cached(directory: Path, digest: object) -> JsonObject:
    if type(digest) is not str:
        raise Error("INPUT_HASH_MISMATCH")
    need(e.HEX.fullmatch(digest) is not None, "INPUT_HASH_MISMATCH")
    item = _read(directory / (digest + ".json"))
    need(v.digest(item) == digest, "INPUT_HASH_MISMATCH")
    return item


FAILURES = (
    e.EvidenceError,
    r.ReleaseError,
    p.ProfileError,
    OSError,
    ValueError,
    TypeError,
    KeyError,
    IndexError,
)


def _code(exc: BaseException) -> str:
    code = getattr(exc, "code", None)
    return code if isinstance(code, str) else "INVALID_EVIDENCE"


def _failed_operation(
    plan: LivePlan,
    operation: str,
    inputs: Mapping[str, LiveCaptureRecord],
    code: str,
) -> OperationReceipt:
    source, target = ("target", "original") if operation == "rollback" else ("original", "target")
    return {
        "schema_version": 2,
        "kind": "arb-live-operation",
        "plan_sha256": v.digest(plan),
        "operation": operation,
        "from": source,
        "to": target,
        "status": "FAIL",
        "reason": code,
        "counts": [],
        "started_capture_time": None,
        "input_sha256": {key: v.digest(value) for key, value in inputs.items()},
    }


def _failed_case(
    plan: LivePlan,
    context: LiveContext,
    case: str,
    inputs: Mapping[str, LiveCaptureRecord],
    operation: OperationReceipt | None,
    code: str,
) -> CaseReceipt:
    role = "original" if case == "rollback" else "target"
    return {
        "schema_version": 2,
        "kind": "arb-live-case",
        "plan_sha256": v.digest(plan),
        "context_sha256": v.digest(context),
        "case": case,
        "status": "FAIL",
        "reason": code,
        "bundle_role": role,
        "bundle_identity": plan["bundles"][role]["identity"],
        "counts": {},
        "input_sha256": {key: v.digest(value) for key, value in inputs.items()},
        "operation_sha256": v.digest(operation) if operation else None,
    }


def _verify_saved_operation(plan: LivePlan, item: JsonObject, directory: Path) -> OperationInputs:
    need(
        item.get("kind") == "arb-live-operation" and item.get("plan_sha256") == v.digest(plan),
        "INVALID_OPERATION",
    )
    receipt = cast("OperationReceipt", item)
    inputs = {
        key: cast("LiveCaptureRecord", _cached(directory, value))
        for key, value in receipt["input_sha256"].items()
    }
    need(set(inputs) == {"before", "stopped", "prepared", "started"}, "INVALID_OPERATION")
    if item.get("status") == "FAIL":
        try:
            v.verify_operation(plan, receipt["operation"], **inputs)
        except FAILURES as exc:
            need(
                item == _failed_operation(plan, receipt["operation"], inputs, _code(exc)),
                "FORGED_OPERATION",
            )
        else:
            need(False, "FORGED_OPERATION")
    else:
        need(v.verify_operation(plan, receipt["operation"], **inputs) == item, "FORGED_OPERATION")
    return cast("OperationInputs", inputs)


def _verify_saved_case(
    plan: LivePlan,
    context: LiveContext,
    item: JsonObject,
    directory: Path,
    operations: Mapping[str, OperationReceipt],
) -> CaseInputs:
    need(
        item.get("kind") == "arb-live-case" and item.get("context_sha256") == v.digest(context),
        "INVALID_CASE",
    )
    receipt = cast("CaseReceipt", item)
    inputs = {
        key: cast("LiveCaptureRecord", _cached(directory, value))
        for key, value in receipt["input_sha256"].items()
    }
    op = operations.get("final-update" if receipt["case"] == "catalog-stale" else receipt["case"])
    if item.get("status") == "FAIL":
        try:
            v.verify_case(
                plan,
                context,
                receipt["case"],
                inputs["before"],
                inputs["after"],
                during=inputs.get("during"),
                operation=op,
            )
        except FAILURES as exc:
            need(
                item == _failed_case(plan, context, receipt["case"], inputs, op, _code(exc)),
                "FORGED_CASE",
            )
        else:
            need(False, "FORGED_CASE")
    else:
        expected = v.verify_case(
            plan,
            context,
            receipt["case"],
            inputs["before"],
            inputs["after"],
            during=inputs.get("during"),
            operation=op,
        )
        need(expected == item, "FORGED_CASE")
    return cast("CaseInputs", inputs)


def parser() -> argparse.ArgumentParser:
    top = argparse.ArgumentParser(description=__doc__)
    commands = top.add_subparsers(dest="command", required=True)
    cap = commands.add_parser("capture")
    for arg in ("project", "settings", "output"):
        cap.add_argument("--" + arg, required=True, type=Path if arg != "project" else str)
    cap.add_argument("--plan", type=Path)
    cap.add_argument("--runtime-only", action="store_true")
    plan = commands.add_parser("plan")
    for arg in (
        "project",
        "settings",
        "offline-evidence",
        "baseline",
        "target-bundle",
        "ui-probes",
        "output",
    ):
        plan.add_argument("--" + arg, required=True, type=str if arg == "project" else Path)
    op = commands.add_parser("verify-operation")
    op.add_argument("--operation", choices=v.OPERATIONS, required=True)
    for arg in ("plan", "before", "stopped", "prepared", "started", "output"):
        op.add_argument("--" + arg, type=Path, required=True)
    bind = commands.add_parser("bind-actor")
    for arg in ("plan", "initial-operation", "after", "output"):
        bind.add_argument("--" + arg, type=Path, required=True)
    case = commands.add_parser("verify")
    case.add_argument("--case", choices=v.CASES, required=True)
    for arg in ("context", "before", "after", "output"):
        case.add_argument("--" + arg, type=Path, required=True)
    case.add_argument("--during", type=Path)
    case.add_argument("--operation", type=Path)
    summary = commands.add_parser("summarize")
    for arg in ("plan", "operations-dir", "results-dir", "final", "output", "public-output"):
        summary.add_argument("--" + arg, type=Path, required=True)
    summary.add_argument("--context", type=Path)
    summary.add_argument("--observations", type=Path)
    return top


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "capture":
            if args.plan:
                plan = v.validate_plan(_read(args.plan), verify_offline=False)
                need(args.project == plan["service"]["project"], "WRONG_PROJECT")
            value = capture(args.project, args.settings, runtime_only=args.runtime_only)
            if args.plan:
                v._valid_capture(value, plan, full=not args.runtime_only)
            _save(args.output, value)
        elif args.command == "plan":
            make_plan(
                args.project,
                args.settings,
                args.offline_evidence,
                args.baseline,
                args.target_bundle,
                args.ui_probes,
                args.output,
            )
        elif args.command == "verify-operation":
            plan = v.validate_plan(_read(args.plan), verify_offline=False)
            values: dict[str, LiveCaptureRecord] = {
                key: cast("LiveCaptureRecord", _read(getattr(args, key)))
                for key in ("before", "stopped", "prepared", "started")
            }
            try:
                result = v.verify_operation(plan, args.operation, **values)
            except FAILURES as exc:
                result = _failed_operation(plan, args.operation, values, _code(exc))
            directory = args.output.parent / "inputs"
            need(directory.is_dir(), "PRIVATE_DIRECTORY_REQUIRED")
            for capture_value in values.values():
                _cache(directory, capture_value)
            _save(args.output, result)
            if result["status"] == "FAIL":
                print("FAIL " + result["reason"], file=sys.stderr)
                return 1
        elif args.command == "bind-actor":
            plan = v.validate_plan(_read(args.plan), verify_offline=False)
            operation_raw = _read(args.initial_operation)
            directory = args.initial_operation.parent / "inputs"
            bound_inputs = _verify_saved_operation(plan, operation_raw, directory)
            initial_operation = cast("OperationReceipt", operation_raw)
            after = cast("LiveCaptureRecord", _read(args.after))
            bound_context = v.bind_context(plan, initial_operation, bound_inputs["started"], after)
            cast("dict[str, object]", bound_context)["plan_path"] = str(args.plan)
            context_inputs = args.output.parent / "inputs"
            need(context_inputs.is_dir(), "PRIVATE_DIRECTORY_REQUIRED")
            _cache(context_inputs, bound_inputs["started"])
            _cache(context_inputs, after)
            _save(args.output, bound_context)
        elif args.command == "verify":
            context_raw = _read(args.context)
            context_path = context_raw.get("plan_path")
            if not isinstance(context_path, str):
                raise TypeError("plan_path must be a string")
            plan_path = Path(context_path)
            plan = v.validate_plan(_read(plan_path), verify_offline=False)
            case_context = cast("LiveContext", context_raw)
            case_inputs_for_verify: dict[str, LiveCaptureRecord] = {
                "before": cast("LiveCaptureRecord", _read(args.before)),
                "after": cast("LiveCaptureRecord", _read(args.after)),
            }
            if args.during:
                case_inputs_for_verify["during"] = cast("LiveCaptureRecord", _read(args.during))
            case_operation_raw = _read(args.operation) if args.operation else None
            if case_operation_raw:
                _verify_saved_operation(plan, case_operation_raw, args.operation.parent / "inputs")
            case_operation = cast("OperationReceipt | None", case_operation_raw)
            try:
                case_result = v.verify_case(
                    plan,
                    case_context,
                    args.case,
                    case_inputs_for_verify["before"],
                    case_inputs_for_verify["after"],
                    during=case_inputs_for_verify.get("during"),
                    operation=case_operation,
                )
            except FAILURES as exc:
                case_result = _failed_case(
                    plan,
                    case_context,
                    args.case,
                    case_inputs_for_verify,
                    case_operation,
                    _code(exc),
                )
            directory = args.output.parent / "inputs"
            need(directory.is_dir(), "PRIVATE_DIRECTORY_REQUIRED")
            for capture_value in case_inputs_for_verify.values():
                _cache(directory, capture_value)
            _save(args.output, case_result)
            if case_result["status"] == "FAIL":
                print("FAIL " + case_result["reason"], file=sys.stderr)
                return 1
        elif args.command == "summarize":
            plan = v.validate_plan(_read(args.plan))
            summary_context_raw = _read(args.context) if args.context else None
            summary_context = cast("LiveContext | None", summary_context_raw)
            observations = _read(args.observations) if args.observations else None
            operations: dict[str, OperationReceipt] = {}
            operation_inputs: dict[str, OperationInputs] = {}
            for path in args.operations_dir.glob("*.json"):
                item = _read(path)
                need(item["operation"] not in operations, "DUPLICATE_OPERATION")
                operation_key = cast("str", item["operation"])
                operation_inputs[operation_key] = _verify_saved_operation(
                    plan, item, args.operations_dir / "inputs"
                )
                operations[operation_key] = cast("OperationReceipt", item)
            if summary_context is not None:
                assert summary_context_raw is not None
                need(
                    summary_context_raw.get("plan_path") == str(args.plan)
                    and summary_context.get("initial_operation_sha256")
                    == v.digest(operations["initial-update"]),
                    "INVALID_CONTEXT",
                )
                anchors = {
                    key: cast("LiveCaptureRecord", _cached(args.context.parent / "inputs", value))
                    for key, value in summary_context["anchor_input_sha256"].items()
                }
                rebound = v.bind_context(
                    plan, operations["initial-update"], anchors["before"], anchors["after"]
                )
                cast("dict[str, object]", rebound)["plan_path"] = str(args.plan)
                need(rebound == summary_context, "FORGED_CONTEXT")
            results: dict[str, CaseReceipt] = {}
            case_inputs: dict[str, CaseInputs] = {}
            for path in args.results_dir.glob("*.json"):
                item = _read(path)
                need(item["case"] not in results and summary_context is not None, "DUPLICATE_CASE")
                if summary_context is None:
                    raise Error("DUPLICATE_CASE")
                case_key = cast("str", item["case"])
                case_inputs[case_key] = _verify_saved_case(
                    plan, summary_context, item, args.results_dir / "inputs", operations
                )
                results[case_key] = cast("CaseReceipt", item)
            final = cast("LiveCaptureRecord", _read(args.final))
            summary_value = v.summarize_evidence(
                plan,
                operations,
                results,
                final,
                summary_context,
                observations,
                operation_inputs,
                case_inputs,
            )
            _save(args.output, summary_value)
            summary_fields = cast("Mapping[str, object]", summary_value)
            public = {
                key: summary_fields[key]
                for key in (
                    "schema_version",
                    "kind",
                    "offline_compatibility",
                    "operations",
                    "production_history_preservation",
                    "machine",
                    "live_ui_acceptance",
                    "live_acceptance",
                    "selected",
                    "poller_scope",
                    "limits",
                )
            }
            _save(args.public_output, public)
            print(summary_value["live_acceptance"])
            return 0 if summary_value["live_acceptance"] == "PASS" else 1
        print("PASS")
        return 0
    except (
        Error,
        r.ReleaseError,
        p.ProfileError,
        OSError,
        ValueError,
        TypeError,
        KeyError,
        IndexError,
        subprocess.SubprocessError,
    ) as exc:
        code = _code(exc)
        print("FAIL " + code, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
