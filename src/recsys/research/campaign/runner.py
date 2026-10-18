"""Fixed, supervised and resumable validation-only ML-02 campaign.

The worker entry uses only the standard library until the generation gate has
been acquired and checked. All expensive scientific work runs in workers.
"""

from __future__ import annotations

import fcntl
import hashlib
import os
import sys
import time
import uuid
from pathlib import Path
from typing import NotRequired, TypedDict, cast

from recsys.research.campaign.artifacts import _phase_directory, _phase_path, _verify_phase
from recsys.research.campaign.common import (
    _SAFE_CODE,
    EVALUATION_SECONDS,
    FIT_SECONDS,
    PHASES,
    RSS_LIMIT_BYTES,
    VARIANTS,
    _canonical,
    _digest,
    _ensure_private,
    _fail,
    _now,
    _parse_iso,
    _private_file,
    _read_private,
    _replace_private,
    _sync_dir,
    _touch_lock,
)
from recsys.research.campaign.journal import CampaignEvent, PhaseKey, _Journal, _successes
from recsys.research.campaign.plan import (
    CampaignPlan,
    _expected_nodes,
    _plan,
    _validate_plan_structure,
)
from recsys.research.campaign.report import _native_public_allowed, _public_projection
from recsys.research.campaign.spec import (
    LEGACY_CAMPAIGN,
    NATIVE_CAMPAIGN,
    NATIVE_VARIANT,
    CampaignName,
    NativeTrainDevice,
    campaign_variants,
)
from recsys.research.campaign.storage import (
    STOP_BYTES,
    admission_bytes,
    check_binding,
    require_storage,
)
from recsys.research.campaign.worker import _worker_request
from recsys.research.models.spec import require_cpu_backend
from recsys.research.resources import JobHandle, ProcessResult


class ActivePhase(TypedDict):
    variant: str
    phase: str
    attempt: int
    generation: str
    observed_elapsed: float
    warned: bool
    result: NotRequired[ProcessResult]


def _worker_env() -> dict[str, str]:
    environment = dict(os.environ)
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        environment[name] = "1"
    return environment


def _native_worker_env(*, cuda_fit: bool = False) -> dict[str, str]:
    environment = _worker_env()
    environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    if not cuda_fit:
        environment["CUDA_VISIBLE_DEVICES"] = ""
    return environment


def _process_fields(result: ProcessResult) -> dict[str, object]:
    return {
        "outcome": result.outcome,
        "returncode": result.returncode,
        "elapsed_seconds": result.elapsed_seconds,
        "sampled_peak_rss_bytes": result.sampled_peak_rss_bytes,
        "wait4_peak_rss_bytes": result.wait4_peak_rss_bytes,
        "observed_peak_rss_bytes": result.observed_peak_rss_bytes,
        "rss_samples": result.rss_samples,
        "max_sample_gap_seconds": result.max_sample_gap_seconds,
        "cleanup_complete": result.cleanup_complete,
    }


def _next_ready(
    successes: dict[PhaseKey, CampaignEvent],
    failed: set[PhaseKey],
    active_keys: set[PhaseKey],
    cpu_used: int,
    cpu_workers: int,
    variants: tuple[str, ...] = VARIANTS,
) -> tuple[str, str] | None:
    for variant in variants:
        for phase in PHASES:
            key = (variant, phase)
            if key in successes or key in failed or key in active_keys:
                continue
            index = PHASES.index(phase)
            if index and (variant, PHASES[index - 1]) not in successes:
                continue
            if (
                phase == "fit"
                and variant == "hybrid"
                and any(
                    (dependency, "fit") not in successes
                    for dependency in ("content-mean", "puresvd")
                )
            ):
                continue
            if cpu_used >= cpu_workers:
                continue
            return variant, phase
    return None


def _phase_budget_remaining(charged: dict[PhaseKey, float], variant: str, phase: str) -> float:
    if phase not in PHASES:
        _fail("research_campaign_invalid")
    budget = FIT_SECONDS if phase == "fit" else EVALUATION_SECONDS
    used = sum(
        charged.get((variant, step), 0.0) for step in (("fit",) if phase == "fit" else PHASES[1:])
    )
    return budget - used


def _remaining(
    plan: CampaignPlan, charged: dict[PhaseKey, float], variant: str, phase: str
) -> float:
    until = (_parse_iso(plan["deadline_utc"]) - _now()).total_seconds()
    return min(_phase_budget_remaining(charged, variant, phase), until)


def _prepare_attempt(
    root: Path,
    plan: CampaignPlan,
    plan_identity: str,
    generation: str,
    successes: dict[PhaseKey, CampaignEvent],
    attempts: dict[PhaseKey, int],
    variant: str,
    phase: str,
    prepared_dir: Path,
    glove: Path,
    campaign_name: CampaignName = LEGACY_CAMPAIGN,
) -> tuple[Path, int]:
    number = attempts.get((variant, phase), 0) + 1
    directory = _phase_directory(root, variant, phase, number)
    _ensure_private(directory)
    _ensure_private(directory / "resources")
    model = _phase_path(root, successes, variant, "fit")
    predictions = _phase_path(root, successes, variant, "predict")
    content = _phase_path(root, successes, "content-mean", "fit")
    svd = _phase_path(root, successes, "puresvd", "fit")
    request = _worker_request(
        root,
        plan_identity,
        generation,
        variant,
        phase,
        number,
        prepared_dir,
        glove,
        model,
        predictions,
        content,
        svd,
        campaign_name=campaign_name,
    )
    _private_file(directory / "request.json", _canonical(request))
    return directory, number


def run_campaign(
    prepared_dir: str | Path,
    output_dir: str | Path,
    *,
    expected_prepared_identity: str,
    code_revision: str,
    glove: Path,
    backend: str = "cpu",
    cpu_workers: int = 2,
    resume: bool = False,
    expected_campaign_identity: str | None = None,
    campaign_name: CampaignName = LEGACY_CAMPAIGN,
    native_train_device: NativeTrainDevice | None = None,
    physical_storage_root: Path | None = None,
) -> Path:
    """Run exactly seven fixed validation variants with durable, bounded workers."""
    if campaign_name == NATIVE_CAMPAIGN:
        return _run_native_campaign(
            prepared_dir,
            output_dir,
            expected_prepared_identity=expected_prepared_identity,
            code_revision=code_revision,
            glove=glove,
            backend=backend,
            cpu_workers=cpu_workers,
            resume=resume,
            expected_campaign_identity=expected_campaign_identity,
            native_train_device=native_train_device,
            physical_storage_root=physical_storage_root,
        )
    if (
        campaign_name != LEGACY_CAMPAIGN
        or native_train_device is not None
        or physical_storage_root is not None
    ):
        _fail("research_campaign_profile_unsupported")
    require_cpu_backend(backend)
    from recsys.research.artifacts import safe_path, separate
    from recsys.research.resources import ResourceSupervisor

    prepared_dir = Path(prepared_dir).absolute()
    glove = Path(glove).absolute()
    root = Path(output_dir).absolute()
    separate(prepared_dir, glove, root)
    safe_path(prepared_dir, directory=True)
    safe_path(glove, directory=False)
    if (
        type(resume) is not bool
        or (resume and expected_campaign_identity is None)
        or (not resume and expected_campaign_identity is not None)
    ):
        _fail()
    if resume:
        if root.is_symlink() or not root.is_dir() or root.stat().st_mode & 0o077:
            _fail()
    else:
        if root.exists() or root.is_symlink():
            _fail()
        root.mkdir(mode=0o700)
        _sync_dir(root.parent)
    lock = _touch_lock(root / ".campaign.lock")
    gate = None
    try:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            _fail("research_campaign_busy")
        if resume:
            prior_value, raw = _read_private(root / "campaign-plan.json")
            _validate_plan_structure(prior_value)
            prior = cast("CampaignPlan", prior_value)
            created = _parse_iso(prior["created_utc"])
            planned = _plan(
                prepared_dir,
                expected_prepared_identity,
                code_revision,
                glove,
                cpu_workers,
                created,
            )
            if prior != planned or _digest(raw) != expected_campaign_identity:
                _fail("research_campaign_plan_mismatch")
            plan, plan_identity = prior, _digest(raw)
        else:
            created = _now()
            plan = _plan(
                prepared_dir,
                expected_prepared_identity,
                code_revision,
                glove,
                cpu_workers,
                created,
            )
            raw = _canonical(plan)
            _private_file(root / "campaign-plan.json", raw)
            _sync_dir(root)
            plan_identity = _digest(raw)
        _validate_plan_structure(plan)
        if _now() > _parse_iso(plan["deadline_utc"]):
            _fail("research_campaign_wall_limit")
        gate = _touch_lock(root / ".workers.lock")
        try:
            fcntl.flock(gate, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            _fail("research_campaign_workers_active")
        journal = _Journal(root / "journal.jsonl")
        successes, attempts, charged, starts, failed = _successes(journal.events)
        if starts:
            # A crashed coordinator cannot certify that a native descendant was
            # reaped. Preserve every orphan attempt and fail closed.
            _fail("research_campaign_interrupted_uncertain")
        for (variant, phase), event in successes.items():
            # A completed stage is reusable only after all current source/input
            # receipts and every payload/cross-link have been checked.
            identity, _ = _verify_phase(
                root,
                plan,
                successes,
                variant,
                phase,
                event["attempt"],
                event["generation"],
                prepared_dir,
            )
            if identity != event.get("output_identity"):
                _fail("research_campaign_chain_invalid")
        generation = uuid.uuid4().hex
        _replace_private(root / "generation.json", _canonical({"generation": generation}))
        fcntl.flock(gate, fcntl.LOCK_UN)
        journal.append(
            "run_started", generation=generation, plan_identity=plan_identity, resumed=resume
        )
        active: dict[JobHandle, ActivePhase] = {}
        pending: list[ActivePhase] = []
        frozen = bool(failed)
        failure_code = "research_campaign_previous_failure" if failed else None
        supervisor = ResourceSupervisor(rss_limit_bytes=RSS_LIMIT_BYTES)
        original_error: BaseException | None = None
        try:
            supervisor.__enter__()
            while True:
                now = _now()
                if now >= _parse_iso(plan["deadline_utc"]) and not frozen:
                    frozen, failure_code = True, "research_campaign_wall_limit"
                    journal.append("campaign_failure", code=failure_code)
                    for handle in active:
                        supervisor.cancel(handle)
                if not frozen:
                    # Start independent ready phases while slots are free. Heavy
                    # artifact validation is deferred until no child is active.
                    while True:
                        cpu_used = len(active)
                        choice = _next_ready(
                            successes,
                            failed,
                            {(item["variant"], item["phase"]) for item in active.values()}
                            | {(item["variant"], item["phase"]) for item in pending},
                            cpu_used,
                            cpu_workers,
                        )
                        if choice is None:
                            break
                        variant, phase = choice
                        remaining = _remaining(plan, charged, variant, phase)
                        if remaining <= 0:
                            frozen, failure_code = True, "research_campaign_budget_limit"
                            journal.append("campaign_failure", code=failure_code)
                            break
                        directory, number = _prepare_attempt(
                            root,
                            plan,
                            plan_identity,
                            generation,
                            successes,
                            attempts,
                            variant,
                            phase,
                            prepared_dir,
                            glove,
                        )
                        attempts[(variant, phase)] = number
                        journal.append(
                            "attempt_started",
                            generation=generation,
                            variant=variant,
                            phase=phase,
                            attempt=number,
                            remaining_seconds=remaining,
                        )
                        # Preserve the virtualenv executable symlink: resolving it
                        # switches -I workers to the system interpreter.
                        argv = [
                            str(Path(sys.executable).absolute()),
                            "-I",
                            "-m",
                            "recsys.research.campaign.worker",
                            "--worker-request",
                            str(directory / "request.json"),
                        ]
                        started = time.monotonic()
                        try:
                            handle = supervisor.start(
                                argv,
                                cwd=root,
                                log_dir=directory / "logs",
                                remaining_seconds=remaining,
                                env=_worker_env(),
                            )
                        except BaseException:
                            charge = time.monotonic() - started
                            charged[(variant, phase)] = charged.get((variant, phase), 0) + charge
                            failed.add((variant, phase))
                            frozen, failure_code = True, "research_campaign_spawn_failed"
                            journal.append(
                                "attempt_finished",
                                generation=generation,
                                variant=variant,
                                phase=phase,
                                attempt=number,
                                status="failed",
                                code=failure_code,
                                charged_seconds=charge,
                                resources=None,
                            )
                            break
                        active[handle] = {
                            "variant": variant,
                            "phase": phase,
                            "attempt": number,
                            "generation": generation,
                            "observed_elapsed": 0.0,
                            "warned": False,
                        }
                if not active:
                    if pending:
                        for item in pending:
                            variant, phase = item["variant"], item["phase"]
                            result = item["result"]
                            charge = max(item["observed_elapsed"], result.elapsed_seconds)
                            charged[(variant, phase)] = charged.get((variant, phase), 0) + charge
                            success = result.outcome == "ok" and result.cleanup_complete
                            artifact_identity = None
                            if success:
                                try:
                                    artifact_identity, _ = _verify_phase(
                                        root,
                                        plan,
                                        successes,
                                        variant,
                                        phase,
                                        item["attempt"],
                                        generation,
                                        prepared_dir,
                                    )
                                except BaseException:
                                    success = False
                                    failure_code = "research_campaign_artifact_invalid"
                            if success and (
                                _phase_budget_remaining(charged, variant, phase) < 0
                                or _now() > _parse_iso(plan["deadline_utc"])
                            ):
                                success = False
                                failure_code = "research_campaign_budget_limit"
                            event = journal.append(
                                "attempt_finished",
                                generation=generation,
                                variant=variant,
                                phase=phase,
                                attempt=item["attempt"],
                                status="success" if success else "failed",
                                code=None
                                if success
                                else failure_code or "research_campaign_phase_failed",
                                charged_seconds=charge,
                                resources=_process_fields(result),
                                output_identity=artifact_identity if success else None,
                            )
                            if success:
                                successes[(variant, phase)] = event
                            else:
                                failed.add((variant, phase))
                                frozen = True
                                failure_code = event["code"]
                        pending.clear()
                    if frozen or set(successes) == _expected_nodes():
                        break
                    if _next_ready(successes, failed, set(), 0, cpu_workers) is None:
                        frozen, failure_code = True, "research_campaign_dependency_blocked"
                        journal.append("campaign_failure", code=failure_code)
                        break
                    continue
                snapshot = supervisor.poll(wait_seconds=0.1)
                # An interim cleanup failure has no completed event. Inspect every
                # unfinished handle after every poll, keep slots, and freeze work.
                for handle, item in tuple(active.items()):
                    polled_result = supervisor.result(handle)
                    if polled_result is None:
                        continue
                    item["observed_elapsed"] = max(
                        item["observed_elapsed"], polled_result.elapsed_seconds
                    )
                    if not polled_result.cleanup_complete:
                        if not item["warned"]:
                            item["warned"] = True
                            journal.append(
                                "cleanup_warning",
                                generation=generation,
                                variant=item["variant"],
                                phase=item["phase"],
                                attempt=item["attempt"],
                                code="research_campaign_cleanup_failed",
                            )
                        frozen, failure_code = True, "research_campaign_cleanup_failed"
                        for other in active:
                            if other != handle:
                                supervisor.cancel(other)
                        continue
                    if handle not in snapshot.completed:
                        _fail("research_campaign_supervisor_invalid")
                    item["result"] = polled_result
                    pending.append(item)
                    del active[handle]
                if frozen and active:
                    for handle in active:
                        supervisor.cancel(handle)
        except BaseException as error:
            original_error = error
            try:
                journal.append("campaign_failure", code="research_campaign_controller_failed")
            except BaseException:
                # A backward UTC clock can reject this second event too. Keep
                # the first failure and leave started attempts unconfirmed.
                pass
            raise
        finally:
            try:
                supervisor.close()
            except BaseException:
                if original_error is None:
                    raise
            if original_error is not None:
                # Only a confirmed clean supervisor result permits a future
                # retry; orphan or failed cleanup keeps its started record open.
                for handle, item in active.items():
                    final_result = supervisor.result(handle)
                    if final_result is None or not final_result.cleanup_complete:
                        continue
                    try:
                        journal.append(
                            "attempt_interrupted",
                            generation=generation,
                            variant=item["variant"],
                            phase=item["phase"],
                            attempt=item["attempt"],
                            charged_seconds=max(
                                item["observed_elapsed"], final_result.elapsed_seconds
                            ),
                            resources=_process_fields(final_result),
                        )
                    except BaseException:
                        pass
                for item in pending:
                    result = item["result"]
                    if result.cleanup_complete:
                        try:
                            journal.append(
                                "attempt_interrupted",
                                generation=generation,
                                variant=item["variant"],
                                phase=item["phase"],
                                attempt=item["attempt"],
                                charged_seconds=max(
                                    item["observed_elapsed"], result.elapsed_seconds
                                ),
                                resources=_process_fields(result),
                            )
                        except BaseException:
                            pass
        complete = not frozen and set(successes) == _expected_nodes()
        if complete:
            report = _public_projection(root, plan, successes, prepared_dir)
            if _now() > _parse_iso(plan["deadline_utc"]):
                complete, failure_code = False, "research_campaign_wall_limit"
        else:
            report = {"variants": [], "primary_ndcg5": None, "paired_vs_popularity": None}
        report.update(
            {
                "schema_version": 2,
                "kind": "arb-ml02-campaign-report",
                "plan_identity": plan_identity,
                "prepared_identity": plan["prepared_identity"],
                "status": "complete" if complete else "incomplete",
                "failure_code": None if complete else failure_code,
                "diagnostics": {
                    "single_query_latency": "not_measured",
                    "batch_throughput": "not_measured",
                    "reason": "standalone_benchmark_not_scheduled",
                },
            }
        )
        journal.append(
            "campaign_finished",
            generation=generation,
            status=report["status"],
            code=report["failure_code"],
        )
        _replace_private(root / "campaign-report.json", _canonical(report))
        manifest = {
            "schema_version": 1,
            "kind": "arb-ml02-campaign-report-manifest",
            "plan_identity": plan_identity,
            "status": report["status"],
            "report": {
                "bytes": len(_canonical(report)),
                "sha256": hashlib.sha256(_canonical(report)).hexdigest(),
            },
        }
        _replace_private(root / "campaign-report-manifest.json", _canonical(manifest))
        return root / "campaign-report-manifest.json"
    finally:
        if gate is not None:
            os.close(gate)
        fcntl.flock(lock, fcntl.LOCK_UN)
        os.close(lock)


def _coordinator_rss_bytes() -> int:
    try:
        pages = int(Path("/proc/self/statm").read_text().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        _fail("research_campaign_rss_probe_failed")


def _native_storage_check(
    root: Path, plan: CampaignPlan, remaining_arms: int, *, admission: bool = True
) -> tuple[int, int]:
    storage = plan["storage"]
    snapshot = check_binding(
        root,
        Path(storage["physical_storage_root"]),
        output_device=storage["output_filesystem_device"],
        physical_device=storage["physical_filesystem_device"],
    )
    require_storage(
        snapshot, required_bytes=admission_bytes(remaining_arms) if admission else STOP_BYTES
    )
    return snapshot.output_available_bytes, snapshot.physical_available_bytes


def _run_native_job(
    supervisor: object,
    *,
    argv: list[str],
    root: Path,
    log_dir: Path,
    remaining_seconds: float,
    plan: CampaignPlan,
    remaining_arms: int,
    cuda_fit: bool = False,
) -> tuple[ProcessResult, str | None, int, int, int]:
    """Keep one process group supervised through cleanup and physical-space checks."""
    from recsys.research.resources import ResourceSupervisor

    owner = cast("ResourceSupervisor", supervisor)
    handle = owner.start(
        argv,
        cwd=root,
        log_dir=log_dir,
        remaining_seconds=remaining_seconds,
        env=_native_worker_env(cuda_fit=cuda_fit),
    )
    failure_code: str | None = None
    output_min = physical_min = 2**63 - 1
    aggregate_peak = 0
    last_storage_check = 0.0
    while True:
        snapshot = owner.poll(wait_seconds=0.1)
        aggregate = snapshot.sampled_rss_bytes + _coordinator_rss_bytes()
        aggregate_peak = max(aggregate_peak, aggregate)
        if aggregate > plan["execution_profile"]["aggregate_rss_limit_bytes"]:
            failure_code = "research_campaign_aggregate_rss_limit"
            owner.cancel(handle)
        now = time.monotonic()
        if now - last_storage_check >= plan["storage"]["policy"]["poll_interval_seconds"]:
            last_storage_check = now
            try:
                available_output, available_physical = _native_storage_check(
                    root, plan, remaining_arms, admission=False
                )
                output_min = min(output_min, available_output)
                physical_min = min(physical_min, available_physical)
            except BaseException as error:
                code = str(error)
                failure_code = (
                    code if _SAFE_CODE.fullmatch(code) else "research_campaign_storage_probe_failed"
                )
                owner.cancel(handle)
        result = owner.result(handle)
        if result is None:
            continue
        if not result.cleanup_complete:
            failure_code = "research_campaign_cleanup_failed"
            owner.cancel(handle)
            continue
        if handle not in snapshot.completed:
            _fail("research_campaign_supervisor_invalid")
        return result, failure_code, aggregate_peak, output_min, physical_min


def _native_verification_request(
    root: Path,
    plan_identity: str,
    generation: str,
    prepared_dir: Path,
    journal: _Journal,
    *,
    operation: str,
    variant: str | None = None,
    phase: str | None = None,
    attempt: int | None = None,
    execution_identity: str | None = None,
) -> tuple[Path, dict[str, object]]:
    request: dict[str, object] = {
        "schema_version": 1,
        "kind": "arb-ml02-campaign-verification-request",
        "operation": operation,
        "root": str(root),
        "plan_identity": plan_identity,
        "generation": generation,
        "prepared_dir": str(prepared_dir),
        "journal_identity": _digest(journal.path.read_bytes()),
    }
    if operation == "verify-phase":
        if variant is None or phase is None or attempt is None or execution_identity is None:
            _fail("research_campaign_verification_invalid")
        request.update(
            variant=variant,
            phase=phase,
            attempt=attempt,
            execution_identity=execution_identity,
        )
        path = _phase_directory(root, variant, phase, attempt) / "verify-request.json"
    else:
        directory = (
            root / ("projections" if operation == "project-report" else "resumptions") / generation
        )
        _ensure_private(directory)
        path = directory / "request.json"
    _private_file(path, _canonical(request))
    return path, request


def _native_verify_job(
    supervisor: object,
    root: Path,
    plan: CampaignPlan,
    plan_identity: str,
    generation: str,
    prepared_dir: Path,
    journal: _Journal,
    *,
    operation: str,
    remaining_seconds: float,
    remaining_arms: int,
    variant: str | None = None,
    phase: str | None = None,
    attempt: int | None = None,
    execution_identity: str | None = None,
) -> tuple[dict[str, object], ProcessResult, int, int, int]:
    request_path, request = _native_verification_request(
        root,
        plan_identity,
        generation,
        prepared_dir,
        journal,
        operation=operation,
        variant=variant,
        phase=phase,
        attempt=attempt,
        execution_identity=execution_identity,
    )
    container = request_path.parent
    result, failure_code, aggregate, output_min, physical_min = _run_native_job(
        supervisor,
        argv=[
            str(Path(sys.executable).absolute()),
            "-I",
            "-m",
            "recsys.research.campaign.verification",
            "--verification-request",
            str(request_path),
        ],
        root=root,
        log_dir=container / "verify-logs",
        remaining_seconds=remaining_seconds,
        plan=plan,
        remaining_arms=remaining_arms,
    )
    _private_file(container / "verification-process.json", _canonical(_process_fields(result)))
    if (
        failure_code
        or result.outcome != "ok"
        or result.returncode != 0
        or not result.cleanup_complete
    ):
        _fail(failure_code or "research_campaign_verification_failed")
    receipt, _ = _read_private(
        container / "verification-result.json"
        if operation == "verify-phase"
        else container / "result.json",
        limit=4096,
    )
    if (
        type(receipt) is not dict
        or receipt.get("schema_version") != 1
        or receipt.get("kind") != "arb-ml02-campaign-verification-receipt"
        or receipt.get("plan_identity") != plan_identity
        or receipt.get("request_identity") != _digest(_canonical(request))
        or receipt.get("generation") != generation
    ):
        _fail("research_campaign_verification_invalid")
    return cast("dict[str, object]", receipt), result, aggregate, output_min, physical_min


def _run_native_campaign(
    prepared_dir: str | Path,
    output_dir: str | Path,
    *,
    expected_prepared_identity: str,
    code_revision: str,
    glove: Path,
    backend: str,
    cpu_workers: int,
    resume: bool,
    expected_campaign_identity: str | None,
    native_train_device: NativeTrainDevice | None,
    physical_storage_root: Path | None,
) -> Path:
    from recsys.research.artifacts import safe_path, separate
    from recsys.research.resources import ResourceSupervisor

    require_cpu_backend(backend)
    prepared = Path(prepared_dir).absolute()
    glove = Path(glove).absolute()
    root = Path(output_dir).absolute()
    separate(prepared, glove, root)
    safe_path(prepared, directory=True)
    safe_path(glove, directory=False)
    if (
        type(resume) is not bool
        or (resume and expected_campaign_identity is None)
        or (not resume and expected_campaign_identity is not None)
        or physical_storage_root is None
        or type(cpu_workers) is not int
        or cpu_workers not in (1, 2)
    ):
        _fail("research_campaign_profile_unsupported")
    physical = Path(physical_storage_root)
    if not physical.is_absolute():
        _fail("research_campaign_storage_root_invalid")
    if resume:
        if root.is_symlink() or not root.is_dir() or root.stat().st_mode & 0o077:
            _fail("research_campaign_invalid")
    else:
        if root.exists() or root.is_symlink():
            _fail("research_campaign_invalid")
        root.mkdir(mode=0o700)
        _sync_dir(root.parent)
    lock = _touch_lock(root / ".campaign.lock")
    gate: int | None = None
    try:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            _fail("research_campaign_busy")
        if resume:
            previous, raw = _read_private(root / "campaign-plan.json")
            _validate_plan_structure(previous)
            prior = cast("CampaignPlan", previous)
            planned = _plan(
                prepared,
                expected_prepared_identity,
                code_revision,
                glove,
                cpu_workers,
                _parse_iso(prior["created_utc"]),
                campaign_name=NATIVE_CAMPAIGN,
                native_train_device=native_train_device,
                physical_storage_root=physical,
                output_root=root,
            )
            if prior != planned or _digest(raw) != expected_campaign_identity:
                _fail("research_campaign_plan_mismatch")
            plan, plan_identity = prior, _digest(raw)
        else:
            plan = _plan(
                prepared,
                expected_prepared_identity,
                code_revision,
                glove,
                cpu_workers,
                _now(),
                campaign_name=NATIVE_CAMPAIGN,
                native_train_device=native_train_device,
                physical_storage_root=physical,
                output_root=root,
            )
            raw = _canonical(plan)
            _private_file(root / "campaign-plan.json", raw)
            plan_identity = _digest(raw)
        _validate_plan_structure(plan)
        if _now() > _parse_iso(plan["deadline_utc"]):
            _fail("research_campaign_wall_limit")
        gate = _touch_lock(root / ".workers.lock")
        try:
            fcntl.flock(gate, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            _fail("research_campaign_workers_active")
        journal = _Journal(root / "journal.jsonl")
        variants = campaign_variants(NATIVE_CAMPAIGN)
        successes, attempts, charged, starts, failed = _successes(journal.events, variants=variants)
        if starts:
            _fail("research_campaign_interrupted_uncertain")
        # A finished or interrupted attempt may be retried only after its
        # process group and any verifier were positively observed as closed.
        # All charges remain in the journal.
        for event in journal.events:
            if event["event"] in ("attempt_finished", "attempt_interrupted"):
                evidence = event.get("resources")
                verify_resources = event.get("verification_resources")
                if (
                    type(evidence) is not dict
                    or evidence.get("cleanup_complete") is not True
                    or (
                        verify_resources is not None
                        and (
                            type(verify_resources) is not dict
                            or verify_resources.get("cleanup_complete") is not True
                        )
                    )
                ):
                    _fail("research_campaign_interrupted_uncertain")
        generation = uuid.uuid4().hex
        _replace_private(root / "generation.json", _canonical({"generation": generation}))
        fcntl.flock(gate, fcntl.LOCK_UN)
        journal.append(
            "run_started", generation=generation, plan_identity=plan_identity, resumed=resume
        )
        supervisor = ResourceSupervisor(
            rss_limit_bytes=plan["execution_profile"]["aggregate_rss_limit_bytes"]
        )
        failure_code: str | None = None
        aggregate_peak = 0
        min_output = min_physical = 2**63 - 1
        projection: dict[str, object] | None = None
        projection_resources: dict[str, object] | None = None
        entered = False
        try:
            supervisor.__enter__()
            entered = True
            if resume and successes:
                # Recheck every retained artifact in a bounded child. The
                # coordinator itself only handles small identities and receipts.
                remaining_arms = 8 - sum((variant, "predict") in successes for variant in variants)
                _native_storage_check(root, plan, remaining_arms)
                receipt, result, peak, out_free, host_free = _native_verify_job(
                    supervisor,
                    root,
                    plan,
                    plan_identity,
                    generation,
                    prepared,
                    journal,
                    operation="resume-verify",
                    remaining_seconds=(_parse_iso(plan["deadline_utc"]) - _now()).total_seconds(),
                    remaining_arms=remaining_arms,
                )
                if (
                    receipt.get("operation") != "resume-verify"
                    or receipt.get("projection_identity") is not None
                ):
                    _fail("research_campaign_verification_invalid")
                aggregate_peak = max(aggregate_peak, peak)
                min_output, min_physical = min(min_output, out_free), min(min_physical, host_free)
                journal.append(
                    "resume_verified", generation=generation, resources=_process_fields(result)
                )
            while set(successes) != _expected_nodes(plan):
                if _now() >= _parse_iso(plan["deadline_utc"]):
                    failure_code = "research_campaign_wall_limit"
                    break
                remaining_arms = 8 - sum((variant, "predict") in successes for variant in variants)
                try:
                    out_free, host_free = _native_storage_check(root, plan, remaining_arms)
                    min_output, min_physical = (
                        min(min_output, out_free),
                        min(min_physical, host_free),
                    )
                except BaseException as error:
                    code = str(error)
                    failure_code = (
                        code
                        if _SAFE_CODE.fullmatch(code)
                        else "research_campaign_storage_probe_failed"
                    )
                    break
                choice = _next_ready(successes, set(), set(), 0, cpu_workers, variants)
                if choice is None:
                    failure_code = "research_campaign_dependency_blocked"
                    break
                variant, phase = choice
                remaining = _remaining(plan, charged, variant, phase)
                if remaining <= 0:
                    failure_code = "research_campaign_budget_limit"
                    break
                directory, number = _prepare_attempt(
                    root,
                    plan,
                    plan_identity,
                    generation,
                    successes,
                    attempts,
                    variant,
                    phase,
                    prepared,
                    glove,
                    NATIVE_CAMPAIGN,
                )
                attempts[(variant, phase)] = number
                journal.append(
                    "attempt_started",
                    generation=generation,
                    variant=variant,
                    phase=phase,
                    attempt=number,
                    remaining_seconds=remaining,
                )
                started = time.monotonic()
                execution: ProcessResult | None = None
                verification: ProcessResult | None = None
                verification_identity: str | None = None
                manifest_identity: str | None = None
                attempt_code: str | None = None
                verification_started = False
                try:
                    execution, execution_code, peak, out_free, host_free = _run_native_job(
                        supervisor,
                        argv=[
                            str(Path(sys.executable).absolute()),
                            "-I",
                            "-m",
                            "recsys.research.campaign.worker",
                            "--worker-request",
                            str(directory / "request.json"),
                        ],
                        root=root,
                        log_dir=directory / "logs",
                        remaining_seconds=remaining,
                        plan=plan,
                        remaining_arms=remaining_arms,
                        cuda_fit=variant == NATIVE_VARIANT
                        and phase == "fit"
                        and plan["variants"][-1]["train_device"]["kind"] == "cuda",
                    )
                    aggregate_peak = max(aggregate_peak, peak)
                    min_output, min_physical = (
                        min(min_output, out_free),
                        min(min_physical, host_free),
                    )
                    if execution_code or execution.outcome != "ok" or execution.returncode != 0:
                        attempt_code = execution_code or "research_campaign_phase_failed"
                    else:
                        execution_raw = _canonical(_process_fields(execution))
                        _private_file(directory / "resources" / "execution.json", execution_raw)
                        verification_remaining = remaining - (time.monotonic() - started)
                        if verification_remaining <= 0:
                            _fail("research_campaign_budget_limit")
                        verification_started = True
                        receipt, verification, peak, out_free, host_free = _native_verify_job(
                            supervisor,
                            root,
                            plan,
                            plan_identity,
                            generation,
                            prepared,
                            journal,
                            operation="verify-phase",
                            variant=variant,
                            phase=phase,
                            attempt=number,
                            execution_identity=_digest(execution_raw),
                            remaining_seconds=verification_remaining,
                            remaining_arms=remaining_arms,
                        )
                        aggregate_peak = max(aggregate_peak, peak)
                        min_output = min(min_output, out_free)
                        min_physical = min(min_physical, host_free)
                        if (
                            receipt.get("variant") != variant
                            or receipt.get("phase") != phase
                            or receipt.get("attempt") != number
                            or receipt.get("execution_identity") != _digest(execution_raw)
                            or type(receipt.get("manifest_identity")) is not str
                        ):
                            _fail("research_campaign_verification_invalid")
                        manifest_identity = cast("str", receipt["manifest_identity"])
                        verification_identity = _digest(_canonical(receipt))
                    charge = time.monotonic() - started
                    charged[(variant, phase)] = charged.get((variant, phase), 0.0) + charge
                    if attempt_code is None and (
                        _phase_budget_remaining(charged, variant, phase) < 0
                        or _now() > _parse_iso(plan["deadline_utc"])
                    ):
                        attempt_code = "research_campaign_budget_limit"
                except BaseException as error:
                    code = str(error)
                    attempt_code = (
                        code
                        if _SAFE_CODE.fullmatch(code)
                        else "research_campaign_controller_failed"
                    )
                    if execution is not None and execution.cleanup_complete:
                        verification_record = directory / "verification-process.json"
                        if (
                            verification_started
                            and verification is None
                            and verification_record.is_file()
                        ):
                            observed, _ = _read_private(verification_record, limit=4096)
                            if (
                                type(observed) is not dict
                                or observed.get("cleanup_complete") is not True
                            ):
                                raise
                        elif verification_started and verification is None:
                            raise
                        charge = time.monotonic() - started
                        charged[(variant, phase)] = charged.get((variant, phase), 0.0) + charge
                    else:
                        # No proof of descendant cleanup: retain open start.
                        raise
                event = journal.append(
                    "attempt_finished",
                    generation=generation,
                    variant=variant,
                    phase=phase,
                    attempt=number,
                    status="success" if attempt_code is None else "failed",
                    code=attempt_code,
                    charged_seconds=charge,
                    resources=_process_fields(execution) if execution else None,
                    verification_resources=_process_fields(verification) if verification else None,
                    verification_identity=verification_identity,
                    output_identity=manifest_identity if attempt_code is None else None,
                )
                if attempt_code is not None:
                    failure_code = attempt_code
                    break
                successes[(variant, phase)] = event
            if failure_code is None:
                remaining = (_parse_iso(plan["deadline_utc"]) - _now()).total_seconds()
                if remaining <= 0:
                    failure_code = "research_campaign_wall_limit"
                else:
                    try:
                        receipt, result, peak, out_free, host_free = _native_verify_job(
                            supervisor,
                            root,
                            plan,
                            plan_identity,
                            generation,
                            prepared,
                            journal,
                            operation="project-report",
                            remaining_seconds=remaining,
                            remaining_arms=0,
                        )
                        aggregate_peak = max(aggregate_peak, peak)
                        min_output = min(min_output, out_free)
                        min_physical = min(min_physical, host_free)
                        projected, projection_raw = _read_private(
                            root / "projections" / generation / "projection.json", limit=256 * 1024
                        )
                        if receipt.get("operation") != "project-report" or receipt.get(
                            "projection_identity"
                        ) != _digest(projection_raw):
                            _fail("research_campaign_verification_invalid")
                        projection = _native_public_allowed(projected, plan)
                        projection_resources = _process_fields(result)
                    except BaseException as error:
                        code = str(error)
                        failure_code = (
                            code
                            if _SAFE_CODE.fullmatch(code)
                            else "research_campaign_report_invalid"
                        )
        finally:
            if entered:
                supervisor.close()
        if failure_code is None and _now() > _parse_iso(plan["deadline_utc"]):
            failure_code = "research_campaign_wall_limit"
        complete = failure_code is None and projection is not None
        report: dict[str, object] = (
            dict(projection)
            if complete and projection
            else {
                "variants": [],
                "primary_ndcg5": None,
                "paired_vs_popularity": None,
            }
        )
        report.update(
            {
                "schema_version": 3,
                "kind": "arb-ml02-campaign-report",
                "campaign_id": NATIVE_CAMPAIGN,
                "plan_identity": plan_identity,
                "prepared_identity": plan["prepared_identity"],
                "status": "complete" if complete else "incomplete",
                "failure_code": None if complete else failure_code,
                "diagnostics": {
                    "single_query_latency": "not_measured",
                    "batch_throughput": "not_measured",
                    "reason": "standalone_benchmark_not_scheduled",
                },
                "storage": {
                    "policy": plan["storage"]["policy"]["kind"],
                    "admission_estimate_bytes": admission_bytes(8),
                    "minimum_output_available_bytes": min_output
                    if min_output != 2**63 - 1
                    else None,
                    "minimum_physical_available_bytes": min_physical
                    if min_physical != 2**63 - 1
                    else None,
                    "reserve_bytes": plan["storage"]["policy"]["reserve_bytes"],
                    "stop_margin_bytes": plan["storage"]["policy"]["stop_margin_bytes"],
                },
                "aggregate_peak_rss_bytes": aggregate_peak,
                "projection_resources": projection_resources,
            }
        )
        journal.append(
            "campaign_finished",
            generation=generation,
            status=report["status"],
            code=report["failure_code"],
        )
        report_raw = _canonical(report)
        _replace_private(root / "campaign-report.json", report_raw)
        manifest = {
            "schema_version": 1,
            "kind": "arb-ml02-campaign-report-manifest",
            "plan_identity": plan_identity,
            "status": report["status"],
            "report": {"bytes": len(report_raw), "sha256": hashlib.sha256(report_raw).hexdigest()},
        }
        _replace_private(root / "campaign-report-manifest.json", _canonical(manifest))
        return root / "campaign-report-manifest.json"
    finally:
        if gate is not None:
            os.close(gate)
        fcntl.flock(lock, fcntl.LOCK_UN)
        os.close(lock)
