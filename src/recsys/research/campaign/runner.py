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

from recsys.research.campaign.artifacts import _phase_directory, _phase_path, _verify_phase
from recsys.research.campaign.common import (
    AUX_PHASE,
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
from recsys.research.campaign.journal import _Journal, _successes
from recsys.research.campaign.plan import (
    _component_phase,
    _expected_nodes,
    _plan,
    _validate_plan_structure,
)
from recsys.research.campaign.report import _public_projection
from recsys.research.campaign.worker import _worker_request


def _worker_env():
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


def _process_fields(result):
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


def _next_ready(successes, failed, active_keys, plan, cpu_used, gpu_used, cpu_workers):
    for variant in VARIANTS:
        for phase in (*PHASES, AUX_PHASE):
            key = (variant, phase)
            if key not in _expected_nodes(plan):
                continue
            if key in successes or key in failed or key in active_keys:
                continue
            index = PHASES.index(phase) if phase in PHASES else 0
            if phase == AUX_PHASE and (variant, "fit") not in successes:
                continue
            if index and (variant, PHASES[index - 1]) not in successes:
                continue
            if (
                phase == "fit"
                and variant == "hybrid"
                and any(
                    (dependency, _component_phase(plan)) not in successes
                    for dependency in ("content-mean", "puresvd")
                )
            ):
                continue
            lane = (
                "cpu"
                if phase == AUX_PHASE
                else plan["variants"][VARIANTS.index(variant)]["backend"]
                if phase in ("fit", "predict")
                else "cpu"
            )
            if lane == "gpu" and gpu_used or lane == "cpu" and cpu_used >= cpu_workers:
                continue
            return variant, phase, lane
    return None


def _phase_budget_remaining(charged, variant, phase):
    budget = FIT_SECONDS if phase in ("fit", AUX_PHASE) else EVALUATION_SECONDS
    used = sum(
        charged.get((variant, step), 0.0)
        for step in (("fit", AUX_PHASE) if phase in ("fit", AUX_PHASE) else PHASES[1:])
    )
    return budget - used


def _remaining(plan, charged, variant, phase):
    until = (_parse_iso(plan["deadline_utc"]) - _now()).total_seconds()
    return min(_phase_budget_remaining(charged, variant, phase), until)


def _prepare_attempt(
    root, plan, plan_identity, generation, successes, attempts, variant, phase, prepared_dir, glove
):
    number = attempts.get((variant, phase), 0) + 1
    directory = _phase_directory(root, variant, phase, number)
    _ensure_private(directory)
    _ensure_private(directory / "resources")
    model = _phase_path(root, successes, variant, "fit")
    predictions = _phase_path(root, successes, variant, "predict")
    content = _phase_path(root, successes, "content-mean", _component_phase(plan))
    svd = _phase_path(root, successes, "puresvd", _component_phase(plan))
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
    )
    _private_file(directory / "request.json", _canonical(request))
    return directory, number


def run_campaign(
    prepared_dir,
    output_dir,
    *,
    expected_prepared_identity,
    code_revision,
    glove: Path,
    backend="cpu",
    cpu_workers=2,
    resume=False,
    expected_campaign_identity=None,
):
    """Run exactly seven fixed validation variants with durable, bounded workers."""
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
            prior, raw = _read_private(root / "campaign-plan.json")
            _validate_plan_structure(prior)
            created = _parse_iso(prior["created_utc"])
            planned = _plan(
                prepared_dir,
                expected_prepared_identity,
                code_revision,
                glove,
                backend,
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
                backend,
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
        active = {}
        pending = []
        frozen = bool(failed)
        failure_code = "research_campaign_previous_failure" if failed else None
        supervisor = ResourceSupervisor(rss_limit_bytes=RSS_LIMIT_BYTES)
        original_error = None
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
                        cpu_used = sum(item["lane"] == "cpu" for item in active.values())
                        gpu_used = sum(item["lane"] == "gpu" for item in active.values())
                        choice = _next_ready(
                            successes,
                            failed,
                            {(item["variant"], item["phase"]) for item in active.values()}
                            | {(item["variant"], item["phase"]) for item in pending},
                            plan,
                            cpu_used,
                            gpu_used,
                            cpu_workers,
                        )
                        if choice is None:
                            break
                        variant, phase, lane = choice
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
                            "lane": lane,
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
                    if frozen or set(successes) == _expected_nodes(plan):
                        break
                    if _next_ready(successes, failed, set(), plan, 0, 0, cpu_workers) is None:
                        frozen, failure_code = True, "research_campaign_dependency_blocked"
                        journal.append("campaign_failure", code=failure_code)
                        break
                    continue
                snapshot = supervisor.poll(wait_seconds=0.1)
                # An interim cleanup failure has no completed event. Inspect every
                # unfinished handle after every poll, keep slots, and freeze work.
                for handle, item in tuple(active.items()):
                    result = supervisor.result(handle)
                    if result is None:
                        continue
                    item["observed_elapsed"] = max(item["observed_elapsed"], result.elapsed_seconds)
                    if not result.cleanup_complete:
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
                    item["result"] = result
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
                    result = supervisor.result(handle)
                    if result is None or not result.cleanup_complete:
                        continue
                    try:
                        journal.append(
                            "attempt_interrupted",
                            generation=generation,
                            variant=item["variant"],
                            phase=item["phase"],
                            attempt=item["attempt"],
                            charged_seconds=max(item["observed_elapsed"], result.elapsed_seconds),
                            resources=_process_fields(result),
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
        complete = not frozen and set(successes) == _expected_nodes(plan)
        if complete:
            report = _public_projection(root, plan, successes, prepared_dir)
            if _now() > _parse_iso(plan["deadline_utc"]):
                complete, failure_code = False, "research_campaign_wall_limit"
        else:
            report = {"variants": [], "primary_ndcg5": None, "paired_vs_popularity": None}
        report.update(
            {
                "schema_version": 1,
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
