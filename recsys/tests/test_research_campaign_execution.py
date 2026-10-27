"""Eight-arm synthetic campaign exercises the closed native worker lifecycle."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

from research_model_support import patch_release_info
from test_research_campaign_report_limits import synthetic_groups
from test_research_model_content import _glove
from training_support import SeedTrainingFixture

from recsys.research.campaign import common
from recsys.research.campaign.common import FIT_SECONDS, _read_native_public
from recsys.research.campaign.journal import CampaignEvent, PhaseKey, _Journal, _successes
from recsys.research.campaign.plan import CampaignPlan
from recsys.research.campaign.report import validate_campaign
from recsys.research.campaign.runner import _canonical, run_campaign
from recsys.research.campaign.spec import CampaignName
from recsys.research.campaign.storage import GIB, STOP_BYTES
from recsys.research.campaign.verification import _verification_main
from recsys.research.campaign.worker import WorkerRequest, _worker_main
from recsys.research.protocol import ResearchError
from recsys.research.resources import JobHandle, PollSnapshot, ProcessResult, ResourceSupervisor


class InlineSupervisor:
    """Retain worker protocol while replacing only process launch for tiny fixtures."""

    def __init__(self, *, rss_limit_bytes: int) -> None:
        self.results: dict[object, ProcessResult] = {}
        self.delivered: set[object] = set()

    def __enter__(self) -> InlineSupervisor:
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.close()

    def start(
        self,
        argv: list[str],
        *,
        cwd: Path,
        log_dir: Path,
        remaining_seconds: float,
        env: dict[str, str],
    ) -> object:
        log_dir.mkdir(mode=0o700)
        code = (
            _verification_main(Path(argv[-1]))
            if "verification" in argv[3]
            else _worker_main(Path(argv[-1]))
        )
        handle = object()
        self.results[handle] = ProcessResult(
            outcome="ok" if code == 0 else "exit_nonzero",
            returncode=code,
            elapsed_seconds=0.01,
            sampled_peak_rss_bytes=1000,
            wait4_peak_rss_bytes=1000,
            observed_peak_rss_bytes=1000,
            rss_samples=1,
            max_sample_gap_seconds=0.01,
            cleanup_complete=True,
        )
        return handle

    def poll(self, *, wait_seconds: float) -> SimpleNamespace | PollSnapshot:
        completed = tuple(handle for handle in self.results if handle not in self.delivered)
        self.delivered.update(completed)
        return SimpleNamespace(completed=completed, sampled_rss_bytes=1000)

    def result(self, handle: object) -> ProcessResult | None:
        return self.results[handle]

    def cancel(self, handle: object) -> None:
        return None

    def close(self) -> None:
        return None


class TimedOutVerifierSupervisor(InlineSupervisor):
    """Let execution publish its artifact, then time out the supervised verifier."""

    def start(
        self,
        argv: list[str],
        *,
        cwd: Path,
        log_dir: Path,
        remaining_seconds: float,
        env: dict[str, str],
    ) -> object:
        handle = super().start(
            argv,
            cwd=cwd,
            log_dir=log_dir,
            remaining_seconds=remaining_seconds,
            env=env,
        )
        if (
            "verification" in argv[3]
            and json.loads(Path(argv[-1]).read_text())["operation"] == "verify-phase"
        ):
            time.sleep(0.15)
            self.results[handle] = replace(
                self.results[handle], outcome="timeout", returncode=None, elapsed_seconds=900.0
            )
        return handle


class InterruptedVerifierSupervisor(InlineSupervisor):
    """Interrupt polling while a verifier is active, then prove cleanup on close."""

    def __init__(self, *, rss_limit_bytes: int) -> None:
        super().__init__(rss_limit_bytes=rss_limit_bytes)
        self.uncertain_handle: object | None = None

    def start(
        self,
        argv: list[str],
        *,
        cwd: Path,
        log_dir: Path,
        remaining_seconds: float,
        env: dict[str, str],
    ) -> object:
        handle = super().start(
            argv,
            cwd=cwd,
            log_dir=log_dir,
            remaining_seconds=remaining_seconds,
            env=env,
        )
        if (
            "verification" in argv[3]
            and json.loads(Path(argv[-1]).read_text())["operation"] == "verify-phase"
        ):
            self.uncertain_handle = handle
            self.results[handle] = replace(
                self.results[handle], outcome="cancelled", returncode=None, cleanup_complete=False
            )
        return handle

    def poll(self, *, wait_seconds: float) -> SimpleNamespace | PollSnapshot:
        if self.uncertain_handle is not None:
            raise RuntimeError("simulated verifier supervision loss")
        return super().poll(wait_seconds=wait_seconds)

    def close(self) -> None:
        if self.uncertain_handle is not None:
            handle = self.uncertain_handle
            self.results[handle] = replace(self.results[handle], cleanup_complete=True)


class UnprovenCleanupVerifierSupervisor(InterruptedVerifierSupervisor):
    """Leave the verifier unclosed after losing supervision."""

    def close(self) -> None:
        return None


class RunningVerifierSupervisor(InlineSupervisor):
    """Run the worker inline, but interrupt an actual supervised verifier process."""

    def __init__(self, *, rss_limit_bytes: int) -> None:
        super().__init__(rss_limit_bytes=rss_limit_bytes)
        self.owner = ResourceSupervisor(rss_limit_bytes=rss_limit_bytes)
        self.verifier_handle: JobHandle | None = None
        self.interrupt_once = True

    def __enter__(self) -> RunningVerifierSupervisor:
        self.owner.__enter__()
        return self

    def start(
        self,
        argv: list[str],
        *,
        cwd: Path,
        log_dir: Path,
        remaining_seconds: float,
        env: dict[str, str],
    ) -> object:
        if "verification" not in argv[3]:
            return super().start(
                argv,
                cwd=cwd,
                log_dir=log_dir,
                remaining_seconds=remaining_seconds,
                env=env,
            )
        self.verifier_handle = self.owner.start(
            argv,
            cwd=cwd,
            log_dir=log_dir,
            remaining_seconds=remaining_seconds,
            env=env,
        )
        return self.verifier_handle

    def poll(self, *, wait_seconds: float) -> SimpleNamespace | PollSnapshot:
        if self.verifier_handle is not None:
            if self.interrupt_once:
                self.interrupt_once = False
                raise RuntimeError("simulated running-verifier interruption")
            return self.owner.poll(wait_seconds=wait_seconds)
        return super().poll(wait_seconds=wait_seconds)

    def result(self, handle: object) -> ProcessResult | None:
        if isinstance(handle, JobHandle):
            return self.owner.result(handle)
        return super().result(handle)

    def cancel(self, handle: object) -> None:
        if isinstance(handle, JobHandle):
            self.owner.cancel(handle)

    def close(self) -> None:
        self.owner.close()


class InterruptedRealSupervisor(ResourceSupervisor):
    """Lose coordinator polling once while a real owned process is running."""

    def __init__(self, *, rss_limit_bytes: int) -> None:
        super().__init__(rss_limit_bytes=rss_limit_bytes)
        self.interrupt_once = True

    def poll(self, *, wait_seconds: float = 0.1) -> PollSnapshot:
        if self.interrupt_once:
            self.interrupt_once = False
            raise RuntimeError("simulated running-worker interruption")
        return super().poll(wait_seconds=wait_seconds)


@unittest.skipUnless(os.environ.get("ARB_TEST_TORCH") == "1", "opt-in native CPU campaign")
class NativeCampaignExecutionTests(unittest.TestCase):
    def test_full_shape_crosses_all_native_public_boundaries(self) -> None:
        from recsys.research.campaign.report import _public_projection

        with tempfile.TemporaryDirectory() as temporary:
            fixture = SeedTrainingFixture(temporary)
            root = Path(temporary)
            glove = root / "glove.txt"
            fixture.lock["inputs"]["embeddings"] = _glove(glove)
            prepared, identity = fixture.make_prepared()
            output = root / "native-large-public"
            groups = synthetic_groups()

            def complete_shape(
                campaign_root: Path,
                plan: CampaignPlan,
                successes: dict[PhaseKey, CampaignEvent],
                prepared_dir: Path,
            ) -> dict[str, object]:
                projection = _public_projection(campaign_root, plan, successes, prepared_dir)
                arms = cast("list[dict[str, object]]", projection["variants"])
                names = [cast("str", arm["variant_id"]) for arm in arms]
                for arm in arms:
                    arm["checked_groups"] = groups
                projection["primary_users"] = 1000
                projection["small_cohort_suppressed"] = False
                projection["primary_ndcg5"] = {
                    name: {"users": 1000, "mean_ndcg5_by_n": [0.125] * 10} for name in names
                }
                projection["paired_vs_popularity"] = {
                    name: {
                        "users": 1000,
                        "mean_difference_by_n": [0.0] * 10,
                        "pointwise_ci95_by_n": [[-0.01, 0.01] for _ in range(10)],
                        "bootstrap_replicates": 2000,
                        "bootstrap_seed": 366,
                    }
                    for name in names[1:]
                }
                return projection

            def generous(path: Path) -> tuple[int, int]:
                return path.stat().st_dev, 1 << 40

            with (
                patch_release_info(fixture.lock),
                patch("recsys.research.resources.ResourceSupervisor", InlineSupervisor),
                patch("recsys.research.campaign.storage._filesystem", generous),
                patch(
                    "recsys.research.campaign.report._public_projection",
                    side_effect=complete_shape,
                ),
            ):
                result = run_campaign(
                    prepared,
                    output,
                    expected_prepared_identity=identity,
                    code_revision="a" * 40,
                    glove=glove,
                    campaign_name="ml02-native-v3",
                    native_train_device="cpu",
                    physical_storage_root=root,
                )
                manifest = json.loads(result.read_text())
                report = validate_campaign(
                    prepared,
                    output,
                    expected_prepared_identity=identity,
                    expected_campaign_identity=manifest["plan_identity"],
                )
                journal_path = output / "journal.jsonl"
                complete_journal = journal_path.read_bytes()
                journal_path.write_bytes(b"".join(complete_journal.splitlines(keepends=True)[:-1]))
                try:
                    with self.assertRaises(ResearchError):
                        validate_campaign(
                            prepared,
                            output,
                            expected_prepared_identity=identity,
                            expected_campaign_identity=manifest["plan_identity"],
                        )
                finally:
                    journal_path.write_bytes(complete_journal)
            self.assertEqual(report["status"], "complete")
            arms = cast("list[dict[str, object]]", report["variants"])
            self.assertEqual(len(arms), 8)
            self.assertTrue(
                all(len(cast("list[object]", arm["checked_groups"])) == 164 for arm in arms)
            )
            self.assertEqual(report["primary_users"], 1000)
            self.assertIs(report["small_cohort_suppressed"], False)
            primary = cast("dict[str, dict[str, object]]", report["primary_ndcg5"])
            paired = cast("dict[str, dict[str, object]]", report["paired_vs_popularity"])
            self.assertEqual(len(primary), 8)
            self.assertEqual(len(paired), 7)
            self.assertTrue(
                all(
                    len(cast("list[float]", item["mean_ndcg5_by_n"])) == 10
                    for item in primary.values()
                )
            )
            self.assertTrue(
                all(
                    len(cast("list[object]", item["pointwise_ci95_by_n"])) == 10
                    for item in paired.values()
                )
            )
            generation = json.loads((output / "generation.json").read_text())["generation"]
            projected, projection_raw = _read_native_public(
                output / "projections" / generation / "projection.json"
            )
            published, report_raw = _read_native_public(output / "campaign-report.json")
            self.assertGreater(len(projection_raw), 5_500_000)
            self.assertLess(len(projection_raw), common.NATIVE_PUBLIC_REPORT_MAX_BYTES)
            self.assertGreater(len(report_raw), len(projection_raw))
            self.assertLess(len(report_raw), common.NATIVE_PUBLIC_REPORT_MAX_BYTES)
            projected_arms = cast(
                "list[dict[str, object]]", cast("dict[str, object]", projected)["variants"]
            )
            self.assertEqual(projected_arms[0]["checked_groups"], groups)
            self.assertEqual(published, report)

    def test_full_envelope_overflow_publishes_minimal_incomplete(self) -> None:
        from recsys.research.campaign.report import _public_projection

        with tempfile.TemporaryDirectory() as temporary:
            fixture = SeedTrainingFixture(temporary)
            root = Path(temporary)
            glove = root / "glove.txt"
            fixture.lock["inputs"]["embeddings"] = _glove(glove)
            prepared, identity = fixture.make_prepared()
            output = root / "native-envelope-overflow"

            def small_cap(
                campaign_root: Path,
                plan: CampaignPlan,
                successes: dict[PhaseKey, CampaignEvent],
                prepared_dir: Path,
            ) -> dict[str, object]:
                projection = _public_projection(campaign_root, plan, successes, prepared_dir)
                common.NATIVE_PUBLIC_REPORT_MAX_BYTES = len(_canonical(projection)) + 100
                return projection

            def generous(path: Path) -> tuple[int, int]:
                return path.stat().st_dev, 1 << 40

            with (
                patch_release_info(fixture.lock),
                patch("recsys.research.resources.ResourceSupervisor", InlineSupervisor),
                patch("recsys.research.campaign.storage._filesystem", generous),
                patch.object(common, "NATIVE_PUBLIC_REPORT_MAX_BYTES"),
                patch("recsys.research.campaign.report._public_projection", side_effect=small_cap),
            ):
                result = run_campaign(
                    prepared,
                    output,
                    expected_prepared_identity=identity,
                    code_revision="a" * 40,
                    glove=glove,
                    campaign_name="ml02-native-v3",
                    native_train_device="cpu",
                    physical_storage_root=root,
                )
            manifest = json.loads(result.read_text())
            report = json.loads((output / "campaign-report.json").read_text())
            self.assertEqual(manifest["status"], "incomplete")
            self.assertEqual(report["failure_code"], "research_campaign_report_invalid")
            self.assertEqual(report["variants"], [])
            self.assertEqual(report["primary_ndcg5"], None)
            self.assertEqual(report["paired_vs_popularity"], None)
            self.assertEqual(len(list(output.glob("variants/*/attempts/*/1/artifact"))), 32)
            terminal = _Journal(output / "journal.jsonl").events[-1]
            self.assertEqual(
                (terminal["event"], terminal["status"]), ("campaign_finished", "incomplete")
            )

    def test_second_publication_failure_cannot_claim_complete(self) -> None:
        from recsys.research.campaign.runner import _replace_private

        with tempfile.TemporaryDirectory() as temporary:
            fixture = SeedTrainingFixture(temporary)
            root = Path(temporary)
            glove = root / "glove.txt"
            fixture.lock["inputs"]["embeddings"] = _glove(glove)
            prepared, identity = fixture.make_prepared()
            output = root / "native-second-rename-fails"
            failed = False

            def fail_once(path: Path, raw: bytes) -> None:
                nonlocal failed
                if path.name == "campaign-report-manifest.json" and not failed:
                    failed = True
                    raise OSError("injected second publication failure")
                _replace_private(path, raw)

            def generous(path: Path) -> tuple[int, int]:
                return path.stat().st_dev, 1 << 40

            with (
                patch_release_info(fixture.lock),
                patch("recsys.research.resources.ResourceSupervisor", InlineSupervisor),
                patch("recsys.research.campaign.storage._filesystem", generous),
                patch("recsys.research.campaign.runner._replace_private", side_effect=fail_once),
            ):
                result = run_campaign(
                    prepared,
                    output,
                    expected_prepared_identity=identity,
                    code_revision="a" * 40,
                    glove=glove,
                    campaign_name="ml02-native-v3",
                    native_train_device="cpu",
                    physical_storage_root=root,
                )
                manifest = json.loads(result.read_text())
                report = json.loads((output / "campaign-report.json").read_text())
                with self.assertRaises(ResearchError):
                    validate_campaign(
                        prepared,
                        output,
                        expected_prepared_identity=identity,
                        expected_campaign_identity=manifest["plan_identity"],
                    )
            self.assertTrue(failed)
            self.assertEqual(manifest["status"], "incomplete")
            self.assertEqual(report["failure_code"], "research_campaign_report_invalid")
            self.assertEqual(len(list(output.glob("variants/*/attempts/*/1/artifact"))), 32)
            self.assertEqual(_Journal(output / "journal.jsonl").events[-1]["status"], "incomplete")

    def test_projection_storage_gate_refuses_child_and_keeps_all_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SeedTrainingFixture(temporary)
            root = Path(temporary)
            glove = root / "glove.txt"
            fixture.lock["inputs"]["embeddings"] = _glove(glove)
            prepared, identity = fixture.make_prepared()
            output = root / "native-projection-storage-trip"
            state = {"free": 1 << 40, "completed": 0}
            launches: list[str] = []
            original_append = _Journal.append

            def current_space(path: Path) -> tuple[int, int]:
                return path.stat().st_dev, state["free"]

            def drop_after_last_phase(
                journal: _Journal, name: str, **fields: object
            ) -> CampaignEvent:
                event = original_append(journal, name, **fields)
                if name == "attempt_finished" and fields.get("status") == "success":
                    state["completed"] += 1
                    if state["completed"] == 32:
                        state["free"] = STOP_BYTES - 1
                return event

            class CountingSupervisor(InlineSupervisor):
                def start(
                    self,
                    argv: list[str],
                    *,
                    cwd: Path,
                    log_dir: Path,
                    remaining_seconds: float,
                    env: dict[str, str],
                ) -> object:
                    launches.append(argv[3])
                    return super().start(
                        argv,
                        cwd=cwd,
                        log_dir=log_dir,
                        remaining_seconds=remaining_seconds,
                        env=env,
                    )

            with (
                patch_release_info(fixture.lock),
                patch("recsys.research.resources.ResourceSupervisor", CountingSupervisor),
                patch("recsys.research.campaign.storage._filesystem", current_space),
                patch("recsys.research.campaign.journal._Journal.append", drop_after_last_phase),
            ):
                result = run_campaign(
                    prepared,
                    output,
                    expected_prepared_identity=identity,
                    code_revision="a" * 40,
                    glove=glove,
                    campaign_name="ml02-native-v3",
                    native_train_device="cpu",
                    physical_storage_root=root,
                )
            self.assertEqual(json.loads(result.read_text())["status"], "incomplete")
            self.assertEqual(
                json.loads((output / "campaign-report.json").read_text())["failure_code"],
                "research_campaign_storage_low",
            )
            self.assertEqual(state["completed"], 32)
            self.assertEqual(len(launches), 64)
            self.assertEqual(launches.count("recsys.research.campaign.verification"), 32)
            self.assertEqual(len(list(output.glob("variants/*/attempts/*/1/artifact"))), 32)

    def test_resume_failures_replace_prior_complete_report_without_science(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SeedTrainingFixture(temporary)
            root = Path(temporary)
            glove = root / "glove.txt"
            fixture.lock["inputs"]["embeddings"] = _glove(glove)
            prepared, identity = fixture.make_prepared()
            output = root / "native-complete-before-resume-trip"

            def generous(path: Path) -> tuple[int, int]:
                return path.stat().st_dev, 1 << 40

            with (
                patch_release_info(fixture.lock),
                patch("recsys.research.resources.ResourceSupervisor", InlineSupervisor),
                patch("recsys.research.campaign.storage._filesystem", generous),
            ):
                complete = run_campaign(
                    prepared,
                    output,
                    expected_prepared_identity=identity,
                    code_revision="a" * 40,
                    glove=glove,
                    campaign_name="ml02-native-v3",
                    native_train_device="cpu",
                    physical_storage_root=root,
                )
            manifest = json.loads(complete.read_text())
            self.assertEqual(manifest["status"], "complete")
            report_path = output / "campaign-report.json"
            saved_report = report_path.read_bytes()
            saved_manifest = complete.read_bytes()
            artifact = (
                output
                / "variants"
                / "popularity"
                / "attempts"
                / "fit"
                / "1"
                / "artifact"
                / "model-manifest.json"
            )
            artifact_bytes = artifact.read_bytes()
            starts_before = sum(
                event["event"] == "attempt_started"
                for event in _Journal(output / "journal.jsonl").events
            )

            class NoLaunchSupervisor(InlineSupervisor):
                def start(
                    self,
                    argv: list[str],
                    *,
                    cwd: Path,
                    log_dir: Path,
                    remaining_seconds: float,
                    env: dict[str, str],
                ) -> object:
                    raise AssertionError("resume storage failure launched a child")

            class FailedResumeVerifier(InlineSupervisor):
                launches: list[str] = []

                def start(
                    self,
                    argv: list[str],
                    *,
                    cwd: Path,
                    log_dir: Path,
                    remaining_seconds: float,
                    env: dict[str, str],
                ) -> object:
                    request = json.loads(Path(argv[-1]).read_text())
                    self.launches.append(request["operation"])
                    handle = super().start(
                        argv,
                        cwd=cwd,
                        log_dir=log_dir,
                        remaining_seconds=remaining_seconds,
                        env=env,
                    )
                    if request["operation"] == "resume-verify":
                        self.results[handle] = replace(
                            self.results[handle], outcome="exit_nonzero", returncode=2
                        )
                    return handle

            def probe_failure(_path: Path) -> tuple[int, int]:
                raise OSError("simulated statvfs failure")

            cases: tuple[
                tuple[str, Callable[[Path], tuple[int, int]], type[InlineSupervisor]], ...
            ] = (
                (
                    "research_campaign_storage_low",
                    lambda path: (path.stat().st_dev, 1),
                    NoLaunchSupervisor,
                ),
                ("research_campaign_storage_probe_failed", probe_failure, NoLaunchSupervisor),
                (
                    "research_campaign_storage_binding_changed",
                    lambda path: (path.stat().st_dev + 1, 1 << 40),
                    NoLaunchSupervisor,
                ),
                ("research_campaign_verification_failed", generous, FailedResumeVerifier),
            )
            for expected_code, filesystem, supervisor_type in cases:
                with self.subTest(expected_code=expected_code):
                    report_path.write_bytes(saved_report)
                    complete.write_bytes(saved_manifest)
                    with (
                        patch_release_info(fixture.lock),
                        patch("recsys.research.resources.ResourceSupervisor", supervisor_type),
                        patch("recsys.research.campaign.storage._filesystem", filesystem),
                    ):
                        result = run_campaign(
                            prepared,
                            output,
                            expected_prepared_identity=identity,
                            code_revision="a" * 40,
                            glove=glove,
                            campaign_name="ml02-native-v3",
                            native_train_device="cpu",
                            physical_storage_root=root,
                            resume=True,
                            expected_campaign_identity=manifest["plan_identity"],
                        )
                    self.assertEqual(json.loads(result.read_text())["status"], "incomplete")
                    published = json.loads(report_path.read_text())
                    self.assertEqual(published["status"], "incomplete")
                    self.assertEqual(published["failure_code"], expected_code)
                    self.assertEqual(published["variants"], [])
                    self.assertEqual(artifact.read_bytes(), artifact_bytes)
                    self.assertEqual(
                        sum(
                            event["event"] == "attempt_started"
                            for event in _Journal(output / "journal.jsonl").events
                        ),
                        starts_before,
                    )
            self.assertEqual(FailedResumeVerifier.launches, ["resume-verify"])
            report_path.write_bytes(saved_report)
            complete.write_bytes(saved_manifest)
            journal_before = (output / "journal.jsonl").read_bytes()
            real_stat = os.stat

            def swapped_physical_stat(path: object, **kwargs: object) -> os.stat_result:
                details = real_stat(
                    cast("Path", path),
                    follow_symlinks=cast("bool", kwargs.get("follow_symlinks", True)),
                )
                if path == root:
                    fields = list(details)
                    fields[2] = details.st_dev + 1
                    return os.stat_result(fields)
                return details

            with self.subTest(pre_plan_mount_swap=True):
                with (
                    patch_release_info(fixture.lock),
                    patch("recsys.research.resources.ResourceSupervisor", NoLaunchSupervisor),
                    patch("recsys.research.campaign.plan.os.stat", swapped_physical_stat),
                ):
                    with self.assertRaisesRegex(ResearchError, "research_campaign_plan_mismatch"):
                        run_campaign(
                            prepared,
                            output,
                            expected_prepared_identity=identity,
                            code_revision="a" * 40,
                            glove=glove,
                            campaign_name="ml02-native-v3",
                            native_train_device="cpu",
                            physical_storage_root=root,
                            resume=True,
                            expected_campaign_identity=manifest["plan_identity"],
                        )
                self.assertEqual(report_path.read_bytes(), saved_report)
                self.assertEqual(complete.read_bytes(), saved_manifest)
                self.assertEqual((output / "journal.jsonl").read_bytes(), journal_before)
                self.assertEqual(artifact.read_bytes(), artifact_bytes)

    def test_verifier_storage_gate_refuses_spawn_after_execution(self) -> None:
        import recsys.research.campaign.runner as runner_module

        with tempfile.TemporaryDirectory() as temporary:
            fixture = SeedTrainingFixture(temporary)
            root = Path(temporary)
            glove = root / "glove.txt"
            fixture.lock["inputs"]["embeddings"] = _glove(glove)
            prepared, identity = fixture.make_prepared()
            output = root / "native-verifier-storage-trip"
            state = {"free": 1 << 40}
            launches: list[str] = []
            original_private = runner_module._private_file

            def current_space(path: Path) -> tuple[int, int]:
                return path.stat().st_dev, state["free"]

            def drop_after_execution(path: Path, raw: bytes) -> None:
                original_private(path, raw)
                if path.name == "execution.json":
                    state["free"] = STOP_BYTES - 1

            class CountingSupervisor(InlineSupervisor):
                def start(
                    self,
                    argv: list[str],
                    *,
                    cwd: Path,
                    log_dir: Path,
                    remaining_seconds: float,
                    env: dict[str, str],
                ) -> object:
                    launches.append(argv[3])
                    return super().start(
                        argv,
                        cwd=cwd,
                        log_dir=log_dir,
                        remaining_seconds=remaining_seconds,
                        env=env,
                    )

            with (
                patch_release_info(fixture.lock),
                patch("recsys.research.resources.ResourceSupervisor", CountingSupervisor),
                patch("recsys.research.campaign.storage._filesystem", current_space),
                patch("recsys.research.campaign.runner._private_file", drop_after_execution),
            ):
                manifest_path = run_campaign(
                    prepared,
                    output,
                    expected_prepared_identity=identity,
                    code_revision="a" * 40,
                    glove=glove,
                    campaign_name="ml02-native-v3",
                    native_train_device="cpu",
                    physical_storage_root=root,
                )
            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(manifest["status"], "incomplete")
            self.assertEqual(
                json.loads((output / "campaign-report.json").read_text())["failure_code"],
                "research_campaign_storage_low",
            )
            self.assertEqual(launches, ["recsys.research.campaign.worker"])
            first = output / "variants" / "popularity" / "attempts" / "fit" / "1"
            self.assertTrue((first / "artifact" / "model-manifest.json").is_file())
            events = _Journal(output / "journal.jsonl").events
            self.assertEqual(
                next(e for e in events if e["event"] == "attempt_finished")["verification_spawned"],
                False,
            )
            self.assertTrue(
                cast(
                    "dict[str, object]",
                    next(e for e in events if e["event"] == "attempt_finished")["resources"],
                )["cleanup_complete"]
            )
            self.assertGreater(
                next(e for e in events if e["event"] == "attempt_finished")["charged_seconds"], 0
            )
            with (
                patch_release_info(fixture.lock),
                patch("recsys.research.resources.ResourceSupervisor", InlineSupervisor),
                patch(
                    "recsys.research.campaign.storage._filesystem",
                    lambda path: (path.stat().st_dev, 1 << 40),
                ),
            ):
                resumed = run_campaign(
                    prepared,
                    output,
                    expected_prepared_identity=identity,
                    code_revision="a" * 40,
                    glove=glove,
                    campaign_name="ml02-native-v3",
                    native_train_device="cpu",
                    physical_storage_root=root,
                    resume=True,
                    expected_campaign_identity=manifest["plan_identity"],
                )
            self.assertEqual(json.loads(resumed.read_text())["status"], "complete")

    def test_execution_storage_gate_records_true_no_spawn_before_worker(self) -> None:
        import recsys.research.campaign.runner as runner_module

        with tempfile.TemporaryDirectory() as temporary:
            fixture = SeedTrainingFixture(temporary)
            root = Path(temporary)
            glove = root / "glove.txt"
            fixture.lock["inputs"]["embeddings"] = _glove(glove)
            prepared, identity = fixture.make_prepared()
            output = root / "native-execution-storage-trip"
            state = {"free": 1 << 40}
            original_request = runner_module._worker_request

            def current_space(path: Path) -> tuple[int, int]:
                return path.stat().st_dev, state["free"]

            def drop_before_launch(
                root_path: Path,
                plan_identity: str,
                generation: str,
                variant: str,
                phase: str,
                attempt: int,
                prepared_dir: Path,
                glove_path: Path,
                model_dir: Path | None,
                prediction_dir: Path | None,
                content_model: Path | None,
                svd_model: Path | None,
                *,
                campaign_name: CampaignName = "ml02-cpu-v1",
            ) -> WorkerRequest:
                request = original_request(
                    root_path,
                    plan_identity,
                    generation,
                    variant,
                    phase,
                    attempt,
                    prepared_dir,
                    glove_path,
                    model_dir,
                    prediction_dir,
                    content_model,
                    svd_model,
                    campaign_name=campaign_name,
                )
                state["free"] = STOP_BYTES - 1
                return request

            with (
                patch_release_info(fixture.lock),
                patch("recsys.research.resources.ResourceSupervisor", InlineSupervisor),
                patch("recsys.research.campaign.storage._filesystem", current_space),
                patch("recsys.research.campaign.runner._worker_request", drop_before_launch),
            ):
                manifest_path = run_campaign(
                    prepared,
                    output,
                    expected_prepared_identity=identity,
                    code_revision="a" * 40,
                    glove=glove,
                    campaign_name="ml02-native-v3",
                    native_train_device="cpu",
                    physical_storage_root=root,
                )
            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(manifest["status"], "incomplete")
            events = _Journal(output / "journal.jsonl").events
            self.assertEqual(
                next(e for e in events if e["event"] == "attempt_finished")["execution_spawned"],
                False,
            )
            self.assertGreater(
                next(e for e in events if e["event"] == "attempt_finished")["charged_seconds"], 0
            )
            first = output / "variants" / "popularity" / "attempts" / "fit" / "1"
            self.assertFalse((first / "artifact").exists())
            self.assertFalse((first / "logs").exists())
            with (
                patch_release_info(fixture.lock),
                patch("recsys.research.resources.ResourceSupervisor", InlineSupervisor),
                patch(
                    "recsys.research.campaign.storage._filesystem",
                    lambda path: (path.stat().st_dev, 1 << 40),
                ),
            ):
                resumed = run_campaign(
                    prepared,
                    output,
                    expected_prepared_identity=identity,
                    code_revision="a" * 40,
                    glove=glove,
                    campaign_name="ml02-native-v3",
                    native_train_device="cpu",
                    physical_storage_root=root,
                    resume=True,
                    expected_campaign_identity=manifest["plan_identity"],
                )
            self.assertEqual(json.loads(resumed.read_text())["status"], "complete")

    def test_running_process_interruption_is_closed_charged_and_resumable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SeedTrainingFixture(temporary)
            root = Path(temporary)
            glove = root / "glove.txt"
            fixture.lock["inputs"]["embeddings"] = _glove(glove)
            prepared, identity = fixture.make_prepared()
            output = root / "native-real-interruption"
            sleeper = root / "sleeping-python"
            sleeper.write_text("#!/usr/bin/env python3\nimport time\ntime.sleep(60)\n")
            sleeper.chmod(0o700)

            def generous(path: Path) -> tuple[int, int]:
                return path.stat().st_dev, 1 << 40

            with (
                patch_release_info(fixture.lock),
                patch("recsys.research.resources.ResourceSupervisor", InterruptedRealSupervisor),
                patch("recsys.research.campaign.storage._filesystem", generous),
                patch("recsys.research.campaign.runner.sys.executable", str(sleeper)),
            ):
                manifest_path = run_campaign(
                    prepared,
                    output,
                    expected_prepared_identity=identity,
                    code_revision="a" * 40,
                    glove=glove,
                    campaign_name="ml02-native-v3",
                    native_train_device="cpu",
                    physical_storage_root=root,
                )
            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(manifest["status"], "incomplete")
            events = _Journal(output / "journal.jsonl").events
            self.assertEqual(events[3]["event"], "attempt_interrupted")
            execution = cast("dict[str, object]", events[3]["resources"])
            self.assertTrue(execution["cleanup_complete"])
            self.assertEqual(execution["outcome"], "cancelled")
            spent = events[3]["charged_seconds"]
            self.assertGreater(spent, 0)
            self.assertFalse(
                (
                    output / "variants" / "popularity" / "attempts" / "fit" / "1" / "artifact"
                ).exists()
            )
            with (
                patch_release_info(fixture.lock),
                patch("recsys.research.resources.ResourceSupervisor", InlineSupervisor),
                patch("recsys.research.campaign.storage._filesystem", generous),
            ):
                resumed = run_campaign(
                    prepared,
                    output,
                    expected_prepared_identity=identity,
                    code_revision="a" * 40,
                    glove=glove,
                    campaign_name="ml02-native-v3",
                    native_train_device="cpu",
                    physical_storage_root=root,
                    resume=True,
                    expected_campaign_identity=manifest["plan_identity"],
                )
            self.assertEqual(json.loads(resumed.read_text())["status"], "complete")
            retry = next(
                event
                for event in _Journal(output / "journal.jsonl").events
                if event["event"] == "attempt_started" and event["attempt"] == 2
            )
            self.assertAlmostEqual(
                cast("float", cast("dict[str, object]", retry)["remaining_seconds"]),
                FIT_SECONDS - spent,
                delta=1.0,
            )

    def test_verifier_timeout_rejects_published_artifact_and_charges_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SeedTrainingFixture(temporary)
            root = Path(temporary)
            glove = root / "glove.txt"
            fixture.lock["inputs"]["embeddings"] = _glove(glove)
            prepared, identity = fixture.make_prepared()
            output = root / "native-verifier-timeout"

            def generous(path: Path) -> tuple[int, int]:
                return path.stat().st_dev, 1 << 40

            with (
                patch_release_info(fixture.lock),
                patch("recsys.research.resources.ResourceSupervisor", TimedOutVerifierSupervisor),
                patch("recsys.research.campaign.storage._filesystem", generous),
            ):
                manifest_path = run_campaign(
                    prepared,
                    output,
                    expected_prepared_identity=identity,
                    code_revision="a" * 40,
                    glove=glove,
                    campaign_name="ml02-native-v3",
                    native_train_device="cpu",
                    physical_storage_root=root,
                )
            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(manifest["status"], "incomplete")
            report = json.loads((output / "campaign-report.json").read_text())
            self.assertEqual(report["failure_code"], "research_campaign_verification_failed")
            first = output / "variants" / "popularity" / "attempts" / "fit" / "1"
            self.assertTrue((first / "artifact" / "model-manifest.json").is_file())
            self.assertTrue((first / "verification-result.json").is_file())
            events = _Journal(output / "journal.jsonl").events
            successes, attempts, charged, starts, failed = _successes(
                events,
                variants=(
                    "popularity",
                    "content-mean",
                    "content-max",
                    "content-union",
                    "itemknn",
                    "puresvd",
                    "hybrid",
                    "item-seed-mean-bpr-v1",
                ),
            )
            self.assertFalse(successes)
            self.assertFalse(starts)
            self.assertEqual(failed, {("popularity", "fit")})
            self.assertEqual(attempts[("popularity", "fit")], 1)
            elapsed = charged[("popularity", "fit")]
            self.assertGreaterEqual(elapsed, 0.15)
            self.assertLess(elapsed, 30.0)
            self.assertEqual(
                events[-2]["charged_seconds"], elapsed
            )  # ProcessResult.elapsed_seconds=900 is not added.
            with (
                patch_release_info(fixture.lock),
                patch("recsys.research.resources.ResourceSupervisor", InlineSupervisor),
                patch("recsys.research.campaign.storage._filesystem", generous),
            ):
                resumed = run_campaign(
                    prepared,
                    output,
                    expected_prepared_identity=identity,
                    code_revision="a" * 40,
                    glove=glove,
                    campaign_name="ml02-native-v3",
                    native_train_device="cpu",
                    physical_storage_root=root,
                    resume=True,
                    expected_campaign_identity=manifest["plan_identity"],
                )
            self.assertEqual(json.loads(resumed.read_text())["status"], "complete")
            resumed_events = _Journal(output / "journal.jsonl").events
            retry = next(
                event
                for event in resumed_events
                if event["event"] == "attempt_started"
                and event["variant"] == "popularity"
                and event["phase"] == "fit"
                and event["attempt"] == 2
            )
            self.assertAlmostEqual(
                cast("float", cast("dict[str, object]", retry)["remaining_seconds"]),
                FIT_SECONDS - elapsed,
                delta=1.0,
            )

    def test_clean_verifier_interruption_auto_records_charge_and_resumes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SeedTrainingFixture(temporary)
            root = Path(temporary)
            glove = root / "glove.txt"
            fixture.lock["inputs"]["embeddings"] = _glove(glove)
            prepared, identity = fixture.make_prepared()
            output = root / "native-uncertain-verifier"
            sleeper = root / "sleeping-verifier-python"
            sleeper.write_text("#!/usr/bin/env python3\nimport time\ntime.sleep(60)\n")
            sleeper.chmod(0o700)

            def generous(path: Path) -> tuple[int, int]:
                return path.stat().st_dev, 1 << 40

            with (
                patch_release_info(fixture.lock),
                patch("recsys.research.resources.ResourceSupervisor", RunningVerifierSupervisor),
                patch("recsys.research.campaign.storage._filesystem", generous),
                patch("recsys.research.campaign.runner.sys.executable", str(sleeper)),
            ):
                interrupted = run_campaign(
                    prepared,
                    output,
                    expected_prepared_identity=identity,
                    code_revision="a" * 40,
                    glove=glove,
                    campaign_name="ml02-native-v3",
                    native_train_device="cpu",
                    physical_storage_root=root,
                )
            self.assertEqual(json.loads(interrupted.read_text())["status"], "incomplete")
            self.assertTrue(
                (
                    output
                    / "variants"
                    / "popularity"
                    / "attempts"
                    / "fit"
                    / "1"
                    / "artifact"
                    / "model-manifest.json"
                ).is_file()
            )
            events = _Journal(output / "journal.jsonl").events
            self.assertEqual(
                [event["event"] for event in events],
                [
                    "run_started",
                    "storage_admission",
                    "attempt_started",
                    "attempt_interrupted",
                    "campaign_finished",
                ],
            )
            interruption = events[3]
            self.assertTrue(
                cast("dict[str, object]", interruption["resources"])["cleanup_complete"]
            )
            self.assertTrue(
                cast("dict[str, object]", interruption["verification_resources"])[
                    "cleanup_complete"
                ]
            )
            spent = interruption["charged_seconds"]
            self.assertGreater(spent, 0)
            plan_identity = (
                "sha256:" + hashlib.sha256((output / "campaign-plan.json").read_bytes()).hexdigest()
            )
            with (
                patch_release_info(fixture.lock),
                patch("recsys.research.resources.ResourceSupervisor", InlineSupervisor),
                patch("recsys.research.campaign.storage._filesystem", generous),
            ):
                resumed = run_campaign(
                    prepared,
                    output,
                    expected_prepared_identity=identity,
                    code_revision="a" * 40,
                    glove=glove,
                    campaign_name="ml02-native-v3",
                    native_train_device="cpu",
                    physical_storage_root=root,
                    resume=True,
                    expected_campaign_identity=plan_identity,
                )
            self.assertEqual(json.loads(resumed.read_text())["status"], "complete")
            retry = next(
                event
                for event in _Journal(output / "journal.jsonl").events
                if event["event"] == "attempt_started"
                and event["variant"] == "popularity"
                and event["phase"] == "fit"
                and event["attempt"] == 2
            )
            self.assertAlmostEqual(
                cast("float", cast("dict[str, object]", retry)["remaining_seconds"]),
                FIT_SECONDS - spent,
                delta=1.0,
            )

    def test_interrupted_attempt_without_cleanup_proof_refuses_resume(self) -> None:
        cases = (
            (None, None),
            ({"cleanup_complete": False}, None),
            ({"cleanup_complete": True}, {"cleanup_complete": False}),
        )
        for evidence, verification in cases:
            with (
                self.subTest(evidence=evidence, verification=verification),
                tempfile.TemporaryDirectory() as temporary,
            ):
                fixture = SeedTrainingFixture(temporary)
                root = Path(temporary)
                glove = root / "glove.txt"
                fixture.lock["inputs"]["embeddings"] = _glove(glove)
                prepared, identity = fixture.make_prepared()
                output = root / "native-uncertain-interruption"

                def generous(path: Path) -> tuple[int, int]:
                    return path.stat().st_dev, 1 << 40

                with (
                    patch_release_info(fixture.lock),
                    patch(
                        "recsys.research.resources.ResourceSupervisor",
                        UnprovenCleanupVerifierSupervisor,
                    ),
                    patch("recsys.research.campaign.storage._filesystem", generous),
                ):
                    incomplete = run_campaign(
                        prepared,
                        output,
                        expected_prepared_identity=identity,
                        code_revision="a" * 40,
                        glove=glove,
                        campaign_name="ml02-native-v3",
                        native_train_device="cpu",
                        physical_storage_root=root,
                    )
                self.assertEqual(json.loads(incomplete.read_text())["status"], "incomplete")
                journal = _Journal(output / "journal.jsonl")
                self.assertEqual(
                    [event["event"] for event in journal.events],
                    ["run_started", "storage_admission", "attempt_started", "campaign_finished"],
                )
                journal.append(
                    "attempt_interrupted",
                    generation=journal.events[1]["generation"],
                    variant="popularity",
                    phase="fit",
                    attempt=1,
                    charged_seconds=120.0,
                    resources=evidence,
                    verification_resources=verification,
                )
                plan_identity = (
                    "sha256:"
                    + hashlib.sha256((output / "campaign-plan.json").read_bytes()).hexdigest()
                )
                with (
                    patch_release_info(fixture.lock),
                    patch("recsys.research.campaign.storage._filesystem", generous),
                ):
                    with self.assertRaisesRegex(
                        ResearchError, "research_campaign_interrupted_uncertain"
                    ):
                        run_campaign(
                            prepared,
                            output,
                            expected_prepared_identity=identity,
                            code_revision="a" * 40,
                            glove=glove,
                            campaign_name="ml02-native-v3",
                            native_train_device="cpu",
                            physical_storage_root=root,
                            resume=True,
                            expected_campaign_identity=plan_identity,
                        )

    def test_inapplicable_request_path_rejected_before_fit(self) -> None:
        import recsys.research.campaign.runner as runner_module

        with tempfile.TemporaryDirectory() as temporary:
            fixture = SeedTrainingFixture(temporary)
            root = Path(temporary)
            glove = root / "glove.txt"
            fixture.lock["inputs"]["embeddings"] = _glove(glove)
            prepared, identity = fixture.make_prepared()
            output = root / "native-bad-request"
            original_request = runner_module._worker_request

            def corrupted_request(
                root_path: Path,
                plan_identity: str,
                generation: str,
                variant: str,
                phase: str,
                attempt: int,
                prepared_dir: Path,
                glove_path: Path,
                model_dir: Path | None,
                prediction_dir: Path | None,
                content_model: Path | None,
                svd_model: Path | None,
                *,
                campaign_name: CampaignName = "ml02-cpu-v1",
            ) -> WorkerRequest:
                request = original_request(
                    root_path,
                    plan_identity,
                    generation,
                    variant,
                    phase,
                    attempt,
                    prepared_dir,
                    glove_path,
                    model_dir,
                    prediction_dir,
                    content_model,
                    svd_model,
                    campaign_name=campaign_name,
                )
                request["model_dir"] = "/tmp/inapplicable"
                return request

            def generous(path: Path) -> tuple[int, int]:
                return path.stat().st_dev, 1 << 40

            with (
                patch_release_info(fixture.lock),
                patch("recsys.research.resources.ResourceSupervisor", InlineSupervisor),
                patch("recsys.research.campaign.storage._filesystem", generous),
                patch("recsys.research.campaign.runner._worker_request", corrupted_request),
                patch("recsys.research.models.package.fit_model") as forbidden_fit,
            ):
                result = run_campaign(
                    prepared,
                    output,
                    expected_prepared_identity=identity,
                    code_revision="a" * 40,
                    glove=glove,
                    campaign_name="ml02-native-v3",
                    native_train_device="cpu",
                    physical_storage_root=root,
                )
            forbidden_fit.assert_not_called()
            self.assertEqual(json.loads(result.read_text())["status"], "incomplete")
            self.assertFalse(list(output.glob("variants/*/attempts/fit/1/artifact")))

    def test_low_physical_space_publishes_incomplete_without_starting_science(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SeedTrainingFixture(temporary)
            root = Path(temporary)
            glove = root / "glove.txt"
            fixture.lock["inputs"]["embeddings"] = _glove(glove)
            prepared, identity = fixture.make_prepared()
            output = root / "native-low-space"

            def low_space(path: Path) -> tuple[int, int]:
                return path.stat().st_dev, STOP_BYTES - 1

            with (
                patch_release_info(fixture.lock),
                patch("recsys.research.resources.ResourceSupervisor", InlineSupervisor),
                patch("recsys.research.campaign.storage._filesystem", low_space),
            ):
                result = run_campaign(
                    prepared,
                    output,
                    expected_prepared_identity=identity,
                    code_revision="a" * 40,
                    glove=glove,
                    campaign_name="ml02-native-v3",
                    native_train_device="cpu",
                    physical_storage_root=root,
                )
            manifest = json.loads(result.read_text())
            report = json.loads((output / "campaign-report.json").read_text())
            self.assertEqual(manifest["status"], "incomplete")
            self.assertEqual(report["failure_code"], "research_campaign_storage_low")
            self.assertFalse((output / "variants").exists())
            admissions = [
                event
                for event in _Journal(output / "journal.jsonl").events
                if event["event"] == "storage_admission"
            ]
            self.assertEqual(len(admissions), 1)
            self.assertEqual(
                admissions[0]["remaining_variants"],
                [
                    "content-max",
                    "content-mean",
                    "content-union",
                    "hybrid",
                    "item-seed-mean-bpr-v1",
                    "itemknn",
                    "popularity",
                    "puresvd",
                ],
            )
            self.assertLess(
                admissions[0]["physical_available_bytes"], admissions[0]["required_bytes"]
            )

    def test_coordinator_plus_worker_rss_stops_acceptance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SeedTrainingFixture(temporary)
            root = Path(temporary)
            glove = root / "glove.txt"
            fixture.lock["inputs"]["embeddings"] = _glove(glove)
            prepared, identity = fixture.make_prepared()
            output = root / "native-rss-limit"

            def generous(path: Path) -> tuple[int, int]:
                return path.stat().st_dev, 1 << 40

            with (
                patch_release_info(fixture.lock),
                patch("recsys.research.resources.ResourceSupervisor", InlineSupervisor),
                patch("recsys.research.campaign.storage._filesystem", generous),
                patch(
                    "recsys.research.campaign.runner._coordinator_rss_bytes", return_value=8 * GIB
                ),
            ):
                result = run_campaign(
                    prepared,
                    output,
                    expected_prepared_identity=identity,
                    code_revision="a" * 40,
                    glove=glove,
                    campaign_name="ml02-native-v3",
                    native_train_device="cpu",
                    physical_storage_root=root,
                )
            report = json.loads((output / "campaign-report.json").read_text())
            self.assertEqual(json.loads(result.read_text())["status"], "incomplete")
            self.assertEqual(report["failure_code"], "research_campaign_aggregate_rss_limit")
            successes, _, _, _, _ = _successes(
                _Journal(output / "journal.jsonl").events,
                variants=(
                    "popularity",
                    "content-mean",
                    "content-max",
                    "content-union",
                    "itemknn",
                    "puresvd",
                    "hybrid",
                    "item-seed-mean-bpr-v1",
                ),
            )
            self.assertFalse(successes)

    def test_all_eight_arms_and_supervised_verification(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SeedTrainingFixture(temporary)
            root = Path(temporary)
            glove = root / "glove.txt"
            fixture.lock["inputs"]["embeddings"] = _glove(glove)
            prepared, identity = fixture.make_prepared()
            output = root / "native-campaign"
            original = os.stat

            def generous_filesystem(path: Path) -> tuple[int, int]:
                return original(path).st_dev, 1 << 40

            with (
                patch_release_info(fixture.lock),
                patch("recsys.research.resources.ResourceSupervisor", InlineSupervisor),
                patch("recsys.research.campaign.storage._filesystem", generous_filesystem),
            ):
                result = run_campaign(
                    prepared,
                    output,
                    expected_prepared_identity=identity,
                    code_revision="a" * 40,
                    glove=glove,
                    campaign_name="ml02-native-v3",
                    native_train_device="cpu",
                    physical_storage_root=root,
                )
                manifest = json.loads(result.read_text())
                report = validate_campaign(
                    prepared,
                    output,
                    expected_prepared_identity=identity,
                    expected_campaign_identity=manifest["plan_identity"],
                )
                report_path = output / "campaign-report.json"
                manifest_path = output / "campaign-report-manifest.json"
                saved_report, saved_manifest = report_path.read_bytes(), manifest_path.read_bytes()
                poisoned = json.loads(saved_report)
                poisoned["variants"][0]["private_input"] = "must-never-publish"
                poisoned_raw = _canonical(poisoned)
                changed_manifest = json.loads(saved_manifest)
                changed_manifest["report"] = {
                    "bytes": len(poisoned_raw),
                    "sha256": hashlib.sha256(poisoned_raw).hexdigest(),
                }
                report_path.write_bytes(poisoned_raw)
                manifest_path.write_bytes(_canonical(changed_manifest))
                try:
                    with self.assertRaises(ResearchError):
                        validate_campaign(
                            prepared,
                            output,
                            expected_prepared_identity=identity,
                            expected_campaign_identity=manifest["plan_identity"],
                        )
                finally:
                    report_path.write_bytes(saved_report)
                    manifest_path.write_bytes(saved_manifest)
                resumed = run_campaign(
                    prepared,
                    output,
                    expected_prepared_identity=identity,
                    code_revision="a" * 40,
                    glove=glove,
                    campaign_name="ml02-native-v3",
                    native_train_device="cpu",
                    physical_storage_root=root,
                    resume=True,
                    expected_campaign_identity=manifest["plan_identity"],
                )
            self.assertTrue(result.is_file())
            self.assertEqual(result, resumed)
            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["execution_config_count"], 8)
            self.assertEqual(report["planned_nodes"], 32)
            self.assertIsInstance(report["variants"], list)
            assert isinstance(report["variants"], list)
            self.assertEqual(len(report["variants"]), 8)
            self.assertEqual(report["actual_attempts_started"], 32)
            self.assertEqual(report["actual_attempts_completed"], 32)
            events = _Journal(output / "journal.jsonl").events
            successes, _, _, starts, failed = _successes(
                events, variants=tuple(entry["variant_id"] for entry in report["variants"])
            )
            self.assertEqual(len(successes), 32)
            self.assertFalse(starts)
            self.assertFalse(failed)
            for event in successes.values():
                execution = event.get("resources")
                verified = event.get("verification_resources")
                self.assertIsInstance(execution, dict)
                self.assertIsInstance(verified, dict)
                assert isinstance(execution, dict) and isinstance(verified, dict)
                self.assertIs(execution["cleanup_complete"], True)
                self.assertIs(verified["cleanup_complete"], True)
            self.assertEqual(
                len(list(output.glob("variants/*/attempts/*/1/verification-result.json"))), 32
            )
            self.assertEqual(len(list(output.glob("variants/*/attempts/*/1/artifact"))), 32)

    @unittest.skipUnless(
        os.environ.get("ARB_TEST_REAL_CAMPAIGN") == "1", "opt-in isolated worker campaign"
    )
    def test_real_isolated_workers_and_verifiers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SeedTrainingFixture(temporary)
            root = Path(temporary)
            glove = root / "glove.txt"
            fixture.lock["inputs"]["embeddings"] = _glove(glove)
            prepared, identity = fixture.make_prepared()
            wrapper = root / "fixture-python"
            subprocess.run(
                [sys.executable, "-m", "venv", str(wrapper)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            version = f"python{sys.version_info.major}.{sys.version_info.minor}"
            site = wrapper / "lib" / version / "site-packages"
            active_site = Path(sys.prefix) / "lib" / version / "site-packages"
            source = Path(__file__).resolve().parents[2] / "src"
            (site / "arb-native-fixture.pth").write_text(
                f"{active_site}\n{source}\nimport arb_native_fixture_hook\n"
            )
            (site / "arb_native_fixture_hook.py").write_text(
                "import json, os\n"
                "if 'ARB_TEST_LOCK_JSON' in os.environ:\n"
                " from pathlib import Path\n"
                " import recsys.release.lock as lock\n"
                " value = json.loads(Path(os.environ['ARB_TEST_LOCK_JSON']).read_text())\n"
                " lock.release_info = lambda: value\n"
            )
            lock_path = root / "fixture-lock.json"
            lock_path.write_text(json.dumps(fixture.lock, sort_keys=True))
            output = root / "native-real"
            with (
                patch_release_info(fixture.lock),
                patch(
                    "recsys.research.campaign.runner.sys.executable",
                    str(wrapper / "bin" / "python"),
                ),
                patch.dict(os.environ, {"ARB_TEST_LOCK_JSON": str(lock_path)}),
            ):
                result = run_campaign(
                    prepared,
                    output,
                    expected_prepared_identity=identity,
                    code_revision="a" * 40,
                    glove=glove,
                    campaign_name="ml02-native-v3",
                    native_train_device="cpu",
                    physical_storage_root=root,
                )
                manifest = json.loads(result.read_text())
                report = validate_campaign(
                    prepared,
                    output,
                    expected_prepared_identity=identity,
                    expected_campaign_identity=manifest["plan_identity"],
                )
            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["planned_nodes"], 32)
