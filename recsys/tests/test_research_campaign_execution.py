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
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

from research_model_support import patch_release_info
from test_research_model_content import _glove
from training_support import SeedTrainingFixture

from recsys.research.campaign.common import FIT_SECONDS
from recsys.research.campaign.journal import _Journal, _successes
from recsys.research.campaign.report import validate_campaign
from recsys.research.campaign.runner import _canonical, run_campaign
from recsys.research.campaign.spec import CampaignName
from recsys.research.campaign.storage import GIB
from recsys.research.campaign.verification import _verification_main
from recsys.research.campaign.worker import WorkerRequest, _worker_main
from recsys.research.protocol import ResearchError
from recsys.research.resources import ProcessResult


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

    def poll(self, *, wait_seconds: float) -> SimpleNamespace:
        completed = tuple(handle for handle in self.results if handle not in self.delivered)
        self.delivered.update(completed)
        return SimpleNamespace(completed=completed, sampled_rss_bytes=1000)

    def result(self, handle: object) -> ProcessResult:
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


class UncertainVerifierSupervisor(InlineSupervisor):
    """Lose proof of verifier cleanup after the worker successfully published."""

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
        return handle

    def poll(self, *, wait_seconds: float) -> SimpleNamespace:
        if self.uncertain_handle is not None:
            raise RuntimeError("simulated verifier supervision loss")
        return super().poll(wait_seconds=wait_seconds)


@unittest.skipUnless(os.environ.get("ARB_TEST_TORCH") == "1", "opt-in native CPU campaign")
class NativeCampaignExecutionTests(unittest.TestCase):
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
                    campaign_name="ml02-native-v2",
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
                    campaign_name="ml02-native-v2",
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

    def test_uncertain_verifier_cleanup_keeps_open_start_and_refuses_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SeedTrainingFixture(temporary)
            root = Path(temporary)
            glove = root / "glove.txt"
            fixture.lock["inputs"]["embeddings"] = _glove(glove)
            prepared, identity = fixture.make_prepared()
            output = root / "native-uncertain-verifier"

            def generous(path: Path) -> tuple[int, int]:
                return path.stat().st_dev, 1 << 40

            with (
                patch_release_info(fixture.lock),
                patch("recsys.research.resources.ResourceSupervisor", UncertainVerifierSupervisor),
                patch("recsys.research.campaign.storage._filesystem", generous),
            ):
                with self.assertRaisesRegex(RuntimeError, "simulated verifier supervision loss"):
                    run_campaign(
                        prepared,
                        output,
                        expected_prepared_identity=identity,
                        code_revision="a" * 40,
                        glove=glove,
                        campaign_name="ml02-native-v2",
                        native_train_device="cpu",
                        physical_storage_root=root,
                    )
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
                [event["event"] for event in events], ["run_started", "attempt_started"]
            )
            plan_identity = (
                "sha256:" + hashlib.sha256((output / "campaign-plan.json").read_bytes()).hexdigest()
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
                        campaign_name="ml02-native-v2",
                        native_train_device="cpu",
                        physical_storage_root=root,
                        resume=True,
                        expected_campaign_identity=plan_identity,
                    )
            _Journal(output / "journal.jsonl").append(
                "attempt_interrupted",
                generation=events[-1]["generation"],
                variant="popularity",
                phase="fit",
                attempt=1,
                charged_seconds=120.0,
                resources={"cleanup_complete": True},
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
                    campaign_name="ml02-native-v2",
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
                FIT_SECONDS - 120.0,
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
                        UncertainVerifierSupervisor,
                    ),
                    patch("recsys.research.campaign.storage._filesystem", generous),
                ):
                    with self.assertRaisesRegex(
                        RuntimeError, "simulated verifier supervision loss"
                    ):
                        run_campaign(
                            prepared,
                            output,
                            expected_prepared_identity=identity,
                            code_revision="a" * 40,
                            glove=glove,
                            campaign_name="ml02-native-v2",
                            native_train_device="cpu",
                            physical_storage_root=root,
                        )
                journal = _Journal(output / "journal.jsonl")
                journal.append(
                    "attempt_interrupted",
                    generation=journal.events[-1]["generation"],
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
                            campaign_name="ml02-native-v2",
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
                    campaign_name="ml02-native-v2",
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
                return path.stat().st_dev, 191 * GIB

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
                    campaign_name="ml02-native-v2",
                    native_train_device="cpu",
                    physical_storage_root=root,
                )
            manifest = json.loads(result.read_text())
            report = json.loads((output / "campaign-report.json").read_text())
            self.assertEqual(manifest["status"], "incomplete")
            self.assertEqual(report["failure_code"], "research_campaign_storage_low")
            self.assertFalse((output / "variants").exists())

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
                    campaign_name="ml02-native-v2",
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
                    campaign_name="ml02-native-v2",
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
                    campaign_name="ml02-native-v2",
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
                    campaign_name="ml02-native-v2",
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
