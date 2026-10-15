"""Seven-variant validation pipeline and durable campaign boundary on tiny inputs."""

import tempfile
import unittest
import os
import fcntl
import sys
import hashlib
import json
import subprocess
import time
import shutil
import uuid
import recsys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from recsys.research_artifacts import read_json
from recsys.research_campaign import (run_campaign, validate_campaign, _worker_main,
    _expected_nodes, _plan, _canonical, _digest, _private_file, _Journal,
    _worker_request, _worker_env, _remaining, _successes, FIT_SECONDS)
from recsys.research_protocol import ResearchError
from research_model_support import ModelFixture
from test_research_model_content import _glove


def _stable_utc_clock():
    # Stabilize only the in-process campaign controller against host UTC jumps.
    # Isolated workers and ResourceSupervisor retain their real clocks.
    start_utc, start_mono = datetime.now(timezone.utc), time.monotonic()
    return lambda: start_utc + timedelta(seconds=time.monotonic() - start_mono)


class _Supervisor:
    """Drive the real closed worker inline so fixture release metadata can be patched."""
    def __init__(self, *, rss_limit_bytes):
        self.results = {}
        self.delivered = set()

    def __enter__(self):
        return self

    def start(self, argv, *, cwd, log_dir, remaining_seconds, env):
        log_dir.mkdir(mode=0o700)
        code = _worker_main(Path(argv[-1]))
        handle = object()
        self.results[handle] = SimpleNamespace(
            outcome="ok" if code == 0 else "exit_nonzero", returncode=code,
            elapsed_seconds=.01, sampled_peak_rss_bytes=1000,
            wait4_peak_rss_bytes=1000, observed_peak_rss_bytes=1000,
            rss_samples=1, max_sample_gap_seconds=.01, cleanup_complete=True)
        return handle

    def poll(self, *, wait_seconds):
        complete = tuple(handle for handle in self.results if handle not in self.delivered)
        self.delivered.update(complete)
        return SimpleNamespace(completed=complete)

    def result(self, handle):
        return self.results[handle]

    def cancel(self, handle):
        pass

    def close(self):
        pass


class _InterimSupervisor(_Supervisor):
    starts = 0

    def __init__(self, *, rss_limit_bytes):
        super().__init__(rss_limit_bytes=rss_limit_bytes)
        self.polls = 0
        self.cancelled = set()

    def start(self, argv, *, cwd, log_dir, remaining_seconds, env):
        handle = object()
        self.results[handle] = None
        type(self).starts += 1
        return handle

    def poll(self, *, wait_seconds):
        self.polls += 1
        return SimpleNamespace(completed=tuple(self.results) if self.polls >= 2 else ())

    def result(self, handle):
        first = next(iter(self.results))
        if self.polls == 1 and handle is not first:
            return None
        if self.polls < 1:
            return None
        return SimpleNamespace(
            outcome="cleanup_failed" if handle is first else "cancelled",
            returncode=None, elapsed_seconds=.2 if self.polls == 1 else .3,
            sampled_peak_rss_bytes=1000, wait4_peak_rss_bytes=1000,
            observed_peak_rss_bytes=1000, rss_samples=1,
            max_sample_gap_seconds=.1, cleanup_complete=self.polls >= 2)

    def cancel(self, handle):
        self.cancelled.add(handle)


class CampaignTests(unittest.TestCase):
    def setUp(self):
        clock = patch("recsys.research_campaign._now", side_effect=_stable_utc_clock())
        clock.start()
        self.addCleanup(clock.stop)

    def test_published_orphan_without_success_event_blocks_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ModelFixture(temporary)
            glove = Path(temporary) / "glove.txt"
            fixture.lock["inputs"]["embeddings"] = _glove(glove)
            prepared, identity = fixture.make_prepared()
            output = Path(temporary) / "campaign"
            with patch("recsys.research_models.release_info", return_value=fixture.lock), \
                 patch("recsys.research_resources.ResourceSupervisor", _Supervisor):
                manifest_path = run_campaign(prepared, output,
                    expected_prepared_identity=identity, code_revision="a" * 40,
                    glove=glove, backend="cpu", cpu_workers=2)
                manifest, _ = read_json(manifest_path)
                old = output / "variants" / "popularity" / "attempts" / "fit" / "1" / "artifact"
                orphan = old.parent.parent / "2" / "artifact"
                orphan.parent.mkdir(mode=0o700)
                shutil.copytree(old, orphan)
                _Journal(output / "journal.jsonl").append(
                    "attempt_started", generation="crashed", variant="popularity",
                    phase="fit", attempt=2, remaining_seconds=FIT_SECONDS)
                before = (orphan / "model-manifest.json").read_bytes()
                with self.assertRaisesRegex(ResearchError, "research_campaign_interrupted_uncertain"):
                    run_campaign(prepared, output, expected_prepared_identity=identity,
                        code_revision="a" * 40, glove=glove, backend="cpu", cpu_workers=2,
                        resume=True, expected_campaign_identity=manifest["plan_identity"])
                self.assertEqual((orphan / "model-manifest.json").read_bytes(), before)

    def test_shared_auxiliary_fit_budget_and_journal_clock_regression(self):
        with tempfile.TemporaryDirectory() as temporary:
            journal = _Journal(Path(temporary) / "journal.jsonl")
            first = datetime.now(timezone.utc) - timedelta(seconds=10)
            with patch("recsys.research_campaign._now", return_value=first):
                journal.append("run_started", generation="g")
            original = journal.path.read_bytes()
            with patch("recsys.research_campaign._now", return_value=first - timedelta(seconds=1)):
                with self.assertRaisesRegex(ResearchError, "research_campaign_clock_invalid"):
                    journal.append("run_started", generation="later")
            self.assertEqual(journal.path.read_bytes(), original)

            journal.append("attempt_started", generation="g", variant="content-mean",
                           phase="fit", attempt=1, remaining_seconds=FIT_SECONDS)
            journal.append("attempt_interrupted", generation="g", variant="content-mean",
                           phase="fit", attempt=1, charged_seconds=300.0,
                           resources={"cleanup_complete": True})
            journal.append("attempt_started", generation="h", variant="content-mean",
                           phase="fit", attempt=2, remaining_seconds=FIT_SECONDS - 300)
            journal.append("attempt_finished", generation="h", variant="content-mean",
                           phase="fit", attempt=2, charged_seconds=500.0,
                           status="success", resources={"cleanup_complete": True},
                           output_identity="sha256:" + "a" * 64)
            journal.append("attempt_started", generation="h", variant="content-mean",
                           phase="fit_cpu_component", attempt=1,
                           remaining_seconds=FIT_SECONDS - 800)
            journal.append("attempt_interrupted", generation="h", variant="content-mean",
                           phase="fit_cpu_component", attempt=1, charged_seconds=400.0,
                           resources={"cleanup_complete": True})
            successes, attempts, charged, starts, failed = _successes(journal.events)
            self.assertFalse(starts)
            self.assertFalse(failed)
            self.assertEqual(attempts[("content-mean", "fit")], 2)
            self.assertIn(("content-mean", "fit"), successes)
            plan = {"deadline_utc": (first + timedelta(days=1)).isoformat().replace("+00:00", "Z")}
            self.assertAlmostEqual(_remaining(plan, charged, "content-mean", "fit_cpu_component"),
                                   FIT_SECONDS - 1200, delta=10)

    def test_interim_cleanup_freezes_scheduling_and_charges_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ModelFixture(temporary)
            glove = Path(temporary) / "glove.txt"
            fixture.lock["inputs"]["embeddings"] = _glove(glove)
            prepared, identity = fixture.make_prepared()
            output = Path(temporary) / "campaign"
            _InterimSupervisor.starts = 0
            with patch("recsys.research_models.release_info", return_value=fixture.lock), \
                 patch("recsys.research_resources.ResourceSupervisor", _InterimSupervisor):
                path = run_campaign(prepared, output, expected_prepared_identity=identity,
                                    code_revision="a" * 40, glove=glove,
                                    backend="cpu", cpu_workers=2)
            report, _ = read_json(output / "campaign-report.json")
            self.assertEqual(report["status"], "incomplete")
            self.assertEqual(report["failure_code"], "research_campaign_cleanup_failed")
            self.assertEqual(_InterimSupervisor.starts, 2)
            events = _Journal(output / "journal.jsonl").events
            self.assertEqual(len([event for event in events
                                  if event["event"] == "cleanup_warning"]), 1)
            finishes = [event for event in events if event["event"] == "attempt_finished"]
            self.assertEqual(len(finishes), 2)
            self.assertTrue(all(event["status"] == "failed" for event in finishes))
            self.assertTrue(all(event["charged_seconds"] == .3 for event in finishes))

    def test_seven_real_variants_replay_public_suppression_and_resume_binding(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ModelFixture(temporary)
            glove = Path(temporary) / "glove.txt"
            fixture.lock["inputs"]["embeddings"] = _glove(glove)
            prepared, identity = fixture.make_prepared()
            output = Path(temporary) / "campaign"
            with patch("recsys.research_models.release_info", return_value=fixture.lock), \
                 patch("recsys.research_resources.ResourceSupervisor", _Supervisor):
                manifest_path = run_campaign(prepared, output,
                    expected_prepared_identity=identity, code_revision="a" * 40,
                    glove=glove, backend="cpu", cpu_workers=2)
                manifest, _ = read_json(manifest_path)
                plan, _ = read_json(output / "campaign-plan.json")
                self.assertEqual(manifest["status"], "complete")
                self.assertEqual(plan["execution_config_count"], 7)
                self.assertEqual(plan["auxiliary_fits"], [])
                self.assertEqual(len(_expected_nodes(plan)), 28)
                self.assertEqual(len(list((output / "variants").iterdir())), 7)
                report = validate_campaign(prepared, output,
                    expected_prepared_identity=identity,
                    expected_campaign_identity=manifest["plan_identity"])
                self.assertEqual(report["status"], "complete")
                self.assertEqual(len(report["variants"]), 7)
                self.assertEqual(report["planned_nodes"], 28)
                self.assertEqual(report["actual_completed_model_packages"], 7)
                self.assertEqual(report["actual_attempts_started"], 28)
                self.assertEqual(report["actual_attempts_completed"], 28)
                self.assertIsNone(report["primary_ndcg5"])
                self.assertIsNone(report["paired_vs_popularity"])
                self.assertTrue(report["small_cohort_suppressed"])
                self.assertEqual(report["diagnostics"]["single_query_latency"], "not_measured")
                self.assertNotIn("chosen_n", str(report))
                self.assertNotIn("u1_", str(report))
                self.assertNotIn("q1_", str(report))
                self.assertNotIn(str(output), str(report))
                before = (output / "journal.jsonl").read_bytes()
                run_campaign(prepared, output, expected_prepared_identity=identity,
                    code_revision="a" * 40, glove=glove, backend="cpu", cpu_workers=2,
                    resume=True, expected_campaign_identity=manifest["plan_identity"])
                self.assertGreater(len((output / "journal.jsonl").read_bytes()), len(before))
                gate = os.open(output / ".workers.lock", os.O_RDWR)
                try:
                    fcntl.flock(gate, fcntl.LOCK_SH)
                    with self.assertRaisesRegex(ResearchError, "research_campaign_workers_active"):
                        run_campaign(prepared, output, expected_prepared_identity=identity,
                            code_revision="a" * 40, glove=glove, backend="cpu", cpu_workers=2,
                            resume=True, expected_campaign_identity=manifest["plan_identity"])
                finally:
                    os.close(gate)
                with self.assertRaises(ResearchError):
                    run_campaign(prepared, output, expected_prepared_identity=identity,
                        code_revision="b" * 40, glove=glove, backend="cpu", cpu_workers=2,
                        resume=True, expected_campaign_identity=manifest["plan_identity"])
                self.assertAlmostEqual(_remaining(plan, {("content-mean", "fit"): 1000.0},
                                                  "content-mean", "fit"), FIT_SECONDS - 1000,
                                       delta=10)
                replay = output / "variants" / "popularity" / "attempts" / "replay" / "1" / "artifact" / "report.json"
                original_replay = replay.read_bytes()
                replay.write_bytes(original_replay + b"x")
                with self.assertRaises(ResearchError):
                    validate_campaign(prepared, output, expected_prepared_identity=identity,
                                      expected_campaign_identity=manifest["plan_identity"])
                replay.write_bytes(original_replay)
                damaged = next((output / "variants" / "hybrid" / "attempts" / "fit" / "1" /
                                "artifact").glob("content_*.npy"))
                with damaged.open("ab") as stream:
                    stream.write(b"x")
                with self.assertRaises(ResearchError):
                    validate_campaign(prepared, output, expected_prepared_identity=identity,
                                      expected_campaign_identity=manifest["plan_identity"])
                with self.assertRaises(ResearchError):
                    run_campaign(prepared, output, expected_prepared_identity=identity,
                        code_revision="a" * 40, glove=glove, backend="cpu", cpu_workers=2,
                        resume=True, expected_campaign_identity=manifest["plan_identity"])

    def test_stale_generation_real_isolated_worker_never_reaches_science(self):
        from recsys.research_resources import ResourceSupervisor

        with tempfile.TemporaryDirectory() as temporary:
            fixture = ModelFixture(temporary)
            glove = Path(temporary) / "glove.txt"
            fixture.lock["inputs"]["embeddings"] = _glove(glove)
            prepared, identity = fixture.make_prepared()
            root = Path(temporary) / "worker-root"
            root.mkdir(mode=0o700)
            with patch("recsys.research_models.release_info", return_value=fixture.lock):
                plan = _plan(prepared, identity, "a" * 40, glove, "cpu", 1,
                             __import__("datetime").datetime.now(__import__("datetime").timezone.utc))
            raw = _canonical(plan)
            _private_file(root / "campaign-plan.json", raw)
            _private_file(root / "generation.json", _canonical({"generation": "new"}))
            gate = os.open(root / ".workers.lock", os.O_RDWR | os.O_CREAT, 0o600)
            os.close(gate)
            attempt = root / "variants" / "popularity" / "attempts" / "fit" / "1"
            attempt.mkdir(parents=True, mode=0o700)
            for parent in (attempt.parent, attempt.parent.parent, attempt.parent.parent.parent):
                parent.chmod(0o700)
            request = _worker_request(root, _digest(raw), "old", "popularity", "fit", 1,
                                      prepared, glove, None, None, None, None)
            _private_file(attempt / "request.json", _canonical(request))
            journal = _Journal(root / "journal.jsonl")
            journal.append("attempt_started", generation="old", variant="popularity",
                           phase="fit", attempt=1, remaining_seconds=5.0)
            with ResourceSupervisor() as supervisor:
                handle = supervisor.start(
                    [sys.executable, "-I", "-m", "recsys.research_campaign",
                     "--worker-request", str(attempt / "request.json")],
                    cwd=root, log_dir=attempt / "logs", remaining_seconds=5.0,
                    env=_worker_env())
                for _ in range(100):
                    supervisor.poll(wait_seconds=.1)
                    result = supervisor.result(handle)
                    if result is not None and result.cleanup_complete:
                        break
                self.assertEqual(result.outcome, "exit_nonzero")
                self.assertEqual(result.returncode, 1)
            receipt, _ = read_json(attempt / "result.json")
            self.assertEqual(receipt["code"], "research_campaign_generation_stale")
            self.assertFalse((attempt / "artifact").exists())


def _isolated_fixture_python(root, lock):
    """Disposable Python shares installed wheels; a test .pth hook supplies the tiny lock."""
    executable = root / "fixture-python" / "bin" / "python"
    subprocess.run([sys.executable, "-m", "venv", str(executable.parent.parent)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    site = executable.parent.parent / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    installed = Path(recsys.__file__).parent.parent
    (site / "recsys-installed.pth").write_text(str(installed) + "\nimport arb_fixture_hook\n")
    (site / "nvidia").symlink_to(installed / "nvidia", target_is_directory=True)
    (site / "arb_fixture_hook.py").write_text(
        "import json, os\n"
        "if 'ARB_TEST_LOCK_JSON' in os.environ:\n"
        " from pathlib import Path\n"
        " import recsys.research_models as models\n"
        " fixture_lock = json.loads(Path(os.environ['ARB_TEST_LOCK_JSON']).read_text())\n"
        " models.release_info = lambda: fixture_lock\n")
    lock_path = root / "tiny-release-lock.json"
    lock_path.write_text(json.dumps(lock, sort_keys=True))
    return executable, lock_path


class RealProcessCampaignTests(unittest.TestCase):
    def _run(self, backend):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ModelFixture(temporary)
            root = Path(temporary)
            glove = root / "glove.txt"
            fixture.lock["inputs"]["embeddings"] = _glove(glove)
            prepared, identity = fixture.make_prepared()
            executable, lock_path = _isolated_fixture_python(root, fixture.lock)
            output = root / f"{backend}-real-campaign"
            with patch("recsys.research_models.release_info", return_value=fixture.lock), \
                 patch("recsys.research_campaign.sys.executable", str(executable)), \
                 patch("recsys.research_campaign._now", side_effect=_stable_utc_clock()), \
                 patch.dict(os.environ, {"ARB_TEST_LOCK_JSON": str(lock_path)}):
                manifest_path = run_campaign(prepared, output,
                    expected_prepared_identity=identity, code_revision="a" * 40,
                    glove=glove, backend=backend, cpu_workers=2)
                manifest, _ = read_json(manifest_path)
                if manifest["status"] != "complete":
                    retention = os.getenv("ARB_RESEARCH_FAILURE_DIR")
                    if backend == "gpu" and retention:
                        target = Path(retention) / ("gpu-fixture-" + uuid.uuid4().hex)
                        shutil.copytree(root, target, symlinks=True)
                    events = _Journal(output / "journal.jsonl").events
                    safe = [(event["event"], event.get("variant"), event.get("phase"),
                             event.get("code"), event.get("resources", {}).get("outcome")
                             if isinstance(event.get("resources"), dict) else None)
                            for event in events[-8:]]
                    receipts = []
                    for path in sorted(output.glob("variants/*/attempts/*/*/result.json")):
                        receipt, _ = read_json(path)
                        if receipt.get("code"):
                            receipts.append((receipt["variant"], receipt["phase"], receipt["code"]))
                    safe.append(("worker_receipts", receipts))
                    self.fail(f"real process campaign incomplete: {safe}")
                report = validate_campaign(prepared, output,
                    expected_prepared_identity=identity,
                    expected_campaign_identity=manifest["plan_identity"])
            self.assertEqual(len(report["variants"]), 7)
            self.assertTrue(all(row["evaluation_replay_bytes_equal"]
                                for row in report["variants"]))
            self.assertEqual(report["execution_config_count"], 9 if backend == "gpu" else 7)
            self.assertEqual(report["planned_nodes"], 30 if backend == "gpu" else 28)
            self.assertEqual(report["actual_completed_model_packages"],
                             9 if backend == "gpu" else 7)
            self.assertEqual(report["actual_attempts_started"], report["planned_nodes"])
            self.assertEqual(report["actual_attempts_completed"], report["planned_nodes"])
            self.assertEqual(len(report["auxiliary_fits"]), 2 if backend == "gpu" else 0)
            self.assertEqual(sum(row["backend"] == "gpu" for row in report["variants"]),
                             4 if backend == "gpu" else 0)
            self.assertEqual(sum(row["predict_measurement"] is not None
                                 for row in report["variants"]), 4 if backend == "gpu" else 0)

    def test_cpu_seven_variant_real_supervised_workers(self):
        self._run("cpu")

    @unittest.skipUnless(os.getenv("ARB_RESEARCH_GPU_TEST") == "1", "hardware opt-in")
    def test_gpu_seven_variant_real_supervised_workers(self):
        self._run("gpu")


@unittest.skipUnless(os.getenv("ARB_RESEARCH_GPU_TEST") == "1", "hardware opt-in")
class GPUCampaignTests(unittest.TestCase):
    def setUp(self):
        clock = patch("recsys.research_campaign._now", side_effect=_stable_utc_clock())
        clock.start()
        self.addCleanup(clock.stop)

    def test_four_gpu_variants_two_cpu_auxiliary_fits_and_cpu_hybrid(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ModelFixture(temporary)
            glove = Path(temporary) / "glove.txt"
            fixture.lock["inputs"]["embeddings"] = _glove(glove)
            prepared, identity = fixture.make_prepared()
            output = Path(temporary) / "gpu-campaign"
            with patch("recsys.research_models.release_info", return_value=fixture.lock), \
                 patch("recsys.research_resources.ResourceSupervisor", _Supervisor):
                path = run_campaign(prepared, output,
                    expected_prepared_identity=identity, code_revision="a" * 40,
                    glove=glove, backend="gpu", cpu_workers=2)
                manifest, _ = read_json(path)
                plan, _ = read_json(output / "campaign-plan.json")
                self.assertEqual(manifest["status"], "complete")
                self.assertEqual(plan["execution_config_count"], 9)
                self.assertEqual(len(_expected_nodes(plan)), 30)
                self.assertEqual([entry["variant_id"] for entry in plan["auxiliary_fits"]],
                                 ["content-mean", "puresvd"])
                report = validate_campaign(prepared, output,
                    expected_prepared_identity=identity,
                    expected_campaign_identity=manifest["plan_identity"])
                self.assertEqual(len(report["variants"]), 7)
                self.assertEqual(len(report["auxiliary_fits"]), 2)
                self.assertEqual(report["hybrid_admission"],
                                 {"status": "not_applicable", "reason": "declared_cpu_execution"})
                self.assertEqual(report["variants"][-1]["backend"], "cpu")
                self.assertEqual([row["backend"] for row in report["variants"]],
                                 ["cpu", "gpu", "gpu", "gpu", "cpu", "gpu", "cpu"])
