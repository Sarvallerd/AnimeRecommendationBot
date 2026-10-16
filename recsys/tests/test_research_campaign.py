"""Seven-variant validation pipeline and durable campaign boundary on tiny inputs."""

from __future__ import annotations

import fcntl
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
import uuid
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, TypedDict, cast
from unittest.mock import patch

import numpy as np
from research_model_support import ModelFixture, patch_release_info
from research_support import FixtureLock
from test_research_model_content import _glove

import recsys
from recsys.research.artifacts import StreamDescriptor, StreamWriter, read_json, stream_jsonl
from recsys.research.campaign.plan import CampaignPlan, _validate_plan_structure
from recsys.research.campaign.report import validate_campaign
from recsys.research.campaign.runner import (
    FIT_SECONDS,
    _canonical,
    _digest,
    _expected_nodes,
    _Journal,
    _plan,
    _private_file,
    _remaining,
    _successes,
    _worker_env,
    run_campaign,
)
from recsys.research.campaign.worker import _worker_main, _worker_request
from recsys.research.evaluation.evaluate import EvaluationManifest, PrivateMetric
from recsys.research.protocol import ResearchError, load_protocol, split_user
from recsys.research.resources import ProcessResult


class _Primary(TypedDict):
    mean_ndcg5_by_n: list[float]


class _Paired(TypedDict):
    mean_difference_by_n: list[float]
    pointwise_ci95_by_n: list[list[float]]
    bootstrap_replicates: int
    bootstrap_seed: int


class _Variant(TypedDict):
    backend: str
    evaluation_replay_bytes_equal: bool
    predict_measurement: object | None


class _Report(TypedDict):
    status: str
    variants: list[_Variant]
    auxiliary_fits: list[object]
    execution_config_count: int
    planned_nodes: int
    actual_completed_model_packages: int
    actual_attempts_started: int
    actual_attempts_completed: int
    primary_users: int
    primary_ndcg5: dict[str, _Primary] | None
    paired_vs_popularity: dict[str, _Paired] | None
    small_cohort_suppressed: bool
    diagnostics: dict[str, object]
    hybrid_admission: dict[str, object]


def _object(path: Path) -> dict[str, object]:
    value, _ = read_json(path)
    assert isinstance(value, dict)
    assert all(isinstance(key, str) for key in value)
    return cast("dict[str, object]", value)


def _manifest(path: Path) -> dict[str, str]:
    value = _object(path)
    assert isinstance(value.get("status"), str)
    assert isinstance(value.get("plan_identity"), str)
    return cast("dict[str, str]", value)


def _plan_file(path: Path) -> CampaignPlan:
    value = _object(path)
    assert isinstance(value.get("deadline_utc"), str)
    assert isinstance(value.get("variants"), list)
    assert isinstance(value.get("scheduling"), dict)
    return cast("CampaignPlan", value)


def _report(prepared: Path, output: Path, identity: str, plan_identity: str) -> _Report:
    value = validate_campaign(
        prepared,
        output,
        expected_prepared_identity=identity,
        expected_campaign_identity=plan_identity,
    )
    assert isinstance(value.get("variants"), list)
    assert isinstance(value.get("auxiliary_fits"), list)
    assert isinstance(value.get("diagnostics"), dict)
    assert isinstance(value.get("hybrid_admission"), dict)
    return cast("_Report", value)


def _stable_utc_clock() -> Callable[[], datetime]:
    # Stabilize only the in-process campaign controller against host UTC jumps.
    # Isolated workers and ResourceSupervisor retain their real clocks.
    start_utc, start_mono = datetime.now(timezone.utc), time.monotonic()
    return lambda: start_utc + timedelta(seconds=time.monotonic() - start_mono)


class _Supervisor:
    """Drive the real closed worker inline so fixture release metadata can be patched."""

    def __init__(self, *, rss_limit_bytes: int) -> None:
        self.results: dict[object, ProcessResult | None] = {}
        self.delivered: set[object] = set()

    def __enter__(self) -> _Supervisor:
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
        log_dir.mkdir(mode=0o700)
        code = _worker_main(Path(argv[-1]))
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
        complete = tuple(handle for handle in self.results if handle not in self.delivered)
        self.delivered.update(complete)
        return SimpleNamespace(completed=complete)

    def result(self, handle: object) -> ProcessResult | None:
        return self.results[handle]

    def cancel(self, handle: object) -> None:
        pass

    def close(self) -> None:
        pass


class _InterimSupervisor(_Supervisor):
    starts = 0

    def __init__(self, *, rss_limit_bytes: int) -> None:
        super().__init__(rss_limit_bytes=rss_limit_bytes)
        self.polls = 0
        self.cancelled: set[object] = set()

    def start(
        self,
        argv: list[str],
        *,
        cwd: Path,
        log_dir: Path,
        remaining_seconds: float,
        env: dict[str, str],
    ) -> object:
        handle = object()
        self.results[handle] = None
        type(self).starts += 1
        return handle

    def poll(self, *, wait_seconds: float) -> SimpleNamespace:
        self.polls += 1
        return SimpleNamespace(completed=tuple(self.results) if self.polls >= 2 else ())

    def result(self, handle: object) -> ProcessResult | None:
        first = next(iter(self.results))
        if self.polls == 1 and handle is not first:
            return None
        if self.polls < 1:
            return None
        return ProcessResult(
            outcome="cleanup_failed" if handle is first else "cancelled",
            returncode=None,
            elapsed_seconds=0.2 if self.polls == 1 else 0.3,
            sampled_peak_rss_bytes=1000,
            wait4_peak_rss_bytes=1000,
            observed_peak_rss_bytes=1000,
            rss_samples=1,
            max_sample_gap_seconds=0.1,
            cleanup_complete=self.polls >= 2,
        )

    def cancel(self, handle: object) -> None:
        self.cancelled.add(handle)


class CampaignCliTests(unittest.TestCase):
    def test_validate_route_calls_report_validator(self) -> None:
        from recsys.cli.main import main

        expected_identity = "sha256:" + "a" * 64
        output = io.StringIO()
        with (
            patch(
                "recsys.research.campaign.report.validate_campaign",
                return_value={"status": "complete"},
            ) as validator,
            redirect_stdout(output),
        ):
            status = main(
                [
                    "research",
                    "campaign-validate",
                    "--prepared-dir",
                    "/synthetic/prepared",
                    "--expected-prepared-identity",
                    expected_identity,
                    "--campaign-dir",
                    "/synthetic/campaign",
                    "--expected-campaign-identity",
                    expected_identity,
                ]
            )
        self.assertEqual(status, 0)
        self.assertEqual(output.getvalue(), f"{expected_identity} complete\n")
        validator.assert_called_once_with(
            Path("/synthetic/prepared"),
            Path("/synthetic/campaign"),
            expected_prepared_identity=expected_identity,
            expected_campaign_identity=expected_identity,
        )


class CampaignTests(unittest.TestCase):
    def setUp(self) -> None:
        clock = patch("recsys.research.campaign.runner._now", side_effect=_stable_utc_clock())
        clock.start()
        self.addCleanup(clock.stop)

    def test_old_campaign_plan_rejects_before_science(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ModelFixture(temporary)
            glove = Path(temporary) / "glove.txt"
            fixture.lock["inputs"]["embeddings"] = _glove(glove)
            prepared, identity = fixture.make_prepared()
            with patch_release_info(fixture.lock):
                plan = _plan(
                    prepared, identity, "a" * 40, glove, "cpu", 2, datetime.now(timezone.utc)
                )
            plan["schema_version"] = 1
            cast("dict[str, object]", plan["producer"]).pop("source_layout")
            with patch(
                "recsys.research.models.spec.campaign", side_effect=AssertionError("science loaded")
            ):
                with self.assertRaisesRegex(ResearchError, "research_source_layout_unsupported"):
                    _validate_plan_structure(plan)

    def test_published_orphan_without_success_event_blocks_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ModelFixture(temporary)
            glove = Path(temporary) / "glove.txt"
            fixture.lock["inputs"]["embeddings"] = _glove(glove)
            prepared, identity = fixture.make_prepared()
            output = Path(temporary) / "campaign"
            with (
                patch_release_info(fixture.lock),
                patch("recsys.research.resources.ResourceSupervisor", _Supervisor),
            ):
                manifest_path = run_campaign(
                    prepared,
                    output,
                    expected_prepared_identity=identity,
                    code_revision="a" * 40,
                    glove=glove,
                    backend="cpu",
                    cpu_workers=2,
                )
                manifest = _manifest(manifest_path)
                old = output / "variants" / "popularity" / "attempts" / "fit" / "1" / "artifact"
                orphan = old.parent.parent / "2" / "artifact"
                orphan.parent.mkdir(mode=0o700)
                shutil.copytree(old, orphan)
                _Journal(output / "journal.jsonl").append(
                    "attempt_started",
                    generation="crashed",
                    variant="popularity",
                    phase="fit",
                    attempt=2,
                    remaining_seconds=FIT_SECONDS,
                )
                before = (orphan / "model-manifest.json").read_bytes()
                with self.assertRaisesRegex(
                    ResearchError, "research_campaign_interrupted_uncertain"
                ):
                    run_campaign(
                        prepared,
                        output,
                        expected_prepared_identity=identity,
                        code_revision="a" * 40,
                        glove=glove,
                        backend="cpu",
                        cpu_workers=2,
                        resume=True,
                        expected_campaign_identity=manifest["plan_identity"],
                    )
                self.assertEqual((orphan / "model-manifest.json").read_bytes(), before)

    def test_twenty_users_publish_paired_three_order_means_and_fixed_bootstrap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ModelFixture(temporary)
            protocol = load_protocol()
            validation = {
                user for user in fixture.users if split_user(user, protocol) == "validation"
            }
            extra: set[str] = set()
            candidate = 0
            while len(validation | extra) < 20:
                user = "u1_" + hashlib.sha256(f"campaign-public-{candidate}".encode()).hexdigest()
                if user not in fixture.users and split_user(user, protocol) == "validation":
                    extra.add(user)
                candidate += 1
            self.assertEqual(len(validation | extra), 20)
            training = fixture.dataset / "interactions.tsv.gz"
            training.unlink()
            writer = StreamWriter(training)
            writer.write(b"dataset_user_key\tmal_id\trating\twatch_status\n")
            for user in sorted(set(fixture.users) | extra):
                for item in fixture.items[:17]:
                    writer.write(f"{user}\t{item}\t{8 + item % 3}\tcompleted\n".encode())
                writer.write(f"{user}\t18\t7\twatching\n".encode())
            writer.close()
            glove = Path(temporary) / "glove.txt"
            fixture.lock["inputs"]["embeddings"] = _glove(glove)
            prepared, identity = fixture.make_prepared()
            output = Path(temporary) / "campaign"
            with (
                patch_release_info(fixture.lock),
                patch("recsys.research.resources.ResourceSupervisor", _Supervisor),
            ):
                manifest_path = run_campaign(
                    prepared,
                    output,
                    expected_prepared_identity=identity,
                    code_revision="a" * 40,
                    glove=glove,
                    backend="cpu",
                    cpu_workers=2,
                )
                manifest = _manifest(manifest_path)
                self.assertEqual(manifest["status"], "complete")
                report = _report(prepared, output, identity, manifest["plan_identity"])
            self.assertEqual(report["primary_users"], 20)
            self.assertFalse(report["small_cohort_suppressed"])
            matrices = {}
            for variant in ("popularity", "hybrid"):
                artifact = (
                    output / "variants" / variant / "attempts" / "evaluate" / "1" / "artifact"
                )
                evaluation_value = _object(artifact / "evaluation-manifest.json")
                assert isinstance(evaluation_value.get("files"), dict)
                evaluation = cast("EvaluationManifest", evaluation_value)
                metric_file = evaluation["files"]["user-metrics.jsonl.gz"]
                assert "rows" in metric_file
                assert "uncompressed_bytes" in metric_file
                assert "canonical_sha256" in metric_file
                rows = stream_jsonl(
                    artifact / "user-metrics.jsonl.gz", cast("StreamDescriptor", metric_file)
                )
                cells: dict[tuple[str, int, str], float] = {}
                for value in rows:
                    assert isinstance(value, dict)
                    assert isinstance(value.get("order"), str)
                    assert isinstance(value.get("metrics"), dict)
                    row = cast("PrivateMetric", value)
                    if (
                        row["cohort"] == "main"
                        and row["mask"] == "controlled"
                        and row["order"].startswith("random-liked-")
                        and row["n"] > 0
                    ):
                        cells[(row["user_key"], row["n"], row["order"])] = row["metrics"]["ndcg5"]
                users = sorted({user for user, _n, _order in cells})
                self.assertEqual(len(users), 20)
                matrix = np.asarray(
                    [
                        [
                            sum(cells[(user, n, f"random-liked-{order}")] for order in (1, 2, 3))
                            / 3
                            for n in range(1, 11)
                        ]
                        for user in users
                    ],
                    dtype=np.float64,
                )
                matrices[variant] = matrix
                primary = report["primary_ndcg5"]
                assert primary is not None
                np.testing.assert_allclose(
                    np.asarray(primary[variant]["mean_ndcg5_by_n"], dtype=np.float64),
                    np.asarray(matrix.mean(axis=0), dtype=np.float64),
                    rtol=0,
                    atol=1e-15,
                )
            delta = matrices["hybrid"] - matrices["popularity"]
            pairs = report["paired_vs_popularity"]
            assert pairs is not None
            paired = pairs["hybrid"]
            np.testing.assert_allclose(
                np.asarray(paired["mean_difference_by_n"], dtype=np.float64),
                np.asarray(delta.mean(axis=0), dtype=np.float64),
                rtol=0,
                atol=1e-15,
            )
            self.assertEqual(
                (paired["bootstrap_replicates"], paired["bootstrap_seed"]), (2000, 366)
            )
            rng = np.random.Generator(np.random.PCG64(366))
            draws = np.asarray([delta[rng.integers(0, 20, 20)].mean(axis=0) for _ in range(2000)])
            expected = [
                [
                    float(np.quantile(draws[:, index], 0.025, method="lower")),
                    float(np.quantile(draws[:, index], 0.975, method="higher")),
                ]
                for index in range(10)
            ]
            np.testing.assert_allclose(paired["pointwise_ci95_by_n"], expected, rtol=0, atol=1e-15)
            public = json.dumps(report, ensure_ascii=False)
            self.assertNotIn("chosen_n", public)
            self.assertNotIn(str(output), public)
            self.assertNotRegex(public, r"u1_[0-9a-f]{64}")
            for private_key in ("user_key", "query_id", "seed_mal_ids", "ranking"):
                self.assertNotIn(private_key, public)

    def test_shared_auxiliary_fit_budget_and_journal_clock_regression(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            journal = _Journal(Path(temporary) / "journal.jsonl")
            first = datetime.now(timezone.utc) - timedelta(seconds=10)
            with patch("recsys.research.campaign.journal._now", return_value=first):
                journal.append("run_started", generation="g")
            original = journal.path.read_bytes()
            with patch(
                "recsys.research.campaign.journal._now", return_value=first - timedelta(seconds=1)
            ):
                with self.assertRaisesRegex(ResearchError, "research_campaign_clock_invalid"):
                    journal.append("run_started", generation="later")
            self.assertEqual(journal.path.read_bytes(), original)

            journal.append(
                "attempt_started",
                generation="g",
                variant="content-mean",
                phase="fit",
                attempt=1,
                remaining_seconds=FIT_SECONDS,
            )
            journal.append(
                "attempt_interrupted",
                generation="g",
                variant="content-mean",
                phase="fit",
                attempt=1,
                charged_seconds=300.0,
                resources={"cleanup_complete": True},
            )
            journal.append(
                "attempt_started",
                generation="h",
                variant="content-mean",
                phase="fit",
                attempt=2,
                remaining_seconds=FIT_SECONDS - 300,
            )
            journal.append(
                "attempt_finished",
                generation="h",
                variant="content-mean",
                phase="fit",
                attempt=2,
                charged_seconds=500.0,
                status="success",
                resources={"cleanup_complete": True},
                output_identity="sha256:" + "a" * 64,
            )
            journal.append(
                "attempt_started",
                generation="h",
                variant="content-mean",
                phase="fit_cpu_component",
                attempt=1,
                remaining_seconds=FIT_SECONDS - 800,
            )
            journal.append(
                "attempt_interrupted",
                generation="h",
                variant="content-mean",
                phase="fit_cpu_component",
                attempt=1,
                charged_seconds=400.0,
                resources={"cleanup_complete": True},
            )
            successes, attempts, charged, starts, failed = _successes(journal.events)
            self.assertFalse(starts)
            self.assertFalse(failed)
            self.assertEqual(attempts[("content-mean", "fit")], 2)
            self.assertIn(("content-mean", "fit"), successes)
            fixture = ModelFixture(temporary)
            glove = Path(temporary) / "glove.txt"
            fixture.lock["inputs"]["embeddings"] = _glove(glove)
            prepared, identity = fixture.make_prepared()
            with patch_release_info(fixture.lock):
                plan = _plan(prepared, identity, "a" * 40, glove, "cpu", 2, first)
            plan["deadline_utc"] = (first + timedelta(days=1)).isoformat().replace("+00:00", "Z")
            self.assertAlmostEqual(
                _remaining(plan, charged, "content-mean", "fit_cpu_component"),
                FIT_SECONDS - 1200,
                delta=10,
            )

    def test_interim_cleanup_freezes_scheduling_and_charges_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ModelFixture(temporary)
            glove = Path(temporary) / "glove.txt"
            fixture.lock["inputs"]["embeddings"] = _glove(glove)
            prepared, identity = fixture.make_prepared()
            output = Path(temporary) / "campaign"
            _InterimSupervisor.starts = 0
            with (
                patch_release_info(fixture.lock),
                patch("recsys.research.resources.ResourceSupervisor", _InterimSupervisor),
            ):
                run_campaign(
                    prepared,
                    output,
                    expected_prepared_identity=identity,
                    code_revision="a" * 40,
                    glove=glove,
                    backend="cpu",
                    cpu_workers=2,
                )
            report = _object(output / "campaign-report.json")
            self.assertEqual(report["status"], "incomplete")
            self.assertEqual(report["failure_code"], "research_campaign_cleanup_failed")
            self.assertEqual(_InterimSupervisor.starts, 2)
            events = _Journal(output / "journal.jsonl").events
            self.assertEqual(
                len([event for event in events if event["event"] == "cleanup_warning"]), 1
            )
            finishes = [event for event in events if event["event"] == "attempt_finished"]
            self.assertEqual(len(finishes), 2)
            self.assertTrue(all(event["status"] == "failed" for event in finishes))
            self.assertTrue(all(event["charged_seconds"] == 0.3 for event in finishes))

    def test_seven_real_variants_replay_public_suppression_and_resume_binding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ModelFixture(temporary)
            glove = Path(temporary) / "glove.txt"
            fixture.lock["inputs"]["embeddings"] = _glove(glove)
            prepared, identity = fixture.make_prepared()
            output = Path(temporary) / "campaign"
            with (
                patch_release_info(fixture.lock),
                patch("recsys.research.resources.ResourceSupervisor", _Supervisor),
            ):
                manifest_path = run_campaign(
                    prepared,
                    output,
                    expected_prepared_identity=identity,
                    code_revision="a" * 40,
                    glove=glove,
                    backend="cpu",
                    cpu_workers=2,
                )
                manifest = _manifest(manifest_path)
                plan = _plan_file(output / "campaign-plan.json")
                self.assertEqual(manifest["status"], "complete")
                self.assertEqual(plan["execution_config_count"], 7)
                self.assertEqual(plan["auxiliary_fits"], [])
                self.assertEqual(len(_expected_nodes(plan)), 28)
                self.assertEqual(len(list((output / "variants").iterdir())), 7)
                report = _report(prepared, output, identity, manifest["plan_identity"])
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
                run_campaign(
                    prepared,
                    output,
                    expected_prepared_identity=identity,
                    code_revision="a" * 40,
                    glove=glove,
                    backend="cpu",
                    cpu_workers=2,
                    resume=True,
                    expected_campaign_identity=manifest["plan_identity"],
                )
                self.assertGreater(len((output / "journal.jsonl").read_bytes()), len(before))
                gate = os.open(output / ".workers.lock", os.O_RDWR)
                try:
                    fcntl.flock(gate, fcntl.LOCK_SH)
                    with self.assertRaisesRegex(ResearchError, "research_campaign_workers_active"):
                        run_campaign(
                            prepared,
                            output,
                            expected_prepared_identity=identity,
                            code_revision="a" * 40,
                            glove=glove,
                            backend="cpu",
                            cpu_workers=2,
                            resume=True,
                            expected_campaign_identity=manifest["plan_identity"],
                        )
                finally:
                    os.close(gate)
                with self.assertRaises(ResearchError):
                    run_campaign(
                        prepared,
                        output,
                        expected_prepared_identity=identity,
                        code_revision="b" * 40,
                        glove=glove,
                        backend="cpu",
                        cpu_workers=2,
                        resume=True,
                        expected_campaign_identity=manifest["plan_identity"],
                    )
                self.assertAlmostEqual(
                    _remaining(plan, {("content-mean", "fit"): 1000.0}, "content-mean", "fit"),
                    FIT_SECONDS - 1000,
                    delta=10,
                )
                replay = (
                    output
                    / "variants"
                    / "popularity"
                    / "attempts"
                    / "replay"
                    / "1"
                    / "artifact"
                    / "report.json"
                )
                original_replay = replay.read_bytes()
                replay.write_bytes(original_replay + b"x")
                with self.assertRaises(ResearchError):
                    validate_campaign(
                        prepared,
                        output,
                        expected_prepared_identity=identity,
                        expected_campaign_identity=manifest["plan_identity"],
                    )
                replay.write_bytes(original_replay)
                damaged = next(
                    (output / "variants" / "hybrid" / "attempts" / "fit" / "1" / "artifact").glob(
                        "content_*.npy"
                    )
                )
                with damaged.open("ab") as stream:
                    stream.write(b"x")
                with self.assertRaises(ResearchError):
                    validate_campaign(
                        prepared,
                        output,
                        expected_prepared_identity=identity,
                        expected_campaign_identity=manifest["plan_identity"],
                    )
                with self.assertRaises(ResearchError):
                    run_campaign(
                        prepared,
                        output,
                        expected_prepared_identity=identity,
                        code_revision="a" * 40,
                        glove=glove,
                        backend="cpu",
                        cpu_workers=2,
                        resume=True,
                        expected_campaign_identity=manifest["plan_identity"],
                    )

    def test_stale_generation_real_isolated_worker_never_reaches_science(self) -> None:
        from recsys.research.resources import ResourceSupervisor

        with tempfile.TemporaryDirectory() as temporary:
            fixture = ModelFixture(temporary)
            glove = Path(temporary) / "glove.txt"
            fixture.lock["inputs"]["embeddings"] = _glove(glove)
            prepared, identity = fixture.make_prepared()
            root = Path(temporary) / "worker-root"
            root.mkdir(mode=0o700)
            with patch_release_info(fixture.lock):
                plan = _plan(
                    prepared,
                    identity,
                    "a" * 40,
                    glove,
                    "cpu",
                    1,
                    __import__("datetime").datetime.now(__import__("datetime").timezone.utc),
                )
            raw = _canonical(plan)
            _private_file(root / "campaign-plan.json", raw)
            _private_file(root / "generation.json", _canonical({"generation": "new"}))
            gate = os.open(root / ".workers.lock", os.O_RDWR | os.O_CREAT, 0o600)
            os.close(gate)
            attempt = root / "variants" / "popularity" / "attempts" / "fit" / "1"
            attempt.mkdir(parents=True, mode=0o700)
            for parent in (attempt.parent, attempt.parent.parent, attempt.parent.parent.parent):
                parent.chmod(0o700)
            request = _worker_request(
                root,
                _digest(raw),
                "old",
                "popularity",
                "fit",
                1,
                prepared,
                glove,
                None,
                None,
                None,
                None,
            )
            _private_file(attempt / "request.json", _canonical(request))
            journal = _Journal(root / "journal.jsonl")
            journal.append(
                "attempt_started",
                generation="old",
                variant="popularity",
                phase="fit",
                attempt=1,
                remaining_seconds=5.0,
            )
            with ResourceSupervisor() as supervisor:
                handle = supervisor.start(
                    [
                        sys.executable,
                        "-I",
                        "-m",
                        "recsys.research.campaign.worker",
                        "--worker-request",
                        str(attempt / "request.json"),
                    ],
                    cwd=root,
                    log_dir=attempt / "logs",
                    remaining_seconds=5.0,
                    env=_worker_env(),
                )
                for _ in range(100):
                    supervisor.poll(wait_seconds=0.1)
                    result = supervisor.result(handle)
                    if result is not None and result.cleanup_complete:
                        break
                assert result is not None
                self.assertEqual(result.outcome, "exit_nonzero")
                self.assertEqual(result.returncode, 1)
            receipt = _object(attempt / "result.json")
            self.assertEqual(receipt["code"], "research_campaign_generation_stale")
            self.assertFalse((attempt / "artifact").exists())


def _isolated_fixture_python(root: Path, lock: FixtureLock) -> tuple[Path, Path]:
    """Disposable Python shares installed wheels; a test .pth hook supplies the tiny lock."""
    executable = root / "fixture-python" / "bin" / "python"
    subprocess.run(
        [sys.executable, "-m", "venv", str(executable.parent.parent)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    site = (
        executable.parent.parent
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    installed = Path(recsys.__file__).parent.parent
    (site / "recsys-installed.pth").write_text(str(installed) + "\nimport arb_fixture_hook\n")
    (site / "nvidia").symlink_to(installed / "nvidia", target_is_directory=True)
    (site / "arb_fixture_hook.py").write_text(
        "import json, os\n"
        "if 'ARB_TEST_LOCK_JSON' in os.environ:\n"
        " from pathlib import Path\n"
        " import recsys.release.lock as models\n"
        " fixture_lock = json.loads(Path(os.environ['ARB_TEST_LOCK_JSON']).read_text())\n"
        " models.release_info = lambda: fixture_lock\n"
    )
    lock_path = root / "tiny-release-lock.json"
    lock_path.write_text(json.dumps(lock, sort_keys=True))
    return executable, lock_path


class RealProcessCampaignTests(unittest.TestCase):
    def _run(self, backend: str) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ModelFixture(temporary)
            root = Path(temporary)
            glove = root / "glove.txt"
            fixture.lock["inputs"]["embeddings"] = _glove(glove)
            prepared, identity = fixture.make_prepared()
            executable, lock_path = _isolated_fixture_python(root, fixture.lock)
            output = root / f"{backend}-real-campaign"
            with (
                patch_release_info(fixture.lock),
                patch("recsys.research.campaign.runner.sys.executable", str(executable)),
                patch("recsys.research.campaign.runner._now", side_effect=_stable_utc_clock()),
                patch.dict(os.environ, {"ARB_TEST_LOCK_JSON": str(lock_path)}),
            ):
                manifest_path = run_campaign(
                    prepared,
                    output,
                    expected_prepared_identity=identity,
                    code_revision="a" * 40,
                    glove=glove,
                    backend=backend,
                    cpu_workers=2,
                )
                manifest = _manifest(manifest_path)
                if manifest["status"] != "complete":
                    retention = os.getenv("ARB_RESEARCH_FAILURE_DIR")
                    if retention:
                        target = Path(retention) / ("gpu-fixture-" + uuid.uuid4().hex)
                        shutil.copytree(root, target, symlinks=True)
                    events = _Journal(output / "journal.jsonl").events
                    safe: list[object] = [
                        (
                            event["event"],
                            event.get("variant"),
                            event.get("phase"),
                            event.get("code"),
                            resources.get("outcome")
                            if isinstance((resources := event.get("resources")), dict)
                            else None,
                        )
                        for event in events[-8:]
                    ]
                    receipts: list[tuple[object, object, object]] = []
                    for path in sorted(output.glob("variants/*/attempts/*/*/result.json")):
                        receipt = _object(path)
                        if receipt.get("code"):
                            receipts.append((receipt["variant"], receipt["phase"], receipt["code"]))
                    safe.append(("worker_receipts", receipts))
                    self.fail(f"real process campaign incomplete: {safe}")
                report = _report(prepared, output, identity, manifest["plan_identity"])
            self.assertEqual(len(report["variants"]), 7)
            self.assertTrue(all(row["evaluation_replay_bytes_equal"] for row in report["variants"]))
            self.assertEqual(report["execution_config_count"], 9 if backend == "gpu" else 7)
            self.assertEqual(report["planned_nodes"], 30 if backend == "gpu" else 28)
            self.assertEqual(
                report["actual_completed_model_packages"], 9 if backend == "gpu" else 7
            )
            self.assertEqual(report["actual_attempts_started"], report["planned_nodes"])
            self.assertEqual(report["actual_attempts_completed"], report["planned_nodes"])
            self.assertEqual(len(report["auxiliary_fits"]), 2 if backend == "gpu" else 0)
            self.assertEqual(
                sum(row["backend"] == "gpu" for row in report["variants"]),
                4 if backend == "gpu" else 0,
            )
            self.assertEqual(
                sum(row["predict_measurement"] is not None for row in report["variants"]),
                4 if backend == "gpu" else 0,
            )

    def test_cpu_seven_variant_real_supervised_workers(self) -> None:
        self._run("cpu")

    @unittest.skipUnless(os.getenv("ARB_RESEARCH_GPU_TEST") == "1", "hardware opt-in")
    def test_gpu_seven_variant_real_supervised_workers(self) -> None:
        self._run("gpu")


@unittest.skipUnless(os.getenv("ARB_RESEARCH_GPU_TEST") == "1", "hardware opt-in")
class GPUCampaignTests(unittest.TestCase):
    def setUp(self) -> None:
        clock = patch("recsys.research.campaign.runner._now", side_effect=_stable_utc_clock())
        clock.start()
        self.addCleanup(clock.stop)

    def test_four_gpu_variants_two_cpu_auxiliary_fits_and_cpu_hybrid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ModelFixture(temporary)
            glove = Path(temporary) / "glove.txt"
            fixture.lock["inputs"]["embeddings"] = _glove(glove)
            prepared, identity = fixture.make_prepared()
            output = Path(temporary) / "gpu-campaign"
            with (
                patch_release_info(fixture.lock),
                patch("recsys.research.resources.ResourceSupervisor", _Supervisor),
            ):
                path = run_campaign(
                    prepared,
                    output,
                    expected_prepared_identity=identity,
                    code_revision="a" * 40,
                    glove=glove,
                    backend="gpu",
                    cpu_workers=2,
                )
                manifest = _manifest(path)
                plan = _plan_file(output / "campaign-plan.json")
                self.assertEqual(manifest["status"], "complete")
                self.assertEqual(plan["execution_config_count"], 9)
                self.assertEqual(len(_expected_nodes(plan)), 30)
                self.assertEqual(
                    [entry["variant_id"] for entry in plan["auxiliary_fits"]],
                    ["content-mean", "puresvd"],
                )
                report = _report(prepared, output, identity, manifest["plan_identity"])
                self.assertEqual(len(report["variants"]), 7)
                self.assertEqual(len(report["auxiliary_fits"]), 2)
                self.assertEqual(
                    report["hybrid_admission"],
                    {"status": "not_applicable", "reason": "declared_cpu_execution"},
                )
                self.assertEqual(report["variants"][-1]["backend"], "cpu")
                self.assertEqual(
                    [row["backend"] for row in report["variants"]],
                    ["cpu", "gpu", "gpu", "gpu", "cpu", "gpu", "cpu"],
                )
