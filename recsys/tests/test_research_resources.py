"""Controlled Linux process tests for the ML-02 resource supervisor."""

from __future__ import annotations

import ctypes
import os
import resource
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from collections.abc import Sequence
from pathlib import Path
from unittest import mock

from recsys.research import resources
from recsys.research.protocol import ResearchError

StatFields = tuple[str, int, int, int, int, int]
ProcessGroups = dict[int, dict[int, tuple[str, int, int, int]]]


class ResourceSupervisorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.number = 0

    def launch(
        self,
        supervisor: resources.ResourceSupervisor,
        code: str,
        *,
        seconds: float = 4,
        args: tuple[str, ...] = (),
    ) -> resources.JobHandle:
        self.number += 1
        return supervisor.start(
            [sys.executable, "-I", "-c", code, *args],
            cwd=self.root,
            log_dir=self.root / f"log-{self.number}",
            remaining_seconds=seconds,
        )

    def await_result(
        self,
        supervisor: resources.ResourceSupervisor,
        handle: resources.JobHandle,
        *,
        wall: float = 8,
    ) -> tuple[resources.ProcessResult, list[resources.JobHandle]]:
        deadline = time.monotonic() + wall
        completions: list[resources.JobHandle] = []
        while supervisor.result(handle) is None and time.monotonic() < deadline:
            completions.extend(supervisor.poll().completed)
        result = supervisor.result(handle)
        assert result is not None, "worker completion missing"
        return result, completions

    def await_text(
        self,
        supervisor: resources.ResourceSupervisor,
        handle: resources.JobHandle,
        path: Path,
        expected: str,
        *,
        wall: float = 5,
    ) -> None:
        deadline = time.monotonic() + wall
        while time.monotonic() < deadline:
            supervisor.poll()
            if path.exists() and expected in path.read_text():
                return
        self.fail("worker handshake missing")

    def launch_held(
        self, supervisor: resources.ResourceSupervisor
    ) -> tuple[resources.JobHandle, Path]:
        ready = self.root / f"ready-{self.number + 1}"
        release = self.root / f"release-{self.number + 1}"
        code = (
            "from pathlib import Path\nimport sys,time\n"
            "Path(sys.argv[1]).write_text('ready')\n"
            "while not Path(sys.argv[2]).exists():\n time.sleep(.01)\n"
        )
        handle = self.launch(supervisor, code, args=(str(ready), str(release)))
        deadline = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertTrue(ready.exists(), "worker handshake missing")
        return handle, release

    @staticmethod
    def threaded_exit_code() -> str:
        return (
            "import ctypes, os, sys, threading, time\n"
            "from pathlib import Path\n"
            "ready, release, count, mib = sys.argv[1:]\n"
            "count, mib = int(count), int(mib)\n"
            "barrier = threading.Barrier(count + 1)\n"
            "def worker():\n"
            " data = bytearray(mib * 1024 * 1024)\n"
            " for offset in range(0, len(data), 4096):\n"
            "  data[offset] = 88\n"
            " barrier.wait()\n"
            " while not Path(release).exists():\n"
            "  time.sleep(.01)\n"
            " os._exit(0)\n"
            "for _ in range(count):\n"
            " threading.Thread(target=worker).start()\n"
            "barrier.wait()\n"
            "Path(ready).write_text('ready')\n"
            "libc = ctypes.CDLL(None)\n"
            "libc.pthread_exit.argtypes = [ctypes.c_void_p]\n"
            "libc.pthread_exit.restype = None\n"
            "libc.pthread_exit(None)\n"
        )

    def wait_for_threaded_zombie(self, pid: int, *, waitable: bool = False) -> list[int]:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                state = resources._stat_fields(pid)[0]
                with os.scandir(f"/proc/{pid}/task") as tasks:
                    live = [
                        int(task.name)
                        for task in tasks
                        if task.name.isdecimal()
                        and resources._stat_path(f"/proc/{pid}/task/{task.name}/stat")[0] != "Z"
                    ]
                if state == "Z" and live:
                    if waitable:
                        self.assertIsNone(
                            os.waitid(os.P_PID, pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
                        )
                    return live
            except (FileNotFoundError, ProcessLookupError):
                pass
            time.sleep(0.01)
        self.fail("zombie leader with live threads not observed")

    def launch_threaded(
        self,
        supervisor: resources.ResourceSupervisor,
        *,
        count: int = 1,
        mib: int = 0,
        seconds: float = 4,
    ) -> tuple[resources.JobHandle, Path]:
        ready = self.root / f"ready-{self.number + 1}"
        release = self.root / f"release-{self.number + 1}"
        handle = self.launch(
            supervisor,
            self.threaded_exit_code(),
            seconds=seconds,
            args=(str(ready), str(release), str(count), str(mib)),
        )
        deadline = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertTrue(ready.exists(), "threaded worker handshake missing")
        pid = supervisor._jobs[handle].pid
        assert pid is not None
        self.wait_for_threaded_zombie(pid, waitable=True)
        return handle, release

    def test_zombie_leader_live_thread_waits_then_completes_once(self) -> None:
        with resources.ResourceSupervisor() as supervisor:
            handle, release = self.launch_threaded(supervisor)
            job = supervisor._jobs[handle]
            for _ in range(3):
                snapshot = supervisor.poll(wait_seconds=0.05)
                self.assertEqual(snapshot.completed, ())
                self.assertEqual(snapshot.active_count, 1)
                self.assertIsNone(supervisor.result(handle))
                self.assertIsNone(job.term_at)
                self.assertIsNone(job.returncode)
                self.assertFalse(job.empty_zombie_seen)
            release.write_text("go")
            result, completed = self.await_result(supervisor, handle)
            self.assertEqual(
                (result.outcome, result.returncode, result.cleanup_complete), ("ok", 0, True)
            )
            self.assertEqual(completed, [handle])
            self.assertEqual(supervisor.poll(wait_seconds=0).completed, ())

    def test_two_empty_scans_must_both_observe_waitable_leader(self) -> None:
        with resources.ResourceSupervisor() as supervisor:
            handle = self.launch(supervisor, "print('done')")
            leader = supervisor._jobs[handle].pid
            assert leader is not None
            deadline = time.monotonic() + 3
            while os.waitid(os.P_PID, leader, os.WEXITED | os.WNOHANG | os.WNOWAIT) is None:
                self.assertLess(time.monotonic(), deadline)
                time.sleep(0.01)
            supervisor.poll(wait_seconds=0)
            self.assertTrue(supervisor._jobs[handle].empty_zombie_seen)
            with mock.patch.object(resources.os, "waitid", return_value=None):
                supervisor.poll(wait_seconds=0)
            self.assertFalse(supervisor._jobs[handle].empty_zombie_seen)
            supervisor.poll(wait_seconds=0)
            self.assertIsNone(supervisor.result(handle))
            self.assertTrue(supervisor._jobs[handle].empty_zombie_seen)
            self.assertEqual(supervisor.poll(wait_seconds=0).completed, (handle,))
            result = supervisor.result(handle)
            assert result is not None
            self.assertEqual((result.outcome, result.returncode), ("ok", 0))

    def test_zombie_leader_timeout_and_cancel_clean_live_threads(self) -> None:
        for action in ("timeout", "cancelled"):
            with self.subTest(action=action):
                with resources.ResourceSupervisor() as supervisor:
                    handle, _release = self.launch_threaded(
                        supervisor, seconds=0.6 if action == "timeout" else 4
                    )
                    pid = supervisor._jobs[handle].pid
                    self.assertIsNone(supervisor.result(handle))
                    if action == "cancelled":
                        supervisor.cancel(handle)
                    result, completed = self.await_result(supervisor, handle)
                    self.assertEqual((result.outcome, result.cleanup_complete), (action, True))
                    self.assertEqual(completed, [handle])
                    self.assertFalse(Path(f"/proc/{pid}").exists())

    def test_adopted_zombie_with_live_thread_requires_descendant_cleanup(self) -> None:
        child_ready = self.root / "child-ready"
        child_release = self.root / "child-release"
        parent_release = self.root / "parent-release"
        child_pid = self.root / "child-pid"
        code = (
            "import subprocess, sys, time\n"
            "from pathlib import Path\n"
            "child = subprocess.Popen([sys.executable, '-I', '-c', sys.argv[1], *sys.argv[2:6]])\n"
            "Path(sys.argv[6]).write_text(str(child.pid))\n"
            "while not Path(sys.argv[7]).exists():\n time.sleep(.01)\n"
        )
        with resources.ResourceSupervisor() as supervisor:
            handle = self.launch(
                supervisor,
                code,
                args=(
                    self.threaded_exit_code(),
                    str(child_ready),
                    str(child_release),
                    "1",
                    "0",
                    str(child_pid),
                    str(parent_release),
                ),
            )
            deadline = time.monotonic() + 5
            while not child_pid.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue(child_pid.exists())
            adopted_pid = int(child_pid.read_text())
            self.wait_for_threaded_zombie(adopted_pid)
            real_wait = resources.ResourceSupervisor._wait4_one
            pending: list[int] = []

            def record_wait(job: resources._Job, pid: int) -> bool | None:
                result = real_wait(job, pid)
                if pid == adopted_pid and result is None:
                    pending.append(pid)
                return result

            with mock.patch.object(
                resources.ResourceSupervisor, "_wait4_one", side_effect=record_wait
            ):
                parent_release.write_text("go")
                result, completed = self.await_result(supervisor, handle)
            self.assertIn(adopted_pid, pending)
            self.assertEqual(
                (result.outcome, result.cleanup_complete), ("descendants_remaining", True)
            )
            self.assertEqual(completed, [handle])
            self.assertFalse(Path(f"/proc/{adopted_pid}").exists())

    def test_zombie_thread_rss_is_counted_once_per_process_and_aggregate(self) -> None:
        with resources.ResourceSupervisor() as supervisor:
            handle, release = self.launch_threaded(supervisor, count=2, mib=16)
            pid = supervisor._jobs[handle].pid
            assert pid is not None
            live = self.wait_for_threaded_zombie(pid, waitable=True)
            self.assertGreaterEqual(len(live), 2)
            thread_rss = [resources._stat_path(f"/proc/{pid}/task/{tid}/stat")[5] for tid in live]
            expected = max(thread_rss) * os.sysconf("SC_PAGE_SIZE")
            self.assertGreater(expected, 24 * 1024**2)
            supervisor.poll(wait_seconds=0)
            sampled = supervisor._jobs[handle].current_rss
            self.assertGreaterEqual(sampled, expected - 1024**2)
            self.assertLess(sampled, expected + 1024**2)
            self.assertLess(sampled, sum(thread_rss) * os.sysconf("SC_PAGE_SIZE"))
            release.write_text("go")
            self.assertEqual(self.await_result(supervisor, handle)[0].outcome, "ok")
        with resources.ResourceSupervisor(rss_limit_bytes=65 * 1024**2) as supervisor:
            handles = []
            for _ in range(2):
                handle, _release = self.launch_threaded(supervisor, count=2, mib=16)
                handles.append(handle)
            snapshot = supervisor.poll(wait_seconds=0)
            self.assertGreater(snapshot.sampled_rss_bytes, supervisor.rss_limit_bytes)
            results = [self.await_result(supervisor, handle)[0] for handle in handles]
            self.assertEqual(
                [result.outcome for result in results],
                ["aggregate_rss_limit", "aggregate_rss_limit"],
            )
            self.assertTrue(
                all(
                    result.sampled_peak_rss_bytes < supervisor.rss_limit_bytes for result in results
                )
            )
            self.assertTrue(all(result.cleanup_complete for result in results))

    def test_genuine_wait4_error_is_sticky_after_later_cleanup(self) -> None:
        with resources.ResourceSupervisor() as supervisor:
            handle = self.launch(supervisor, "print('done')")
            leader = supervisor._jobs[handle].pid
            assert leader is not None
            real_wait4 = os.wait4
            failed: list[int] = []

            def once(pid: int, flags: int) -> tuple[int, int, resource.struct_rusage]:
                if pid == leader and not failed:
                    failed.append(pid)
                    raise OSError("injected genuine wait failure")
                return real_wait4(pid, flags)

            with mock.patch.object(resources.os, "wait4", side_effect=once):
                result, completed = self.await_result(supervisor, handle)
            self.assertEqual(failed, [leader])
            self.assertEqual(
                (result.outcome, result.returncode, result.cleanup_complete),
                ("monitor_failed", 0, True),
            )
            self.assertEqual(completed, [handle])

    def test_malformed_live_thread_stat_fails_closed(self) -> None:
        with resources.ResourceSupervisor() as supervisor:
            handle, _release = self.launch_threaded(supervisor)
            real_stat = resources._stat_path

            def malformed(path: str) -> StatFields:
                if "/task/" in path:
                    raise ValueError("malformed live thread stat")
                return real_stat(path)

            with mock.patch.object(resources, "_stat_path", side_effect=malformed):
                supervisor.poll(wait_seconds=0)
            result, completed = self.await_result(supervisor, handle)
            self.assertEqual((result.outcome, result.cleanup_complete), ("monitor_failed", True))
            self.assertEqual(completed, [handle])

    def test_initial_transient_proc_read_accepts_late_identity(self) -> None:
        for missing in (FileNotFoundError, ProcessLookupError):
            with self.subTest(missing=missing.__name__):
                with resources.ResourceSupervisor() as supervisor:
                    with mock.patch.object(
                        resources, "_stat_fields", side_effect=missing("transient")
                    ) as stat:
                        handle, release = self.launch_held(supervisor)
                    self.assertEqual(stat.call_count, 1)
                    job = supervisor._jobs[handle]
                    self.assertIsNone(job.starttime)
                    self.assertIsNone(job.outcome)
                    self.assertIsNone(job.term_at)
                    self.assertIsNone(supervisor.result(handle))

                    self.assertEqual(supervisor.poll(wait_seconds=0).completed, ())
                    self.assertIsNotNone(job.starttime)
                    self.assertIsNone(job.outcome)
                    self.assertIsNone(job.term_at)
                    self.assertIsNone(supervisor.result(handle))
                    release.write_text("go")
                    result, completed = self.await_result(supervisor, handle)
                    self.assertEqual(
                        (result.outcome, result.returncode, result.cleanup_complete),
                        ("ok", 0, True),
                    )
                    self.assertEqual(completed, [handle])
                    self.assertEqual(supervisor.poll(wait_seconds=0).completed, ())

    def test_initial_nontransient_read_or_wrong_identity_remains_failed(self) -> None:
        real_stat = resources._stat_fields
        for wrong in ("group", "session", "io"):
            with self.subTest(wrong=wrong):

                def wrong_initial(pid: int) -> StatFields:
                    if wrong == "io":
                        raise OSError("nontransient")
                    state, ppid, pgrp, session, birth, rss = real_stat(pid)
                    if wrong == "group":
                        pgrp += 1
                    else:
                        session += 1
                    return state, ppid, pgrp, session, birth, rss

                with resources.ResourceSupervisor() as supervisor:
                    with mock.patch.object(
                        resources, "_stat_fields", side_effect=wrong_initial
                    ) as stat:
                        handle, _release = self.launch_held(supervisor)
                    self.assertEqual(stat.call_count, 1)
                    job = supervisor._jobs[handle]
                    self.assertEqual(job.outcome, "monitor_failed")
                    self.assertIsNone(job.starttime)
                    supervisor.poll(wait_seconds=0)
                    self.assertEqual(job.outcome, "monitor_failed")
                    self.assertIsNotNone(job.starttime)
                    result, completed = self.await_result(supervisor, handle)
                    self.assertEqual(
                        (result.outcome, result.returncode, result.cleanup_complete),
                        ("monitor_failed", -signal.SIGTERM, True),
                    )
                    self.assertEqual(completed, [handle])

    def test_inconsistent_late_identity_cannot_signal_unverified_group(self) -> None:
        real_stat = resources._stat_fields
        for wrong in ("birth", "session"):
            with self.subTest(wrong=wrong):
                with resources.ResourceSupervisor() as supervisor:
                    with mock.patch.object(
                        resources, "_stat_fields", side_effect=FileNotFoundError("transient")
                    ):
                        handle, release = self.launch_held(supervisor)
                    job = supervisor._jobs[handle]
                    leader_reads = 0

                    def inconsistent_second_read(pid: int) -> StatFields:
                        nonlocal leader_reads
                        state, ppid, pgrp, session, birth, rss = real_stat(pid)
                        if pid == job.pid:
                            leader_reads += 1
                            if leader_reads == 2:
                                if wrong == "birth":
                                    birth += 1
                                else:
                                    session += 1
                        return state, ppid, pgrp, session, birth, rss

                    with mock.patch.object(
                        resources, "_stat_fields", side_effect=inconsistent_second_read
                    ):
                        with mock.patch.object(
                            resources.os, "killpg", side_effect=AssertionError("unverified killpg")
                        ) as kill:
                            supervisor.poll(wait_seconds=0)
                    self.assertGreaterEqual(leader_reads, 2)
                    kill.assert_not_called()
                    self.assertIsNone(job.starttime)
                    self.assertEqual(job.outcome, "monitor_failed")
                    self.assertIsNone(supervisor.result(handle))
                    release.write_text("go")
                    result, completed = self.await_result(supervisor, handle)
                    self.assertEqual(
                        (result.outcome, result.returncode, result.cleanup_complete),
                        ("monitor_failed", 0, True),
                    )
                    self.assertEqual(completed, [handle])

    def test_success_nonzero_signal_and_literal_argv(self) -> None:
        with resources.ResourceSupervisor() as supervisor:
            literal = "; touch should-not-exist"
            h = self.launch(
                supervisor,
                "import sys; print(sys.argv[1]); print('err', file=sys.stderr); sys.exit(0)",
                args=(literal,),
            )
            result, done = self.await_result(supervisor, h)
            self.assertEqual((result.outcome, result.returncode, done), ("ok", 0, [h]))
            self.assertEqual(supervisor.poll(wait_seconds=0).completed, ())
            self.assertIs(result, supervisor.result(h))
            self.assertEqual((self.root / "log-1/stdout.log").read_text().strip(), literal)
            self.assertEqual((self.root / "log-1/stderr.log").read_text().strip(), "err")
            self.assertFalse((self.root / "should-not-exist").exists())
            h2 = self.launch(supervisor, "raise SystemExit(7)")
            h2_result, _ = self.await_result(supervisor, h2)
            self.assertEqual(
                (h2_result.outcome, h2_result.returncode),
                ("exit_nonzero", 7),
            )
            h3 = self.launch(supervisor, "import os,signal; os.kill(os.getpid(),signal.SIGUSR1)")
            h3_result, _ = self.await_result(supervisor, h3)
            self.assertEqual(
                (h3_result.outcome, h3_result.returncode),
                ("exit_nonzero", -signal.SIGUSR1),
            )

    def test_inheritable_parent_fd_is_closed_in_worker(self) -> None:
        sentinel = self.root / "parent-only"
        sentinel.write_text("private")
        fd = os.open(sentinel, os.O_RDONLY)
        try:
            os.set_inheritable(fd, True)
            code = (
                "import os,sys; "
                "p='/proc/self/fd/'+sys.argv[1]; "
                "target=os.readlink(p) if os.path.exists(p) else ''; "
                "assert target != sys.argv[2]"
            )
            with resources.ResourceSupervisor() as supervisor:
                h = self.launch(supervisor, code, args=(str(fd), str(sentinel)))
                self.assertEqual(self.await_result(supervisor, h)[0].outcome, "ok")
        finally:
            os.close(fd)

    def test_timeout_kills_term_ignoring_grandchild_and_reaps(self) -> None:
        code = (
            "import os,signal,subprocess,sys,time; "
            "p=subprocess.Popen([sys.executable,'-I','-c',"
            "'import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)']); "
            "print(p.pid,flush=True); time.sleep(60)"
        )
        with resources.ResourceSupervisor() as supervisor:
            h = self.launch(supervisor, code, seconds=0.35)
            self.await_text(supervisor, h, self.root / "log-1/stdout.log", "\n")
            child = int((self.root / "log-1/stdout.log").read_text().strip())
            result, _ = self.await_result(supervisor, h)
            self.assertEqual(result.outcome, "timeout")
            self.assertTrue(result.cleanup_complete)
            self.assertGreaterEqual(result.elapsed_seconds, 0.35)
            self.assertFalse(Path(f"/proc/{child}").exists())

    def test_leader_exits_first_and_orphan_is_cleaned(self) -> None:
        code = (
            "import subprocess,sys; "
            "p=subprocess.Popen([sys.executable,'-I','-c','import time; time.sleep(60)']); "
            "print(p.pid,flush=True)"
        )
        with resources.ResourceSupervisor() as supervisor:
            h = self.launch(supervisor, code)
            result, _ = self.await_result(supervisor, h)
            self.assertEqual(result.outcome, "descendants_remaining")
            self.assertTrue(result.cleanup_complete)
            child = int((self.root / "log-1/stdout.log").read_text().strip())
            self.assertFalse(Path(f"/proc/{child}").exists())

    def test_incomplete_first_scan_cannot_reap_leader_with_live_descendant(self) -> None:
        code = (
            "import subprocess,sys; "
            "p=subprocess.Popen([sys.executable,'-I','-c','import time; time.sleep(60)']); "
            "print(p.pid,flush=True)"
        )
        with resources.ResourceSupervisor() as supervisor:
            h = self.launch(supervisor, code)
            leader = supervisor._jobs[h].pid
            assert leader is not None
            log = self.root / "log-1/stdout.log"
            deadline = time.monotonic() + 3
            while (not log.exists() or not log.read_text().strip()) and time.monotonic() < deadline:
                time.sleep(0.01)
            child = int(log.read_text().strip())
            while (
                os.waitid(os.P_PID, leader, os.WEXITED | os.WNOHANG | os.WNOWAIT) is None
                and time.monotonic() < deadline
            ):
                time.sleep(0.01)
            self.assertIsNotNone(os.waitid(os.P_PID, leader, os.WEXITED | os.WNOWAIT))
            real_scan = resources._scan_groups
            hidden: list[bool] = []

            def incomplete_once(jobs: Sequence[resources._Job]) -> ProcessGroups:
                groups = real_scan(jobs)
                if not hidden and child in groups[leader]:
                    groups[leader].pop(child)
                    hidden.append(True)
                return groups

            with mock.patch.object(resources, "_scan_groups", side_effect=incomplete_once):
                supervisor.poll(wait_seconds=0)
            self.assertEqual(hidden, [True])
            self.assertIsNone(supervisor.result(h))
            result, _ = self.await_result(supervisor, h)
            self.assertEqual(result.outcome, "descendants_remaining")
            self.assertFalse(Path(f"/proc/{child}").exists())

    def test_local_and_aggregate_rss_limits(self) -> None:
        held = "import time; b=bytearray(24*1024*1024); b[:]=b'X'*len(b); print('ready',flush=True); time.sleep(60)"
        with resources.ResourceSupervisor(rss_limit_bytes=18 * 1024**2) as supervisor:
            h = self.launch(supervisor, held)
            result, _ = self.await_result(supervisor, h)
            self.assertEqual(result.outcome, "rss_limit")
            self.assertGreater(result.sampled_peak_rss_bytes, supervisor.rss_limit_bytes)
            self.assertEqual(
                result.observed_peak_rss_bytes,
                max(result.sampled_peak_rss_bytes, result.wait4_peak_rss_bytes),
            )
        smaller = "import time; b=bytearray(14*1024*1024); b[:]=b'X'*len(b); print('ready',flush=True); time.sleep(60)"
        with resources.ResourceSupervisor(rss_limit_bytes=38 * 1024**2) as supervisor:
            a = self.launch(supervisor, smaller)
            b = self.launch(supervisor, smaller)
            first, _ = self.await_result(supervisor, a)
            second, _ = self.await_result(supervisor, b)
            self.assertEqual(
                (first.outcome, second.outcome), ("aggregate_rss_limit", "aggregate_rss_limit")
            )
            self.assertGreater(
                supervisor.poll(wait_seconds=0).sampled_peak_rss_bytes, supervisor.rss_limit_bytes
            )
            self.assertLess(first.sampled_peak_rss_bytes, supervisor.rss_limit_bytes)
            self.assertLess(second.sampled_peak_rss_bytes, supervisor.rss_limit_bytes)

    def test_final_wait4_highwater_is_not_aggregate(self) -> None:
        real_wait4 = os.wait4

        def inflated(pid: int, flags: int) -> tuple[int, int, resource.struct_rusage | mock.Mock]:
            got, status, usage = real_wait4(pid, flags)
            if got:
                return got, status, mock.Mock(ru_maxrss=262144)
            return got, status, usage

        with resources.ResourceSupervisor(rss_limit_bytes=128 * 1024**2) as supervisor:
            with mock.patch.object(resources.os, "wait4", side_effect=inflated):
                h = self.launch(supervisor, "print('done')")
                result, _ = self.await_result(supervisor, h)
            self.assertEqual(result.outcome, "rss_limit")
            self.assertLess(result.sampled_peak_rss_bytes, supervisor.rss_limit_bytes)
            self.assertEqual(result.wait4_peak_rss_bytes, 256 * 1024**2)

    def test_cancel_one_and_context_exception_cleans_all(self) -> None:
        code = "import time; print('ready',flush=True); time.sleep(60)"
        with resources.ResourceSupervisor() as supervisor:
            a = self.launch(supervisor, code)
            b = self.launch(supervisor, "print('done')")
            self.await_text(supervisor, a, self.root / "log-1/stdout.log", "ready")
            supervisor.cancel(a)
            supervisor.cancel(a)
            self.assertEqual(self.await_result(supervisor, a)[0].outcome, "cancelled")
            self.assertEqual(self.await_result(supervisor, b)[0].outcome, "ok")
        supervisor = resources.ResourceSupervisor()
        with self.assertRaisesRegex(RuntimeError, "sentinel"):
            with supervisor:
                h = self.launch(supervisor, code)
                self.await_text(supervisor, h, self.root / "log-3/stdout.log", "ready")
                raise RuntimeError("sentinel")
        result = supervisor.result(h)
        assert result is not None
        self.assertEqual(result.outcome, "cancelled")
        self.assertTrue(result.cleanup_complete)

    def test_spawn_failure_private_logs_and_validation(self) -> None:
        with resources.ResourceSupervisor() as supervisor:
            h = supervisor.start(
                ["/definitely/missing/executable"],
                cwd=self.root,
                log_dir=self.root / "failed",
                remaining_seconds=1,
            )
            result = supervisor.result(h)
            assert result is not None
            self.assertEqual(result.outcome, "spawn_failed")
            self.assertEqual(supervisor.poll(wait_seconds=0).completed, (h,))
            self.assertEqual((self.root / "failed").stat().st_mode & 0o777, 0o700)
            self.assertEqual((self.root / "failed/stdout.log").stat().st_mode & 0o777, 0o600)
            sentinel = self.root / "sentinel"
            sentinel.write_text("unchanged")
            with self.assertRaisesRegex(ResearchError, "research_resources_log_invalid"):
                supervisor.start(
                    [sys.executable],
                    cwd=self.root,
                    log_dir=self.root / "failed",
                    remaining_seconds=1,
                )
            (self.root / "link").symlink_to(sentinel)
            with self.assertRaisesRegex(ResearchError, "research_resources_log_invalid"):
                supervisor.start(
                    [sys.executable],
                    cwd=self.root,
                    log_dir=self.root / "link/next",
                    remaining_seconds=1,
                )
            self.assertEqual(sentinel.read_text(), "unchanged")
            for args in ([], ["relative"], [sys.executable, "bad\0arg"]):
                with self.assertRaisesRegex(ResearchError, "research_resources_invalid"):
                    supervisor.start(
                        args, cwd=self.root, log_dir=self.root / "unused", remaining_seconds=1
                    )

    def test_partial_log_creation_preserves_private_artifact_without_spawning(self) -> None:
        real_open = os.open

        def fail_second(
            name: str | bytes | os.PathLike[str] | os.PathLike[bytes],
            flags: int,
            mode: int = 0o777,
            *,
            dir_fd: int | None = None,
        ) -> int:
            if name == "stderr.log":
                raise OSError("injected private failure")
            return real_open(name, flags, mode, dir_fd=dir_fd)

        with resources.ResourceSupervisor() as supervisor:
            with mock.patch.object(resources.os, "open", side_effect=fail_second):
                with mock.patch.object(resources.subprocess, "Popen") as spawn:
                    with self.assertRaisesRegex(ResearchError, "research_resources_log_invalid"):
                        self.launch(supervisor, "print('unreachable')")
                    spawn.assert_not_called()
            self.assertEqual((self.root / "log-1").stat().st_mode & 0o777, 0o700)
            self.assertTrue((self.root / "log-1/stdout.log").exists())
            self.assertFalse((self.root / "log-1/stderr.log").exists())

    def test_restrictive_umask_and_log_sync_failure(self) -> None:
        previous_umask = os.umask(0o777)
        try:
            with resources.ResourceSupervisor() as supervisor:
                h = self.launch(supervisor, "print('done')")
                self.assertEqual((self.root / "log-1").stat().st_mode & 0o777, 0o700)
                self.assertEqual((self.root / "log-1/stdout.log").stat().st_mode & 0o777, 0o600)
                with mock.patch.object(resources.os, "fsync", side_effect=OSError("injected")):
                    result, _ = self.await_result(supervisor, h)
                self.assertEqual(result.outcome, "log_io_failed")
                self.assertEqual(result.returncode, 0)
        finally:
            os.umask(previous_umask)

    def test_elapsed_includes_delayed_final_log_cleanup(self) -> None:
        close_logs = resources._close_logs

        def delayed_close(job: resources._Job) -> bool:
            self.assertIsNone(job.outcome)
            time.sleep(1.15)
            return close_logs(job)

        with resources.ResourceSupervisor() as supervisor:
            started = time.monotonic()
            with mock.patch.object(resources, "_close_logs", side_effect=delayed_close):
                h = self.launch(supervisor, "print('done')", seconds=1.0)
                result, completed = self.await_result(supervisor, h, wall=4)
            wall_elapsed = time.monotonic() - started
            self.assertEqual((result.outcome, result.returncode, completed), ("timeout", 0, [h]))
            self.assertTrue(result.cleanup_complete)
            self.assertGreaterEqual(result.elapsed_seconds, 1.15)
            self.assertGreaterEqual(result.max_sample_gap_seconds, 1.10)
            self.assertGreaterEqual(wall_elapsed, result.elapsed_seconds)
            self.assertLess(wall_elapsed - result.elapsed_seconds, 0.5)

    def test_invalid_time_and_memory_inputs_have_fixed_errors(self) -> None:
        for limit in (0, -1, True, 1.5):
            with self.assertRaisesRegex(ResearchError, "research_resources_invalid"):
                resources.ResourceSupervisor(rss_limit_bytes=limit)  # type: ignore[arg-type]
        with resources.ResourceSupervisor() as supervisor:
            for value in (0, -1, float("nan"), float("inf"), 10**1000):
                with self.assertRaisesRegex(ResearchError, "research_resources_invalid"):
                    supervisor.start(
                        [sys.executable],
                        cwd=self.root,
                        log_dir=self.root / "unused",
                        remaining_seconds=value,
                    )
            for value in (-1, float("nan"), float("inf"), 10**1000):
                with self.assertRaisesRegex(ResearchError, "research_resources_invalid"):
                    supervisor.poll(wait_seconds=value)
            self.assertFalse((self.root / "unused").exists())

    def test_proc_scan_failure_stops_worker_and_restores_owner(self) -> None:
        code = "import time; print('ready',flush=True); time.sleep(60)"
        with resources.ResourceSupervisor() as supervisor:
            h = self.launch(supervisor, code)
            self.await_text(supervisor, h, self.root / "log-1/stdout.log", "ready")
            with mock.patch.object(resources, "_scan_groups", side_effect=OSError("private error")):
                supervisor.poll(wait_seconds=0)
            result, _ = self.await_result(supervisor, h)
            self.assertEqual(result.outcome, "monitor_failed")
            self.assertTrue(result.cleanup_complete)

    def test_cleanup_failure_retains_ownership_until_later_recovery(self) -> None:
        supervisor = resources.ResourceSupervisor()
        supervisor.__enter__()
        try:
            h = self.launch(supervisor, "print('done')")
            pid = supervisor._jobs[h].pid
            self.assertIsNotNone(pid)
            # A controlled false-positive live group member exercises the
            # post-KILL failure state without creating an unkillable process.
            real_scan = resources._scan_groups

            def held_group(jobs: Sequence[resources._Job]) -> ProcessGroups:
                groups = real_scan(jobs)
                if pid in groups and pid in groups[pid]:
                    groups[pid][pid + 100000] = ("S", pid, 1, 0)
                return groups

            with mock.patch.object(resources, "_scan_groups", side_effect=held_group):
                deadline = time.monotonic() + 8
                while supervisor.result(h) is None and time.monotonic() < deadline:
                    supervisor.poll()
                result = supervisor.result(h)
                assert result is not None
                self.assertEqual(
                    (result.outcome, result.cleanup_complete), ("cleanup_failed", False)
                )
                self.assertEqual(supervisor.poll(wait_seconds=0).completed, ())
                with self.assertRaisesRegex(ResearchError, "research_resources_cleanup_failed"):
                    self.launch(supervisor, "print('blocked')")
                with self.assertRaisesRegex(ResearchError, "research_resources_cleanup_failed"):
                    supervisor.close()
            completed: list[resources.JobHandle] = []
            for _ in range(3):
                completed.extend(supervisor.poll(wait_seconds=0).completed)
                current = supervisor.result(h)
                assert current is not None
                if current.cleanup_complete:
                    break
            final = supervisor.result(h)
            assert final is not None
            self.assertEqual((final.outcome, final.cleanup_complete), ("cleanup_failed", True))
            self.assertEqual(completed, [h])
            self.assertEqual(supervisor.poll(wait_seconds=0).completed, ())
        finally:
            supervisor.close()
        final = supervisor.result(h)
        assert final is not None
        self.assertEqual(
            (final.outcome, final.cleanup_complete),
            ("cleanup_failed", True),
        )

    def test_sample_gap_records_blocking_caller_delay(self) -> None:
        with resources.ResourceSupervisor() as supervisor:
            h = self.launch(supervisor, "import time; time.sleep(60)")
            supervisor.poll(wait_seconds=0)
            time.sleep(0.18)
            snap = supervisor.poll(wait_seconds=0)
            self.assertGreaterEqual(snap.max_sample_gap_seconds, 0.15)
            supervisor.cancel(h)
            result, _ = self.await_result(supervisor, h)
            self.assertGreaterEqual(result.max_sample_gap_seconds, 0.15)

    def test_unrelated_child_and_subreaper_restored(self) -> None:
        original = resources._subreaper_state()
        sig = signal.getsignal(signal.SIGCHLD)
        outsider = subprocess.Popen([sys.executable, "-I", "-c", "raise SystemExit(11)"])
        try:
            with resources.ResourceSupervisor() as supervisor:
                h = self.launch(supervisor, "print('done')")
                self.await_result(supervisor, h)
                state = os.waitid(os.P_PID, outsider.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
                if state is None:
                    time.sleep(0.1)
                    state = os.waitid(os.P_PID, outsider.pid, os.WEXITED | os.WNOWAIT)
                self.assertIsNotNone(state)
            self.assertEqual(outsider.wait(), 11)
            self.assertEqual(resources._subreaper_state(), original)
            self.assertIs(signal.getsignal(signal.SIGCHLD), sig)
        finally:
            if outsider.returncode is None:
                outsider.wait()

    def test_subreaper_query_passes_real_void_pointer(self) -> None:
        def fake_prctl(option: int, value: int | ctypes.c_void_p) -> None:
            self.assertEqual(option, resources._GET_SUBREAPER)
            assert isinstance(value, ctypes.c_void_p)
            ctypes.cast(value, ctypes.POINTER(ctypes.c_int)).contents.value = 1

        with mock.patch.object(resources, "_prctl", side_effect=fake_prctl) as prctl:
            self.assertEqual(resources._subreaper_state(), 1)
        prctl.assert_called_once()

    def test_thread_fork_and_handle_boundaries(self) -> None:
        with resources.ResourceSupervisor() as supervisor:
            other_errors = []

            def thread_call() -> None:
                try:
                    supervisor.poll(wait_seconds=0)
                except ResearchError as error:
                    other_errors.append(str(error))

            thread = threading.Thread(target=thread_call)
            thread.start()
            thread.join()
            self.assertEqual(other_errors, ["research_resources_busy"])
            with self.assertRaisesRegex(ResearchError, "research_resources_busy"):
                supervisor.__enter__()
            with self.assertRaisesRegex(ResearchError, "research_resources_busy"):
                with resources.ResourceSupervisor():
                    pass
            # A second active owner is rejected without changing the original.
            with self.assertRaisesRegex(ResearchError, "research_resources_invalid"):
                supervisor.result(resources.JobHandle(b"foreign", 1))
        source = str(Path(__file__).resolve().parents[2] / "src")
        code = (
            "import os,sys; sys.path.insert(0," + repr(source) + "); "
            "from recsys.research.resources import ResourceSupervisor; "
            "from recsys.research.protocol import ResearchError; "
            "s=ResourceSupervisor(); s.__enter__(); pid=os.fork(); "
            "\nif pid==0:\n"
            " try: s.close()\n"
            " except ResearchError as e: os._exit(0 if str(e)=='research_resources_process_mismatch' else 2)\n"
            " os._exit(3)\n"
            "assert os.waitstatus_to_exitcode(os.waitpid(pid,0)[1])==0; s.close()"
        )
        self.assertEqual(
            subprocess.run(
                [sys.executable, "-I", "-c", code], capture_output=True, check=False
            ).returncode,
            0,
        )

    def test_no_import_side_effects(self) -> None:
        source = str(Path(__file__).resolve().parents[2] / "src")
        code = (
            "import sys,signal,ctypes,os; sys.path.insert(0," + repr(source) + "); "
            "libc=ctypes.CDLL(None); before=ctypes.c_int(); libc.prctl(37,ctypes.byref(before),0,0,0); "
            "handler=signal.getsignal(signal.SIGCHLD); import recsys.research.resources; "
            "after=ctypes.c_int(); libc.prctl(37,ctypes.byref(after),0,0,0); "
            "assert before.value==after.value and signal.getsignal(signal.SIGCHLD)==handler; "
            "assert 'numpy' not in sys.modules and 'cupy' not in sys.modules"
        )
        self.assertEqual(
            subprocess.run(
                [sys.executable, "-I", "-c", code], capture_output=True, check=False
            ).returncode,
            0,
        )


if __name__ == "__main__":
    unittest.main()
