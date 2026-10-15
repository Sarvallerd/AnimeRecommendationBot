"""Controlled Linux process tests for the ML-02 resource supervisor."""

from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

from recsys.research_protocol import ResearchError
from recsys import research_resources as resources


class ResourceSupervisorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.number = 0

    def launch(self, supervisor, code, *, seconds=4, args=()):
        self.number += 1
        return supervisor.start([sys.executable, "-I", "-c", code, *args], cwd=self.root,
                                log_dir=self.root / f"log-{self.number}", remaining_seconds=seconds)

    def await_result(self, supervisor, handle, *, wall=8):
        deadline = time.monotonic() + wall
        completions = []
        while supervisor.result(handle) is None and time.monotonic() < deadline:
            completions.extend(supervisor.poll().completed)
        self.assertIsNotNone(supervisor.result(handle))
        return supervisor.result(handle), completions

    def await_text(self, supervisor, handle, path, expected, *, wall=5):
        deadline = time.monotonic() + wall
        while time.monotonic() < deadline:
            supervisor.poll()
            if path.exists() and expected in path.read_text():
                return
        self.fail("worker handshake missing")

    def test_success_nonzero_signal_and_literal_argv(self):
        with resources.ResourceSupervisor() as supervisor:
            literal = "; touch should-not-exist"
            h = self.launch(supervisor, "import sys; print(sys.argv[1]); print('err', file=sys.stderr); sys.exit(0)", args=(literal,))
            result, done = self.await_result(supervisor, h)
            self.assertEqual((result.outcome, result.returncode, done), ("ok", 0, [h]))
            self.assertEqual(supervisor.poll(wait_seconds=0).completed, ())
            self.assertIs(result, supervisor.result(h))
            self.assertEqual((self.root / "log-1/stdout.log").read_text().strip(), literal)
            self.assertEqual((self.root / "log-1/stderr.log").read_text().strip(), "err")
            self.assertFalse((self.root / "should-not-exist").exists())
            h2 = self.launch(supervisor, "raise SystemExit(7)")
            self.assertEqual((self.await_result(supervisor, h2)[0].outcome,
                              supervisor.result(h2).returncode), ("exit_nonzero", 7))
            h3 = self.launch(supervisor, "import os,signal; os.kill(os.getpid(),signal.SIGUSR1)")
            self.assertEqual((self.await_result(supervisor, h3)[0].outcome,
                              supervisor.result(h3).returncode), ("exit_nonzero", -signal.SIGUSR1))

    def test_inheritable_parent_fd_is_closed_in_worker(self):
        sentinel = self.root / "parent-only"
        sentinel.write_text("private")
        fd = os.open(sentinel, os.O_RDONLY)
        try:
            os.set_inheritable(fd, True)
            code = ("import os,sys; "
                    "p='/proc/self/fd/'+sys.argv[1]; "
                    "target=os.readlink(p) if os.path.exists(p) else ''; "
                    "assert target != sys.argv[2]")
            with resources.ResourceSupervisor() as supervisor:
                h = self.launch(supervisor, code, args=(str(fd), str(sentinel)))
                self.assertEqual(self.await_result(supervisor, h)[0].outcome, "ok")
        finally:
            os.close(fd)

    def test_timeout_kills_term_ignoring_grandchild_and_reaps(self):
        code = ("import os,signal,subprocess,sys,time; "
                "p=subprocess.Popen([sys.executable,'-I','-c',"
                "'import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)']); "
                "print(p.pid,flush=True); time.sleep(60)")
        with resources.ResourceSupervisor() as supervisor:
            h = self.launch(supervisor, code, seconds=.35)
            self.await_text(supervisor, h, self.root / "log-1/stdout.log", "\n")
            child = int((self.root / "log-1/stdout.log").read_text().strip())
            result, _ = self.await_result(supervisor, h)
            self.assertEqual(result.outcome, "timeout")
            self.assertTrue(result.cleanup_complete)
            self.assertGreaterEqual(result.elapsed_seconds, .35)
            self.assertFalse(Path(f"/proc/{child}").exists())

    def test_leader_exits_first_and_orphan_is_cleaned(self):
        code = ("import subprocess,sys; "
                "p=subprocess.Popen([sys.executable,'-I','-c','import time; time.sleep(60)']); "
                "print(p.pid,flush=True)")
        with resources.ResourceSupervisor() as supervisor:
            h = self.launch(supervisor, code)
            result, _ = self.await_result(supervisor, h)
            self.assertEqual(result.outcome, "descendants_remaining")
            self.assertTrue(result.cleanup_complete)
            child = int((self.root / "log-1/stdout.log").read_text().strip())
            self.assertFalse(Path(f"/proc/{child}").exists())

    def test_incomplete_first_scan_cannot_reap_leader_with_live_descendant(self):
        code = ("import subprocess,sys; "
                "p=subprocess.Popen([sys.executable,'-I','-c','import time; time.sleep(60)']); "
                "print(p.pid,flush=True)")
        with resources.ResourceSupervisor() as supervisor:
            h = self.launch(supervisor, code)
            leader = supervisor._jobs[h].pid
            log = self.root / "log-1/stdout.log"
            deadline = time.monotonic() + 3
            while (not log.exists() or not log.read_text().strip()) and time.monotonic() < deadline:
                time.sleep(.01)
            child = int(log.read_text().strip())
            while (os.waitid(os.P_PID, leader, os.WEXITED | os.WNOHANG | os.WNOWAIT) is None
                   and time.monotonic() < deadline):
                time.sleep(.01)
            self.assertIsNotNone(os.waitid(os.P_PID, leader, os.WEXITED | os.WNOWAIT))
            real_scan = resources._scan_groups
            hidden = []

            def incomplete_once(jobs):
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

    def test_local_and_aggregate_rss_limits(self):
        held = "import time; b=bytearray(24*1024*1024); b[:]=b'X'*len(b); print('ready',flush=True); time.sleep(60)"
        with resources.ResourceSupervisor(rss_limit_bytes=18 * 1024**2) as supervisor:
            h = self.launch(supervisor, held)
            result, _ = self.await_result(supervisor, h)
            self.assertEqual(result.outcome, "rss_limit")
            self.assertGreater(result.sampled_peak_rss_bytes, supervisor.rss_limit_bytes)
            self.assertEqual(result.observed_peak_rss_bytes,
                             max(result.sampled_peak_rss_bytes, result.wait4_peak_rss_bytes))
        smaller = "import time; b=bytearray(14*1024*1024); b[:]=b'X'*len(b); print('ready',flush=True); time.sleep(60)"
        with resources.ResourceSupervisor(rss_limit_bytes=38 * 1024**2) as supervisor:
            a = self.launch(supervisor, smaller)
            b = self.launch(supervisor, smaller)
            first, _ = self.await_result(supervisor, a)
            second, _ = self.await_result(supervisor, b)
            self.assertEqual((first.outcome, second.outcome),
                             ("aggregate_rss_limit", "aggregate_rss_limit"))
            self.assertGreater(supervisor.poll(wait_seconds=0).sampled_peak_rss_bytes,
                               supervisor.rss_limit_bytes)
            self.assertLess(first.sampled_peak_rss_bytes, supervisor.rss_limit_bytes)
            self.assertLess(second.sampled_peak_rss_bytes, supervisor.rss_limit_bytes)

    def test_final_wait4_highwater_is_not_aggregate(self):
        real_wait4 = os.wait4

        def inflated(pid, flags):
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

    def test_cancel_one_and_context_exception_cleans_all(self):
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
        self.assertEqual(supervisor.result(h).outcome, "cancelled")
        self.assertTrue(supervisor.result(h).cleanup_complete)

    def test_spawn_failure_private_logs_and_validation(self):
        with resources.ResourceSupervisor() as supervisor:
            h = supervisor.start(["/definitely/missing/executable"], cwd=self.root,
                                 log_dir=self.root / "failed", remaining_seconds=1)
            self.assertEqual(supervisor.result(h).outcome, "spawn_failed")
            self.assertEqual(supervisor.poll(wait_seconds=0).completed, (h,))
            self.assertEqual((self.root / "failed").stat().st_mode & 0o777, 0o700)
            self.assertEqual((self.root / "failed/stdout.log").stat().st_mode & 0o777, 0o600)
            sentinel = self.root / "sentinel"
            sentinel.write_text("unchanged")
            with self.assertRaisesRegex(ResearchError, "research_resources_log_invalid"):
                supervisor.start([sys.executable], cwd=self.root, log_dir=self.root / "failed", remaining_seconds=1)
            (self.root / "link").symlink_to(sentinel)
            with self.assertRaisesRegex(ResearchError, "research_resources_log_invalid"):
                supervisor.start([sys.executable], cwd=self.root, log_dir=self.root / "link/next", remaining_seconds=1)
            self.assertEqual(sentinel.read_text(), "unchanged")
            for args in ([], ["relative"], [sys.executable, "bad\0arg"]):
                with self.assertRaisesRegex(ResearchError, "research_resources_invalid"):
                    supervisor.start(args, cwd=self.root, log_dir=self.root / "unused", remaining_seconds=1)

    def test_partial_log_creation_preserves_private_artifact_without_spawning(self):
        real_open = os.open

        def fail_second(name, *args, **kwargs):
            if name == "stderr.log":
                raise OSError("injected private failure")
            return real_open(name, *args, **kwargs)

        with resources.ResourceSupervisor() as supervisor:
            with mock.patch.object(resources.os, "open", side_effect=fail_second):
                with mock.patch.object(resources.subprocess, "Popen") as spawn:
                    with self.assertRaisesRegex(ResearchError, "research_resources_log_invalid"):
                        self.launch(supervisor, "print('unreachable')")
                    spawn.assert_not_called()
            self.assertEqual((self.root / "log-1").stat().st_mode & 0o777, 0o700)
            self.assertTrue((self.root / "log-1/stdout.log").exists())
            self.assertFalse((self.root / "log-1/stderr.log").exists())

    def test_restrictive_umask_and_log_sync_failure(self):
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

    def test_elapsed_includes_delayed_final_log_cleanup(self):
        close_logs = resources._close_logs

        def delayed_close(job):
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
            self.assertLess(wall_elapsed - result.elapsed_seconds, .5)

    def test_invalid_time_and_memory_inputs_have_fixed_errors(self):
        for limit in (0, -1, True, 1.5):
            with self.assertRaisesRegex(ResearchError, "research_resources_invalid"):
                resources.ResourceSupervisor(rss_limit_bytes=limit)
        with resources.ResourceSupervisor() as supervisor:
            for value in (0, -1, float("nan"), float("inf"), 10**1000):
                with self.assertRaisesRegex(ResearchError, "research_resources_invalid"):
                    supervisor.start([sys.executable], cwd=self.root,
                                     log_dir=self.root / "unused", remaining_seconds=value)
            for value in (-1, float("nan"), float("inf"), 10**1000):
                with self.assertRaisesRegex(ResearchError, "research_resources_invalid"):
                    supervisor.poll(wait_seconds=value)
            self.assertFalse((self.root / "unused").exists())

    def test_proc_scan_failure_stops_worker_and_restores_owner(self):
        code = "import time; print('ready',flush=True); time.sleep(60)"
        with resources.ResourceSupervisor() as supervisor:
            h = self.launch(supervisor, code)
            self.await_text(supervisor, h, self.root / "log-1/stdout.log", "ready")
            with mock.patch.object(resources, "_scan_groups", side_effect=OSError("private error")):
                supervisor.poll(wait_seconds=0)
            result, _ = self.await_result(supervisor, h)
            self.assertEqual(result.outcome, "monitor_failed")
            self.assertTrue(result.cleanup_complete)

    def test_cleanup_failure_retains_ownership_until_later_recovery(self):
        supervisor = resources.ResourceSupervisor()
        supervisor.__enter__()
        try:
            h = self.launch(supervisor, "print('done')")
            pid = supervisor._jobs[h].pid
            self.assertIsNotNone(pid)
            # A controlled false-positive live group member exercises the
            # post-KILL failure state without creating an unkillable process.
            real_scan = resources._scan_groups

            def held_group(jobs):
                groups = real_scan(jobs)
                if pid in groups and pid in groups[pid]:
                    groups[pid][pid + 100000] = ("S", pid, 1, 0)
                return groups

            with mock.patch.object(resources, "_scan_groups", side_effect=held_group):
                deadline = time.monotonic() + 8
                while supervisor.result(h) is None and time.monotonic() < deadline:
                    supervisor.poll()
                result = supervisor.result(h)
                self.assertIsNotNone(result)
                self.assertEqual((result.outcome, result.cleanup_complete), ("cleanup_failed", False))
                self.assertEqual(supervisor.poll(wait_seconds=0).completed, ())
                with self.assertRaisesRegex(ResearchError, "research_resources_cleanup_failed"):
                    self.launch(supervisor, "print('blocked')")
                with self.assertRaisesRegex(ResearchError, "research_resources_cleanup_failed"):
                    supervisor.close()
            completed = []
            for _ in range(3):
                completed.extend(supervisor.poll(wait_seconds=0).completed)
                if supervisor.result(h).cleanup_complete:
                    break
            final = supervisor.result(h)
            self.assertEqual((final.outcome, final.cleanup_complete), ("cleanup_failed", True))
            self.assertEqual(completed, [h])
            self.assertEqual(supervisor.poll(wait_seconds=0).completed, ())
        finally:
            supervisor.close()
        self.assertEqual((supervisor.result(h).outcome, supervisor.result(h).cleanup_complete),
                         ("cleanup_failed", True))

    def test_sample_gap_records_blocking_caller_delay(self):
        with resources.ResourceSupervisor() as supervisor:
            h = self.launch(supervisor, "import time; time.sleep(60)")
            supervisor.poll(wait_seconds=0)
            time.sleep(.18)
            snap = supervisor.poll(wait_seconds=0)
            self.assertGreaterEqual(snap.max_sample_gap_seconds, .15)
            supervisor.cancel(h)
            result, _ = self.await_result(supervisor, h)
            self.assertGreaterEqual(result.max_sample_gap_seconds, .15)

    def test_unrelated_child_and_subreaper_restored(self):
        original = resources._subreaper_state()
        sig = signal.getsignal(signal.SIGCHLD)
        outsider = subprocess.Popen([sys.executable, "-I", "-c", "raise SystemExit(11)"])
        try:
            with resources.ResourceSupervisor() as supervisor:
                h = self.launch(supervisor, "print('done')")
                self.await_result(supervisor, h)
                state = os.waitid(os.P_PID, outsider.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
                if state is None:
                    time.sleep(.1)
                    state = os.waitid(os.P_PID, outsider.pid, os.WEXITED | os.WNOWAIT)
                self.assertIsNotNone(state)
            self.assertEqual(outsider.wait(), 11)
            self.assertEqual(resources._subreaper_state(), original)
            self.assertIs(signal.getsignal(signal.SIGCHLD), sig)
        finally:
            if outsider.returncode is None:
                outsider.wait()

    def test_thread_fork_and_handle_boundaries(self):
        with resources.ResourceSupervisor() as supervisor:
            other_errors = []

            def thread_call():
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
        source = str(Path(__file__).parents[1] / "src")
        code = ("import os,sys; sys.path.insert(0," + repr(source) + "); "
                "from recsys.research_resources import ResourceSupervisor; "
                "from recsys.research_protocol import ResearchError; "
                "s=ResourceSupervisor(); s.__enter__(); pid=os.fork(); "
                "\nif pid==0:\n"
                " try: s.close()\n"
                " except ResearchError as e: os._exit(0 if str(e)=='research_resources_process_mismatch' else 2)\n"
                " os._exit(3)\n"
                "assert os.waitstatus_to_exitcode(os.waitpid(pid,0)[1])==0; s.close()")
        self.assertEqual(subprocess.run([sys.executable, "-I", "-c", code],
                                        capture_output=True, check=False).returncode, 0)

    def test_no_import_side_effects(self):
        source = str(Path(__file__).parents[1] / "src")
        code = ("import sys,signal,ctypes,os; sys.path.insert(0," + repr(source) + "); "
                "libc=ctypes.CDLL(None); before=ctypes.c_int(); libc.prctl(37,ctypes.byref(before),0,0,0); "
                "handler=signal.getsignal(signal.SIGCHLD); import recsys.research_resources; "
                "after=ctypes.c_int(); libc.prctl(37,ctypes.byref(after),0,0,0); "
                "assert before.value==after.value and signal.getsignal(signal.SIGCHLD)==handler; "
                "assert 'numpy' not in sys.modules and 'cupy' not in sys.modules")
        self.assertEqual(subprocess.run([sys.executable, "-I", "-c", code],
                                        capture_output=True, check=False).returncode, 0)


if __name__ == "__main__":
    unittest.main()
