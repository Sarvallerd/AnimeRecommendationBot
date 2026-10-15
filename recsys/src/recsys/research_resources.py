"""Synchronous, bounded observation of trusted Linux worker process groups."""

from __future__ import annotations

import ctypes
import math
import os
from pathlib import Path
import secrets
import signal
import stat
import subprocess
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from .research_protocol import ResearchError

SAMPLE_INTERVAL_SECONDS = 0.1
_TERM_GRACE = 1.0
_KILL_GRACE = 5.0
_GET_SUBREAPER = 37
_SET_SUBREAPER = 36
_active_owner: ResourceSupervisor | None = None
_owner_lock = threading.Lock()


def _error(suffix: str) -> ResearchError:
    return ResearchError("research_resources_" + suffix)


def _valid_seconds(value: object, *, zero_allowed: bool = False) -> bool:
    if type(value) not in (int, float):
        return False
    try:
        number = float(value)
    except OverflowError:
        return False
    return math.isfinite(number) and (number >= 0 if zero_allowed else number > 0)


@dataclass(frozen=True, slots=True, repr=False)
class JobHandle:
    _token: bytes
    _sequence: int


@dataclass(frozen=True, slots=True, repr=False)
class ProcessResult:
    outcome: str
    returncode: int | None
    elapsed_seconds: float
    sampled_peak_rss_bytes: int
    wait4_peak_rss_bytes: int
    observed_peak_rss_bytes: int
    rss_samples: int
    max_sample_gap_seconds: float
    cleanup_complete: bool


@dataclass(frozen=True, slots=True, repr=False)
class PollSnapshot:
    completed: tuple[JobHandle, ...]
    active_count: int
    sampled_rss_bytes: int
    sampled_peak_rss_bytes: int
    rss_samples: int
    max_sample_gap_seconds: float


@dataclass(slots=True)
class _Job:
    handle: JobHandle
    started: float
    deadline: float
    log_fds: tuple[int, ...]
    process: subprocess.Popen[bytes] | None = None
    pid: int | None = None
    starttime: int | None = None
    outcome: str | None = None
    term_at: float | None = None
    kill_at: float | None = None
    result: ProcessResult | None = None
    returncode: int | None = None
    wait4_peak: int = 0
    sampled_peak: int = 0
    samples: int = 0
    last_sample: float | None = None
    max_gap: float = 0.0
    current_rss: int = 0
    owned_descendants: set[int] = field(default_factory=set)
    known_members: set[int] = field(default_factory=set)
    empty_zombie_seen: bool = False


def _prctl(option: int, value: int | ctypes.c_void_p) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(option, value, 0, 0, 0) != 0:
        raise _error("unavailable") from None


def _subreaper_state() -> int:
    state = ctypes.c_int()
    _prctl(_GET_SUBREAPER, ctypes.byref(state))
    return state.value


def _stat_fields(pid: int) -> tuple[str, int, int, int, int, int]:
    """Return state, ppid, pgrp, session, starttime, RSS pages."""
    with open(f"/proc/{pid}/stat", "rb") as stream:
        raw = stream.read(4096)
    end = raw.rfind(b")")
    if end < 0 or raw[end + 1 : end + 2] != b" ":
        raise ValueError("stat")
    fields = raw[end + 2 :].split()
    if len(fields) < 22:
        raise ValueError("stat")
    state = fields[0].decode("ascii")
    if len(state) != 1:
        raise ValueError("stat")
    rss_pages = int(fields[21])
    if rss_pages < 0:
        raise ValueError("stat")
    return state, int(fields[1]), int(fields[2]), int(fields[3]), int(fields[19]), rss_pages


def _scan_groups(jobs: Sequence[_Job]) -> dict[int, dict[int, tuple[str, int, int, int]]]:
    wanted = {job.pid for job in jobs if job.pid is not None}
    groups: dict[int, dict[int, tuple[str, int, int, int]]] = {pid: {} for pid in wanted}
    with os.scandir("/proc") as entries:
        for entry in entries:
            if not entry.name.isdecimal():
                continue
            pid = int(entry.name)
            try:
                state, ppid, pgrp, session, starttime, rss_pages = _stat_fields(pid)
            except (FileNotFoundError, ProcessLookupError):
                continue
            except (PermissionError, ValueError, OSError):
                if pid in wanted or any(pid in j.known_members for j in jobs):
                    raise _error("unavailable") from None
                continue
            if pgrp in wanted:
                groups[pgrp][pid] = (state, ppid, starttime, rss_pages)
    return groups


def _open_private_logs(path: Path) -> tuple[int, int]:
    if not path.is_absolute() or not path.name or path == Path("/"):
        raise _error("log_invalid")
    parent_fd: int | None = None
    dir_fd: int | None = None
    opened: list[int] = []
    try:
        parent_fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        for part in path.parent.parts[1:]:
            next_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=parent_fd)
            os.close(parent_fd)
            parent_fd = next_fd
        info = os.fstat(parent_fd)
        if info.st_uid != os.getuid() or info.st_mode & 0o077 or not stat.S_ISDIR(info.st_mode):
            raise _error("log_invalid")
        os.mkdir(path.name, 0o700, dir_fd=parent_fd)
        os.chmod(path.name, 0o700, dir_fd=parent_fd, follow_symlinks=False)
        dir_fd = os.open(path.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=parent_fd)
        for name in ("stdout.log", "stderr.log"):
            fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600, dir_fd=dir_fd)
            opened.append(fd)
            os.fchmod(fd, 0o600)
        os.fsync(dir_fd)
        return opened[0], opened[1]
    except (OSError, ResearchError):
        for fd in opened:
            os.close(fd)
        raise _error("log_invalid") from None
    finally:
        if dir_fd is not None:
            os.close(dir_fd)
        if parent_fd is not None:
            os.close(parent_fd)


def _close_logs(job: _Job) -> bool:
    ok = True
    for fd in job.log_fds:
        try:
            os.fsync(fd)
        except OSError:
            ok = False
        try:
            os.close(fd)
        except OSError:
            ok = False
    job.log_fds = ()
    return ok


class ResourceSupervisor:
    """One synchronous owner for a set of trusted, same-session worker groups."""

    def __init__(self, *, rss_limit_bytes: int = 8 * 1024**3):
        if type(rss_limit_bytes) is not int or rss_limit_bytes <= 0:
            raise _error("invalid")
        self.rss_limit_bytes = rss_limit_bytes
        self._pid = os.getpid()
        self._thread = threading.get_ident()
        self._token = secrets.token_bytes(24)
        self._sequence = 0
        self._jobs: dict[JobHandle, _Job] = {}
        self._entered = False
        self._closed = False
        self._blocked = False
        self._previous_subreaper: int | None = None
        self._sampled_peak = 0
        self._samples = 0
        self._last_sample: float | None = None
        self._max_gap = 0.0
        self._pending_completed: list[JobHandle] = []

    def _check(self) -> None:
        if os.getpid() != self._pid:
            raise _error("process_mismatch")
        if threading.get_ident() != self._thread:
            raise _error("busy")

    def __enter__(self) -> ResourceSupervisor:
        global _active_owner
        self._check()
        if self._closed:
            raise _error("closed")
        if self._entered:
            raise _error("busy")
        with _owner_lock:
            if _active_owner is not None:
                raise _error("busy")
            if (not sys.platform.startswith("linux") or not os.path.isdir("/proc")
                    or not all(hasattr(os, x) for x in ("wait4", "waitid", "WNOWAIT"))
                    or signal.getsignal(signal.SIGCHLD) != signal.SIG_DFL):
                raise _error("unavailable")
            previous = _subreaper_state()
            _prctl(_SET_SUBREAPER, 1)
            self._previous_subreaper = previous
            self._entered = True
            _active_owner = self
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> bool:
        self.close()
        return False

    def _require_entered(self) -> None:
        self._check()
        if self._closed or not self._entered:
            raise _error("closed")

    def _job(self, handle: JobHandle) -> _Job:
        self._check()
        if type(handle) is not JobHandle or handle._token != self._token or handle not in self._jobs:
            raise _error("invalid")
        return self._jobs[handle]

    def start(self, argv: Sequence[str], *, cwd: Path, log_dir: Path,
              remaining_seconds: float, env: Mapping[str, str] | None = None) -> JobHandle:
        self._require_entered()
        if self._blocked:
            raise _error("cleanup_failed")
        if (isinstance(argv, (str, bytes)) or not isinstance(argv, Sequence)
                or not argv or any(type(a) is not str or "\0" in a for a in argv)
                or not os.path.isabs(argv[0]) or not isinstance(cwd, Path)
                or not cwd.is_absolute() or not cwd.is_dir() or not isinstance(log_dir, Path)
                or not _valid_seconds(remaining_seconds)
                or not math.isfinite(time.monotonic() + float(remaining_seconds))
                or (env is not None and (not isinstance(env, Mapping) or any(
                    type(k) is not str or type(v) is not str or not k or "=" in k or "\0" in k or "\0" in v
                    for k, v in env.items())))):
            raise _error("invalid")
        fds = _open_private_logs(log_dir)
        self._sequence += 1
        handle = JobHandle(self._token, self._sequence)
        begun = time.monotonic()
        if not any(j.result is None or not j.result.cleanup_complete for j in self._jobs.values()):
            self._last_sample = begun
        job = _Job(handle, begun, begun + float(remaining_seconds), fds, last_sample=begun)
        self._jobs[handle] = job
        try:
            proc = subprocess.Popen(list(argv), cwd=cwd, env=None if env is None else dict(env),
                                    stdin=subprocess.DEVNULL, stdout=fds[0], stderr=fds[1],
                                    shell=False, close_fds=True, start_new_session=True)
        except (OSError, ValueError):
            self._finish(job, "spawn_failed")
            return handle
        job.process = proc
        job.pid = proc.pid
        try:
            _state, _ppid, group, session, birth, _rss = _stat_fields(proc.pid)
            if group != proc.pid or session != proc.pid:
                raise ValueError("group")
            job.starttime = birth
        except (OSError, ValueError):
            job.outcome = "monitor_failed"
            self._blocked = True
            # The leader PID remains owned; a subsequent poll attempts identity-safe cleanup.
        return handle

    def _signal_group(self, job: _Job, which: signal.Signals) -> bool:
        assert job.pid is not None
        try:
            _state, _ppid, pgrp, session, birth, _rss = _stat_fields(job.pid)
            if job.starttime is None or birth != job.starttime or pgrp != job.pid or session != job.pid:
                return False
            os.killpg(job.pid, which)
            return True
        except ProcessLookupError:
            return False
        except (OSError, ValueError):
            return False

    def _stop(self, job: _Job, reason: str) -> None:
        if job.result is not None:
            return
        if job.outcome is None:
            job.outcome = reason
        if job.pid is None:
            return
        if job.term_at is None:
            sent = self._signal_group(job, signal.SIGTERM)
            job.term_at = time.monotonic()
            if not sent:
                self._blocked = True

    @staticmethod
    def _wait4_one(job: _Job, pid: int) -> bool:
        try:
            got, status, usage = os.wait4(pid, os.WNOHANG)
        except OSError:
            return False
        if got == 0:
            return False
        job.wait4_peak = max(job.wait4_peak, max(0, usage.ru_maxrss) * 1024)
        if pid == job.pid:
            job.returncode = os.waitstatus_to_exitcode(status)
            assert job.process is not None
            job.process.returncode = job.returncode
        else:
            job.owned_descendants.discard(pid)
        return True

    def _finish(self, job: _Job, outcome: str, *, clean: bool = True) -> None:
        if job.result is not None and job.result.cleanup_complete:
            return
        logs_ok = _close_logs(job)
        if not clean:
            outcome = "cleanup_failed"
            self._blocked = True
        elif not logs_ok and outcome == "ok":
            outcome = "log_io_failed"
        elif job.wait4_peak > self.rss_limit_bytes and outcome == "ok":
            outcome = "rss_limit"
        finished = time.monotonic()
        elapsed = max(0.0, finished - job.started)
        if elapsed > job.deadline - job.started and outcome == "ok":
            outcome = "timeout"
        if job.last_sample is not None:
            job.max_gap = max(job.max_gap, max(0.0, finished - job.last_sample))
        job.result = ProcessResult(outcome, job.returncode, elapsed, job.sampled_peak,
                                   job.wait4_peak, max(job.sampled_peak, job.wait4_peak),
                                   job.samples, job.max_gap, clean)
        if clean:
            self._pending_completed.append(job.handle)

    def _sample(self) -> None:
        active = [j for j in self._jobs.values() if j.result is None or not j.result.cleanup_complete]
        if not active:
            return
        try:
            groups = _scan_groups(active)
        except (OSError, ResearchError):
            for job in active:
                self._stop(job, "monitor_failed")
                if job.term_at is not None and job.kill_at is None and time.monotonic() - job.term_at >= _TERM_GRACE:
                    self._signal_group(job, signal.SIGKILL)
                    job.kill_at = time.monotonic()
                if job.kill_at is not None and time.monotonic() - job.kill_at >= _KILL_GRACE:
                    self._finish(job, "cleanup_failed", clean=False)
            self._blocked = True
            return
        now = time.monotonic()
        current_by_pid = {
            pid: sum(rss * os.sysconf("SC_PAGE_SIZE") for state, _ppid, _birth, rss in group.values() if state != "Z")
            for pid, group in groups.items()
        }
        current_total = sum(current_by_pid.values())
        if current_total > self.rss_limit_bytes:
            for job in active:
                if job.pid is not None and groups.get(job.pid):
                    reason = "rss_limit" if current_by_pid[job.pid] > self.rss_limit_bytes else "aggregate_rss_limit"
                    self._stop(job, reason)
        for job in active:
            if job.pid is None:
                continue
            group = groups.get(job.pid, {})
            job.known_members = set(group)
            leader = group.get(job.pid)
            if leader is not None and job.starttime is None:
                try:
                    _state, _ppid, pgrp, session, birth, _rss = _stat_fields(job.pid)
                    if pgrp == job.pid and session == job.pid and birth == leader[2]:
                        job.starttime = birth
                        self._stop(job, "monitor_failed")
                except (OSError, ValueError):
                    pass
            if job.starttime is None and job.term_at is None:
                self._stop(job, "monitor_failed")
            leader_valid = leader is not None and job.starttime is not None and leader[2] == job.starttime
            if not leader_valid:
                self._stop(job, "monitor_failed")
                self._blocked = True
            current = current_by_pid[job.pid]
            job.current_rss = current
            job.sampled_peak = max(job.sampled_peak, current)
            job.samples += 1
            if job.last_sample is not None:
                job.max_gap = max(job.max_gap, now - job.last_sample)
            job.last_sample = now
            if current > self.rss_limit_bytes:
                self._stop(job, "rss_limit")
            if time.monotonic() >= job.deadline:
                self._stop(job, "timeout")
            for pid, (state, ppid, birth, _rss) in group.items():
                if pid == job.pid:
                    continue
                if ppid == self._pid and state == "Z":
                    job.owned_descendants.add(pid)
                    if not self._wait4_one(job, pid):
                        self._stop(job, "monitor_failed")
            if leader is not None:
                try:
                    observed = os.waitid(os.P_PID, job.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
                except OSError:
                    observed = None
                    self._stop(job, "monitor_failed")
                remaining = any(pid != job.pid and (state != "Z" or ppid != self._pid)
                                for pid, (state, ppid, _birth, _rss) in group.items())
                if observed is not None and job.outcome is None and remaining:
                    self._stop(job, "descendants_remaining")
            if job.term_at is not None and job.kill_at is None and time.monotonic() - job.term_at >= _TERM_GRACE:
                if not self._signal_group(job, signal.SIGKILL):
                    self._blocked = True
                job.kill_at = time.monotonic()
            if job.kill_at is not None and time.monotonic() - job.kill_at >= _KILL_GRACE and (
                    leader is None or leader[0] != "Z" or len(group) > 1):
                self._finish(job, "cleanup_failed", clean=False)
                continue
            if leader_valid and len(group) == 1 and leader[0] == "Z" and job.empty_zombie_seen:
                if self._wait4_one(job, job.pid):
                    outcome = job.outcome or ("ok" if job.returncode == 0 else "exit_nonzero")
                    self._finish(job, outcome)
                else:
                    self._stop(job, "monitor_failed")
            else:
                # The unreaped zombie pins the PGID. A concurrent /proc walk
                # can miss a descendant when the leader exits; a later walk
                # checks again while inherited-group survivors remain visible.
                job.empty_zombie_seen = leader_valid and len(group) == 1 and leader[0] == "Z"
        self._sampled_peak = max(self._sampled_peak, current_total)
        self._samples += 1
        finished_sample = time.monotonic()
        if self._last_sample is not None:
            self._max_gap = max(self._max_gap, finished_sample - self._last_sample)
        self._last_sample = finished_sample
        if not any(j.result is None or not j.result.cleanup_complete for j in self._jobs.values()):
            self._blocked = False
        return

    def poll(self, *, wait_seconds: float = SAMPLE_INTERVAL_SECONDS) -> PollSnapshot:
        self._require_entered()
        if not _valid_seconds(wait_seconds, zero_allowed=True) or wait_seconds > SAMPLE_INTERVAL_SECONDS:
            raise _error("invalid")
        self._sample()
        active = [j for j in self._jobs.values() if j.result is None or not j.result.cleanup_complete]
        if active and wait_seconds:
            next_event = min([time.monotonic() + SAMPLE_INTERVAL_SECONDS] +
                             [j.deadline for j in active if j.term_at is None] +
                             [j.term_at + _TERM_GRACE for j in active if j.term_at is not None and j.kill_at is None] +
                             [j.kill_at + _KILL_GRACE for j in active if j.kill_at is not None])
            delay = max(0.0, min(float(wait_seconds), next_event - time.monotonic()))
            if delay:
                time.sleep(delay)
            self._sample()
        active = [j for j in self._jobs.values() if j.result is None or not j.result.cleanup_complete]
        completed = tuple(self._pending_completed)
        self._pending_completed.clear()
        return PollSnapshot(completed, len(active), sum(j.current_rss for j in active),
                            self._sampled_peak, self._samples, self._max_gap)

    def result(self, handle: JobHandle) -> ProcessResult | None:
        return self._job(handle).result

    def cancel(self, handle: JobHandle) -> None:
        self._require_entered()
        self._stop(self._job(handle), "cancelled")

    def close(self) -> None:
        global _active_owner
        self._check()
        if self._closed:
            return
        if not self._entered:
            self._closed = True
            return
        for job in self._jobs.values():
            if job.result is None:
                self._stop(job, "cancelled")
        while any(j.result is None or not j.result.cleanup_complete for j in self._jobs.values()):
            self.poll()
            if any(j.result is not None and not j.result.cleanup_complete for j in self._jobs.values()):
                raise _error("cleanup_failed")
        with _owner_lock:
            try:
                _prctl(_SET_SUBREAPER, self._previous_subreaper or 0)
            except ResearchError:
                raise _error("cleanup_failed") from None
            _active_owner = None
            self._closed = True
            self._entered = False
