"""Optional, bounded float64 CUDA dot products for offline research scoring.

The public boundary accepts numeric matrices only.  CUDA and NumPy are imported
when a caller acquires a session, never while importing this module.
"""

from __future__ import annotations

import atexit
import errno
import fcntl
import os
import platform
import stat
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Never, Protocol, TypedDict, cast

from recsys.research.protocol import ResearchError

if TYPE_CHECKING:
    import cupy as cp
    import numpy as np
    from numpy.typing import NDArray


class GpuMetadata(TypedDict):
    backend: str
    dtype: str
    device_index: int
    device_name: str
    compute_capability: str
    cupy_version: str
    cuda_runtime_version: int
    cuda_driver_version: int
    memory_pool_max_bytes: int
    query_batch_size: int
    dot_block_rows: int


class GpuMemory(TypedDict):
    pool_used_bytes: int
    pool_reserved_bytes: int
    pool_limit_bytes: int
    pool_peak_used_bytes: int
    pool_peak_reserved_bytes: int


class GpuAdapter(Protocol):
    def open(self) -> tuple[GpuMetadata, int]: ...
    def used_bytes(self) -> int: ...
    def memory_usage(self) -> GpuMemory: ...
    def upload(self, matrix: NDArray[np.float64]) -> object: ...
    def dot(self, left: object, right: object) -> object: ...
    def download(self, matrix: object) -> NDArray[np.float64]: ...
    def close(self) -> None: ...


_POOL_MAX = 6 * 1024**3
_DEVICE_RESERVE = 2 * 1024**3
_BATCH_MAX = 128
_DEVICE_INDEX = 0
_MEMORY_FIELDS = {
    "pool_used_bytes",
    "pool_reserved_bytes",
    "pool_limit_bytes",
    "pool_peak_used_bytes",
    "pool_peak_reserved_bytes",
}
_METADATA_FIELDS = {
    "backend",
    "dtype",
    "device_index",
    "device_name",
    "compute_capability",
    "cupy_version",
    "cuda_runtime_version",
    "cuda_driver_version",
    "memory_pool_max_bytes",
    "query_batch_size",
    "dot_block_rows",
}
_GUARD = threading.RLock()
_SESSION: _Session | None = None


def _error(code: str) -> Never:
    raise ResearchError(code)


def _lease() -> int:
    uid = os.getuid()
    directory = Path("/tmp") / f"arb-ml02-gpu-{uid}"
    try:
        try:
            directory.mkdir(mode=0o700)
        except FileExistsError:
            pass
        info = directory.lstat()
        if (
            not stat.S_ISDIR(info.st_mode)
            or info.st_uid != uid
            or stat.S_IMODE(info.st_mode) != 0o700
        ):
            _error("research_gpu_unavailable")
        fd = os.open(
            directory / f"device-{_DEVICE_INDEX}.lock",
            os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o600,
        )
        try:
            info = os.fstat(fd)
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_uid != uid
                or stat.S_IMODE(info.st_mode) != 0o600
                or info.st_nlink != 1
            ):
                _error("research_gpu_unavailable")
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                if exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                    _error("research_gpu_busy")
                raise
            return fd
        except BaseException:
            os.close(fd)
            raise
    except ResearchError:
        raise
    except (OSError, ValueError):
        _error("research_gpu_unavailable")


class _CudaAdapter:
    """Private CUDA boundary. Tests replace its factory, not public APIs."""

    def __init__(self) -> None:
        import cupy as cp

        self.cp = cp
        self.pool = None
        self.previous_allocator = None
        self.allocator_set = False
        self.peak_used_bytes = 0
        self.peak_reserved_bytes = 0

    def _pool_malloc(self, size: int) -> cp.cuda.MemoryPointer:
        if self.pool is None:
            _error("research_gpu_closed")
        pointer = self.pool.malloc(size)
        self.peak_used_bytes = max(self.peak_used_bytes, int(self.pool.used_bytes()))
        self.peak_reserved_bytes = max(self.peak_reserved_bytes, int(self.pool.total_bytes()))
        return pointer

    def memory_usage(self) -> GpuMemory:
        if self.pool is None:
            _error("research_gpu_closed")
        return {
            "pool_used_bytes": int(self.pool.used_bytes()),
            "pool_reserved_bytes": int(self.pool.total_bytes()),
            "pool_limit_bytes": int(self.pool.get_limit()),
            "pool_peak_used_bytes": self.peak_used_bytes,
            "pool_peak_reserved_bytes": self.peak_reserved_bytes,
        }

    def open(self) -> tuple[GpuMetadata, int]:
        cp = self.cp
        cp.cuda.Device(_DEVICE_INDEX).use()
        properties = cp.cuda.runtime.getDeviceProperties(_DEVICE_INDEX)
        free_bytes, _ = cp.cuda.runtime.memGetInfo()
        available = int(free_bytes) - _DEVICE_RESERVE
        if available <= 0:
            _error("research_gpu_resource_limit")
        limit = min(_POOL_MAX, available)
        pool = cp.cuda.MemoryPool()
        pool.set_limit(size=limit)
        self.previous_allocator = cp.cuda.get_allocator()
        self.pool = pool
        cp.cuda.set_allocator(self._pool_malloc)
        self.allocator_set = True
        # Check the pinned toolkit's compiler as well as the driver and BLAS path.
        kernel = cp.ElementwiseKernel("float64 x", "float64 y", "y = x", "arb_research_gpu_probe")
        signal = cp.asarray([1.0], dtype=cp.float64)
        echoed = kernel(signal)
        if cp.asnumpy(echoed, blocking=True).tolist() != [1.0]:
            _error("research_gpu_unavailable")
        del echoed, signal
        name = properties["name"]
        if isinstance(name, bytes):
            name = name.decode("utf-8")
        metadata: GpuMetadata = {
            "backend": "cupy",
            "dtype": "float64",
            "device_index": _DEVICE_INDEX,
            "device_name": str(name),
            "compute_capability": f"{properties['major']}{properties['minor']}",
            "cupy_version": str(cp.__version__),
            "cuda_runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
            "cuda_driver_version": int(cp.cuda.runtime.driverGetVersion()),
            "memory_pool_max_bytes": _POOL_MAX,
            "query_batch_size": _BATCH_MAX,
            "dot_block_rows": _BATCH_MAX,
        }
        return metadata, limit

    def used_bytes(self) -> int:
        if self.pool is None:
            _error("research_gpu_closed")
        return int(self.pool.used_bytes())

    def upload(self, matrix: NDArray[np.float64]) -> cp.ndarray:
        return self.cp.asarray(matrix)

    def dot(self, left: object, right: object) -> cp.ndarray:
        return cast("cp.ndarray", left) @ cast("cp.ndarray", right).T

    def download(self, matrix: object) -> NDArray[np.float64]:
        return self.cp.asnumpy(cast("cp.ndarray", matrix), blocking=True)

    def close(self) -> None:
        try:
            if self.pool is not None:
                self.cp.cuda.get_current_stream().synchronize()
        finally:
            try:
                if self.allocator_set:
                    self.cp.cuda.set_allocator(self.previous_allocator)
            finally:
                if self.pool is not None:
                    self.pool.free_all_blocks()


def _new_adapter() -> GpuAdapter:
    return _CudaAdapter()


class _Session:
    def __init__(self, fd: int, adapter: GpuAdapter, metadata: GpuMetadata, limit: int) -> None:
        self.fd = fd
        self.adapter = adapter
        self.metadata = metadata
        self.limit = limit
        self.pid = os.getpid()
        self.thread = threading.get_ident()
        self.refs = 1


def _acquire() -> _Session:
    global _SESSION
    with _GUARD:
        current = _SESSION
        if current is not None:
            if current.pid != os.getpid():
                _error("research_gpu_process_mismatch")
            if current.thread != threading.get_ident():
                _error("research_gpu_busy")
            current.refs += 1
            return current
        if sys.platform != "linux" or platform.machine() != "x86_64":
            _error("research_gpu_unavailable")
        fd = _lease()
        adapter = None
        try:
            adapter = _new_adapter()
            metadata, limit = adapter.open()
            if (
                type(metadata) is not dict
                or set(metadata) != _METADATA_FIELDS
                or type(limit) is not int
                or not 0 < limit <= _POOL_MAX
            ):
                _error("research_gpu_unavailable")
            current = _Session(fd, adapter, metadata, limit)
            _SESSION = current
            return current
        except BaseException as exc:
            cleanup_error = None
            try:
                if adapter is not None:
                    adapter.close()
            except BaseException as close_error:
                cleanup_error = close_error
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                cleanup_error = cleanup_error or ResearchError("research_gpu_execution_failed")
            finally:
                os.close(fd)
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            if cleanup_error is not None and not isinstance(cleanup_error, Exception):
                raise cleanup_error
            if cleanup_error is not None:
                _error("research_gpu_execution_failed")
            if isinstance(exc, ResearchError):
                raise
            _error(
                "research_gpu_resource_limit"
                if isinstance(exc, MemoryError)
                else "research_gpu_unavailable"
            )


def _release(current: _Session) -> None:
    global _SESSION
    with _GUARD:
        if current.pid != os.getpid():
            _error("research_gpu_process_mismatch")
        if current.thread != threading.get_ident():
            _error("research_gpu_busy")
        current.refs -= 1
        if current.refs:
            return
        _SESSION = None
        cleanup_error = None
        try:
            current.adapter.close()
        except BaseException as exc:
            cleanup_error = exc
        try:
            fcntl.flock(current.fd, fcntl.LOCK_UN)
        except OSError:
            cleanup_error = cleanup_error or ResearchError("research_gpu_execution_failed")
        finally:
            os.close(current.fd)
        if cleanup_error is not None and not isinstance(cleanup_error, Exception):
            raise cleanup_error
        if cleanup_error is not None:
            _error("research_gpu_execution_failed")


def _check_expected(expected: object, actual: GpuMetadata) -> None:
    if expected is None:
        return
    actual_fields: dict[str, object] = dict(actual)
    if (
        type(expected) is not dict
        or set(expected) != _METADATA_FIELDS
        or any(
            type(expected[key]) is not type(actual_fields[key])
            or expected[key] != actual_fields[key]
            for key in _METADATA_FIELDS
        )
    ):
        _error("research_backend_mismatch")


@contextmanager
def gpu_execution(expected_scoring: object = None) -> Iterator[GpuMetadata]:
    """Lease the process-local GPU session and yield immutable provenance by copy."""
    current = _acquire()
    try:
        _check_expected(expected_scoring, current.metadata)
        yield cast("GpuMetadata", dict(current.metadata))
    finally:
        _release(current)


def gpu_memory_usage() -> GpuMemory:
    """Return measured bytes of the active dedicated pool and its session peaks."""
    with _GUARD:
        current = _SESSION
        if current is None:
            _error("research_gpu_closed")
        if current.pid != os.getpid():
            _error("research_gpu_process_mismatch")
        if current.thread != threading.get_ident():
            _error("research_gpu_busy")
        try:
            usage = current.adapter.memory_usage()
        except Exception:
            _error("research_gpu_execution_failed")
        if (
            type(usage) is not dict
            or set(usage) != _MEMORY_FIELDS
            or any(type(value) is not int or value < 0 for value in usage.values())
            or usage["pool_used_bytes"] > usage["pool_peak_used_bytes"]
            or usage["pool_reserved_bytes"] > usage["pool_peak_reserved_bytes"]
            or usage["pool_used_bytes"] > usage["pool_reserved_bytes"]
            or usage["pool_reserved_bytes"] > usage["pool_limit_bytes"]
        ):
            _error("research_gpu_execution_failed")
        return cast("GpuMemory", dict(usage))


def _matrix(
    value: object, *, rows_max: int | None = None, columns: int | None = None
) -> NDArray[np.float64]:
    import numpy as np

    if (
        type(value) is not np.ndarray
        or value.ndim != 2
        or value.dtype != np.dtype("float64")
        or not value.dtype.isnative
        or not value.flags.c_contiguous
        or value.shape[0] < 1
        or value.shape[1] < 1
        or (rows_max is not None and value.shape[0] > rows_max)
        or (columns is not None and value.shape[1] != columns)
        or not bool(np.isfinite(value).all())
    ):
        _error("research_gpu_input_invalid")
    return cast("NDArray[np.float64]", value)


def _ensure_capacity(current: _Session, required: int) -> None:
    if required < 0 or current.adapter.used_bytes() + required > current.limit:
        _error("research_gpu_resource_limit")


class CupyDenseDots:
    """Upload a catalog matrix once and score contiguous float64 query blocks."""

    def __init__(self, features: NDArray[np.float64]) -> None:
        features = _matrix(features)
        self._session = None
        self._features = None
        self._columns = features.shape[1]
        self._rows = features.shape[0]
        current = _acquire()
        try:
            _ensure_capacity(current, features.nbytes)
            self._features = current.adapter.upload(features)
            self._session = current
        except BaseException as exc:
            self._features = None
            try:
                _release(current)
            finally:
                if isinstance(exc, (KeyboardInterrupt, SystemExit, ResearchError)):
                    raise
                _error(
                    "research_gpu_resource_limit"
                    if isinstance(exc, MemoryError)
                    else "research_gpu_execution_failed"
                )

    def dot(self, left: NDArray[np.float64]) -> NDArray[np.float64]:
        current = self._session
        if current is None:
            _error("research_gpu_closed")
        if current.pid != os.getpid():
            _error("research_gpu_process_mismatch")
        if current.thread != threading.get_ident():
            _error("research_gpu_busy")
        left = _matrix(left, rows_max=_BATCH_MAX, columns=self._columns)
        if self._features is None:
            _error("research_gpu_closed")
        _ensure_capacity(current, left.nbytes + left.shape[0] * self._rows * 8)
        left_gpu = score_gpu = None
        try:
            left_gpu = current.adapter.upload(left)
            score_gpu = current.adapter.dot(left_gpu, self._features)
            result = current.adapter.download(score_gpu)
            if (
                type(result) is not type(left)
                or result.dtype != left.dtype
                or result.shape != (left.shape[0], self._rows)
                or not bool(__import__("numpy").isfinite(result).all())
            ):
                _error("research_gpu_execution_failed")
            return result
        except ResearchError:
            raise
        except MemoryError:
            _error("research_gpu_resource_limit")
        except Exception:
            _error("research_gpu_execution_failed")
        finally:
            score_gpu = None
            left_gpu = None

    def close(self) -> None:
        current = self._session
        if current is None:
            return
        if current.pid != os.getpid():
            _error("research_gpu_process_mismatch")
        if current.thread != threading.get_ident():
            _error("research_gpu_busy")
        self._features = None
        self._session = None
        _release(current)

    def __enter__(self) -> CupyDenseDots:
        if self._session is None:
            _error("research_gpu_closed")
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.close()


def probe_gpu() -> GpuMetadata:
    """Exercise acquisition and a tiny float64 dot before reporting the device."""
    import numpy as np

    with gpu_execution() as metadata:
        with CupyDenseDots(np.asarray([[1.0, 2.0], [3.0, 4.0]])) as dots:
            result = dots.dot(np.asarray([[2.0, 1.0]]))
            if result.tolist() != [[4.0, 10.0]]:
                _error("research_gpu_execution_failed")
        return metadata


def _cleanup_at_exit() -> None:
    global _SESSION
    current = _SESSION
    if current is None or current.pid != os.getpid():
        return
    _SESSION = None
    try:
        current.adapter.close()
    except Exception:
        pass
    try:
        fcntl.flock(current.fd, fcntl.LOCK_UN)
    finally:
        os.close(current.fd)


atexit.register(_cleanup_at_exit)
