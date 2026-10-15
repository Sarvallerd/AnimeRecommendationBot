import os
import subprocess
import sys
import threading
import textwrap
import unittest
import weakref
from unittest.mock import patch

import numpy as np

from recsys import research_gpu as gpu
from recsys.research_protocol import ResearchError


class FakeCuda:
    instances = []
    fail_next_upload = False

    def __init__(self):
        self.closed = False
        self.uploads = 0
        self.fail_upload = FakeCuda.fail_next_upload
        FakeCuda.fail_next_upload = False
        self.fail_dot = False
        self.fail_close = False
        self.oom = False
        self.base_used = 0
        self.current_used = 0
        self.reserved = 0
        self.peak_used = 0
        self.peak_reserved = 0
        FakeCuda.instances.append(self)

    def open(self):
        return ({
            "backend": "cupy", "dtype": "float64", "device_index": 0,
            "device_name": "Fake CUDA", "compute_capability": "89",
            "cupy_version": "14.2.0", "cuda_runtime_version": 12090,
            "cuda_driver_version": 13030, "memory_pool_max_bytes": gpu._POOL_MAX,
            "query_batch_size": 128, "dot_block_rows": 128,
        }, 256)

    def used_bytes(self):
        return self.current_used + self.base_used

    def _track(self, value):
        size = value.nbytes
        self.current_used += size
        self.reserved = max(self.reserved, self.used_bytes())
        self.peak_used = max(self.peak_used, self.used_bytes())
        self.peak_reserved = max(self.peak_reserved, self.reserved)
        weakref.finalize(value, self._free, size)
        return value

    def _free(self, size):
        self.current_used -= size

    def memory_usage(self):
        return {
            "pool_used_bytes": self.used_bytes(),
            "pool_reserved_bytes": self.reserved,
            "pool_limit_bytes": 256,
            "pool_peak_used_bytes": self.peak_used,
            "pool_peak_reserved_bytes": self.peak_reserved,
        }

    def upload(self, value):
        self.uploads += 1
        if self.oom:
            raise MemoryError("private CUDA out of memory")
        if self.fail_upload:
            raise RuntimeError("private CUDA failure")
        return self._track(value.copy())

    def dot(self, left, right):
        if self.oom:
            raise MemoryError("private CUDA out of memory")
        if self.fail_dot:
            raise RuntimeError("private CUDA failure")
        return self._track(left @ right.T)

    def download(self, value):
        return value.copy()

    def close(self):
        self.closed = True
        if self.fail_close:
            raise RuntimeError("private CUDA cleanup failure")


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNone(gpu._SESSION)
        FakeCuda.instances.clear()
        FakeCuda.fail_next_upload = False
        self.factory = patch.object(gpu, "_new_adapter", FakeCuda)
        self.factory.start()
        self.addCleanup(self.factory.stop)
        self.features = np.asarray([[1.0, 2.0], [2.0, 1.0]])
        self.left = np.asarray([[2.0, 1.0]])

    def tearDown(self):
        self.assertIsNone(gpu._SESSION)

    def assert_code(self, code, callable_):
        with self.assertRaises(ResearchError) as caught:
            callable_()
        self.assertEqual(str(caught.exception), code)

    def test_import_has_no_array_or_cuda_side_effects(self):
        code = ("import os, sys; before = set(os.listdir('/proc/self/fd')); "
                "import recsys.research_gpu as gpu; "
                "assert 'numpy' not in sys.modules and 'cupy' not in sys.modules; "
                "assert gpu._SESSION is None and before == set(os.listdir('/proc/self/fd'))")
        completed = subprocess.run([sys.executable, "-I", "-c", code],
                                   capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_missing_extra_and_platform_are_unavailable(self):
        with patch.object(gpu, "_new_adapter", side_effect=ModuleNotFoundError("cupy")):
            self.assert_code("research_gpu_unavailable", gpu.probe_gpu)
        with patch.object(gpu.sys, "platform", "darwin"):
            self.assert_code("research_gpu_unavailable", gpu.probe_gpu)

    def test_memory_usage_tracks_short_lived_peak_and_resets(self):
        self.assert_code("research_gpu_closed", gpu.gpu_memory_usage)
        with gpu.gpu_execution():
            first = gpu.gpu_memory_usage()
            fields = {"pool_used_bytes", "pool_reserved_bytes", "pool_limit_bytes",
                      "pool_peak_used_bytes", "pool_peak_reserved_bytes"}
            self.assertEqual(set(first), fields)
            self.assertTrue(all(type(value) is int for value in first.values()))
            self.assertEqual(first["pool_peak_used_bytes"], 0)
            with gpu.CupyDenseDots(self.features) as dots:
                after_upload = gpu.gpu_memory_usage()
                self.assertEqual(after_upload["pool_used_bytes"], self.features.nbytes)
                with gpu.gpu_execution():
                    self.assertEqual(gpu.gpu_memory_usage(), after_upload)
                    self.assertEqual(dots.dot(self.left).tolist(), [[4.0, 5.0]])
                after_dot = gpu.gpu_memory_usage()
                self.assertEqual(after_dot["pool_used_bytes"], self.features.nbytes)
                self.assertGreater(after_dot["pool_peak_used_bytes"],
                                   after_dot["pool_used_bytes"])
                self.assertGreaterEqual(after_dot["pool_peak_reserved_bytes"],
                                        after_dot["pool_reserved_bytes"])
            after_close = gpu.gpu_memory_usage()
            self.assertEqual(after_close["pool_used_bytes"], 0)
            self.assertEqual(after_close["pool_peak_used_bytes"],
                             after_dot["pool_peak_used_bytes"])
        self.assert_code("research_gpu_closed", gpu.gpu_memory_usage)
        with gpu.gpu_execution():
            self.assertEqual(gpu.gpu_memory_usage()["pool_peak_used_bytes"], 0)
        self.assertEqual(len(FakeCuda.instances), 2)

    def test_metadata_copy_strict_expected_and_nested_probe(self):
        with gpu.gpu_execution() as metadata:
            expected = dict(metadata)
            metadata["device_name"] = "tampered"
            with gpu.gpu_execution(expected) as nested:
                self.assertEqual(nested, expected)
                self.assertEqual(gpu.probe_gpu(), expected)
            self.assertEqual(len(FakeCuda.instances), 1)
            for invalid in ({**expected, "query_batch_size": True},
                            {**expected, "device_name": "wrong"},
                            {**expected, "extra": 1}, [], None):
                if invalid is None:
                    continue
                self.assert_code("research_backend_mismatch",
                                 lambda invalid=invalid: gpu.gpu_execution(invalid).__enter__())
        self.assertTrue(FakeCuda.instances[0].closed)

    def test_inputs_outputs_and_closed_state(self):
        with gpu.CupyDenseDots(self.features) as dots:
            self.assertEqual(dots.dot(self.left).tolist(), [[4.0, 5.0]])
            for bad in (self.left.astype(np.float32), self.left[:, :1],
                        np.asarray([[float("nan"), 1.0]]),
                        np.asarray([[float("inf"), 1.0]]),
                        np.zeros((129, 2)), [[1.0, 2.0]]):
                self.assert_code("research_gpu_input_invalid", lambda bad=bad: dots.dot(bad))
            for bad in (self.features.astype(np.float32),
                        np.asarray([[float("nan"), 1.0]]),
                        np.zeros((0, 2)), self.features[:, ::-1]):
                self.assert_code("research_gpu_input_invalid", lambda bad=bad: gpu.CupyDenseDots(bad))
        dots.close()
        self.assert_code("research_gpu_closed", lambda: dots.dot(self.left))

    def test_two_dots_share_session_and_constructor_failure_unwinds(self):
        first = gpu.CupyDenseDots(self.features)
        second = gpu.CupyDenseDots(self.features)
        try:
            self.assertEqual(len(FakeCuda.instances), 1)
            self.assertEqual(gpu._SESSION.refs, 2)
            first.close()
            self.assertEqual(second.dot(self.left).tolist(), [[4.0, 5.0]])
            FakeCuda.instances[0].base_used = 240
            self.assert_code("research_gpu_resource_limit", lambda: gpu.CupyDenseDots(self.features))
            self.assertEqual(gpu._SESSION.refs, 1)
        finally:
            second.close()
        self.assertTrue(FakeCuda.instances[0].closed)
        FakeCuda.fail_next_upload = True
        self.assert_code("research_gpu_execution_failed", lambda: gpu.CupyDenseDots(self.features))
        self.assertTrue(FakeCuda.instances[-1].closed)

    def test_dot_failure_keeps_handle_reusable_and_releases(self):
        with gpu.CupyDenseDots(self.features) as dots:
            fake = FakeCuda.instances[-1]
            fake.fail_dot = True
            self.assert_code("research_gpu_execution_failed", lambda: dots.dot(self.left))
            fake.fail_dot = False
            self.assertEqual(dots.dot(self.left).tolist(), [[4.0, 5.0]])
        self.assertTrue(fake.closed)

    def test_out_of_memory_and_cleanup_error_release_session(self):
        with gpu.CupyDenseDots(self.features) as dots:
            fake = FakeCuda.instances[-1]
            fake.oom = True
            self.assert_code("research_gpu_resource_limit", lambda: dots.dot(self.left))
            fake.oom = False
            self.assertEqual(dots.dot(self.left).tolist(), [[4.0, 5.0]])
            fake.fail_close = True
            self.assert_code("research_gpu_execution_failed", dots.close)
            dots.close()
        self.assertIsNone(gpu._SESSION)
        with gpu.CupyDenseDots(self.features) as next_dots:
            self.assertEqual(next_dots.dot(self.left).tolist(), [[4.0, 5.0]])

    def test_constructor_control_flow_preserved_after_cleanup(self):
        for interruption in (KeyboardInterrupt(), SystemExit(17)):
            with self.subTest(interruption=type(interruption).__name__):
                with patch.object(FakeCuda, "upload", side_effect=interruption):
                    with self.assertRaises(type(interruption)) as caught:
                        gpu.CupyDenseDots(self.features)
                self.assertIs(caught.exception, interruption)
                self.assertIsNone(gpu._SESSION)
                self.assertTrue(FakeCuda.instances[-1].closed)
                with gpu.CupyDenseDots(self.features) as again:
                    self.assertEqual(again.dot(self.left).tolist(), [[4.0, 5.0]])

    def test_release_control_flow_preserved_and_lease_freed(self):
        for interruption in (KeyboardInterrupt(), SystemExit(23)):
            with self.subTest(interruption=type(interruption).__name__):
                dots = gpu.CupyDenseDots(self.features)
                with patch.object(FakeCuda, "close", side_effect=interruption):
                    with self.assertRaises(type(interruption)) as caught:
                        dots.close()
                self.assertIs(caught.exception, interruption)
                self.assertIsNone(gpu._SESSION)
                dots.close()
                with gpu.CupyDenseDots(self.features) as again:
                    self.assertEqual(again.dot(self.left).tolist(), [[4.0, 5.0]])

    def test_acquire_control_flow_preserved_after_cleanup(self):
        for interruption in (KeyboardInterrupt(), SystemExit(31)):
            with self.subTest(interruption=type(interruption).__name__):
                with patch.object(FakeCuda, "open", side_effect=interruption):
                    with self.assertRaises(type(interruption)) as caught:
                        gpu.gpu_execution().__enter__()
                self.assertIs(caught.exception, interruption)
                self.assertIsNone(gpu._SESSION)
                self.assertTrue(FakeCuda.instances[-1].closed)
                with gpu.gpu_execution():
                    self.assertIsNotNone(gpu._SESSION)

    def test_real_fork_cannot_release_parent_lease(self):
        # Fork before NumPy or CUDA initializes any worker threads.
        code = textwrap.dedent("""\
            import fcntl
            import os
            from recsys import research_gpu as gpu
            from recsys.research_protocol import ResearchError
            fd = gpu._lease()
            gpu._SESSION = gpu._Session(fd, None, {}, 1)
            child = os.fork()
            if child == 0:
                try:
                    gpu._acquire()
                    outcome = 2
                except ResearchError as exc:
                    outcome = 0 if str(exc) == 'research_gpu_process_mismatch' else 3
                gpu._cleanup_at_exit()
                os._exit(outcome)
            _, status = os.waitpid(child, 0)
            assert status == 0
            other = os.open(f'/tmp/arb-ml02-gpu-{os.getuid()}/device-0.lock', os.O_RDONLY)
            try:
                try:
                    fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    raise AssertionError('child released parent lease')
                except BlockingIOError:
                    pass
            finally:
                os.close(other)
                gpu._SESSION = None
                fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)
        """)
        completed = subprocess.run([sys.executable, "-I", "-c", code],
                                   capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_thread_and_fork_pid_guards(self):
        with gpu.CupyDenseDots(self.features) as dots:
            found = []

            def other_thread():
                for call in (gpu.probe_gpu, gpu.gpu_memory_usage,
                             lambda: dots.dot(self.left), dots.close):
                    try:
                        call()
                    except ResearchError as exc:
                        found.append(str(exc))

            worker = threading.Thread(target=other_thread)
            worker.start()
            worker.join()
            self.assertEqual(found, ["research_gpu_busy"] * 4)
            with patch.object(gpu.os, "getpid", return_value=os.getpid() + 1):
                self.assert_code("research_gpu_process_mismatch", gpu.probe_gpu)
                self.assert_code("research_gpu_process_mismatch", gpu.gpu_memory_usage)
                self.assert_code("research_gpu_process_mismatch", lambda: dots.dot(self.left))
                self.assert_code("research_gpu_process_mismatch", dots.close)
            self.assertEqual(dots.dot(self.left).tolist(), [[4.0, 5.0]])


@unittest.skipUnless(os.environ.get("ARB_RESEARCH_GPU_TEST") == "1", "GPU hardware opt-in")
class HardwareTests(unittest.TestCase):
    def assert_code(self, code, callable_):
        with self.assertRaises(ResearchError) as caught:
            callable_()
        self.assertEqual(str(caught.exception), code)

    def test_float64_parity_shared_pool_and_cleanup(self):
        metadata = gpu.probe_gpu()  # Opt-in missing hardware must fail.
        self.assert_code("research_gpu_closed", gpu.gpu_memory_usage)
        self.assertEqual(metadata["backend"], "cupy")
        features = np.arange(1, 257, dtype=np.float64).reshape(128, 2) / 11
        left = np.arange(1, 257, dtype=np.float64).reshape(128, 2) / 13
        with patch.object(gpu, "_new_adapter", wraps=gpu._new_adapter) as factory:
            with gpu.gpu_execution(metadata) as scoring:
                self.assertEqual(scoring, metadata)
                before_dots = gpu.gpu_memory_usage()
                self.assertEqual(set(before_dots), {
                    "pool_used_bytes", "pool_reserved_bytes", "pool_limit_bytes",
                    "pool_peak_used_bytes", "pool_peak_reserved_bytes"})
                self.assertTrue(all(type(value) is int for value in before_dots.values()))
                first = gpu.CupyDenseDots(features)
                second = gpu.CupyDenseDots(features.copy())
                try:
                    self.assertEqual(factory.call_count, 1)
                    self.assertEqual(gpu.probe_gpu(), metadata)
                    first.close()
                    output = second.dot(left)
                    measured = gpu.gpu_memory_usage()
                    self.assertGreater(measured["pool_peak_used_bytes"],
                                       measured["pool_used_bytes"])
                    self.assertGreaterEqual(measured["pool_peak_reserved_bytes"],
                                            measured["pool_reserved_bytes"])
                    self.assertEqual(measured["pool_limit_bytes"],
                                     gpu._SESSION.adapter.pool.get_limit())
                    self.assertEqual(output.shape, (128, 128))
                    self.assertEqual(output.dtype, np.float64)
                    self.assertTrue(np.isfinite(output).all())
                    np.testing.assert_allclose(output, left @ features.T,
                                               rtol=1e-12, atol=1e-12)
                    integer_features = np.asarray([[1., 0.], [1., 0.], [0., 1.]])
                    with gpu.CupyDenseDots(integer_features) as ties:
                        self.assertEqual(ties.dot(np.asarray([[1., 0.]])).tolist(),
                                         [[1.0, 1.0, 0.0]])
                    self.assertLessEqual(gpu._SESSION.adapter.pool.get_limit(), gpu._POOL_MAX)
                    self.assertLessEqual(gpu._SESSION.adapter.used_bytes(), gpu._SESSION.limit)
                finally:
                    second.close()
                    first.close()
                after_dots = gpu.gpu_memory_usage()
                self.assertEqual(after_dots["pool_used_bytes"], 0)
                self.assertEqual(after_dots["pool_peak_used_bytes"],
                                 measured["pool_peak_used_bytes"])
        self.assertIsNone(gpu._SESSION)
        self.assert_code("research_gpu_closed", gpu.gpu_memory_usage)
        with gpu.CupyDenseDots(np.asarray([[1.0]])) as again:
            self.assertLess(gpu.gpu_memory_usage()["pool_peak_used_bytes"],
                            measured["pool_peak_used_bytes"])
            self.assertEqual(again.dot(np.asarray([[2.0]])).tolist(), [[2.0]])

    def test_other_process_busy_then_acquires(self):
        gpu.probe_gpu()
        code = ("from recsys.research_gpu import probe_gpu; "
                "from recsys.research_protocol import ResearchError; "
                "import sys; "
                "exec('try:\\n probe_gpu()\\nexcept ResearchError as e:\\n "
                "sys.exit(0 if str(e) == \\\"research_gpu_busy\\\" else 2)\\n'); "
                "sys.exit(3)")
        with gpu.gpu_execution():
            busy = subprocess.run([sys.executable, "-I", "-c", code],
                                  capture_output=True, text=True)
            self.assertEqual(busy.returncode, 0, busy.stderr)
        released = subprocess.run([sys.executable, "-I", "-c",
                                   "from recsys.research_gpu import probe_gpu; probe_gpu()"],
                                  capture_output=True, text=True)
        self.assertEqual(released.returncode, 0, released.stderr)


if __name__ == "__main__":
    unittest.main()
