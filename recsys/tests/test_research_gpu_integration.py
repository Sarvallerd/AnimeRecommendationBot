"""GPU model integration against the small frozen research fixture."""

import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from recsys.research_artifacts import read_json, write_json
from recsys.research_evaluate import evaluate_predictions
from recsys.research_model_content import ContentScorer
from recsys.research_model_cf import _SVD, fit as fit_cf, load as load_cf
from recsys.research_models import ScoreVector, fit_model, load_model, read_scoring_inputs
from recsys.research_predict import _admit_gpu, gpu_preflight, predict_validation
from recsys.research_protocol import ResearchError
from research_model_support import ModelFixture
from test_research_model_cf import hand_inputs
from test_research_model_content import _glove


class BackendContractTests(unittest.TestCase):
    def test_publication_failure_writes_failed_measurement_without_output(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = ModelFixture(directory)
            prepared, identity = fixture.make_prepared()
            model = fixture.fit_popularity(prepared, identity)
            evidence = fixture.root / "evidence"
            evidence.mkdir(mode=0o700)
            measurement = evidence / "measure.json"
            output = fixture.root / "prediction"
            measured = {}
            with patch("recsys.research_models.release_info", return_value=fixture.lock), \
                 patch("recsys.research_predict.publish",
                       side_effect=OSError("injected publication failure")):
                with self.assertRaises(OSError):
                    predict_validation(prepared, model, output,
                                       expected_prepared_identity=identity,
                                       measurements=measured,
                                       measurement_output=measurement)
            self.assertFalse(output.exists())
            self.assertFalse(list(fixture.root.glob(".prediction.*")))
            self.assertEqual(measured["status"], "failed")
            self.assertEqual(read_json(measurement)[0], measured)
            self.assertEqual(measurement.stat().st_mode & 0o777, 0o600)

    def test_measurement_write_failure_keeps_published_output_honest(self):
        from recsys.research_predict import write_json as original_write

        with tempfile.TemporaryDirectory() as directory:
            fixture = ModelFixture(directory)
            prepared, identity = fixture.make_prepared()
            model = fixture.fit_popularity(prepared, identity)
            evidence = fixture.root / "evidence"
            evidence.mkdir(mode=0o700)
            measurement = evidence / "measure.json"
            output = fixture.root / "prediction"
            measured = {}
            failures = 0

            def fail_once(path, body):
                nonlocal failures
                if path == measurement and failures == 0:
                    failures += 1
                    raise OSError("injected measurement failure")
                return original_write(path, body)

            with patch("recsys.research_models.release_info", return_value=fixture.lock), \
                 patch("recsys.research_predict.write_json", side_effect=fail_once):
                with self.assertRaises(OSError):
                    predict_validation(prepared, model, output,
                                       expected_prepared_identity=identity,
                                       measurements=measured,
                                       measurement_output=measurement)
            self.assertTrue((output / "prediction-manifest.json").exists())
            self.assertEqual(measured["status"], "published_measurement_failed")
            self.assertEqual(read_json(measurement)[0], measured)

    def test_measurement_output_is_private_separate_and_no_replace(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = ModelFixture(directory)
            prepared, identity = fixture.make_prepared()
            model = fixture.fit_popularity(prepared, identity)
            evidence = fixture.root / "evidence"
            evidence.mkdir(mode=0o700)
            measurement = evidence / "measure.json"
            measurement.write_text("existing", encoding="utf-8")
            output = fixture.root / "prediction"
            with patch("recsys.research_models.release_info", return_value=fixture.lock):
                with self.assertRaisesRegex(ResearchError, "research_unsafe_path"):
                    predict_validation(prepared, model, output,
                                       expected_prepared_identity=identity,
                                       measurement_output=measurement)
            self.assertFalse(output.exists())
            self.assertEqual(measurement.read_text(encoding="utf-8"), "existing")

    def test_svd_gpu_batch_keeps_negative_zero_and_cold_semantics(self):
        inputs, rows = hand_inputs()
        fake_svd = (np.empty((3, 2)), np.asarray([9.0, 3.0]),
                    np.asarray([[1.0, 1e-16, -1.0, 7.0],
                                [0.0, 1.0, 0.0, 7.0]]))

        class FakeDots:
            def __init__(self, features):
                self.features = features

            def dot(self, left):
                return left @ self.features.T

            def close(self):
                pass

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("recsys.research_model_cf.iter_training_rows", return_value=iter(rows)), \
                 patch("sklearn.utils.extmath.randomized_svd", return_value=fake_svd):
                metadata = fit_cf(inputs, dict(_SVD), {}, root)
            manifest = {"variant_id": "puresvd", "parameters": dict(_SVD),
                        "metadata": metadata, "files": {"factors.npy": {}}}
            cpu = load_cf(root, manifest, inputs)
            with patch("recsys.research_gpu.CupyDenseDots", FakeDots):
                gpu = load_cf(root, manifest, inputs, backend="gpu")
                try:
                    queries = ((1,), (1, 4), (4,), (3, 1, 2))
                    actual = gpu.score_many(queries)
                    for expected, observed in zip(cpu.score_many(queries), actual, strict=True):
                        self.assertEqual(expected.supported_seeds, observed.supported_seeds)
                        np.testing.assert_array_equal(expected.available, observed.available)
                        np.testing.assert_allclose(expected.values, observed.values,
                                                   rtol=0, atol=1e-14)
                    self.assertEqual(actual[0].values.tolist(), [1, 1e-16, -1, 0])
                    self.assertEqual(actual[1].values.tolist(), [.5, 5e-17, -.5, 0])
                    self.assertFalse(actual[2].available.any())
                finally:
                    gpu.close()

    def test_gpu_admission_rejects_full_rank_flip_inside_raw_tolerance(self):
        low, high = .5000000004999, .5000000005001
        mask = np.asarray([True, True, False], dtype=np.bool_)
        cpu = ScoreVector(np.asarray([low, high, 0.0]), mask, 1)
        gpu = ScoreVector(np.asarray([high, low, 0.0]), mask, 1)
        reference = SimpleNamespace(score_many=lambda groups: (cpu,), close=lambda: None)
        scored = SimpleNamespace(score_many=lambda groups: (gpu,))
        module = SimpleNamespace(load=lambda *_args, **_kwargs: reference)
        model = {"variant_id": "puresvd",
                 "producer": {"execution": {"scoring": {"backend": "cupy"}},
                              "code_files_sha256": {"research_gpu.py": "a" * 64}}}
        first = (({"query_id": "q1_" + "a" * 64}, (3,)),)
        inputs = SimpleNamespace(mal_ids=(1, 2, 3))
        with patch("recsys.research_predict._backend", return_value=module):
            with self.assertRaisesRegex(ResearchError, "research_gpu_parity_failed"):
                _admit_gpu(first, Path("unused"), model, scored, inputs,
                           {"sha256": "b" * 64})

    def test_rank_zero_gpu_svd_never_creates_zero_dimension_dots(self):
        with tempfile.TemporaryDirectory() as directory:
            inputs = SimpleNamespace(mal_ids=(1, 2), train_stats={
                "train_users": 1, "train_completed_rows": 1,
                "items": {
                    "1": {"completed_users": 1, "positive_users": 1,
                          "rating_sum": 8, "rating_count": 1},
                    "2": {"completed_users": 0, "positive_users": 0,
                          "rating_sum": 0, "rating_count": 0}}})
            root = Path(directory)
            with patch("recsys.research_model_cf.iter_training_rows",
                       return_value=iter([("a", 1, 8)])):
                metadata = fit_cf(inputs, dict(_SVD), {}, root)
            self.assertEqual(metadata["effective_rank"], 0)
            with patch("recsys.research_gpu.CupyDenseDots",
                       side_effect=AssertionError("rank-zero upload")):
                scorer = load_cf(root, {"variant_id": "puresvd",
                                        "parameters": dict(_SVD), "metadata": metadata,
                                        "files": {"factors.npy": {}}}, inputs,
                                 backend="gpu")
                self.assertEqual(scorer.score((1,)).supported_seeds, 0)
                scorer.close()
                scorer.close()
                with self.assertRaisesRegex(ResearchError, "research_gpu_closed"):
                    scorer.score((1,))

    def test_content_gpu_batch_owns_cached_rows_and_preserves_query_order(self):
        ids = tuple(range(1, 131))
        vectors = np.zeros((len(ids), 300), dtype=np.float64)
        vectors[:, 0] = 1.0
        calls = []

        class FakeDots:
            def __init__(self, features):
                self.features = features
                self.closed = False

            def dot(self, left):
                calls.append(left.shape[0])
                return left @ self.features.T

            def close(self):
                self.closed = True

        with patch("recsys.research_gpu.CupyDenseDots", FakeDots):
            scorer = ContentScorer(ids, vectors, np.ones(len(ids), dtype=np.bool_),
                                   "content-mean", backend="gpu")
            queries = tuple((item,) for item in reversed(ids[:128]))
            rows = scorer.score_many(queries)
            self.assertEqual(len(rows), 128)
            self.assertEqual(calls, [128])
            self.assertEqual(rows[0].supported_seeds, 1)
            self.assertEqual(rows[-1].supported_seeds, 1)
            self.assertTrue(all(entry.base is None for entry in scorer.cache.entries.values()))
            self.assertLessEqual(scorer.cache.bytes, 512 * 1024 * 1024)
            self.assertEqual(scorer.score_many(()), ())
            with self.assertRaises(ResearchError):
                scorer.score_many(tuple((1,) for _ in range(129)))
            scorer.close()
            self.assertTrue(scorer.dots is None)

    def test_unsupported_gpu_variants_fail_explicitly(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = ModelFixture(directory)
            prepared, identity = fixture.make_prepared()
            for variant in ("popularity", "itemknn"):
                with self.subTest(variant=variant), self.assertRaisesRegex(
                        ResearchError, "research_backend_unsupported"):
                    with patch("recsys.research_models.release_info", return_value=fixture.lock):
                        fit_model(prepared, fixture.root / variant,
                                  expected_prepared_identity=identity, variant_id=variant,
                                  code_revision="a" * 40, backend="gpu")

    def test_cpu_manifest_keeps_original_execution_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = ModelFixture(directory)
            prepared, identity = fixture.make_prepared()
            model = fixture.fit_popularity(prepared, identity)
            manifest, _ = read_json(model / "model-manifest.json")
            self.assertEqual(manifest["producer"]["execution"],
                             {"backend": "cpu", "dtype": "float64", "blas_threads": 1})
            self.assertNotIn("research_gpu.py", manifest["producer"]["code_files_sha256"])


@unittest.skipUnless(os.getenv("ARB_RESEARCH_GPU_TEST") == "1", "hardware opt-in")
class HardwareIntegrationTests(unittest.TestCase):
    def test_four_variants_paired_payloads_prediction_parity_and_preflight(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = ModelFixture(directory)
            glove = Path(directory) / "glove.txt"
            fixture.lock["inputs"]["embeddings"] = _glove(glove)
            prepared, identity = fixture.make_prepared()
            evidence = fixture.root / "gpu-evidence"
            evidence.mkdir(mode=0o700)
            original_open = Path.open

            def development_only(path, *args, **kwargs):
                if path.name in {"train-interactions.tsv.gz", "validation-labels.jsonl.gz",
                                 "test-labels.jsonl.gz", "splits.tsv.gz"}:
                    raise AssertionError("nondevelopment stream opened during scoring")
                return original_open(path, *args, **kwargs)

            with patch("recsys.research_models.release_info", return_value=fixture.lock):
                inputs = read_scoring_inputs(prepared, expected_prepared_identity=identity)
                for variant in ("content-mean", "content-max", "content-union", "puresvd"):
                    with self.subTest(variant=variant):
                        assets = {"glove": glove} if variant.startswith("content-") else None
                        cpu = fixture.root / f"{variant}-cpu"
                        gpu = fixture.root / f"{variant}-gpu"
                        fit_model(prepared, cpu, expected_prepared_identity=identity,
                                  variant_id=variant, code_revision="a" * 40,
                                  assets=assets, backend="cpu")
                        fit_model(prepared, gpu, expected_prepared_identity=identity,
                                  variant_id=variant, code_revision="a" * 40,
                                  assets=assets, backend="gpu")
                        cpu_manifest, cpu_scorer = load_model(cpu, inputs=inputs)
                        gpu_manifest, gpu_scorer = load_model(gpu, inputs=inputs, backend="gpu")
                        try:
                            self.assertEqual(cpu_manifest["files"], gpu_manifest["files"])
                            self.assertIn("research_gpu.py",
                                          gpu_manifest["producer"]["code_files_sha256"])
                            self.assertEqual(gpu_manifest["producer"]["execution"]["scoring"]["backend"],
                                             "cupy")
                            for seeds in ((1,), (2, 1), (21,), (1, 2, 3)):
                                actual = gpu_scorer.score(seeds)
                                expected = cpu_scorer.score(seeds)
                                self.assertEqual(actual.supported_seeds, expected.supported_seeds)
                                np.testing.assert_array_equal(actual.available, expected.available)
                                np.testing.assert_allclose(actual.values, expected.values,
                                                           atol=1e-12, rtol=1e-12)
                        finally:
                            gpu_scorer.close()
                            close = getattr(cpu_scorer, "close", None)
                            if callable(close):
                                close()
                        output = fixture.root / f"{variant}-pred"
                        measured = {}
                        measurement_path = evidence / f"{variant}.json"
                        with patch.object(Path, "open", development_only):
                            predict_validation(prepared, gpu, output,
                                               expected_prepared_identity=identity,
                                               backend="gpu", measurements=measured,
                                               measurement_output=measurement_path)
                        self.assertEqual(read_json(measurement_path)[0], measured)
                        self.assertEqual(measurement_path.stat().st_mode & 0o777, 0o600)
                        self.assertEqual(measured["status"], "published")
                        self.assertTrue(measured["gpu_admission"]["complete_available_rankings_equal"])
                        self.assertGreater(measured["queries"], 0)
                        self.assertGreaterEqual(measured["dedicated_pool_session_including_admission"]
                                                ["pool_peak_reserved_bytes"], 0)
                        evaluate_predictions(prepared, output,
                                             fixture.root / f"{variant}-evaluation",
                                             split="validation",
                                             expected_prepared_identity=identity)
                        if variant == "content-mean":
                            diagnostic = fixture.root / "preflight"
                            with patch.object(Path, "open", development_only):
                                gpu_preflight(prepared, gpu, diagnostic,
                                              expected_prepared_identity=identity)
                            report, _ = read_json(diagnostic / "preflight.json")
                            self.assertEqual(report["status"], "passed")
                            self.assertIsNotNone(report["single_query_latency"])
                            self.assertIsNotNone(report["score_many_batch_throughput"])

    def test_gpu_manifest_runtime_type_tamper_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = ModelFixture(directory)
            prepared, identity = fixture.make_prepared()
            model = fixture.root / "gpu"
            with patch("recsys.research_models.release_info", return_value=fixture.lock):
                fit_model(prepared, model, expected_prepared_identity=identity,
                          variant_id="puresvd", code_revision="a" * 40, backend="gpu")
                inputs = read_scoring_inputs(prepared, expected_prepared_identity=identity)
                manifest, _ = read_json(model / "model-manifest.json")
                manifest["producer"]["execution"]["scoring"]["device_index"] = False
                (model / "model-manifest.json").unlink()
                write_json(model / "model-manifest.json", manifest)
                with self.assertRaisesRegex(ResearchError, "research_backend_mismatch"):
                    load_model(model, inputs=inputs, backend="gpu")

    def test_admission_and_writer_failures_release_gpu_and_never_publish(self):
        from recsys.research_gpu import probe_gpu
        from recsys.research_predict import GzipOneWriter

        with tempfile.TemporaryDirectory() as directory:
            fixture = ModelFixture(directory)
            prepared, identity = fixture.make_prepared()
            model = fixture.root / "gpu"
            with patch("recsys.research_models.release_info", return_value=fixture.lock):
                fit_model(prepared, model, expected_prepared_identity=identity,
                          variant_id="puresvd", code_revision="a" * 40, backend="gpu")
                with patch("recsys.research_predict._admit_gpu",
                           side_effect=ResearchError("research_gpu_parity_failed")):
                    with self.assertRaisesRegex(ResearchError, "research_gpu_parity_failed"):
                        predict_validation(prepared, model, fixture.root / "admission-fail",
                                           expected_prepared_identity=identity, backend="gpu")
                self.assertFalse((fixture.root / "admission-fail").exists())
                self.assertEqual(probe_gpu()["backend"], "cupy")
                with patch.object(GzipOneWriter, "write_json",
                                  side_effect=OSError("fixture writer failure")):
                    with self.assertRaises(OSError):
                        predict_validation(prepared, model, fixture.root / "writer-fail",
                                           expected_prepared_identity=identity, backend="gpu")
                self.assertFalse((fixture.root / "writer-fail").exists())
                self.assertEqual(probe_gpu()["backend"], "cupy")
                with patch("recsys.research_predict.publish",
                           side_effect=OSError("fixture publication failure")):
                    with self.assertRaises(OSError):
                        predict_validation(prepared, model, fixture.root / "publish-fail",
                                           expected_prepared_identity=identity, backend="gpu")
                self.assertFalse((fixture.root / "publish-fail").exists())
                self.assertEqual(probe_gpu()["backend"], "cupy")

    def test_two_loaded_gpu_scorers_share_session_and_close_independently(self):
        from recsys.research_gpu import gpu_execution, gpu_memory_usage, probe_gpu

        with tempfile.TemporaryDirectory() as directory:
            fixture = ModelFixture(directory)
            glove = Path(directory) / "glove.txt"
            fixture.lock["inputs"]["embeddings"] = _glove(glove)
            prepared, identity = fixture.make_prepared()
            content = fixture.root / "content-gpu"
            svd = fixture.root / "svd-gpu"
            with patch("recsys.research_models.release_info", return_value=fixture.lock):
                fit_model(prepared, content, expected_prepared_identity=identity,
                          variant_id="content-mean", code_revision="a" * 40,
                          assets={"glove": glove}, backend="gpu")
                fit_model(prepared, svd, expected_prepared_identity=identity,
                          variant_id="puresvd", code_revision="a" * 40, backend="gpu")
                inputs = read_scoring_inputs(prepared, expected_prepared_identity=identity)
                with gpu_execution() as expected:
                    _, content_scorer = load_model(content, inputs=inputs, backend="gpu")
                    _, svd_scorer = load_model(svd, inputs=inputs, backend="gpu")
                    try:
                        self.assertEqual(probe_gpu(), expected)
                        self.assertGreater(gpu_memory_usage()["pool_used_bytes"], 0)
                        content_scorer.close()
                        self.assertEqual(svd_scorer.score((1,)).supported_seeds, 1)
                    finally:
                        content_scorer.close()
                        svd_scorer.close()
            self.assertEqual(probe_gpu()["backend"], "cupy")


if __name__ == "__main__":
    unittest.main()
