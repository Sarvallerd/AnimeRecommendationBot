"""Small exact examples and integrity checks for the two CF baselines."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from recsys.bundle import canonical_bytes
from recsys.research_artifacts import read_json, stream_jsonl
from recsys.research_models import fit_model, load_model, read_scoring_inputs
from recsys.research_model_cf import _KNN, _SVD, _seed_columns, _training_matrix, fit, load
from recsys.research_protocol import ResearchError, sha
from research_model_support import ModelFixture


def hand_inputs():
    rows = [("a", 1, 8), ("a", 2, 9), ("b", 1, 10), ("b", 3, 8),
            ("c", 2, 8), ("c", 3, 10), ("c", 4, 7)]
    stats = {str(item): {"completed_users": 0, "positive_users": 0,
                         "rating_sum": 0, "rating_count": 0, "band": "fixture"}
             for item in range(1, 5)}
    for _, item, rating in rows:
        entry = stats[str(item)]
        entry["completed_users"] += 1
        entry["positive_users"] += rating >= 8
        entry["rating_sum"] += rating
        entry["rating_count"] += 1
    return SimpleNamespace(mal_ids=(1, 2, 3, 4), train_stats={
        "train_users": 3, "train_completed_rows": len(rows), "items": stats}), rows


class HandMatrixTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.inputs, self.rows = hand_inputs()

    def matrix(self):
        with patch("recsys.research_model_cf.iter_training_rows", return_value=iter(self.rows)):
            return _training_matrix(self.inputs)

    def test_binary_matrix_counts_and_all_stat_fields(self):
        matrix, users, positive = self.matrix()
        self.assertEqual(users, 3)
        self.assertEqual(positive.tolist(), [2, 2, 2, 0])
        self.assertEqual(matrix.toarray().tolist(), [[1, 1, 0, 0], [1, 0, 1, 0],
                                                      [0, 1, 1, 0]])
        for field in ("completed_users", "positive_users", "rating_sum", "rating_count"):
            with self.subTest(field=field):
                changed, _ = hand_inputs()
                changed.train_stats["items"]["4"][field] += 1
                with patch("recsys.research_model_cf.iter_training_rows", return_value=iter(self.rows)):
                    with self.assertRaises(ResearchError):
                        _training_matrix(changed)
        changed, _ = hand_inputs()
        changed.train_stats["items"]["1"]["positive_users"] += 1
        with patch("recsys.research_model_cf.iter_training_rows", return_value=iter(self.rows)):
            with self.assertRaises(ResearchError):
                _training_matrix(changed)

    def test_knn_exact_similarity_cold_seed_and_permutation(self):
        with patch("recsys.research_model_cf.iter_training_rows", return_value=iter(self.rows)):
            metadata = fit(self.inputs, dict(_KNN), {}, self.root)
        scorer = load(self.root, {"variant_id": "itemknn", "parameters": dict(_KNN),
                                  "metadata": metadata,
                                  "files": {name: {} for name in ("indptr.npy", "indices.npy",
                                                                   "weights.npy")}}, self.inputs)
        first = scorer.score((1,))
        self.assertEqual(first.supported_seeds, 1)
        self.assertEqual(first.values.tolist(), [0, 1 / 22, 1 / 22, 0])
        self.assertEqual(first.available.tolist(), [False, True, True, False])
        half = scorer.score((1, 4))
        self.assertEqual(half.supported_seeds, 1)
        np.testing.assert_array_equal(half.values, first.values / 2)
        self.assertEqual(scorer.score((4,)).supported_seeds, 0)
        self.assertFalse(scorer.score((4,)).available.any())
        np.testing.assert_array_equal(scorer.score((1, 2)).values,
                                      scorer.score((2, 1)).values)
        np.testing.assert_array_equal(scorer.score((3, 1, 2)).values,
                                      scorer.score((2, 3, 1)).values)
        self.assertEqual(_seed_columns((3, 1, 2), {1: 0, 2: 1, 3: 2}), (0, 1, 2))
        for seeds in ((1, 1), (True,), (99,), [1], tuple(range(11))):
            with self.subTest(seeds=seeds), self.assertRaises(ResearchError):
                scorer.score(seeds)

    def test_knn_top_200_exact_boundary_ties(self):
        ids = tuple(range(1, 204))
        inputs = SimpleNamespace(mal_ids=ids, train_stats={
            "train_users": 1, "train_completed_rows": 203,
            "items": {str(item): {"completed_users": 1, "positive_users": 1,
                                  "rating_sum": 8, "rating_count": 1, "band": "fixture"}
                      for item in ids}})
        rows = [("a", item, 8) for item in ids]
        with patch("recsys.research_model_cf.iter_training_rows", return_value=iter(rows)):
            metadata = fit(inputs, dict(_KNN), {}, self.root)
        scorer = load(self.root, {"variant_id": "itemknn", "parameters": dict(_KNN),
                                  "metadata": metadata,
                                  "files": {name: {} for name in ("indptr.npy", "indices.npy",
                                                                   "weights.npy")}}, inputs)
        one = scorer.score((1,))
        self.assertEqual(np.flatnonzero(one.available).tolist(), list(range(1, 201)))
        self.assertTrue(np.all(one.values[one.available] == 1 / 11))
        self.assertFalse(one.available[201:].any())

    def test_knn_three_unequal_seed_contributions_have_exact_order_invariance(self):
        ids = (1, 2, 3, 4)
        inputs = SimpleNamespace(mal_ids=ids, train_stats={
            "train_users": 1, "train_completed_rows": 4,
            "items": {str(item): {"completed_users": 1, "positive_users": 1,
                                  "rating_sum": 8, "rating_count": 1}
                      for item in ids}})
        np.save(self.root / "indptr.npy", np.array([0, 1, 2, 3, 3], dtype=np.int32))
        np.save(self.root / "indices.npy", np.array([3, 3, 3], dtype=np.int32))
        np.save(self.root / "weights.npy", np.array([.1, .2, .3], dtype=np.float64))
        metadata = {"kind": "itemknn", "source": "fit/train-interactions.tsv.gz",
                    "items": 4, "train_users": 1, "matrix_users": 1,
                    "train_completed_rows": 4, "positive_rows": 4,
                    "supported_items": 4, "edges": 3}
        scorer = load(self.root, {"variant_id": "itemknn", "parameters": dict(_KNN),
                                  "metadata": metadata,
                                  "files": {name: {} for name in ("indptr.npy", "indices.npy",
                                                                   "weights.npy")}}, inputs)
        first = scorer.score((1, 2, 3))
        permuted = scorer.score((2, 3, 1))
        np.testing.assert_array_equal(first.values, permuted.values)
        np.testing.assert_array_equal(first.available, permuted.available)
        self.assertEqual(first.supported_seeds, 3)
        self.assertEqual(first.available.tolist(), [False, False, False, True])

    def test_svd_exact_call_no_sigma_negative_scores_and_rank_zero(self):
        right = np.array([[1., 1e-16, -1., 7.], [0., 1., 0., 7.]])
        fake = (np.empty((3, 2)), np.array([9., 3.]), right)
        with patch("recsys.research_model_cf.iter_training_rows", return_value=iter(self.rows)), \
             patch("sklearn.utils.extmath.randomized_svd", return_value=fake) as svd:
            metadata = fit(self.inputs, dict(_SVD), {}, self.root)
        self.assertEqual(svd.call_args.kwargs, {
            "n_components": 2, "n_oversamples": 16, "n_iter": 4,
            "power_iteration_normalizer": "LU", "transpose": False,
            "flip_sign": True, "random_state": 3737, "svd_lapack_driver": "gesdd"})
        self.assertEqual(metadata["effective_rank"], 2)
        self.assertEqual(np.load(self.root / "factors.npy").tolist(),
                         [[1, 0], [1e-16, 1], [-1, 0], [0, 0]])
        scorer = load(self.root, {"variant_id": "puresvd", "parameters": dict(_SVD),
                                  "metadata": metadata, "files": {"factors.npy": {}}}, self.inputs)
        answer = scorer.score((1,))
        self.assertEqual(answer.values.tolist(), [1, 1e-16, -1, 0])
        self.assertEqual(answer.available.tolist(), [True, True, True, False])
        self.assertEqual(scorer.score((1, 4)).values.tolist(), [.5, 5e-17, -.5, 0])
        self.assertFalse(scorer.score((4,)).available.any())
        np.testing.assert_array_equal(scorer.score((3, 1, 2)).values,
                                      scorer.score((2, 3, 1)).values)
        np.testing.assert_array_equal(scorer.score((3, 1, 2)).available,
                                      scorer.score((2, 3, 1)).available)

        tiny = SimpleNamespace(mal_ids=(1, 2), train_stats={"train_users": 1,
            "train_completed_rows": 1, "items": {
                "1": {"completed_users": 1, "positive_users": 1, "rating_sum": 8, "rating_count": 1},
                "2": {"completed_users": 0, "positive_users": 0, "rating_sum": 0, "rating_count": 0}}})
        zero = self.root / "zero"
        zero.mkdir()
        with patch("recsys.research_model_cf.iter_training_rows", return_value=iter([("a", 1, 8)])), \
             patch("sklearn.utils.extmath.randomized_svd") as skipped:
            zero_metadata = fit(tiny, dict(_SVD), {}, zero)
        skipped.assert_not_called()
        self.assertEqual(zero_metadata["effective_rank"], 0)
        zero_scorer = load(zero, {"variant_id": "puresvd", "parameters": dict(_SVD),
                                  "metadata": zero_metadata, "files": {"factors.npy": {}}}, tiny)
        self.assertEqual(zero_scorer.score((1,)).supported_seeds, 0)
        self.assertFalse(zero_scorer.score((1,)).available.any())


class PublisherTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.fixture = ModelFixture(tmp.name)
        self.prepared, self.identity = self.fixture.make_prepared()

    def fit_variant(self, variant, name):
        model = self.fixture.root / name
        with patch("recsys.research_models.release_info", return_value=self.fixture.lock):
            fit_model(self.prepared, model, expected_prepared_identity=self.identity,
                      variant_id=variant, code_revision="a" * 40)
        return model

    def load_variant(self, model):
        with patch("recsys.research_models.release_info", return_value=self.fixture.lock):
            inputs = read_scoring_inputs(self.prepared, expected_prepared_identity=self.identity)
            return load_model(model, inputs=inputs)

    def test_fit_load_repeat_bytes_and_private_source_guard(self):
        original_open = Path.open
        def fit_open(path, *args, **kwargs):
            if path.name in {"splits.tsv.gz", "validation-labels.jsonl.gz", "test-labels.jsonl.gz"}:
                raise AssertionError("evaluator data opened by training")
            return original_open(path, *args, **kwargs)
        for variant in ("itemknn", "puresvd"):
            with self.subTest(variant=variant), patch.object(Path, "open", fit_open):
                first = self.fit_variant(variant, f"{variant}-a")
                second = self.fit_variant(variant, f"{variant}-b")
            for path in first.iterdir():
                self.assertEqual(path.read_bytes(), (second / path.name).read_bytes())
                self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(first.stat().st_mode & 0o777, 0o700)
            def score_open(path, *args, **kwargs):
                if path.name in {"train-interactions.tsv.gz", "splits.tsv.gz",
                                 "validation-labels.jsonl.gz", "test-labels.jsonl.gz"}:
                    raise AssertionError("private stream opened by scoring")
                return original_open(path, *args, **kwargs)
            with patch.object(Path, "open", score_open):
                _, scorer = self.load_variant(first)
                self.assertEqual(scorer.score((1, 2)).supported_seeds, 2)

    def test_completed_stream_corruption_fails_without_publishing(self):
        entry = self.prepared / "fit/train-interactions.tsv.gz"
        with entry.open("ab") as stream:
            stream.write(b"tail")
        with self.assertRaises(ResearchError):
            self.fit_variant("itemknn", "corrupt")
        self.assertFalse((self.fixture.root / "corrupt").exists())

    def test_both_models_publish_full_catalog_validation_scores(self):
        for variant in ("itemknn", "puresvd"):
            with self.subTest(variant=variant):
                model = self.fit_variant(variant, variant)
                prediction = self.fixture.predict(self.prepared, self.identity, model,
                                                  f"{variant}-prediction")
                manifest, _ = read_json(prediction / "prediction-manifest.json")
                self.assertEqual(manifest["model_id"], variant)
                rows = stream_jsonl(prediction / "scores.jsonl.gz", manifest["scores"])
                first = next(rows)
                self.assertEqual([item for item, _ in first["scores"]], self.fixture.items)
                self.assertGreater(sum(1 for _ in rows), 0)

    def test_semantic_tamper_with_rehashed_payload_fails(self):
        for variant, filename, change in (
                ("itemknn", "indices.npy", lambda a: np.full_like(a, 999)),
                ("puresvd", "factors.npy", lambda a: np.full_like(a, np.nan))):
            with self.subTest(variant=variant):
                model = self.fit_variant(variant, variant)
                path = model / filename
                np.save(path, change(np.load(path)), allow_pickle=False)
                manifest_path = model / "model-manifest.json"
                manifest, _ = read_json(manifest_path)
                raw = path.read_bytes()
                manifest["files"][filename]["bytes"] = len(raw)
                manifest["files"][filename]["sha256"] = sha(raw)
                manifest_path.write_bytes(canonical_bytes(manifest))
                with self.assertRaises(ResearchError):
                    self.load_variant(model)

    def test_rehashed_manifest_metadata_is_semantically_rejected(self):
        for variant in ("itemknn", "puresvd"):
            with self.subTest(variant=variant):
                model = self.fit_variant(variant, f"{variant}-metadata")
                path = model / "model-manifest.json"
                manifest, _ = read_json(path)
                manifest["metadata"]["matrix_users"] = 0
                path.write_bytes(canonical_bytes(manifest))
                with self.assertRaises(ResearchError):
                    self.load_variant(model)

    def test_import_remains_lazy(self):
        result = subprocess.run([sys.executable, "-I", "-c",
            "import sys, recsys.research_model_cf; "
            "assert 'scipy' not in sys.modules; "
            "assert 'sklearn' not in sys.modules; "
            "assert 'threadpoolctl' not in sys.modules"],
            capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
