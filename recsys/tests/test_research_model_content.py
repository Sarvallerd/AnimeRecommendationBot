"""Content model semantics, locked inputs, and private validation workflow."""

import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from recsys.research_artifacts import read_json, write_json
from recsys.research_evaluate import evaluate_predictions
from recsys.research_model_content import ContentScorer, fit
from recsys.research_models import fit_model, load_model, read_scoring_inputs
from research_model_support import ModelFixture


def _vectors(rows):
    result = np.zeros((len(rows), 300), dtype=np.float64)
    result[:, :2] = rows
    result.setflags(write=False)
    return result


def _scorer(rows, variant):
    vectors = _vectors(rows)
    return ContentScorer(tuple(range(1, len(rows) + 1)), vectors,
                         np.any(vectors != 0, axis=1), variant)


def _glove(path, keep=None):
    words = {"a": (1, 0), "story": (1, 0), "title": (0, 1),
             "drama": (1, 0), "action": (0, 1)}
    if keep is not None:
        words = {word: pair for word, pair in words.items() if word in keep}
    with path.open("wb") as stream:
        for word, pair in words.items():
            components = [*pair, *([0] * 298)]
            stream.write((word + " " + " ".join(map(str, components)) + "\n").encode())
    raw = path.read_bytes()
    return {"text_size": len(raw), "text_sha256": hashlib.sha256(raw).hexdigest()}


class ContentMathTests(unittest.TestCase):
    def test_mean_max_raw_signed_cosines_and_oov(self):
        rows = [(1, 0), (0, 1), (.6, .8), (1, 0), (-.8, .6), (-1, 0), (0, 0)]
        mean = _scorer(rows, "content-mean").score((2, 1))
        maximum = _scorer(rows, "content-max").score((1, 2))
        np.testing.assert_allclose(mean.values[2:6], [.7, .5, -.1, -.5], atol=1e-14)
        np.testing.assert_allclose(maximum.values[2:6], [.8, 1, .6, 0], atol=1e-14)
        self.assertEqual(mean.available[2:7].tolist(), [True, True, False, False, False])
        self.assertEqual(maximum.available[2:7].tolist(), [True, True, True, False, False])
        self.assertEqual(mean.supported_seeds, 2)
        with_oov = _scorer(rows, "content-mean").score((7, 1))
        np.testing.assert_array_equal(with_oov.values, _scorer(rows, "content-mean").score((1,)).values)
        self.assertEqual(with_oov.supported_seeds, 1)
        all_oov = _scorer(rows, "content-mean").score((7,))
        self.assertEqual(all_oov.supported_seeds, 0)
        self.assertFalse(all_oov.available.any())
        self.assertFalse(all_oov.values.any())
        cancellation = _scorer(rows, "content-mean").score((1, 6))
        self.assertFalse(cancellation.available.any())

    def test_union_removes_visible_before_ranks_and_sums_overlap(self):
        scorer = _scorer([(1, 0)] * 5, "content-union")
        scorer._source = lambda source: np.asarray((3, 2) if source == 0 else (4, 2), dtype=np.int32)
        answer = scorer.score((2, 1))
        self.assertAlmostEqual(answer.values[2], 1 / 62)
        self.assertAlmostEqual(answer.values[3], 1 / 122)
        self.assertAlmostEqual(answer.values[4], 1 / 122)
        self.assertFalse(answer.available[:2].any())
        self.assertEqual(answer.supported_seeds, 2)

    def test_union_top110_handles_ten_visible_and_stable_boundaries(self):
        scorer = _scorer([(1, 0)] * 121, "content-union")
        seeds = tuple(range(1, 11))
        original = scorer.score(seeds)
        self.assertEqual(np.flatnonzero(original.available).tolist(), list(range(10, 110)))
        self.assertFalse(original.available[110])
        repeat = scorer.score(tuple(reversed(seeds)))
        np.testing.assert_array_equal(original.values, repeat.values)
        scorer.cache.entries.clear()
        scorer.cache.bytes = 0
        with patch("recsys.research_model_content.CACHE_BYTES", 440):
            np.testing.assert_array_equal(original.values, scorer.score(seeds).values)
            self.assertLessEqual(scorer.cache.bytes, 440)
            np.testing.assert_array_equal(original.values, scorer.score(tuple(reversed(seeds))).values)

    def test_seed_contract(self):
        scorer = _scorer([(1, 0)], "content-max")
        for seeds in ([1], (True,), (1, 1), (2,), tuple(range(11))):
            with self.subTest(seeds=seeds), self.assertRaisesRegex(ValueError, "research_model_invalid"):
                scorer.score(seeds)


class ContentFitTests(unittest.TestCase):
    def test_locked_glove_repeat_fit_load_prediction_and_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = ModelFixture(directory)
            glove = Path(directory) / "glove.txt"
            fixture.lock["inputs"]["embeddings"] = _glove(glove)
            prepared, prepared_identity = fixture.make_prepared()
            assets = {"glove": glove}
            with patch("recsys.research_models.release_info", return_value=fixture.lock):
                first = Path(directory) / "mean-a"
                second = Path(directory) / "mean-b"
                fit_model(prepared, first, expected_prepared_identity=prepared_identity,
                          variant_id="content-mean", code_revision="a" * 40, assets=assets)
                fit_model(prepared, second, expected_prepared_identity=prepared_identity,
                          variant_id="content-mean", code_revision="a" * 40, assets=assets)
                inputs = read_scoring_inputs(prepared, expected_prepared_identity=prepared_identity)
                manifest, scorer = load_model(first, inputs=inputs)
            for variant in ("content-max", "content-union"):
                sibling = Path(directory) / variant
                with patch("recsys.research_models.release_info", return_value=fixture.lock):
                    fit_model(prepared, sibling, expected_prepared_identity=prepared_identity,
                              variant_id=variant, code_revision="a" * 40, assets=assets)
                    variant_manifest, variant_scorer = load_model(sibling, inputs=inputs)
                self.assertEqual(variant_manifest["variant_id"], variant)
                self.assertEqual(variant_scorer.score((1,)).supported_seeds, 1)
                sibling_prediction = fixture.predict(prepared, prepared_identity, sibling,
                                                     name=variant + "-prediction")
                sibling_pred_manifest, _ = read_json(sibling_prediction / "prediction-manifest.json")
                self.assertEqual(sibling_pred_manifest["model_id"], variant)
            for name in ("vectors.npy", "ready.npy", "model-manifest.json"):
                self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())
                self.assertEqual((first / name).stat().st_mode & 0o777, 0o600)
            self.assertEqual(manifest["metadata"]["ready_items"], len(fixture.items))
            self.assertEqual(scorer.score((1,)).supported_seeds, 1)
            prediction = fixture.predict(prepared, prepared_identity, first)
            output = Path(directory) / "validation"
            evaluate_predictions(prepared, prediction, output, split="validation",
                                 expected_prepared_identity=prepared_identity)
            report, _ = read_json(output / "evaluation-manifest.json")
            self.assertEqual(report["model_id"], "content-mean")

    def test_real_oov_seed_abstains_and_fit_does_not_open_private_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = ModelFixture(directory)
            glove = Path(directory) / "glove.txt"
            fixture.lock["inputs"]["embeddings"] = _glove(glove, keep={"drama"})
            prepared, prepared_identity = fixture.make_prepared()
            model = Path(directory) / "model"
            original_open = Path.open
            protected = {"train-interactions.tsv.gz", "validation-labels.jsonl.gz",
                         "test-labels.jsonl.gz", "splits.tsv.gz", "queries.jsonl.gz"}

            def guard(path, *args, **kwargs):
                if path.name in protected:
                    raise AssertionError("private or query data opened during content fit")
                return original_open(path, *args, **kwargs)

            with patch("recsys.research_models.release_info", return_value=fixture.lock), \
                 patch.object(Path, "open", guard):
                fit_model(prepared, model, expected_prepared_identity=prepared_identity,
                          variant_id="content-mean", code_revision="a" * 40,
                          assets={"glove": glove})
            with patch("recsys.research_models.release_info", return_value=fixture.lock):
                inputs = read_scoring_inputs(prepared, expected_prepared_identity=prepared_identity)
                manifest, scorer = load_model(model, inputs=inputs)
            self.assertEqual(manifest["metadata"]["ready_items"], 11)
            answer = scorer.score((2,))
            self.assertEqual(answer.supported_seeds, 0)
            self.assertFalse(answer.available.any())
            self.assertFalse(answer.values.any())

    def test_wrong_glove_and_strict_parameters_fail_without_publish(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = ModelFixture(directory)
            glove = Path(directory) / "glove.txt"
            fixture.lock["inputs"]["embeddings"] = _glove(glove)
            prepared, prepared_identity = fixture.make_prepared()
            with patch("recsys.research_models.release_info", return_value=fixture.lock):
                inputs = read_scoring_inputs(prepared, expected_prepared_identity=prepared_identity)
                stage = Path(directory) / "stage"
                stage.mkdir()
                for invalid in ({"aggregation": "mean", "dimension": True, "positive_only": True},
                                {"aggregation": "mean", "dimension": 300, "positive_only": 1}):
                    with self.assertRaisesRegex(ValueError, "research_model_invalid"):
                        fit(inputs, invalid, {"glove": glove}, stage)
                glove.write_bytes(glove.read_bytes() + b"corrupt\n")
                target = Path(directory) / "bad-model"
                with self.assertRaisesRegex(ValueError, "research_model_invalid"):
                    fit_model(prepared, target, expected_prepared_identity=prepared_identity,
                              variant_id="content-max", code_revision="a" * 40,
                              assets={"glove": glove})
                self.assertFalse(target.exists())

    def test_payload_semantics_checked_even_after_manifest_descriptor_repair(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = ModelFixture(directory)
            glove = Path(directory) / "glove.txt"
            fixture.lock["inputs"]["embeddings"] = _glove(glove)
            prepared, prepared_identity = fixture.make_prepared()
            model = Path(directory) / "model"
            with patch("recsys.research_models.release_info", return_value=fixture.lock):
                fit_model(prepared, model, expected_prepared_identity=prepared_identity,
                          variant_id="content-union", code_revision="a" * 40,
                          assets={"glove": glove})
                inputs = read_scoring_inputs(prepared, expected_prepared_identity=prepared_identity)
                manifest, _ = load_model(model, inputs=inputs)
                vectors = np.load(model / "vectors.npy", allow_pickle=False)
                vectors[0, :] = 0
                with (model / "vectors.npy").open("wb") as stream:
                    np.save(stream, vectors, allow_pickle=False)
                raw = (model / "vectors.npy").read_bytes()
                manifest["files"]["vectors.npy"]["bytes"] = len(raw)
                manifest["files"]["vectors.npy"]["sha256"] = hashlib.sha256(raw).hexdigest()
                (model / "model-manifest.json").unlink()
                write_json(model / "model-manifest.json", manifest)
                with self.assertRaisesRegex(ValueError, "research_model_invalid"):
                    load_model(model, inputs=inputs)


if __name__ == "__main__":
    unittest.main()
