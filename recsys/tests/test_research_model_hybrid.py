"""Fixed raw-score RRF and portable embedded component checks."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from research_model_support import ModelFixture, patch_release_info
from test_research_model_content import _glove

from recsys.research.models.hybrid import _fusion
from recsys.research.models.package import fit_model, load_model, read_scoring_inputs
from recsys.research.models.types import ScoreVector
from recsys.research.protocol import ResearchError


class _Fake:
    def __init__(self, values, available, supported):
        self.vector = ScoreVector(
            np.asarray(values, dtype=np.float64), np.asarray(available, dtype=np.bool_), 1
        )
        self.supported = supported

    def score_many(self, seeds):
        return tuple(self.vector for _ in seeds)

    def supported_seed_ids(self, seeds):
        return frozenset(self.supported & set(seeds))


class FusionTests(unittest.TestCase):
    def test_raw_ranks_negative_svd_visible_exclusion_and_support_union(self):
        ids = (1, 2, 3, 4, 5)
        # Values 1.01e-10 and 1.02e-10 both publish as integer zero; raw order differs.
        content = _Fake([1, 1.01e-10, 1.02e-10, 0, 0], [True, True, True, False, False], {1})
        svd = _Fake([10, -3, -2, -1, 0], [True, True, True, True, False], {1, 4})
        row = _fusion(((1, 4),), ids, content, svd)[0]
        self.assertEqual(row.supported_seeds, 2)
        self.assertEqual(row.available.tolist(), [False, True, True, False, False])
        self.assertAlmostEqual(row.values[2], 1 / 61)
        self.assertAlmostEqual(row.values[1], 1 / 62)
        negative = _fusion(((1,),), ids, content, svd)[0]
        self.assertTrue(negative.available[3])
        self.assertAlmostEqual(negative.values[3], 0.5 / 61)

    def test_empty_and_invalid_seed_contract(self):
        scorer = _Fake([0, 0], [False, False], set())
        self.assertFalse(_fusion(((),), (1, 2), scorer, scorer)[0].available.any())
        for seeds in ((1, 1), (True,), (3,), [1]):
            with self.subTest(seeds=seeds), self.assertRaises(ResearchError):
                _fusion((seeds,), (1, 2), scorer, scorer)


class PortableTests(unittest.TestCase):
    def test_fit_embed_remove_origins_then_score_and_tamper(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ModelFixture(temporary)
            glove = Path(temporary) / "glove.txt"
            fixture.lock["inputs"]["embeddings"] = _glove(glove)
            prepared, identity = fixture.make_prepared()
            base = Path(temporary)
            paths = {"content_model": base / "content-model", "puresvd_model": base / "svd-model"}
            with patch_release_info(fixture.lock):
                for variant, path, assets in (
                    ("content-mean", paths["content_model"], {"glove": glove}),
                    ("puresvd", paths["puresvd_model"], None),
                ):
                    fit_model(
                        prepared,
                        path,
                        expected_prepared_identity=identity,
                        variant_id=variant,
                        code_revision="a" * 40,
                        assets=assets,
                    )
                hybrid = base / "hybrid-model"
                fit_model(
                    prepared,
                    hybrid,
                    expected_prepared_identity=identity,
                    variant_id="hybrid",
                    code_revision="a" * 40,
                    assets=paths,
                )
                inputs = read_scoring_inputs(prepared, expected_prepared_identity=identity)
                from recsys.research.predict import gpu_preflight, predict_validation

                with (
                    patch(
                        "recsys.research.gpu.probe_gpu",
                        side_effect=AssertionError("probe forbidden"),
                    ),
                    patch(
                        "recsys.research.gpu.gpu_execution",
                        side_effect=AssertionError("session forbidden"),
                    ),
                ):
                    for action in (
                        lambda: fit_model(
                            prepared,
                            base / "hybrid-gpu",
                            expected_prepared_identity=identity,
                            variant_id="hybrid",
                            code_revision="a" * 40,
                            assets=paths,
                            backend="gpu",
                        ),
                        lambda: load_model(hybrid, inputs=inputs, backend="gpu"),
                        lambda: predict_validation(
                            prepared,
                            hybrid,
                            base / "hybrid-gpu-pred",
                            expected_prepared_identity=identity,
                            backend="gpu",
                        ),
                        lambda: gpu_preflight(
                            prepared,
                            hybrid,
                            base / "hybrid-gpu-preflight",
                            expected_prepared_identity=identity,
                        ),
                    ):
                        with self.assertRaisesRegex(ResearchError, "research_backend_unsupported"):
                            action()
                self.assertFalse((base / "hybrid-gpu").exists())
                self.assertFalse((base / "hybrid-gpu-pred").exists())
                self.assertFalse((base / "hybrid-gpu-preflight").exists())
                original = load_model(hybrid, inputs=inputs)[1].score((1, 2))
                for source in paths.values():
                    for entry in source.iterdir():
                        entry.unlink()
                    source.rmdir()
                loaded = load_model(hybrid, inputs=inputs)[1].score((1, 2))
                np.testing.assert_array_equal(original.values, loaded.values)
                np.testing.assert_array_equal(original.available, loaded.available)
                self.assertEqual(original.supported_seeds, loaded.supported_seeds)
                with (hybrid / "content_vectors.npy").open("ab") as stream:
                    stream.write(b"x")
                with self.assertRaises(ResearchError):
                    load_model(hybrid, inputs=inputs)


@unittest.skipUnless(os.getenv("ARB_RESEARCH_GPU_TEST") == "1", "hardware opt-in")
class HardwareTests(unittest.TestCase):
    def test_four_gpu_components_and_cpu_hybrid_preserve_raw_ranks(self):
        from recsys.research.gpu import probe_gpu
        from recsys.research.predict import gpu_preflight, predict_validation

        with tempfile.TemporaryDirectory() as temporary:
            fixture = ModelFixture(temporary)
            glove = Path(temporary) / "glove.txt"
            fixture.lock["inputs"]["embeddings"] = _glove(glove)
            prepared, identity = fixture.make_prepared()
            base = Path(temporary)
            parts = {"content_model": base / "content-cpu", "puresvd_model": base / "svd-cpu"}
            evidence = base / "evidence"
            evidence.mkdir(mode=0o700)
            with patch_release_info(fixture.lock):
                for variant in ("content-mean", "content-max", "content-union", "puresvd"):
                    fit_model(
                        prepared,
                        base / (variant + "-gpu"),
                        expected_prepared_identity=identity,
                        variant_id=variant,
                        code_revision="a" * 40,
                        assets={"glove": glove} if variant.startswith("content-") else None,
                        backend="gpu",
                    )
                fit_model(
                    prepared,
                    parts["content_model"],
                    expected_prepared_identity=identity,
                    variant_id="content-mean",
                    code_revision="a" * 40,
                    assets={"glove": glove},
                    backend="cpu",
                )
                fit_model(
                    prepared,
                    parts["puresvd_model"],
                    expected_prepared_identity=identity,
                    variant_id="puresvd",
                    code_revision="a" * 40,
                    backend="cpu",
                )
                hybrid = base / "hybrid"
                fit_model(
                    prepared,
                    hybrid,
                    expected_prepared_identity=identity,
                    variant_id="hybrid",
                    code_revision="a" * 40,
                    assets=parts,
                    backend="cpu",
                )
                inputs = read_scoring_inputs(prepared, expected_prepared_identity=identity)
                for source in parts.values():
                    for entry in source.iterdir():
                        entry.unlink()
                    source.rmdir()
                manifest, scorer = load_model(hybrid, inputs=inputs, backend="cpu")
                self.assertEqual(scorer.score((1,)).supported_seeds, 1)
                scorer.close()
                measured = {}
                predict_validation(
                    prepared,
                    hybrid,
                    base / "predictions",
                    expected_prepared_identity=identity,
                    backend="cpu",
                    measurements=measured,
                    measurement_output=evidence / "prediction.json",
                )
                self.assertEqual(measured["status"], "published")
                with (
                    patch(
                        "recsys.research.gpu.probe_gpu",
                        side_effect=AssertionError("probe forbidden"),
                    ),
                    patch(
                        "recsys.research.gpu.gpu_execution",
                        side_effect=AssertionError("session forbidden"),
                    ),
                ):
                    with self.assertRaisesRegex(ResearchError, "research_backend_unsupported"):
                        fit_model(
                            prepared,
                            base / "hybrid-gpu",
                            expected_prepared_identity=identity,
                            variant_id="hybrid",
                            code_revision="a" * 40,
                            assets=parts,
                            backend="gpu",
                        )
                    with self.assertRaisesRegex(ResearchError, "research_backend_unsupported"):
                        load_model(hybrid, inputs=inputs, backend="gpu")
                    with self.assertRaisesRegex(ResearchError, "research_backend_unsupported"):
                        predict_validation(
                            prepared,
                            hybrid,
                            base / "hybrid-gpu-pred",
                            expected_prepared_identity=identity,
                            backend="gpu",
                        )
                    with self.assertRaisesRegex(ResearchError, "research_backend_unsupported"):
                        gpu_preflight(
                            prepared,
                            hybrid,
                            base / "preflight",
                            expected_prepared_identity=identity,
                        )
                self.assertEqual(probe_gpu()["backend"], "cupy")
