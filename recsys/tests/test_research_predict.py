import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import IO, cast
from unittest.mock import patch

import numpy as np
from research_model_support import ModelFixture, patch_release_info

from recsys.research.artifacts import read_json, stream_jsonl
from recsys.research.evaluation.compare import compare_evaluations
from recsys.research.evaluation.evaluate import PredictionManifest, evaluate_predictions
from recsys.research.models.inputs import read_scoring_inputs
from recsys.research.models.types import ScoreVector
from recsys.research.predict import _encode_vector, _query, predict_validation
from recsys.research.protocol import ResearchError, load_protocol


class PredictionTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.fixture = ModelFixture(tmp.name)
        self.prepared, self.prepared_identity = self.fixture.make_prepared()
        self.model = self.fixture.fit_popularity(self.prepared, self.prepared_identity)

    def test_predict_evaluate_compare_and_repeat_bytes(self) -> None:
        source_open = Path.open

        def guarded(
            path: Path,
            mode: str = "r",
            buffering: int = -1,
            encoding: str | None = None,
            errors: str | None = None,
            newline: str | None = None,
        ) -> IO[str] | IO[bytes]:
            if path.name in {
                "train-interactions.tsv.gz",
                "validation-labels.jsonl.gz",
                "test-labels.jsonl.gz",
                "splits.tsv.gz",
            }:
                raise AssertionError("private stream opened")
            return source_open(path, mode, buffering, encoding, errors, newline)

        with patch.object(Path, "open", guarded):
            first = self.fixture.predict(self.prepared, self.prepared_identity, self.model, "first")
            second = self.fixture.predict(
                self.prepared, self.prepared_identity, self.model, "second"
            )
        self.assertEqual(
            (first / "prediction-manifest.json").read_bytes(),
            (second / "prediction-manifest.json").read_bytes(),
        )
        self.assertEqual(
            (first / "scores.jsonl.gz").read_bytes(), (second / "scores.jsonl.gz").read_bytes()
        )
        self.assertTrue(
            (first / "scores.jsonl.gz")
            .read_bytes()
            .startswith(bytes.fromhex("1f8b08000000000000ff"))
        )
        manifest_value, _ = read_json(first / "prediction-manifest.json")
        assert isinstance(manifest_value, dict)
        manifest = cast(
            "PredictionManifest", manifest_value
        )  # Published by predict_validation above.
        rows = list(stream_jsonl(first / "scores.jsonl.gz", manifest["scores"]))
        queries = list(
            stream_jsonl(
                self.prepared / "development/queries.jsonl.gz", manifest["query_descriptor"]
            )
        )
        with patch_release_info(self.fixture.lock):
            stats = read_scoring_inputs(
                self.prepared, expected_prepared_identity=self.prepared_identity
            ).train_stats
        self.assertEqual(len(rows), len(queries))
        assert all(isinstance(row, dict) for row in rows)
        assert all(isinstance(query, dict) for query in queries)
        typed_rows = cast("list[dict[str, object]]", rows)
        typed_queries = cast("list[dict[str, object]]", queries)
        self.assertEqual(
            [row["query_id"] for row in typed_rows],
            [query["query_id"] for query in typed_queries],
        )
        for query, row in zip(typed_queries, typed_rows):
            scores = row["scores"]
            assert isinstance(scores, list)
            self.assertEqual(len(scores), len(self.fixture.items))
            self.assertEqual(
                scores,
                [
                    [item, stats["items"][str(item)]["positive_users"]]
                    for item in self.fixture.items
                ],
            )
        self.assertEqual(first.stat().st_mode & 0o777, 0o700)
        self.assertEqual((first / "scores.jsonl.gz").stat().st_mode & 0o777, 0o600)
        evaluation = self.fixture.root / "evaluation"
        evaluate_predictions(
            self.prepared,
            first,
            evaluation,
            split="validation",
            expected_prepared_identity=self.prepared_identity,
        )
        compare_evaluations(
            [evaluation],
            self.fixture.root / "comparison",
            expected_prepared_identity=self.prepared_identity,
        )
        self.assertTrue((self.fixture.root / "comparison/comparison.json").exists())

    def test_query_and_score_vector_fail_closed(self) -> None:
        valid = {
            "query_id": "q1_" + "a" * 64,
            "cohort": "main",
            "seed_policy": "liked",
            "order": "random-liked-1",
            "n": 1,
            "seed_mal_ids": [1],
        }
        for override in (
            {"n": True},
            {"seed_mal_ids": [True]},
            {"n": 2, "seed_mal_ids": [1, 1]},
            {"seed_policy": "popularity"},
            {"query_id": "invalid"},
            {"extra": 1},
        ):
            with self.subTest(override=override):
                with self.assertRaises(ResearchError):
                    _query({**valid, **override}, {1, 2}, set(), load_protocol())
        ids = (1, 2)
        good = ScoreVector(
            np.array([0.5, 0.0], dtype=np.float64), np.array([True, False], dtype=np.bool_), 1
        )
        self.assertEqual(_encode_vector(good, 1, ids), [[1, 500000000], [2, None]])
        for bad in (
            ScoreVector(np.array([np.nan, 0.0]), good.available, 1),
            ScoreVector(np.array([1, 0], dtype=np.int64), good.available, 1),
            ScoreVector(good.values, np.array([1, 0], dtype=np.int64), 1),
            ScoreVector(good.values, good.available, 2),
            ScoreVector(good.values, good.available, True),
        ):
            with self.subTest(bad=bad):
                with self.assertRaises(ResearchError):
                    _encode_vector(bad, 1, ids)

    def test_bad_model_and_no_replace_do_not_publish(self) -> None:
        output = self.fixture.root / "prediction"
        manifest_path = self.model / "model-manifest.json"
        original = manifest_path.read_bytes()
        manifest_path.write_bytes(original + b" ")
        with self.assertRaises(ResearchError):
            self.fixture.predict(self.prepared, self.prepared_identity, self.model)
        self.assertFalse(output.exists())
        manifest_path.write_bytes(original)
        self.fixture.predict(self.prepared, self.prepared_identity, self.model)
        with self.assertRaises(ResearchError):
            self.fixture.predict(self.prepared, self.prepared_identity, self.model)

    def test_invalid_backend_vector_cleans_partial_stage(self) -> None:
        scorer = SimpleNamespace(
            score=lambda seeds: ScoreVector(
                np.full(len(self.fixture.items), np.nan, dtype=np.float64),
                np.ones(len(self.fixture.items), dtype=np.bool_),
                len(seeds),
            )
        )
        model = {"variant_id": "content-mean", "producer": {"code_revision": "a" * 40}}
        output = self.fixture.root / "partial"
        with (
            patch_release_info(self.fixture.lock),
            patch("recsys.research.predict.load_model", return_value=(model, scorer)),
        ):
            with self.assertRaises(ResearchError):
                predict_validation(
                    self.prepared,
                    self.model,
                    output,
                    expected_prepared_identity=self.prepared_identity,
                )
        self.assertFalse(output.exists())
        self.assertFalse(list(self.fixture.root.glob(".partial.*")))
