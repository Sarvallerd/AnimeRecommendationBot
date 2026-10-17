"""Bounded one-query policy without loading a native framework."""

from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np
from test_training_artifacts import _inputs

from recsys.research.models.types import ScoreVector, ScoringInputs, TrainItem
from recsys.research.protocol import ResearchError
from recsys.research.training.predict import predict_one, validate_seed_query


class TinyScorer:
    def supported_seed_ids(self, seed_mal_ids: tuple[int, ...]) -> frozenset[int]:
        return frozenset(seed for seed in seed_mal_ids if seed in (1, 2))

    def score(self, seed_mal_ids: tuple[int, ...]) -> ScoreVector:
        return ScoreVector(
            np.array([0.0, 0.7, 0.7, -1.0], dtype=np.float64),
            np.array([True, True, True, False], dtype=np.bool_),
            len(seed_mal_ids),
        )

    def score_many(self, seeds: tuple[tuple[int, ...], ...]) -> tuple[ScoreVector, ...]:
        return tuple(self.score(seed) for seed in seeds)


def _scoring_inputs() -> ScoringInputs:
    baseline = _inputs()
    items: dict[str, TrainItem] = {
        str(item): {
            "completed_users": 1 if item != 4 else 0,
            "positive_users": 2 if item in (1, 2) else 1 if item == 3 else 0,
            "rating_sum": 9 if item != 4 else 0,
            "rating_count": 1 if item != 4 else 0,
            "band": "head" if item == 1 else "tail" if item in (2, 3) else "cold",
        }
        for item in (1, 2, 3, 4)
    }
    return replace(
        baseline,
        mal_ids=(1, 2, 3, 4),
        train_stats={
            "schema_version": 1,
            "train_users": 2,
            "train_completed_rows": 3,
            "items": items,
        },
    )


class TrainingPredictTests(unittest.TestCase):
    def setUp(self) -> None:
        self.inputs = _scoring_inputs()
        self.scorer = TinyScorer()

    def test_empty_query_uses_train_positive_popularity(self) -> None:
        doc = predict_one(
            self.inputs,
            self.scorer,
            model_identity="sha256:" + "a" * 64,
            seed_mal_ids=(),
            limit=2,
        )
        self.assertEqual(doc["source"], "train-popularity")
        self.assertEqual([row["mal_id"] for row in doc["recommendations"]], [1, 2])

    def test_mixed_seed_skips_unsupported_and_masks_seed_and_cold(self) -> None:
        doc = predict_one(
            self.inputs,
            self.scorer,
            model_identity="sha256:" + "a" * 64,
            seed_mal_ids=(4, 1),
            limit=3,
        )
        self.assertEqual(doc["seed_mal_ids"], [1, 4])
        self.assertEqual(doc["supported_seed_mal_ids"], [1])
        self.assertEqual(doc["skipped_seed_mal_ids"], [4])
        self.assertEqual([row["mal_id"] for row in doc["recommendations"]], [2, 3])
        self.assertEqual(doc["source"], "seed-embedding")

    def test_all_unsupported_abstains(self) -> None:
        doc = predict_one(
            self.inputs,
            self.scorer,
            model_identity="sha256:" + "a" * 64,
            seed_mal_ids=(4,),
            limit=5,
        )
        self.assertEqual(
            (doc["status"], doc["source"], doc["recommendations"]),
            (
                "abstained",
                "none",
                [],
            ),
        )

    def test_rejects_unknown_duplicates_bools_and_bad_limit(self) -> None:
        for seeds, limit in (((5,), 5), ((1, 1), 5), ((True,), 5), ((1,), 0), ((1,), 21)):
            with self.subTest(seeds=seeds, limit=limit):
                with self.assertRaises(ResearchError):
                    validate_seed_query(seeds, mal_ids=self.inputs.mal_ids, limit=limit)

    def test_query_accepts_ten_distinct_ids_and_rejects_eleven(self) -> None:
        known = tuple(range(1, 13))
        seeds, limit = validate_seed_query(tuple(range(10, 0, -1)), mal_ids=known, limit=20)
        self.assertEqual((seeds, limit), (tuple(range(1, 11)), 20))
        with self.assertRaisesRegex(ResearchError, "training_query_invalid"):
            validate_seed_query(tuple(range(1, 12)), mal_ids=known, limit=20)


if __name__ == "__main__":
    unittest.main()
