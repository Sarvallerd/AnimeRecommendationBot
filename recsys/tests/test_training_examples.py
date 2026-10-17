"""Train-only scan and deterministic per-user sampling contracts."""

import copy
import tempfile
import unittest
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from research_model_support import patch_release_info
from training_support import SeedTrainingFixture

from recsys.contracts.bundle import Catalog
from recsys.research.artifacts import FileDescriptor, StreamDescriptor
from recsys.research.models.inputs import iter_training_rows, read_training_inputs
from recsys.research.models.types import TrainingInputs, TrainItem
from recsys.research.protocol import ResearchError
from recsys.research.training.examples import (
    _user_examples,
    iter_training_examples,
    validate_training_rows,
)


def _synthetic_inputs(rows: list[tuple[str, int, int]], item_count: int) -> TrainingInputs:
    items: dict[str, TrainItem] = {
        str(item): {
            "completed_users": 0,
            "positive_users": 0,
            "rating_sum": 0,
            "rating_count": 0,
            "band": "cold",
        }
        for item in range(1, item_count + 1)
    }
    for _, item, rating in rows:
        stats = items[str(item)]
        stats["completed_users"] += 1
        stats["positive_users"] += rating >= 8
        stats["rating_sum"] += rating
        stats["rating_count"] += 1
    descriptor: FileDescriptor = {"bytes": 1, "sha256": "0" * 64}
    stream: StreamDescriptor = {
        "bytes": 1,
        "sha256": "0" * 64,
        "rows": len(rows) + 1,
        "uncompressed_bytes": 1,
        "canonical_sha256": "0" * 64,
    }
    catalog: Catalog = {"schema_version": 1, "anime": {}}
    return TrainingInputs(
        prepared_identity="sha256:" + "a" * 64,
        protocol_identity="sha256:" + "b" * 64,
        catalog_descriptor=descriptor,
        fit_identity="sha256:" + "c" * 64,
        train_interactions_descriptor=stream,
        train_stats_descriptor=descriptor,
        mal_ids=tuple(range(1, item_count + 1)),
        catalog=catalog,
        train_stats={
            "schema_version": 1,
            "train_users": 2,
            "train_completed_rows": len(rows),
            "items": items,
        },
        train_interactions_path=Path("/synthetic/not-opened"),
    )


class ExampleTests(unittest.TestCase):
    def test_fixture_scan_and_nine_examples(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SeedTrainingFixture(temporary)
            prepared, identity = fixture.make_prepared()
            with patch_release_info(fixture.lock):
                inputs = read_training_inputs(prepared, expected_prepared_identity=identity)
                scan = validate_training_rows(inputs)
                self.assertEqual((scan.completed_rows, scan.observed_users), (18, 5))
                self.assertEqual(scan.warm_mal_ids, tuple(range(1, 9)))
                self.assertEqual(scan.positive_rows, 9)
                epochs = [
                    list(iter_training_examples(inputs, scan=scan, epoch=i)) for i in range(3)
                ]
                self.assertEqual([len(rows) for rows in epochs], [3, 3, 3])
                self.assertEqual(
                    epochs[0], list(iter_training_examples(inputs, scan=scan, epoch=0))
                )
                completed = [{1, 2, 3}, {1, 3, 4}, {2, 5, 6}, {7}, set(range(1, 9))]
                for rows in epochs:
                    for index, row in enumerate(rows):
                        self.assertEqual(len(row.seed_mal_ids), 1)
                        self.assertNotIn(row.positive_mal_id, row.seed_mal_ids)
                        self.assertTrue(row.negative_mal_ids)
                        self.assertEqual(len(row.negative_mal_ids), len(set(row.negative_mal_ids)))
                        self.assertFalse(set(row.negative_mal_ids) & completed[index])

    def test_scan_rejects_stats_mismatch_and_late_stream_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SeedTrainingFixture(temporary)
            prepared, identity = fixture.make_prepared()
            with patch_release_info(fixture.lock):
                inputs = read_training_inputs(prepared, expected_prepared_identity=identity)
                stats = copy.deepcopy(inputs.train_stats)
                stats["items"]["1"]["rating_sum"] += 1
                with self.assertRaisesRegex(ResearchError, "training_data_mismatch"):
                    validate_training_rows(replace(inputs, train_stats=stats))
                rows = list(iter_training_rows(inputs))

                def broken_rows(_inputs: object) -> Iterator[tuple[str, int, int]]:
                    yield from rows
                    raise ResearchError("research_integrity_error")

                with patch(
                    "recsys.research.training.examples.iter_training_rows", side_effect=broken_rows
                ):
                    with self.assertRaises(ResearchError):
                        validate_training_rows(inputs)

    def test_rejection_fallback_uses_same_rng_and_excludes_low_ratings(self) -> None:
        class ControlledRandom:
            def __init__(self, seed: int) -> None:
                self.seed = seed
                self.choices = 0
                self.samples: list[tuple[list[int], int]] = []

            def choice(self, values: list[int] | tuple[int, ...]) -> int:
                self.choices += 1
                return values[0] if self.choices == 1 else 3

            def randint(self, low: int, high: int) -> int:
                return low

            def sample(self, values: list[int], amount: int) -> list[int]:
                self.samples.append((values, amount))
                return values[:amount]

        instance = ControlledRandom(0)
        with patch("recsys.research.training.examples.random.Random", return_value=instance):
            examples = list(
                _user_examples("u1_" + "a" * 64, [(1, 10), (2, 8), (3, 2)], tuple(range(1, 9)), 0)
            )
        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0].negative_mal_ids, (4, 5, 6, 7))
        self.assertEqual(instance.choices, 129)
        self.assertEqual(instance.samples[-1], ([4, 5, 6, 7, 8], 4))

    def test_seed_cap_and_one_available_negative(self) -> None:
        class MaxRandom:
            def __init__(self, seed: int) -> None:
                self.seed = seed
                self.high = 0

            def choice(self, values: list[int] | tuple[int, ...]) -> int:
                return values[0]

            def randint(self, low: int, high: int) -> int:
                self.high = high
                return high

            def sample(self, values: list[int], amount: int) -> list[int]:
                return values[-amount:]

        instance = MaxRandom(0)
        profile = [(item, 10) for item in range(1, 13)]
        with patch("recsys.research.training.examples.random.Random", return_value=instance):
            examples = list(_user_examples("u1_" + "a" * 64, profile, tuple(range(1, 14)), 0))
        self.assertEqual(instance.high, 10)
        self.assertEqual(len(examples), 1)
        self.assertEqual(len(examples[0].seed_mal_ids), 10)
        self.assertEqual(examples[0].negative_mal_ids, (13,))

    def test_stream_sampler_varies_n_and_caps_at_ten(self) -> None:
        class SelectedNRandom:
            def __init__(self, selected_n: int) -> None:
                self.selected_n = selected_n
                self.choice_count = 0

            def choice(self, values: list[int] | tuple[int, ...]) -> int:
                self.choice_count += 1
                return values[0]

            def randint(self, low: int, high: int) -> int:
                self_case.assertEqual(low, 1)
                self_case.assertEqual(high, min(10, positive_count - 1))
                return self.selected_n

            def sample(self, values: list[int], amount: int) -> list[int]:
                return values[:amount]

        self_case = self
        for positive_count, selected_n in ((3, 1), (3, 2), (12, 10)):
            with self.subTest(positives=positive_count, n=selected_n):
                first_user = "u1_" + "a" * 64
                second_user = "u1_" + "b" * 64
                rows = [(first_user, item, 10) for item in range(1, positive_count + 1)] + [
                    (second_user, positive_count + 1, 8)
                ]
                inputs = _synthetic_inputs(rows, positive_count + 1)
                rng = SelectedNRandom(selected_n)
                with (
                    patch(
                        "recsys.research.training.examples.iter_training_rows",
                        side_effect=lambda _inputs: iter(rows),
                    ),
                    patch("recsys.research.training.examples.random.Random", return_value=rng),
                ):
                    scan = validate_training_rows(inputs)
                    examples = list(iter_training_examples(inputs, scan=scan, epoch=0))
                self.assertEqual(scan.observed_users, 2)
                self.assertEqual(scan.positive_rows, positive_count + 1)
                self.assertEqual(len(examples), 1)
                self.assertEqual(examples[0].positive_mal_id, 1)
                self.assertEqual(examples[0].seed_mal_ids, tuple(range(2, selected_n + 2)))
                self.assertEqual(examples[0].negative_mal_ids, (positive_count + 1,))


if __name__ == "__main__":
    unittest.main()
