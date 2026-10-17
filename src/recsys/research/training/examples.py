"""Train-only stream validation and bounded, deterministic BPR examples."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Iterator

from recsys.contracts.bundle import canonical_bytes
from recsys.research.models.inputs import iter_training_rows
from recsys.research.models.types import TrainingInputs
from recsys.research.protocol import ResearchError
from recsys.research.training.types import TrainingExample, TrainingScan

_POSITIVE_MIN = 8
_MAX_N = 10
_MAX_NEGATIVES = 4
_REJECTION_ATTEMPTS = 128


def validate_training_rows(inputs: TrainingInputs) -> TrainingScan:
    """Consume the whole compressed stream and independently reconcile train stats."""
    counts = {item: [0, 0, 0, 0] for item in inputs.mal_ids}
    current_user: str | None = None
    users = 0
    rows = 0
    positives = 0
    for user, item, rating in iter_training_rows(inputs):
        if user != current_user:
            users += 1
            current_user = user
        rows += 1
        counters = counts[item]
        counters[0] += 1
        counters[2] += rating
        counters[3] += 1
        if rating >= _POSITIVE_MIN:
            positives += 1
            counters[1] += 1
    stats = inputs.train_stats
    if (
        rows != stats["train_completed_rows"]
        or users > stats["train_users"]
        or not 0 <= users <= rows
    ):
        raise ResearchError("training_data_mismatch")
    for item, counters in counts.items():
        expected = stats["items"][str(item)]
        if counters != [
            expected["completed_users"],
            expected["positive_users"],
            expected["rating_sum"],
            expected["rating_count"],
        ]:
            raise ResearchError("training_data_mismatch")
    return TrainingScan(
        rows, users, positives, tuple(item for item in inputs.mal_ids if counts[item][1])
    )


def _user_examples(
    user: str, profile: list[tuple[int, int]], warm: tuple[int, ...], epoch: int
) -> Iterator[TrainingExample]:
    completed = {item for item, _ in profile}
    positive = sorted(item for item, rating in profile if rating >= _POSITIVE_MIN)
    if len(positive) < 2 or not any(item not in completed for item in warm):
        return
    raw_seed = canonical_bytes(["arb060-sampling-v1", 6061, epoch, user])
    rng = random.Random(int.from_bytes(hashlib.sha256(raw_seed).digest(), "big"))
    target = rng.choice(positive)
    n = rng.randint(1, min(_MAX_N, len(positive) - 1))
    seeds = tuple(sorted(rng.sample([item for item in positive if item != target], n)))
    negatives: list[int] = []
    chosen: set[int] = set()
    available = sum(item not in completed for item in warm)
    target_negatives = min(_MAX_NEGATIVES, available)
    for _ in range(_REJECTION_ATTEMPTS):
        if len(negatives) == target_negatives:
            break
        item = rng.choice(warm)
        if item not in completed and item not in chosen:
            negatives.append(item)
            chosen.add(item)
    if len(negatives) < target_negatives:
        remaining = sorted(item for item in warm if item not in completed and item not in chosen)
        negatives.extend(rng.sample(remaining, target_negatives - len(negatives)))
    yield TrainingExample(seeds, target, tuple(negatives))


def iter_training_examples(
    inputs: TrainingInputs, *, scan: TrainingScan, epoch: int
) -> Iterator[TrainingExample]:
    """Keep one profile in memory; derive a fresh deterministic stream per epoch."""
    if type(epoch) is not int or not 0 <= epoch < 3:
        raise ResearchError("training_spec_invalid")
    if scan.warm_mal_ids != tuple(
        item for item in inputs.mal_ids if inputs.train_stats["items"][str(item)]["positive_users"]
    ):
        raise ResearchError("training_data_mismatch")
    current_user: str | None = None
    profile: list[tuple[int, int]] = []
    for user, item, rating in iter_training_rows(inputs):
        if current_user is not None and user != current_user:
            yield from _user_examples(current_user, profile, scan.warm_mal_ids, epoch)
            profile.clear()
        current_user = user
        profile.append((item, rating))
    if current_user is not None:
        yield from _user_examples(current_user, profile, scan.warm_mal_ids, epoch)
