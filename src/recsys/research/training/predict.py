"""One bounded seed query against a verified native training scorer."""

from __future__ import annotations

import math
from typing import Literal, TypedDict, cast

from recsys.research.models.types import ScoreVector, ScoringInputs, SupportedBatchScorer
from recsys.research.protocol import ResearchError
from recsys.research.training.types import PredictionRow


class PredictionDocument(TypedDict):
    schema_version: Literal[1]
    kind: Literal["arb-seed-prediction"]
    model_identity: str
    prepared_identity: str
    seed_mal_ids: list[int]
    supported_seed_mal_ids: list[int]
    skipped_seed_mal_ids: list[int]
    limit: int
    status: Literal["ok", "abstained"]
    source: Literal["seed-embedding", "train-popularity", "none"]
    recommendations: list[PredictionRowWire]


class PredictionRowWire(TypedDict):
    mal_id: int
    score: float


def validate_seed_query(
    value: object, *, mal_ids: tuple[int, ...], limit: object
) -> tuple[tuple[int, ...], int]:
    """Reject unknown or duplicate IDs and canonicalize order before aggregation."""
    if type(value) not in (tuple, list):
        raise ResearchError("training_query_invalid")
    seeds = cast("tuple[object, ...] | list[object]", value)
    if len(seeds) > 10 or type(limit) is not int or not 1 <= limit <= 20:
        raise ResearchError("training_query_invalid")
    known = set(mal_ids)
    if any(type(seed) is not int or seed not in known for seed in seeds):
        raise ResearchError("training_query_invalid")
    numbers = cast("tuple[int, ...]", tuple(seeds))
    if len(set(numbers)) != len(numbers):
        raise ResearchError("training_query_invalid")
    return tuple(sorted(numbers)), limit


def _ranked(rows: list[PredictionRow], limit: int) -> list[PredictionRowWire]:
    ranked = sorted(rows, key=lambda row: (-row.score, row.mal_id))[:limit]
    return [{"mal_id": row.mal_id, "score": row.score} for row in ranked]


def predict_one(
    inputs: ScoringInputs,
    scorer: SupportedBatchScorer,
    *,
    model_identity: str,
    seed_mal_ids: tuple[int, ...],
    limit: int,
) -> PredictionDocument:
    import numpy as np

    seeds, count = validate_seed_query(seed_mal_ids, mal_ids=inputs.mal_ids, limit=limit)
    supported = scorer.supported_seed_ids(seeds)
    if type(supported) is not frozenset or not supported <= set(seeds):
        raise ResearchError("training_prediction_invalid")
    used = tuple(seed for seed in seeds if seed in supported)
    skipped = tuple(seed for seed in seeds if seed not in supported)
    status: Literal["ok", "abstained"] = "ok"
    source: Literal["seed-embedding", "train-popularity", "none"]
    recommendations: list[PredictionRowWire]
    if not seeds:
        source = "train-popularity"
        ranked = [
            PredictionRow(item, float(inputs.train_stats["items"][str(item)]["positive_users"]))
            for item in inputs.mal_ids
        ]
        recommendations = _ranked(ranked, count)
    elif not used:
        status, source, recommendations = "abstained", "none", []
    else:
        source = "seed-embedding"
        vector = scorer.score(used)
        if (
            type(vector) is not ScoreVector
            or not isinstance(vector.values, np.ndarray)
            or vector.values.dtype != np.dtype("float64")
            or vector.values.shape != (len(inputs.mal_ids),)
            or not isinstance(vector.available, np.ndarray)
            or vector.available.dtype != np.dtype("bool")
            or vector.available.shape != (len(inputs.mal_ids),)
            or type(vector.supported_seeds) is not int
            or vector.supported_seeds != len(used)
            or not np.isfinite(vector.values).all()
        ):
            raise ResearchError("training_prediction_invalid")
        ranked = []
        for item, score, available in zip(
            inputs.mal_ids, vector.values, vector.available, strict=True
        ):
            if available and item not in seeds:
                if inputs.train_stats["items"][str(item)]["positive_users"] < 1:
                    raise ResearchError("training_prediction_invalid")
                number = float(score)
                if not math.isfinite(number):
                    raise ResearchError("training_prediction_invalid")
                ranked.append(PredictionRow(item, number))
        recommendations = _ranked(ranked, count)
    return {
        "schema_version": 1,
        "kind": "arb-seed-prediction",
        "model_identity": model_identity,
        "prepared_identity": inputs.prepared_identity,
        "seed_mal_ids": list(seeds),
        "supported_seed_mal_ids": list(used),
        "skipped_seed_mal_ids": list(skipped),
        "limit": count,
        "status": status,
        "source": source,
        "recommendations": recommendations,
    }
