"""Private ML-02 model inputs, fixed variants, and immutable model packages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, TypedDict, runtime_checkable

from recsys.contracts.bundle import Catalog
from recsys.research.artifacts import FileDescriptor, StreamDescriptor

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


class TrainItem(TypedDict):
    completed_users: int
    positive_users: int
    rating_sum: int
    rating_count: int
    band: Literal["cold", "head", "tail"]


class TrainStats(TypedDict):
    schema_version: int
    train_users: int
    train_completed_rows: int
    items: dict[str, TrainItem]


@dataclass(frozen=True)
class ScoringInputs:
    prepared_identity: str
    protocol_identity: str
    catalog_descriptor: FileDescriptor
    fit_identity: str
    train_interactions_descriptor: StreamDescriptor
    train_stats_descriptor: FileDescriptor
    mal_ids: tuple[int, ...]
    catalog: Catalog
    train_stats: TrainStats


@dataclass(frozen=True)
class TrainingInputs(ScoringInputs):
    train_interactions_path: Path


@dataclass(frozen=True)
class ScoreVector:
    values: NDArray[np.float64]
    available: NDArray[np.bool_]
    supported_seeds: int


class Scorer(Protocol):
    def score(self, seed_mal_ids: tuple[int, ...]) -> ScoreVector: ...


@runtime_checkable
class BatchScorer(Protocol):
    def score_many(self, seeds: tuple[tuple[int, ...], ...]) -> tuple[ScoreVector, ...]: ...


@runtime_checkable
class CloseableScorer(Protocol):
    def close(self) -> None: ...


@runtime_checkable
class SupportedBatchScorer(Scorer, BatchScorer, CloseableScorer, Protocol):
    def supported_seed_ids(self, seed_mal_ids: tuple[int, ...]) -> frozenset[int]: ...
