"""Private ML-02 model inputs, fixed variants, and immutable model packages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


@dataclass(frozen=True)
class ScoringInputs:
    prepared_identity: str
    protocol_identity: str
    catalog_descriptor: dict
    fit_identity: str
    train_interactions_descriptor: dict
    train_stats_descriptor: dict
    mal_ids: tuple[int, ...]
    catalog: dict
    train_stats: dict


@dataclass(frozen=True)
class TrainingInputs(ScoringInputs):
    train_interactions_path: Path


@dataclass(frozen=True)
class ScoreVector:
    values: np.ndarray
    available: np.ndarray
    supported_seeds: int
