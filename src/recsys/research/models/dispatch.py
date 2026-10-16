"""Private ML-02 model inputs, fixed variants, and immutable model packages."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

from recsys.contracts.bundle import JSONValue
from recsys.research.models.manifest import ModelManifest
from recsys.research.models.types import Scorer, ScoringInputs, TrainingInputs
from recsys.research.protocol import fail

if TYPE_CHECKING:
    from recsys.research.models import collaborative, content, hybrid, popularity


class PopularityBackend(Protocol):
    def fit(
        self,
        inputs: TrainingInputs,
        parameters: dict[str, JSONValue],
        assets: dict[str, Path],
        stage: Path,
    ) -> dict[str, object]: ...
    def load(self, model_dir: Path, manifest: ModelManifest, inputs: ScoringInputs) -> Scorer: ...


class StandardBackend(Protocol):
    def fit(
        self,
        inputs: TrainingInputs,
        parameters: dict[str, JSONValue],
        assets: dict[str, Path],
        stage: Path,
    ) -> dict[str, object]: ...
    def load(
        self,
        model_dir: Path,
        manifest: ModelManifest,
        inputs: ScoringInputs,
        *,
        backend: str = "cpu",
        payload_prefix: str = "",
    ) -> Scorer: ...


class HybridBackend(Protocol):
    def fit(
        self,
        inputs: ScoringInputs,
        parameters: dict[str, JSONValue],
        assets: dict[str, Path],
        stage: Path,
        *,
        code_revision: str,
        backend: str,
    ) -> dict[str, object]: ...
    def load(
        self,
        model_dir: Path,
        manifest: ModelManifest,
        inputs: ScoringInputs,
        *,
        backend: str = "cpu",
    ) -> Scorer: ...


type Backend = PopularityBackend | StandardBackend | HybridBackend

if TYPE_CHECKING:
    _popularity_backend: PopularityBackend = popularity
    _content_backend: StandardBackend = content
    _collaborative_backend: StandardBackend = collaborative
    _hybrid_backend: HybridBackend = hybrid

BACKENDS = {
    "popularity": "popularity",
    "content-mean": "content",
    "content-max": "content",
    "content-union": "content",
    "itemknn": "collaborative",
    "puresvd": "collaborative",
    "hybrid": "hybrid",
}


def _backend(variant_id: str) -> Backend:
    if type(variant_id) is not str or variant_id not in BACKENDS:
        fail("research_model_variant_invalid")
    try:
        return cast("Backend", importlib.import_module("." + BACKENDS[variant_id], __package__))
    except ModuleNotFoundError as exc:
        if exc.name == f"{__package__}.{BACKENDS[variant_id]}":
            fail("research_model_unavailable")
        raise
