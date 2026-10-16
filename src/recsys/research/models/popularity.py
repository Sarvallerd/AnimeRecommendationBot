"""Train-only completed-positive item-count baseline."""

from pathlib import Path

from recsys.contracts.bundle import JSONValue
from recsys.research.models.manifest import ModelManifest
from recsys.research.models.types import Scorer, ScoreVector, ScoringInputs, TrainingInputs
from recsys.research.protocol import exact, fail


def fit(
    inputs: TrainingInputs,
    parameters: dict[str, JSONValue],
    assets: dict[str, Path],
    stage: Path,
) -> dict[str, object]:
    exact(parameters, set())
    if assets:
        fail("research_model_invalid")
    return {"source": "fit/train-stats.json", "items": len(inputs.mal_ids)}


def load(model_dir: Path, manifest: ModelManifest, inputs: ScoringInputs) -> Scorer:
    import numpy as np

    exact(manifest["metadata"], {"source", "items"})
    if (
        manifest["metadata"] != {"source": "fit/train-stats.json", "items": len(inputs.mal_ids)}
        or manifest["files"]
        or manifest["external_inputs"]
    ):
        fail("research_model_invalid")
    values = np.asarray(
        [inputs.train_stats["items"][str(item)]["positive_users"] for item in inputs.mal_ids],
        dtype=np.float64,
    )
    values.setflags(write=False)
    available = np.ones(len(values), dtype=np.bool_)
    available.setflags(write=False)

    class PopularityScorer:
        def score(self, seed_mal_ids: tuple[int, ...]) -> ScoreVector:
            return ScoreVector(values, available, len(seed_mal_ids))

    return PopularityScorer()
