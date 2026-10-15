"""Train-only completed-positive item-count baseline."""

from .research_models import ScoreVector
from .research_protocol import exact, fail

CODE_FILES = ("research_model_popularity.py",)


def fit(inputs, parameters, assets, stage):
    exact(parameters, set())
    if assets:
        fail("research_model_invalid")
    return {"source": "fit/train-stats.json", "items": len(inputs.mal_ids)}


def load(model_dir, manifest, inputs):
    import numpy as np
    exact(manifest["metadata"], {"source", "items"})
    if (manifest["metadata"] != {"source": "fit/train-stats.json", "items": len(inputs.mal_ids)}
            or manifest["files"] or manifest["external_inputs"]):
        fail("research_model_invalid")
    values = np.asarray([inputs.train_stats["items"][str(item)]["positive_users"]
                         for item in inputs.mal_ids], dtype=np.float64)
    values.setflags(write=False)
    available = np.ones(len(values), dtype=np.bool_)
    available.setflags(write=False)

    class PopularityScorer:
        def score(self, seed_mal_ids):
            return ScoreVector(values, available, len(seed_mal_ids))

    return PopularityScorer()
