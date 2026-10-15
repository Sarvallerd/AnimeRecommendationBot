"""Small private-fixture helpers for the ML-02 model and prediction boundary."""

from pathlib import Path
from unittest.mock import patch

from recsys.research_artifacts import identity
from recsys.research_models import fit_model
from recsys.research_predict import predict_validation
from research_support import ResearchFixture


class ModelFixture(ResearchFixture):
    def make_prepared(self):
        prepared = self.prepare()
        return prepared, identity(prepared / "prepare-manifest.json")

    def fit_popularity(self, prepared, prepared_identity, name="model"):
        directory = self.root / name
        with patch("recsys.research_models.release_info", return_value=self.lock):
            fit_model(prepared, directory, expected_prepared_identity=prepared_identity,
                      variant_id="popularity", code_revision="a" * 40)
        return directory

    def predict(self, prepared, prepared_identity, model, name="prediction"):
        directory = self.root / name
        with patch("recsys.research_models.release_info", return_value=self.lock):
            predict_validation(prepared, model, directory,
                               expected_prepared_identity=prepared_identity)
        return directory
