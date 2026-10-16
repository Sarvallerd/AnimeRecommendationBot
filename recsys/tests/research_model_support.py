"""Small private-fixture helpers for the ML-02 model and prediction boundary."""

from contextlib import ExitStack, contextmanager
from unittest.mock import patch

from research_support import ResearchFixture

from recsys.research.artifacts import identity
from recsys.research.models.package import fit_model
from recsys.research.predict import predict_validation


class ModelFixture(ResearchFixture):
    def make_prepared(self):
        prepared = self.prepare()
        return prepared, identity(prepared / "prepare-manifest.json")

    def fit_popularity(self, prepared, prepared_identity, name="model"):
        directory = self.root / name
        with patch_release_info(self.lock):
            fit_model(
                prepared,
                directory,
                expected_prepared_identity=prepared_identity,
                variant_id="popularity",
                code_revision="a" * 40,
            )
        return directory

    def predict(self, prepared, prepared_identity, model, name="prediction"):
        directory = self.root / name
        with patch_release_info(self.lock):
            predict_validation(
                prepared, model, directory, expected_prepared_identity=prepared_identity
            )
        return directory


@contextmanager
def patch_release_info(lock):
    """Inject a tiny lock at each direct source-contract import owner."""
    with ExitStack() as stack:
        for owner in (
            "recsys.release.lock",
            "recsys.research.models.inputs",
            "recsys.research.models.manifest",
            "recsys.research.models.content",
        ):
            stack.enter_context(patch(owner + ".release_info", return_value=lock))
        yield
