import tempfile
import unittest

from research_model_support import ModelFixture

from recsys.research.artifacts import read_json


class PopularityTests(unittest.TestCase):
    def test_counts_include_only_train_completed_ratings_at_least_eight(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = ModelFixture(directory)
            prepared, prepared_identity = fixture.make_prepared()
            model = fixture.fit_popularity(prepared, prepared_identity)
            prediction = fixture.predict(prepared, prepared_identity, model)
            manifest, _ = read_json(prediction / "prediction-manifest.json")
            assert isinstance(manifest, dict)
            self.assertEqual(manifest["model_id"], "popularity")
            self.assertEqual(manifest["zero_seed_policy"], "popularity-train-positive-user-counts")
            stats, _ = read_json(prepared / "fit/train-stats.json")
            assert isinstance(stats, dict)
            items = stats["items"]
            assert isinstance(items, dict)
            cold, warm = items["18"], items["1"]
            assert isinstance(cold, dict) and isinstance(warm, dict)
            self.assertEqual(cold["positive_users"], 0)
            positive_users = warm["positive_users"]
            assert isinstance(positive_users, int)
            self.assertGreater(positive_users, 0)
