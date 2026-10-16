import tempfile
import unittest
from pathlib import Path

import numpy as np
from research_support import ResearchFixture

from recsys.research.artifacts import identity, read_json
from recsys.research.evaluation.compare import _policy, compare_evaluations
from recsys.research.evaluation.evaluate import evaluate_predictions
from recsys.research.protocol import ResearchError, load_protocol


class CompareTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.fixture = ResearchFixture(self.tmp.name)
        self.prepared = self.fixture.prepare()
        self.prepared_identity = identity(self.prepared / "prepare-manifest.json")
        self.prediction = self.fixture.predictions(self.prepared, "prediction")
        self.evaluation = Path(self.tmp.name) / "evaluation"
        evaluate_predictions(
            self.prepared,
            self.prediction,
            self.evaluation,
            split="validation",
            expected_prepared_identity=self.prepared_identity,
        )

    def test_policy_equal_and_inconclusive(self) -> None:
        protocol = load_protocol()
        equal = _policy(np.ones((10, 10)), protocol)
        self.assertEqual((equal["status"], equal["chosen_n"]), ("selected", 1))
        insufficient = _policy(np.ones((1, 10)), protocol)
        self.assertEqual(insufficient["status"], "insufficient-users")
        self.assertIsNone(insufficient["chosen_n"])
        # Every user's best N differs; uncertainty exceeds the fixed .01 tolerance.
        matrix = np.eye(10)
        uncertain = _policy(matrix, protocol)
        self.assertEqual(uncertain["status"], "inconclusive")
        self.assertIsNone(uncertain["chosen_n"])

    def test_comparison_receipt_and_duplicate_rejection(self) -> None:
        output = Path(self.tmp.name) / "comparison"
        compare_evaluations(
            [self.evaluation], output, expected_prepared_identity=self.prepared_identity
        )
        report, _ = read_json(output / "comparison.json")
        assert isinstance(report, dict)
        self.assertEqual(report["users"], 8)
        models = report["models"]
        assert isinstance(models, list) and isinstance(models[0], dict)
        policy = models[0]["n_policy"]
        assert isinstance(policy, dict)
        self.assertEqual(policy["chosen_n"], 1)
        with self.assertRaises(ResearchError):
            compare_evaluations(
                [self.evaluation, self.evaluation],
                Path(self.tmp.name) / "bad",
                expected_prepared_identity=self.prepared_identity,
            )
