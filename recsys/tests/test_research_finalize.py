import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from recsys.research_artifacts import identity, read_json, stream_jsonl, write_json
from recsys.research_compare import compare_evaluations
from recsys.research_evaluate import evaluate_predictions
from recsys.research_finalize import export_test_queries, finalize_selection
from recsys.research_protocol import ResearchError
from research_support import ResearchFixture


class FinalizeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.fixture = ResearchFixture(self.tmp.name)
        self.prepared = self.fixture.prepare()
        self.prepared_identity = identity(self.prepared / "prepare-manifest.json")
        prediction = self.fixture.predictions(self.prepared, "validation-predictions")
        self.evaluation = Path(self.tmp.name) / "evaluation"
        evaluate_predictions(self.prepared, prediction, self.evaluation,
                             split="validation", expected_prepared_identity=self.prepared_identity)
        self.comparison = Path(self.tmp.name) / "comparison"
        compare_evaluations([self.evaluation], self.comparison,
                            expected_prepared_identity=self.prepared_identity)
        report, _ = read_json(self.comparison / "comparison.json")
        model = report["models"][0]
        self.decision = {"schema_version": 1, "prepared_identity": self.prepared_identity,
                         "comparison_identity": identity(self.comparison / "comparison-manifest.json"),
                         "evaluation_identity": model["evaluation_identity"],
                         "model_id": model["model_id"], "config_sha256": model["config_sha256"],
                         "fit_identity": model["fit_identity"],
                         "code_revision": model["code_revision"], "n": 1,
                         "declared_comparison_arms": [f"random-liked-{i}" for i in (1, 2, 3)]}
        self.decision_path = Path(self.tmp.name) / "decision.json"
        write_json(self.decision_path, self.decision)

    def test_final_test_boundary_and_exact_selected_predictions(self):
        selection_dir = Path(self.tmp.name) / "selection"
        selection = finalize_selection(self.prepared, self.decision_path,
                                       [self.evaluation, self.comparison], selection_dir)
        selected_identity = identity(selection)
        test_queries = Path(self.tmp.name) / "test-queries"
        export_test_queries(self.prepared, selection, test_queries,
                            expected_selection_identity=selected_identity)
        manifest, _ = read_json(test_queries / "test-query-manifest.json")
        queries = list(stream_jsonl(test_queries / "test-queries.jsonl.gz", manifest["queries"]))
        self.assertEqual(len(queries), 7 * 3)
        self.assertTrue(all(query["n"] == 1 for query in queries))
        prediction = self.fixture.predictions(self.prepared, "test-predictions", split="test",
                                              queries_path=test_queries / "test-queries.jsonl.gz")
        output = Path(self.tmp.name) / "test-evaluation"
        evaluate_predictions(self.prepared, prediction, output, split="test",
                             expected_prepared_identity=self.prepared_identity,
                             selection_path=selection, expected_selection_identity=selected_identity)
        report, _ = read_json(output / "report.json")
        self.assertEqual(report["queries"], 21)


    def test_finalize_does_not_open_test_labels(self):
        original = Path.open
        def guarded(path, *args, **kwargs):
            if path.name == "test-labels.jsonl.gz":
                raise AssertionError("test labels opened before finalization")
            return original(path, *args, **kwargs)
        with patch.object(Path, "open", guarded):
            finalize_selection(self.prepared, self.decision_path,
                               [self.evaluation, self.comparison], Path(self.tmp.name) / "selected")

    def test_test_prediction_code_revision_cannot_change(self):
        selection = finalize_selection(self.prepared, self.decision_path,
                                       [self.evaluation, self.comparison], Path(self.tmp.name) / "selection")
        selected_identity = identity(selection)
        queries = Path(self.tmp.name) / "queries"
        export_test_queries(self.prepared, selection, queries,
                            expected_selection_identity=selected_identity)
        prediction = self.fixture.predictions(self.prepared, "wrong-code", split="test",
                                              queries_path=queries / "test-queries.jsonl.gz")
        manifest, _ = read_json(prediction / "prediction-manifest.json")
        manifest["code_revision"] = "b" * 40
        from recsys.bundle import canonical_bytes
        (prediction / "prediction-manifest.json").write_bytes(canonical_bytes(manifest))
        with self.assertRaises(ResearchError):
            evaluate_predictions(self.prepared, prediction, Path(self.tmp.name) / "rejected-test",
                                 split="test", expected_prepared_identity=self.prepared_identity,
                                 selection_path=selection, expected_selection_identity=selected_identity)

    def test_changed_n_rejected_before_test_labels(self):
        self.decision["n"] = 2
        self.decision_path.write_bytes(__import__("recsys.bundle", fromlist=["canonical_bytes"]).canonical_bytes(self.decision))
        with self.assertRaises(ResearchError):
            finalize_selection(self.prepared, self.decision_path, [self.evaluation, self.comparison],
                               Path(self.tmp.name) / "rejected")
        self.assertFalse((Path(self.tmp.name) / "rejected").exists())
