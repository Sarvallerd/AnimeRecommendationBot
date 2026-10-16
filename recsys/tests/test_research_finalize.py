import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from research_support import ResearchFixture

from recsys.bundle import canonical_bytes
from recsys.research_artifacts import identity, read_json, stream_jsonl, write_json
from recsys.research_compare import compare_evaluations
from recsys.research_evaluate import evaluate_predictions
from recsys.research_finalize import export_test_queries, finalize_selection
from recsys.research_protocol import ResearchError


class FinalizeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.fixture = ResearchFixture(self.tmp.name)
        self.prepared = self.fixture.prepare()
        self.prepared_identity = identity(self.prepared / "prepare-manifest.json")
        prediction = self.fixture.predictions(self.prepared, "validation-predictions")
        self.evaluation = Path(self.tmp.name) / "evaluation"
        evaluate_predictions(
            self.prepared,
            prediction,
            self.evaluation,
            split="validation",
            expected_prepared_identity=self.prepared_identity,
        )
        self.comparison = Path(self.tmp.name) / "comparison"
        compare_evaluations(
            [self.evaluation], self.comparison, expected_prepared_identity=self.prepared_identity
        )
        report, _ = read_json(self.comparison / "comparison.json")
        model = report["models"][0]
        self.decision = {
            "schema_version": 1,
            "prepared_identity": self.prepared_identity,
            "comparison_identity": identity(self.comparison / "comparison-manifest.json"),
            "evaluation_identity": model["evaluation_identity"],
            "model_id": model["model_id"],
            "config_sha256": model["config_sha256"],
            "fit_identity": model["fit_identity"],
            "code_revision": model["code_revision"],
            "n": 1,
            "declared_comparison_arms": [f"random-liked-{i}" for i in (1, 2, 3)],
        }
        self.decision_path = Path(self.tmp.name) / "decision.json"
        write_json(self.decision_path, self.decision)

    def _rewrite_comparison(self, report=None, manifest=None):
        if report is None:
            report, _ = read_json(self.comparison / "comparison.json")
        if manifest is None:
            manifest, _ = read_json(self.comparison / "comparison-manifest.json")
        raw = canonical_bytes(report)
        (self.comparison / "comparison.json").write_bytes(raw)
        from recsys.research_protocol import sha

        manifest["report"] = {"bytes": len(raw), "sha256": sha(raw)}
        (self.comparison / "comparison-manifest.json").write_bytes(canonical_bytes(manifest))
        self.decision["comparison_identity"] = identity(
            self.comparison / "comparison-manifest.json"
        )
        self.decision_path.write_bytes(canonical_bytes(self.decision))

    def _assert_rejected_without_test_labels(self, name):
        output = Path(self.tmp.name) / name
        original = Path.open

        def guarded(path, *args, **kwargs):
            if path.name == "test-labels.jsonl.gz":
                raise AssertionError("test labels opened before finalization")
            return original(path, *args, **kwargs)

        with patch.object(Path, "open", guarded), self.assertRaises(ResearchError):
            finalize_selection(
                self.prepared, self.decision_path, [self.evaluation, self.comparison], output
            )
        self.assertFalse(output.exists())

    def test_rehashed_policy_tamper_cannot_select_different_n(self):
        report, _ = read_json(self.comparison / "comparison.json")
        self.assertEqual(report["models"][0]["n_policy"]["chosen_n"], 1)
        report["models"][0]["n_policy"].update(status="selected", chosen_n=10)
        self.decision["n"] = 10
        self._rewrite_comparison(report)
        self._assert_rejected_without_test_labels("rehashed-n-rejected")

    def test_comparison_schema_and_evidence_binding(self):
        report, _ = read_json(self.comparison / "comparison.json")
        report["unexpected"] = True
        self._rewrite_comparison(report)
        self._assert_rejected_without_test_labels("schema-rejected")

    def test_duplicate_evaluation_identity_rejected(self):
        report, _ = read_json(self.comparison / "comparison.json")
        manifest, _ = read_json(self.comparison / "comparison-manifest.json")
        report["models"].append(dict(report["models"][0]))
        manifest["evaluation_identities"].append(manifest["evaluation_identities"][0])
        self._rewrite_comparison(report, manifest)
        self._assert_rejected_without_test_labels("duplicate-rejected")

    def test_duplicate_supplied_evaluation_identity_rejected(self):
        duplicate = Path(self.tmp.name) / "duplicate-evaluation"
        shutil.copytree(self.evaluation, duplicate)
        output = Path(self.tmp.name) / "duplicate-supplied-rejected"
        with self.assertRaises(ResearchError):
            finalize_selection(
                self.prepared,
                self.decision_path,
                [self.evaluation, duplicate, self.comparison],
                output,
            )
        self.assertFalse(output.exists())

    def test_comparison_manifest_schema_rejected(self):
        report, _ = read_json(self.comparison / "comparison.json")
        manifest, _ = read_json(self.comparison / "comparison-manifest.json")
        manifest["kind"] = "wrong-kind"
        self._rewrite_comparison(report, manifest)
        self._assert_rejected_without_test_labels("manifest-schema-rejected")

    def test_manifest_and_report_model_mismatch_rejected(self):
        report, _ = read_json(self.comparison / "comparison.json")
        manifest, _ = read_json(self.comparison / "comparison-manifest.json")
        manifest["evaluation_identities"][0] = "sha256:" + "d" * 64
        self._rewrite_comparison(report, manifest)
        self._assert_rejected_without_test_labels("model-mismatch-rejected")

    def test_rehashed_paired_result_tamper_rejected(self):
        prediction = self.fixture.predictions(self.prepared, "other-predictions", model_id="other")
        other = Path(self.tmp.name) / "other-evaluation"
        evaluate_predictions(
            self.prepared,
            prediction,
            other,
            split="validation",
            expected_prepared_identity=self.prepared_identity,
        )
        self.comparison.rename(Path(self.tmp.name) / "old-comparison")
        compare_evaluations(
            [self.evaluation, other],
            self.comparison,
            expected_prepared_identity=self.prepared_identity,
        )
        report, _ = read_json(self.comparison / "comparison.json")
        self.assertEqual(len(report["paired_differences"]), 1)
        report["paired_differences"][0]["mean_difference_by_n"][0] = 0.5
        self._rewrite_comparison(report)
        output = Path(self.tmp.name) / "paired-rejected"
        with self.assertRaises(ResearchError):
            finalize_selection(
                self.prepared, self.decision_path, [self.evaluation, other, self.comparison], output
            )
        self.assertFalse(output.exists())

    def test_final_test_boundary_and_exact_selected_predictions(self):
        selection_dir = Path(self.tmp.name) / "selection"
        selection = finalize_selection(
            self.prepared, self.decision_path, [self.evaluation, self.comparison], selection_dir
        )
        selected_identity = identity(selection)
        test_queries = Path(self.tmp.name) / "test-queries"
        export_test_queries(
            self.prepared, selection, test_queries, expected_selection_identity=selected_identity
        )
        manifest, _ = read_json(test_queries / "test-query-manifest.json")
        queries = list(stream_jsonl(test_queries / "test-queries.jsonl.gz", manifest["queries"]))
        self.assertEqual(len(queries), 7 * 3)
        self.assertTrue(all(query["n"] == 1 for query in queries))
        prediction = self.fixture.predictions(
            self.prepared,
            "test-predictions",
            split="test",
            queries_path=test_queries / "test-queries.jsonl.gz",
        )
        output = Path(self.tmp.name) / "test-evaluation"
        evaluate_predictions(
            self.prepared,
            prediction,
            output,
            split="test",
            expected_prepared_identity=self.prepared_identity,
            selection_path=selection,
            expected_selection_identity=selected_identity,
        )
        report, _ = read_json(output / "report.json")
        self.assertEqual(report["queries"], 21)

    def test_finalize_does_not_open_test_labels(self):
        original = Path.open

        def guarded(path, *args, **kwargs):
            if path.name == "test-labels.jsonl.gz":
                raise AssertionError("test labels opened before finalization")
            return original(path, *args, **kwargs)

        with patch.object(Path, "open", guarded):
            finalize_selection(
                self.prepared,
                self.decision_path,
                [self.evaluation, self.comparison],
                Path(self.tmp.name) / "selected",
            )

    def test_test_prediction_code_revision_cannot_change(self):
        selection = finalize_selection(
            self.prepared,
            self.decision_path,
            [self.evaluation, self.comparison],
            Path(self.tmp.name) / "selection",
        )
        selected_identity = identity(selection)
        queries = Path(self.tmp.name) / "queries"
        export_test_queries(
            self.prepared, selection, queries, expected_selection_identity=selected_identity
        )
        prediction = self.fixture.predictions(
            self.prepared,
            "wrong-code",
            split="test",
            queries_path=queries / "test-queries.jsonl.gz",
        )
        manifest, _ = read_json(prediction / "prediction-manifest.json")
        manifest["code_revision"] = "b" * 40
        from recsys.bundle import canonical_bytes

        (prediction / "prediction-manifest.json").write_bytes(canonical_bytes(manifest))
        with self.assertRaises(ResearchError):
            evaluate_predictions(
                self.prepared,
                prediction,
                Path(self.tmp.name) / "rejected-test",
                split="test",
                expected_prepared_identity=self.prepared_identity,
                selection_path=selection,
                expected_selection_identity=selected_identity,
            )

    def test_changed_n_rejected_before_test_labels(self):
        self.decision["n"] = 2
        self.decision_path.write_bytes(
            __import__("recsys.bundle", fromlist=["canonical_bytes"]).canonical_bytes(self.decision)
        )
        with self.assertRaises(ResearchError):
            finalize_selection(
                self.prepared,
                self.decision_path,
                [self.evaluation, self.comparison],
                Path(self.tmp.name) / "rejected",
            )
        self.assertFalse((Path(self.tmp.name) / "rejected").exists())
