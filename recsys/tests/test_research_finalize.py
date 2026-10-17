import copy
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import IO, cast
from unittest.mock import patch

from research_support import ResearchFixture

from recsys.contracts.bundle import canonical_bytes
from recsys.research.artifacts import (
    StreamDescriptor,
    identity,
    read_json,
    stream_jsonl,
    write_json,
)
from recsys.research.evaluation.compare import compare_evaluations
from recsys.research.evaluation.evaluate import evaluate_predictions
from recsys.research.evaluation.finalize import (
    ComparisonManifest,
    ComparisonReport,
    SelectionDecision,
    _comparison,
    export_test_queries,
    finalize_selection,
)
from recsys.research.evaluation.selection import read_selection
from recsys.research.protocol import ResearchError


def _report(path: Path) -> ComparisonReport:
    value, _ = read_json(path)
    assert isinstance(value, dict) and isinstance(value.get("models"), list)
    return cast("ComparisonReport", value)  # Produced by compare_evaluations in the fixture.


def _manifest(path: Path) -> ComparisonManifest:
    value, _ = read_json(path)
    assert isinstance(value, dict) and isinstance(value.get("evaluation_identities"), list)
    return cast("ComparisonManifest", value)  # Produced by compare_evaluations in the fixture.


class FinalizeTests(unittest.TestCase):
    def setUp(self) -> None:
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
        report = _report(self.comparison / "comparison.json")
        model = report["models"][0]
        self.decision: SelectionDecision = {
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

    def test_rehashed_selection_requires_all_returned_field_types(self) -> None:
        selection_path = Path(self.tmp.name) / "selection.json"
        valid: dict[str, object] = {
            **self.decision,
            "kind": "arb-ml-selection",
            "allowed_arms": [
                {"cohort": "main", "order": f"random-liked-{i}", "n": 1} for i in (1, 2, 3)
            ],
        }
        selection_path.write_bytes(canonical_bytes(valid))
        selection, _ = read_selection(
            selection_path, identity(selection_path), self.prepared_identity
        )
        self.assertEqual(selection["n"], 1)
        malformed: tuple[tuple[str, object], ...] = (
            ("comparison_identity", 3),
            ("evaluation_identity", 4),
            ("model_id", 5),
            ("config_sha256", 6),
            ("fit_identity", 7),
            ("code_revision", 8),
            ("declared_comparison_arms", 9),
            ("allowed_arms", [{"cohort": "main", "order": "random-liked-1", "n": True}]),
        )
        for field, wrong in malformed:
            with self.subTest(field=field):
                selection_path.write_bytes(canonical_bytes({**valid, field: wrong}))
                with self.assertRaisesRegex(ResearchError, "research_selection_invalid"):
                    read_selection(selection_path, identity(selection_path), self.prepared_identity)

    def test_comparison_reader_rejects_wrong_nested_model_types(self) -> None:
        original_report = _report(self.comparison / "comparison.json")
        original_manifest = _manifest(self.comparison / "comparison-manifest.json")
        for field, wrong in (("code_revision", 3), ("n_policy", {"status": 4, "chosen_n": 1})):
            with self.subTest(field=field):
                report = cast("dict[str, object]", copy.deepcopy(original_report))
                models = report["models"]
                assert isinstance(models, list) and isinstance(models[0], dict)
                model = cast("dict[str, object]", models[0])
                model[field] = wrong
                self._rewrite_comparison(report, copy.deepcopy(original_manifest))
                with self.assertRaises(ResearchError):
                    _comparison(self.comparison, self.prepared_identity)

    def _rewrite_comparison(
        self,
        report: ComparisonReport | dict[str, object] | None = None,
        manifest: ComparisonManifest | None = None,
    ) -> None:
        if report is None:
            report = _report(self.comparison / "comparison.json")
        if manifest is None:
            manifest = _manifest(self.comparison / "comparison-manifest.json")
        raw = canonical_bytes(report)
        (self.comparison / "comparison.json").write_bytes(raw)
        from recsys.research.protocol import sha

        manifest["report"] = {"bytes": len(raw), "sha256": sha(raw)}
        (self.comparison / "comparison-manifest.json").write_bytes(canonical_bytes(manifest))
        self.decision["comparison_identity"] = identity(
            self.comparison / "comparison-manifest.json"
        )
        self.decision_path.write_bytes(canonical_bytes(self.decision))

    def _assert_rejected_without_test_labels(self, name: str) -> None:
        output = Path(self.tmp.name) / name
        original = Path.open

        def guarded(
            path: Path,
            mode: str = "r",
            buffering: int = -1,
            encoding: str | None = None,
            errors: str | None = None,
            newline: str | None = None,
        ) -> IO[str] | IO[bytes]:
            if path.name == "test-labels.jsonl.gz":
                raise AssertionError("test labels opened before finalization")
            return original(path, mode, buffering, encoding, errors, newline)

        with patch.object(Path, "open", guarded), self.assertRaises(ResearchError):
            finalize_selection(
                self.prepared, self.decision_path, [self.evaluation, self.comparison], output
            )
        self.assertFalse(output.exists())

    def test_rehashed_policy_tamper_cannot_select_different_n(self) -> None:
        report = _report(self.comparison / "comparison.json")
        self.assertEqual(report["models"][0]["n_policy"]["chosen_n"], 1)
        report["models"][0]["n_policy"]["status"] = "selected"
        report["models"][0]["n_policy"]["chosen_n"] = 10
        self.decision["n"] = 10
        self._rewrite_comparison(report)
        self._assert_rejected_without_test_labels("rehashed-n-rejected")

    def test_comparison_schema_and_evidence_binding(self) -> None:
        report = _report(self.comparison / "comparison.json")
        tampered: dict[str, object] = dict(report)
        tampered["unexpected"] = True
        self._rewrite_comparison(tampered)
        self._assert_rejected_without_test_labels("schema-rejected")

    def test_duplicate_evaluation_identity_rejected(self) -> None:
        report = _report(self.comparison / "comparison.json")
        manifest = _manifest(self.comparison / "comparison-manifest.json")
        report["models"].append(copy.deepcopy(report["models"][0]))
        manifest["evaluation_identities"].append(manifest["evaluation_identities"][0])
        self._rewrite_comparison(report, manifest)
        self._assert_rejected_without_test_labels("duplicate-rejected")

    def test_duplicate_supplied_evaluation_identity_rejected(self) -> None:
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

    def test_comparison_manifest_schema_rejected(self) -> None:
        report = _report(self.comparison / "comparison.json")
        manifest = _manifest(self.comparison / "comparison-manifest.json")
        manifest["kind"] = "wrong-kind"
        self._rewrite_comparison(report, manifest)
        self._assert_rejected_without_test_labels("manifest-schema-rejected")

    def test_manifest_and_report_model_mismatch_rejected(self) -> None:
        report = _report(self.comparison / "comparison.json")
        manifest = _manifest(self.comparison / "comparison-manifest.json")
        manifest["evaluation_identities"][0] = "sha256:" + "d" * 64
        self._rewrite_comparison(report, manifest)
        self._assert_rejected_without_test_labels("model-mismatch-rejected")

    def test_rehashed_paired_result_tamper_rejected(self) -> None:
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
        report = _report(self.comparison / "comparison.json")
        self.assertEqual(len(report["paired_differences"]), 1)
        differences = report["paired_differences"][0]["mean_difference_by_n"]
        assert isinstance(differences, list)
        differences[0] = 0.5
        self._rewrite_comparison(report)
        output = Path(self.tmp.name) / "paired-rejected"
        with self.assertRaises(ResearchError):
            finalize_selection(
                self.prepared, self.decision_path, [self.evaluation, other, self.comparison], output
            )
        self.assertFalse(output.exists())

    def test_final_test_boundary_and_exact_selected_predictions(self) -> None:
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
        assert isinstance(manifest, dict)
        queries_value = list(
            stream_jsonl(
                test_queries / "test-queries.jsonl.gz",
                cast("StreamDescriptor", manifest["queries"]),
            )
        )
        assert all(isinstance(query, dict) for query in queries_value)
        queries = cast("list[dict[str, object]]", queries_value)
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
        assert isinstance(report, dict)
        self.assertEqual(report["queries"], 21)

    def test_finalize_does_not_open_test_labels(self) -> None:
        original = Path.open

        def guarded(
            path: Path,
            mode: str = "r",
            buffering: int = -1,
            encoding: str | None = None,
            errors: str | None = None,
            newline: str | None = None,
        ) -> IO[str] | IO[bytes]:
            if path.name == "test-labels.jsonl.gz":
                raise AssertionError("test labels opened before finalization")
            return original(path, mode, buffering, encoding, errors, newline)

        with patch.object(Path, "open", guarded):
            finalize_selection(
                self.prepared,
                self.decision_path,
                [self.evaluation, self.comparison],
                Path(self.tmp.name) / "selected",
            )

    def test_test_prediction_code_revision_cannot_change(self) -> None:
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
        assert isinstance(manifest, dict)
        manifest["code_revision"] = "b" * 40
        from recsys.contracts.bundle import canonical_bytes

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

    def test_changed_n_rejected_before_test_labels(self) -> None:
        self.decision["n"] = 2
        self.decision_path.write_bytes(
            __import__("recsys.contracts.bundle", fromlist=["canonical_bytes"]).canonical_bytes(
                self.decision
            )
        )
        with self.assertRaises(ResearchError):
            finalize_selection(
                self.prepared,
                self.decision_path,
                [self.evaluation, self.comparison],
                Path(self.tmp.name) / "rejected",
            )
        self.assertFalse((Path(self.tmp.name) / "rejected").exists())
