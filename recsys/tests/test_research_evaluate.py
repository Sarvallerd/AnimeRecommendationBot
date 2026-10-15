import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from recsys.bundle import canonical_bytes
from recsys.research_artifacts import identity, read_json, stream_jsonl, write_json
from recsys.research_evaluate import evaluate_predictions, metrics
from recsys.research_protocol import ResearchError
from research_support import ResearchFixture


class EvaluateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.fixture = ResearchFixture(self.tmp.name)
        self.prepared = self.fixture.prepare()
        self.identity = identity(self.prepared / "prepare-manifest.json")
        self.prediction = self.fixture.predictions(self.prepared, "prediction")

    def evaluate(self, name="evaluation"):
        return evaluate_predictions(self.prepared, self.prediction, Path(self.tmp.name) / name,
                                    split="validation", expected_prepared_identity=self.identity)

    def test_manual_binary_metrics_and_full_vector(self):
        answer = metrics([2, 1], {1, 3})
        self.assertAlmostEqual(answer["ndcg5"], (1 / __import__("math").log2(3)) /
                               (1 + 1 / __import__("math").log2(3)))
        self.assertEqual(answer["recall5"], .5)
        self.assertEqual(answer["hit5"], 1)
        self.assertEqual(metrics([], {1})["ndcg5"], 0)
        self.evaluate()
        report, _ = read_json(Path(self.tmp.name) / "evaluation/report.json")
        self.assertEqual(report["queries"], 8 * 41)
        self.assertEqual(report["groups"], [])  # slices under 20 users stay private

    def test_validation_never_opens_test_labels(self):
        original = Path.open
        def guarded(path, *args, **kwargs):
            if path.name == "test-labels.jsonl.gz":
                raise AssertionError("test labels opened")
            return original(path, *args, **kwargs)
        with patch.object(Path, "open", guarded):
            self.evaluate()

    def test_rejects_missing_score_and_identity_tamper(self):
        manifest_path = self.prediction / "prediction-manifest.json"
        manifest, _ = read_json(manifest_path)
        manifest["catalog_descriptor"]["sha256"] = "0" * 64
        manifest_path.write_bytes(canonical_bytes(manifest))
        with self.assertRaises(ResearchError):
            self.evaluate()
        self.assertFalse((Path(self.tmp.name) / "evaluation").exists())
        self.assertFalse(list(Path(self.tmp.name).glob(".evaluation.*")))


    def test_full_catalog_score_row_failures(self):
        from recsys.research_artifacts import StreamWriter
        manifest_path = self.prediction / "prediction-manifest.json"
        original_manifest, _ = read_json(manifest_path)
        score_path = self.prediction / "scores.jsonl.gz"
        original_rows = list(stream_jsonl(score_path, original_manifest["scores"]))
        for name, mutate in (
            ("missing", lambda scores: scores.pop()),
            ("duplicate", lambda scores: scores.__setitem__(1, scores[0])),
            ("unknown", lambda scores: scores.__setitem__(0, [999999, 1])),
            ("bool_id", lambda scores: scores.__setitem__(0, [True, 1])),
            ("bool_score", lambda scores: scores.__setitem__(0, [1, True])),
        ):
            with self.subTest(name=name):
                rows = [dict(row) for row in original_rows]
                rows[0] = {**rows[0], "scores": [list(pair) for pair in rows[0]["scores"]]}
                mutate(rows[0]["scores"])
                score_path.unlink()
                writer = StreamWriter(score_path)
                for row in rows:
                    writer.write_json(row)
                manifest = {**original_manifest, "scores": writer.close()}
                manifest_path.write_bytes(canonical_bytes(manifest))
                with self.assertRaises(ResearchError):
                    self.evaluate(name)
                self.assertFalse((Path(self.tmp.name) / name).exists())

    def test_null_abstention_keeps_target_denominator(self):
        from recsys.research_artifacts import StreamWriter
        manifest_path = self.prediction / "prediction-manifest.json"
        manifest, _ = read_json(manifest_path)
        score_path = self.prediction / "scores.jsonl.gz"
        rows = list(stream_jsonl(score_path, manifest["scores"]))
        score_path.unlink()
        writer = StreamWriter(score_path)
        for index, row in enumerate(rows):
            if index % 41:
                row["scores"] = [[item, None] for item, _ in row["scores"]]
            writer.write_json(row)
        manifest["scores"] = writer.close()
        manifest_path.write_bytes(canonical_bytes(manifest))
        self.evaluate("abstention")
        out, _ = read_json(Path(self.tmp.name) / "abstention/evaluation-manifest.json")
        private = list(stream_jsonl(Path(self.tmp.name) / "abstention/user-metrics.jsonl.gz",
                                    out["files"]["user-metrics.jsonl.gz"]))
        empty = [row for row in private if row["n"] > 0]
        self.assertTrue(empty)
        self.assertTrue(all(not row["ranking"] and row["metrics"]["ndcg5"] == 0
                            and row["metrics"]["recall5"] == 0 for row in empty))

    def test_no_test_without_selection(self):
        with self.assertRaises(ResearchError):
            evaluate_predictions(self.prepared, self.prediction, Path(self.tmp.name) / "test-eval",
                                 split="test", expected_prepared_identity=self.identity)
