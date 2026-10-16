import copy
import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path
from typing import IO, TypedDict, cast
from unittest.mock import patch

from research_support import ResearchFixture

from recsys.contracts.bundle import canonical_bytes
from recsys.research.artifacts import (
    StreamDescriptor,
    StreamWriter,
    file_descriptor,
    identity,
    read_json,
    stream_jsonl,
)
from recsys.research.evaluation.evaluate import (
    EvaluationManifest,
    PredictionManifest,
    PrivateMetric,
    _paired_rows,
    _prepared,
    evaluate_predictions,
    metrics,
)
from recsys.research.protocol import ResearchError, load_protocol


class ScoreTestRow(TypedDict):
    query_id: str
    scores: list[list[int | None]]


def _prediction_manifest(path: Path) -> PredictionManifest:
    value, _ = read_json(path)
    assert isinstance(value, dict) and isinstance(value.get("scores"), dict)
    return cast("PredictionManifest", value)  # Fixture prediction was published above.


def _score_rows(path: Path, descriptor: StreamDescriptor) -> list[ScoreTestRow]:
    values = list(stream_jsonl(path, descriptor))
    assert all(
        isinstance(row, dict)
        and isinstance(row.get("query_id"), str)
        and isinstance(row.get("scores"), list)
        and all(isinstance(pair, list) and len(pair) == 2 for pair in row["scores"])
        for row in values
    )
    return cast("list[ScoreTestRow]", values)


class EvaluateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.fixture = ResearchFixture(self.tmp.name)
        self.prepared = self.fixture.prepare()
        self.identity = identity(self.prepared / "prepare-manifest.json")
        self.prediction = self.fixture.predictions(self.prepared, "prediction")

    def evaluate(self, name: str = "evaluation") -> Path:
        return evaluate_predictions(
            self.prepared,
            self.prediction,
            Path(self.tmp.name) / name,
            split="validation",
            expected_prepared_identity=self.identity,
        )

    def test_manual_binary_metrics_and_full_vector(self) -> None:
        answer = metrics([2, 1], {1, 3})
        self.assertAlmostEqual(
            answer["ndcg5"], (1 / __import__("math").log2(3)) / (1 + 1 / __import__("math").log2(3))
        )
        self.assertEqual(answer["recall5"], 0.5)
        self.assertEqual(answer["hit5"], 1)
        self.assertEqual(metrics([], {1})["ndcg5"], 0)
        self.evaluate()
        report, _ = read_json(Path(self.tmp.name) / "evaluation/report.json")
        assert isinstance(report, dict)
        self.assertEqual(report["queries"], 8 * 41)
        self.assertEqual(report["groups"], [])  # slices under 20 users stay private

    def test_validation_never_opens_test_labels(self) -> None:
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
                raise AssertionError("test labels opened")
            return original(path, mode, buffering, encoding, errors, newline)

        with patch.object(Path, "open", guarded):
            self.evaluate()

    def test_rejects_missing_score_and_identity_tamper(self) -> None:
        manifest_path = self.prediction / "prediction-manifest.json"
        manifest = _prediction_manifest(manifest_path)
        manifest["catalog_descriptor"]["sha256"] = "0" * 64
        manifest_path.write_bytes(canonical_bytes(manifest))
        with self.assertRaises(ResearchError):
            self.evaluate()
        self.assertFalse((Path(self.tmp.name) / "evaluation").exists())
        self.assertFalse(list(Path(self.tmp.name).glob(".evaluation.*")))

    def test_rehashed_preparation_rejects_missing_files_and_nested_stats(self) -> None:
        manifest_path = self.prepared / "prepare-manifest.json"
        original, _ = read_json(manifest_path)
        assert isinstance(original, dict)
        missing_files = copy.deepcopy(original)
        missing_files.pop("files")
        manifest_path.write_bytes(canonical_bytes(missing_files))
        with self.assertRaises(ResearchError):
            _prepared(self.prepared, identity(manifest_path))

        stats_path = self.prepared / "fit/train-stats.json"
        stats, _ = read_json(stats_path)
        assert isinstance(stats, dict)
        stats["items"] = 3
        stats_path.write_bytes(canonical_bytes(stats))
        malformed_stats = copy.deepcopy(original)
        files = malformed_stats["files"]
        assert isinstance(files, dict)
        files["fit/train-stats.json"] = file_descriptor(stats_path)
        manifest_path.write_bytes(canonical_bytes(malformed_stats))
        with self.assertRaises(ResearchError):
            _prepared(self.prepared, identity(manifest_path))

    def test_malformed_query_id_fails_before_hashing_or_typed_use(self) -> None:
        prepared, _, catalog, _, _, _ = _prepared(self.prepared, self.identity)
        path = self.prepared / "development/queries.jsonl.gz"
        rows = list(
            stream_jsonl(
                path, cast("StreamDescriptor", prepared["files"]["development/queries.jsonl.gz"])
            )
        )
        assert rows and isinstance(rows[0], dict)
        rows[0]["query_id"] = []
        path.unlink()
        writer = StreamWriter(path)
        for row in rows:
            writer.write_json(row)
        changed = copy.deepcopy(prepared)
        changed["files"]["development/queries.jsonl.gz"] = writer.close()
        with self.assertRaises(ResearchError):
            next(
                _paired_rows(
                    self.prepared,
                    changed,
                    "validation",
                    protocol=load_protocol(),
                    ids=set(catalog["anime"]),
                )
            )

    def test_full_catalog_score_row_failures(self) -> None:
        from recsys.research.artifacts import StreamWriter

        manifest_path = self.prediction / "prediction-manifest.json"
        original_manifest = _prediction_manifest(manifest_path)
        score_path = self.prediction / "scores.jsonl.gz"
        original_rows = _score_rows(score_path, original_manifest["scores"])
        cases: tuple[tuple[str, Callable[[list[list[int | None]]], object]], ...] = (
            ("missing", lambda scores: scores.pop()),
            ("duplicate", lambda scores: scores.__setitem__(1, scores[0])),
            ("unknown", lambda scores: scores.__setitem__(0, [999999, 1])),
            ("bool_id", lambda scores: scores.__setitem__(0, [True, 1])),
            ("bool_score", lambda scores: scores.__setitem__(0, [1, True])),
        )
        for name, mutate in cases:
            with self.subTest(name=name):
                rows = [row.copy() for row in original_rows]
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

    def test_null_abstention_keeps_target_denominator(self) -> None:
        from recsys.research.artifacts import StreamWriter

        manifest_path = self.prediction / "prediction-manifest.json"
        manifest = _prediction_manifest(manifest_path)
        score_path = self.prediction / "scores.jsonl.gz"
        rows = _score_rows(score_path, manifest["scores"])
        score_path.unlink()
        writer = StreamWriter(score_path)
        for index, row in enumerate(rows):
            if index % 41:
                row["scores"] = [[item, None] for item, _ in row["scores"]]
            writer.write_json(row)
        manifest["scores"] = writer.close()
        manifest_path.write_bytes(canonical_bytes(manifest))
        self.evaluate("abstention")
        out_value, _ = read_json(Path(self.tmp.name) / "abstention/evaluation-manifest.json")
        assert isinstance(out_value, dict) and isinstance(out_value.get("files"), dict)
        out = cast("EvaluationManifest", out_value)
        private_values = list(
            stream_jsonl(
                Path(self.tmp.name) / "abstention/user-metrics.jsonl.gz",
                cast("StreamDescriptor", out["files"]["user-metrics.jsonl.gz"]),
            )
        )
        assert all(isinstance(row, dict) for row in private_values)
        private = cast("list[PrivateMetric]", private_values)
        empty = [row for row in private if row["n"] > 0]
        self.assertTrue(empty)
        self.assertTrue(
            all(
                not row["ranking"]
                and row["metrics"]["ndcg5"] == 0
                and row["metrics"]["recall5"] == 0
                for row in empty
            )
        )

    def test_no_test_without_selection(self) -> None:
        with self.assertRaises(ResearchError):
            evaluate_predictions(
                self.prepared,
                self.prediction,
                Path(self.tmp.name) / "test-eval",
                split="test",
                expected_prepared_identity=self.identity,
            )
