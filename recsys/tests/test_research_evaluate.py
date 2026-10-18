import copy
import math
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
    EvaluationTrainStats,
    PredictionManifest,
    PrivateMetric,
    _paired_rows,
    _prepared,
    _rank,
    evaluate_predictions,
    metrics,
)
from recsys.research.protocol import ResearchError, fail, finite, load_protocol


def _f2_rank_reference(
    scores: object,
    ids: list[int],
    excluded: set[int],
    *,
    n0: bool = False,
    stats: EvaluationTrainStats | None = None,
) -> list[int]:
    """The frozen F2 ranking body, used only as a small synthetic oracle."""
    if type(scores) is not list or len(scores) != len(ids):
        fail("research_prediction_invalid")
    values: list[tuple[float, int]] = []
    for entry, expected in zip(scores, ids):
        if (
            type(entry) is not list
            or len(entry) != 2
            or type(entry[0]) is not int
            or entry[0] != expected
        ):
            fail("research_prediction_invalid")
        score = entry[1]
        if score is not None:
            finite(score)
        if n0 and (stats is None or score != stats["items"][str(expected)]["positive_users"]):
            fail("research_prediction_invalid")
        if expected not in excluded and score is not None:
            values.append((-cast("float", score), expected))
    values.sort()
    return [item for _, item in values[:10]]


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


class RankDifferentialTests(unittest.TestCase):
    def assert_f2_equivalent(
        self,
        scores: object,
        ids: list[int],
        excluded: set[int],
        *,
        n0: bool = False,
        stats: EvaluationTrainStats | None = None,
    ) -> None:
        def result(reference: bool) -> tuple[type[Exception] | None, object]:
            rank = _f2_rank_reference if reference else _rank
            try:
                return None, rank(scores, ids, excluded, n0=n0, stats=stats)
            except Exception as exc:
                return type(exc), str(exc)

        self.assertEqual(result(True), result(False))

    def test_full_vector_top_ten_and_masks_match_f2(self) -> None:
        for size in (0, 1, 9, 10, 11, 21, 28_955):
            ids = [3 * index + 7 for index in range(size)]
            for order in ("increasing", "decreasing", "shuffled", "ties"):
                if order == "increasing":
                    values: list[int | float | None] = [index - size for index in range(size)]
                elif order == "decreasing":
                    values = [size - index for index in range(size)]
                elif order == "shuffled":
                    values = [((index * 104_729) % max(size, 1)) - size for index in range(size)]
                else:
                    values = [0.0 if index % 2 else -0.0 for index in range(size)]
                if size and order == "shuffled":
                    values[0] = 10**100
                    values[-1] = -(10**100)
                    if size > 2:
                        values[1] = None
                scores = [[item, score] for item, score in zip(ids, values, strict=True)]
                for excluded in (set(), set(ids[:3]), set(ids[::2]), set(ids)):
                    with self.subTest(size=size, order=order, excluded=len(excluded)):
                        self.assert_f2_equivalent(scores, ids, excluded)

    def test_null_zero_negative_and_n0_count_equivalence(self) -> None:
        ids = [7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37]
        mixed: list[int | float | None] = [None, 0, 0.0, -0.0, -3, -3.0, 7, 7.0, 10**100, 1, 1]
        self.assert_f2_equivalent(
            [[item, score] for item, score in zip(ids, mixed, strict=True)], ids, {10, 22}
        )
        stats: EvaluationTrainStats = {
            "schema_version": 1,
            "train_users": 11,
            "train_completed_rows": 11,
            "items": {
                str(item): {
                    "completed_users": 1,
                    "positive_users": index % 3,
                    "rating_sum": 8,
                    "rating_count": 1,
                    "band": "head",
                }
                for index, item in enumerate(ids)
            },
        }
        counts = [
            [item, float(index % 3) if index % 2 else index % 3] for index, item in enumerate(ids)
        ]
        self.assert_f2_equivalent(counts, ids, {7, 22}, n0=True, stats=stats)
        counts[-1][1] = 20
        self.assert_f2_equivalent(counts, ids, {37}, n0=True, stats=stats)
        self.assert_f2_equivalent(counts, ids, set(), n0=True, stats=None)

    def test_late_excluded_and_malformed_entries_keep_f2_error_codes(self) -> None:
        ids = [3 * index + 7 for index in range(21)]
        baseline: list[object] = [[item, 100 - index] for index, item in enumerate(ids)]
        malformed: list[tuple[str, object, set[int]]] = [
            ("missing", baseline[:-1], set()),
            ("extra", baseline + [[9999, 1]], set()),
            ("wrong_container", tuple(baseline), set()),
        ]
        changes: list[tuple[str, int, object, set[int]]] = [
            ("late_bad_id", -1, [ids[-1] + 1, 0], set()),
            ("excluded_bad_id", -1, [ids[-1] + 1, 0], {ids[-1]}),
            ("outside_top_ten", -1, [ids[-1], True], set()),
            ("duplicate", 1, [ids[0], 1], set()),
            ("unknown", 0, [9999, 1], set()),
            ("bool_id", 0, [True, 1], set()),
            ("wrong_pair", 0, (ids[0], 1), set()),
            ("short_pair", 0, [ids[0]], set()),
            ("bool_score", 0, [ids[0], True], set()),
            ("string_score", 0, [ids[0], "1"], set()),
            ("list_score", 0, [ids[0], [1]], set()),
            ("nan", 0, [ids[0], math.nan], set()),
            ("inf", 0, [ids[0], math.inf], set()),
        ]
        for name, index, replacement, excluded in changes:
            rows = list(baseline)
            rows[index] = replacement
            malformed.append((name, rows, excluded))
        for name, scores, excluded in malformed:
            with self.subTest(name=name):
                self.assert_f2_equivalent(scores, ids, excluded)
                with self.assertRaises(ResearchError) as raised:
                    _rank(scores, ids, excluded)
                self.assertIn(
                    str(raised.exception),
                    {"research_prediction_invalid", "research_integrity_error"},
                )


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

    def test_historical_f2_producer_rejected_before_payload_reads(self) -> None:
        migration = (
            Path(__file__).resolve().parents[2]
            / "docs/migrations/arb064-bounded-evaluation-v1.json"
        )
        record = cast("dict[str, object]", __import__("json").loads(migration.read_bytes()))
        changed = cast("dict[str, dict[str, str]]", record["changed_producer_source_sha256"])
        manifest_path = self.prepared / "prepare-manifest.json"
        value, _ = read_json(manifest_path)
        manifest = cast("dict[str, object]", value)
        producer = cast("dict[str, object]", manifest["producer"])
        fingerprints = cast("dict[str, str]", producer["code_files_sha256"])
        for path, digests in changed.items():
            fingerprints[path.removeprefix("src/recsys/")] = digests["from"]
        producer["package_version"] = "0.5.1"
        manifest_path.write_bytes(canonical_bytes(manifest))
        old_identity = identity(manifest_path)
        original = Path.open

        def guarded(
            path: Path,
            mode: str = "r",
            buffering: int = -1,
            encoding: str | None = None,
            errors: str | None = None,
            newline: str | None = None,
        ) -> IO[str] | IO[bytes]:
            if path.is_relative_to(self.prepared) and path.name != "prepare-manifest.json":
                raise AssertionError("historical preparation payload opened")
            return original(path, mode, buffering, encoding, errors, newline)

        with patch.object(Path, "open", guarded):
            with self.assertRaisesRegex(ResearchError, "^research_integrity_error$"):
                _prepared(self.prepared, old_identity)

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
