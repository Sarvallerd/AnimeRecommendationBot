"""Synthetic comparison behavior and measurement validation."""
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from recsys.evaluation import (_baseline_queries, _performance, _worker, compare, EvaluationError)
from recsys.quality import genre_neighbors


class EvaluationTests(unittest.TestCase):
    def test_jaccard_adapter_preserves_baseline_scores_and_order(self):
        anime = {"1": {"genres": ["Action", "Drama"]},
                 "2": {"genres": ["Drama"]}, "3": {"genres": ["Action", "Drama"]},
                 "4": {"genres": []}}
        ids, rows = _baseline_queries(anime, [{"mal_id": 1}, {"mal_id": 4}])
        self.assertEqual(ids, [1, 2, 3, 4])
        self.assertEqual([{"mal_id": x["mal_id"], "score": x["score"]}
                          for x in rows[0]["recommendations"]], genre_neighbors(anime, 1))
        self.assertEqual(rows[1]["status"], "empty_genres")

    def test_selected_worker_calls_adapter_and_records_timing(self):
        pins = {"catalog": {"anime": {"1": {"genres": ["Drama"]}}},
                "queries": [{"mal_id": 1}],
                "registry": {"sources": [{"id": "glove_300d", "size": 1, "sha256": "a" * 64}]}}
        rows = [{"query_mal_id": 1, "status": "no_positive_candidates",
                 "reasons": [], "recommendations": []}]
        def selected(anime, queries, glove, descriptor):
            self.assertEqual(anime, pins["catalog"]["anime"])
            self.assertEqual(queries, pins["queries"])
            self.assertEqual(descriptor["id"], "glove_300d")
            return [1], rows, {"size": 1, "sha256": "a" * 64}, 5
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            request = root / "request.json"
            output = root / "output.json"
            request.write_text(json.dumps({"method": "glove-selected", "paths": {
                name: str(root / name) for name in ("bundle_dir", "normalization_report",
                                                     "build_report", "anime_csv", "synopsis_csv",
                                                     "glove", "lockfile")}}))
            with patch("recsys.evaluation._load_pins", return_value=pins), \
                 patch("recsys.evaluation._selected_queries", side_effect=selected):
                _worker(request, output)
            result = json.loads(output.read_text())
        self.assertEqual(result["result"]["queries"], rows)
        self.assertEqual(result["rank20_wall_ns"], 5)
        self.assertGreater(result["prepare_wall_ns"], 0)

    def test_measurements_need_valid_units_and_complete_repetitions(self):
        comparison_raw = b"{}\n"
        import hashlib
        entry = {"repetition": 1, "exit_code": 0, "end_to_end_wall_ns": 20,
                 "prepare_wall_ns": 10, "rank20_wall_ns": 5, "peak_rss_bytes": 100, "output_sha256": "a" * 64,
                 "effective_threadpools": []}
        measurements = {"schema_version": 1, "protocol_version": "arb018-v1",
                        "comparison_sha256": hashlib.sha256(comparison_raw).hexdigest(),
                        "scope": "fresh-process-prepare-and-rank20-v1",
                        "schedule": [{"repetition": 1, "order": ["glove-selected", "genre-jaccard", "legacy-repaired"]}],
                        "workers": [{"method": m, **entry} for m in
                                    ("glove-selected", "genre-jaccard", "legacy-repaired")]}
        result = _performance(measurements, comparison_raw)
        self.assertEqual(result["glove-selected"]["rank20_wall_ns"]["median"], 5)
        measurements["workers"][0]["peak_rss_bytes"] = -1
        with self.assertRaises(EvaluationError):
            _performance(measurements, comparison_raw)

    def test_bad_repetition_never_creates_output(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "out"
            with self.assertRaisesRegex(EvaluationError, "repetitions"):
                compare(Path("missing"), Path("missing"), Path("missing"), Path("missing"),
                        Path("missing"), Path("missing"), Path("missing"), "a" * 40,
                        out, repetitions=0)
            self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
