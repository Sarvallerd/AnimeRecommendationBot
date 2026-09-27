"""Synthetic comparison behavior and measurement validation."""
import hashlib
import json
import os
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

    def test_selected_worker_uses_real_vectors_and_records_timing(self):
        def record(title):
            return {"title": title, "aliases": [], "genres": [], "score": None,
                    "year": None, "type": None, "episodes": None, "synopsis": None}
        anime = {"1": record("Blue"), "2": record("Blue"), "3": record("Absent")}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            glove = root / "glove.6B.300d.txt"
            glove_raw = ("blue " + " ".join(["1"] + ["0"] * 299) + "\n").encode()
            glove.write_bytes(glove_raw)
            descriptor = {"id": "glove_300d", "size": len(glove_raw),
                          "sha256": hashlib.sha256(glove_raw).hexdigest()}
            pins = {"catalog": {"anime": anime}, "queries": [{"mal_id": 1}, {"mal_id": 3}],
                    "registry": {"sources": [descriptor]}}
            request = root / "request.json"
            output = root / "output.json"
            request.write_text(json.dumps({"method": "glove-selected", "paths": {
                name: str(glove if name == "glove" else root / name)
                for name in ("bundle_dir", "normalization_report", "build_report",
                             "anime_csv", "synopsis_csv", "glove", "lockfile")}}))
            thread_env = {name: "1" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                                 "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                                                 "NUMEXPR_NUM_THREADS")}
            with patch.dict(os.environ, thread_env), patch("recsys.evaluation._load_pins", return_value=pins):
                _worker(request, output)
            result = json.loads(output.read_text())
        rows = result["result"]["queries"]
        self.assertEqual(result["result"]["universe_count"], 3)
        self.assertEqual(rows[0], {"query_mal_id": 1, "status": "ok", "reasons": [],
                                   "recommendations": [{"rank": 1, "mal_id": 2, "score": 1.0}]})
        self.assertEqual(rows[1], {"query_mal_id": 3, "status": "oov_only",
                                   "reasons": [], "recommendations": []})
        self.assertEqual(result["result"]["extra"]["glove"]["sha256"], descriptor["sha256"])
        self.assertGreater(result["rank20_wall_ns"], 0)
        self.assertGreater(result["prepare_wall_ns"], 0)
        self.assertTrue(all(pool["num_threads"] == 1 for pool in result["effective_threadpools"]))

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
