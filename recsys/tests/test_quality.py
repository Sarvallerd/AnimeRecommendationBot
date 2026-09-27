"""Diagnostic baseline checks using small synthetic catalogs."""

import copy
import hashlib
import json
import tempfile
import unittest
from importlib import resources
from pathlib import Path

from recsys.quality import QualityError, genre_neighbors, run_baseline


def canonical(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":")) + "\n").encode("utf-8")


class QualityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.catalog = {"schema_version": 1, "anime": {
            str(mal_id): {"title": f"Title {mal_id}", "genres": ["Action"]}
            for mal_id in range(1, 21)}}
        self.spec = json.loads(resources.files("recsys").joinpath("quality_queries_v1.json").read_bytes())
        self.spec["query_set_version"] = "fixture-v1"
        self.spec["queries"] = [
            {"mal_id": mal_id, "title": f"Title {mal_id}",
             "popularity_rank": (1, 101, 1001, 5001)[(mal_id - 1) // 5],
             "members": 20 - mal_id,
             "popularity_band": ("1-100", "101-1000", "1001-5000", "5001+")[(mal_id - 1) // 5],
             "rationale": f"Fixture case {mal_id}"}
            for mal_id in range(1, 21)]
        self._save_catalog()

    def _save_catalog(self):
        raw = canonical(self.catalog)
        self.path = self.root / "catalog.json"
        self.path.write_bytes(raw)
        self.spec["catalog_sha256"] = hashlib.sha256(raw).hexdigest()
        self.spec["catalog_count"] = len(self.catalog["anime"])

    def test_jaccard_fraction_ties_self_limit_and_distinct_duplicate_titles(self):
        anime = {
            "1": {"title": "Same", "genres": ["A", "B"]},
            "2": {"title": "Same", "genres": ["A"]},
            "10": {"title": "Ten", "genres": ["B"]},
            "3": {"title": "Three", "genres": ["A", "B", "C"]},
            "4": {"title": "Four", "genres": ["Z"]},
            "5": {"title": "Five", "genres": ["A"]},
            "6": {"title": "Six", "genres": ["A"]},
            "7": {"title": "Seven", "genres": ["A"]},
            "8": {"title": "Eight", "genres": []},
        }
        result = genre_neighbors(anime, 1)
        self.assertEqual([item["mal_id"] for item in result], [3, 2, 5, 6, 7])
        self.assertEqual([item["score"] for item in result], [2/3, 1/2, 1/2, 1/2, 1/2])
        self.assertNotIn(1, [item["mal_id"] for item in result])
        self.assertEqual([item["mal_id"] for item in genre_neighbors(anime, 1, 7)],
                         [3, 2, 5, 6, 7, 10])
        self.assertEqual(genre_neighbors(anime, 8), [])
        self.assertEqual(genre_neighbors(anime, 4), [])
        self.assertEqual(genre_neighbors({"1": anime["1"]}, 1), [])
        self.assertEqual([item["mal_id"] for item in genre_neighbors(
            {key: anime[key] for key in ("1", "10", "2")}, 1)], [2, 10])

    def test_outputs_are_deterministic_linked_and_unjudged(self):
        self.catalog["anime"]["2"]["title"] = "Title 1"
        self.spec["queries"][1]["title"] = "Title 1"
        self._save_catalog()
        first = run_baseline(self.path, self.root / "a", query_set=self.spec)
        second = run_baseline(self.path, self.root / "b", query_set=self.spec)
        baseline_raw = first.read_bytes()
        template_raw = (first.parent / "relevance-template.json").read_bytes()
        self.assertEqual(baseline_raw, second.read_bytes())
        self.assertEqual(template_raw, (second.parent / "relevance-template.json").read_bytes())
        self.assertEqual(baseline_raw, canonical(json.loads(baseline_raw)))
        self.assertEqual(template_raw, canonical(json.loads(template_raw)))
        baseline = json.loads(baseline_raw)
        template = json.loads(template_raw)
        self.assertEqual(len(baseline["queries"]), 20)
        self.assertEqual(sum(len(row["recommendations"]) for row in baseline["queries"]), 100)
        self.assertEqual(len(template["judgments"]), 100)
        self.assertEqual(template["runs"], [{"method_id": "genre-jaccard",
                                             "sha256": hashlib.sha256(baseline_raw).hexdigest()}])
        self.assertEqual(baseline["query_set_sha256"], hashlib.sha256(canonical(self.spec)).hexdigest())
        self.assertEqual([(row["query_mal_id"], row["candidate_mal_id"])
                          for row in template["judgments"]],
                         sorted((row["query_mal_id"], row["candidate_mal_id"])
                                for row in template["judgments"]))
        for row in template["judgments"]:
            self.assertIsNone(row["relevance"])
            self.assertEqual(row["assessor_kind"], "unjudged")
            self.assertIsNone(row["assessor_id"])
            self.assertIsNone(row["method_version"])
            self.assertIsNone(row["rationale"])
        self.assertEqual(template["judgments"][0]["candidate_title"], "Title 1")

    def test_empty_genres_and_no_positive_candidates_keep_short_lists(self):
        self.catalog["anime"]["1"]["genres"] = []
        self.catalog["anime"]["2"]["genres"] = ["Mystery"]
        self._save_catalog()
        path = run_baseline(self.path, self.root / "short", query_set=self.spec)
        baseline = json.loads(path.read_bytes())
        template = json.loads((path.parent / "relevance-template.json").read_bytes())
        self.assertEqual(baseline["queries"][0]["status"], "empty_genres")
        self.assertEqual(baseline["queries"][1]["status"], "no_positive_candidates")
        self.assertEqual(baseline["queries"][0]["recommendations"], [])
        self.assertEqual(baseline["queries"][1]["recommendations"], [])
        self.assertEqual(len(template["judgments"]), 90)
        self.assertTrue(all(len(row["recommendations"]) == 5
                            for row in baseline["queries"][2:]))

    def test_invalid_specs_and_catalog_fail_before_writes(self):
        cases = []
        duplicate = copy.deepcopy(self.spec)
        duplicate["queries"][1]["mal_id"] = 1
        cases.append(duplicate)
        missing = copy.deepcopy(self.spec)
        missing["queries"].pop()
        cases.append(missing)
        wrong_band = copy.deepcopy(self.spec)
        wrong_band["queries"][0]["popularity_band"] = "5001+"
        cases.append(wrong_band)
        for invalid_band in ([], {}):
            wrong_band_type = copy.deepcopy(self.spec)
            wrong_band_type["queries"][0]["popularity_band"] = invalid_band
            cases.append(wrong_band_type)
        wrong_title = copy.deepcopy(self.spec)
        wrong_title["queries"][0]["title"] = "Another title"
        cases.append(wrong_title)
        negative_members = copy.deepcopy(self.spec)
        negative_members["queries"][0]["members"] = -1
        cases.append(negative_members)
        wrong_provenance = copy.deepcopy(self.spec)
        wrong_provenance["source_provenance"]["anime_csv_sha256"] = "0" * 64
        cases.append(wrong_provenance)
        for index, spec in enumerate(cases):
            with self.subTest(index=index), self.assertRaises(QualityError):
                run_baseline(self.path, self.root / f"bad-{index}", query_set=spec)
            self.assertFalse((self.root / f"bad-{index}").exists())
        self.path.write_bytes(self.path.read_bytes() + b" ")
        with self.assertRaisesRegex(QualityError, "SHA256 mismatch"):
            run_baseline(self.path, self.root / "bad-hash", query_set=self.spec)
        self.assertFalse((self.root / "bad-hash").exists())

    def test_packaged_spec_provenance_and_frozen_ids(self):
        raw = resources.files("recsys").joinpath("quality_queries_v1.json").read_bytes()
        spec = json.loads(raw)
        self.assertEqual(raw, canonical(spec))
        self.assertEqual([row["mal_id"] for row in spec["queries"]],
                         [1, 20, 30, 199, 1535, 457, 440, 5680, 877, 10162,
                          1065, 326, 885, 16664, 111, 35102, 1391, 3258, 1550, 39619])
        self.assertEqual(spec["catalog_sha256"],
                         "2ae604168ba10be67cca14cd8cba0235fc6f0d0de19fdc25b9fc91175ffd06d2")
        self.assertEqual(spec["registry_sha256"],
                         "49bb399935f0ac7ec649fe2f4bd7715444c09c8513704866fa90b9bea29d182f")
        self.assertEqual(spec["source_provenance"]["mal_repository_commit"],
                         "9a1d7f56482accbde99e24cd7cfcad65e2d24b1f")
        self.assertEqual(spec["source_provenance"]["anime_csv_sha256"],
                         "6e6d723f3e021e084a281d1ec557a0a10dfb9721d256b87596e02c27eb1f77e9")
        self.assertEqual(spec["source_provenance"]["synopsis_csv_sha256"],
                         "fc5065338f1d66063cc6b6ed328701c3e23118110689daa852d5876911674e33")


if __name__ == "__main__":
    unittest.main()
