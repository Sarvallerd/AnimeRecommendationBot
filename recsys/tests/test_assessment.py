"""Blinding and fixed-denominator assessment checks."""
import copy
import unittest

from recsys.assessment import (AssessmentError, make_packet, make_template,
                               score_method, validate_assessment)


class AssessmentTests(unittest.TestCase):
    def fixture(self):
        catalog = {"anime": {str(i): {"title": f"Anime {i}", "aliases": [],
                                      "genres": ["Drama"], "type": "TV", "episodes": 12,
                                      "year": 2020, "synopsis": f"Story {i}"}
                             for i in (1, 2, 3)}}
        comparison = {"catalog_sha256": "a" * 64, "query_set_sha256": "b" * 64,
                      "methods": [{"queries": [{"query_mal_id": 1,
                                                 "recommendations": [{"mal_id": 3, "score": 0.8},
                                                                     {"mal_id": 2, "score": 0.7}]}]},
                                  {"queries": [{"query_mal_id": 1,
                                                "recommendations": [{"mal_id": 2, "score": 0.6}]}]}]}
        return make_packet(comparison, catalog)

    def test_packet_blind_union_and_template_not_complete(self):
        packet = self.fixture()
        self.assertEqual(packet["pairs"], [{"query_mal_id": 1, "candidate_mal_id": 2},
                                           {"query_mal_id": 1, "candidate_mal_id": 3}])
        self.assertEqual(set(packet["anime"]["1"]),
                         {"title", "aliases", "genres", "type", "episodes", "year", "synopsis"})
        self.assertNotIn("score", str(packet))
        with self.assertRaises(AssessmentError):
            validate_assessment(make_template(packet), packet)

    def test_complete_assessment_checks_evidence_and_pair_coverage(self):
        packet = self.fixture()
        value = make_template(packet)
        value["assessor"] = {"id": "reviewer", "kind": "llm", "method_version": "v1",
                             "model": "test-model", "prompt_version": packet["prompt_version"],
                             "prompt_sha256": packet["prompt_sha256"]}
        for row in value["judgments"]:
            row.update(relevance=1, rationale="Related narrative.",
                       evidence=[{"side": "query", "field": "synopsis", "quote": "Story 1"},
                                 {"side": "candidate", "field": "synopsis",
                                  "quote": f"Story {row['candidate_mal_id']}"}],
                       franchise_rationale="Insufficient evidence.")
        self.assertEqual(validate_assessment(value, packet), "reviewer")
        wrong = copy.deepcopy(value)
        wrong["judgments"][0]["evidence"][0]["quote"] = "Invented"
        with self.assertRaises(AssessmentError):
            validate_assessment(wrong, packet)
        wrong = copy.deepcopy(value)
        wrong["judgments"][0]["relevance"] = True
        with self.assertRaises(AssessmentError):
            validate_assessment(wrong, packet)
        wrong = copy.deepcopy(value)
        wrong["judgments"][0] = wrong["judgments"][1]
        with self.assertRaises(AssessmentError):
            validate_assessment(wrong, packet)

    def test_unknown_returned_blocks_point_but_bounds_include_it(self):
        queries = [{"query_mal_id": i, "status": "ok", "recommendations": []}
                   for i in range(1, 21)]
        queries[0]["recommendations"] = [{"mal_id": 22}, {"mal_id": 23}]
        method = {"id": "test", "queries": queries}
        judgments = [{"query_mal_id": 1, "candidate_mal_id": 22,
                      "relevance": 2, "same_franchise": True},
                     {"query_mal_id": 1, "candidate_mal_id": 23,
                      "relevance": None, "same_franchise": None}]
        result = score_method(method, judgments)
        row = result["queries"][0]
        self.assertIsNone(row["mean_grade_at_5"])
        self.assertEqual(row["mean_grade_bounds"], [0.4, 0.8])
        self.assertEqual(row["precision_bounds"], [0.2, 0.4])
        self.assertIsNone(result["macro"]["mean_grade_at_5"])
        self.assertEqual(result["macro"]["mean_grade_bounds"], [0.02, 0.04])
        self.assertEqual(result["macro"]["returned"], 2)
        self.assertEqual(result["macro"]["missing_slots"], 98)


if __name__ == "__main__":
    unittest.main()
