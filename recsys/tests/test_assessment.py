"""Blinding and fixed-denominator assessment checks."""

import copy
import unittest

from recsys.contracts.bundle import Catalog
from recsys.historical.assessment import (
    AssessmentError,
    AssessmentJudgment,
    AssessmentPacket,
    HistoricalComparison,
    HistoricalMethod,
    make_packet,
    make_template,
    score_method,
    validate_assessment,
)


class AssessmentTests(unittest.TestCase):
    def fixture(self) -> AssessmentPacket:
        catalog: Catalog = {
            "schema_version": 1,
            "anime": {
                str(i): {
                    "title": f"Anime {i}",
                    "aliases": [],
                    "genres": ["Drama"],
                    "score": 8.0,
                    "type": "TV",
                    "episodes": 12,
                    "year": 2020,
                    "synopsis": f"Story {i}",
                }
                for i in (1, 2, 3)
            },
        }
        comparison: HistoricalComparison = {
            "catalog_sha256": "a" * 64,
            "query_set_sha256": "b" * 64,
            "methods": [
                {
                    "id": "first",
                    "queries": [
                        {
                            "query_mal_id": 1,
                            "status": "ok",
                            "recommendations": [
                                {"mal_id": 3, "rank": 1, "score": 0.8},
                                {"mal_id": 2, "rank": 2, "score": 0.7},
                            ],
                        }
                    ],
                },
                {
                    "id": "second",
                    "queries": [
                        {
                            "query_mal_id": 1,
                            "status": "ok",
                            "recommendations": [{"mal_id": 2, "rank": 1, "score": 0.6}],
                        }
                    ],
                },
            ],
        }
        return make_packet(comparison, catalog)

    def test_packet_blind_union_and_template_not_complete(self) -> None:
        packet = self.fixture()
        self.assertEqual(
            packet["pairs"],
            [
                {"query_mal_id": 1, "candidate_mal_id": 2},
                {"query_mal_id": 1, "candidate_mal_id": 3},
            ],
        )
        self.assertEqual(
            set(packet["anime"]["1"]),
            {"title", "aliases", "genres", "type", "episodes", "year", "synopsis"},
        )
        self.assertNotIn("score", str(packet))
        with self.assertRaises(AssessmentError):
            validate_assessment(make_template(packet), packet)

    def test_complete_assessment_checks_evidence_and_pair_coverage(self) -> None:
        packet = self.fixture()
        value = make_template(packet)
        value["assessor"] = {
            "id": "reviewer",
            "kind": "llm",
            "method_version": "v1",
            "model": "test-model",
            "prompt_version": packet["prompt_version"],
            "prompt_sha256": packet["prompt_sha256"],
        }
        judgments = value["judgments"]
        assert isinstance(judgments, list)
        for row in judgments:
            assert isinstance(row, dict)
            row.update(
                relevance=1,
                rationale="Related narrative.",
                evidence=[
                    {"side": "query", "field": "synopsis", "quote": "Story 1"},
                    {
                        "side": "candidate",
                        "field": "synopsis",
                        "quote": f"Story {row['candidate_mal_id']}",
                    },
                ],
                franchise_rationale="Insufficient evidence.",
            )
        self.assertEqual(validate_assessment(value, packet), "reviewer")
        wrong = copy.deepcopy(value)
        wrong_rows = wrong["judgments"]
        assert isinstance(wrong_rows, list) and isinstance(wrong_rows[0], dict)
        evidence = wrong_rows[0]["evidence"]
        assert isinstance(evidence, list) and isinstance(evidence[0], dict)
        evidence[0]["quote"] = "Invented"
        with self.assertRaises(AssessmentError):
            validate_assessment(wrong, packet)
        wrong = copy.deepcopy(value)
        wrong_rows = wrong["judgments"]
        assert isinstance(wrong_rows, list) and isinstance(wrong_rows[0], dict)
        wrong_rows[0]["relevance"] = True
        with self.assertRaises(AssessmentError):
            validate_assessment(wrong, packet)
        wrong = copy.deepcopy(value)
        wrong_rows = wrong["judgments"]
        assert isinstance(wrong_rows, list)
        wrong_rows[0] = wrong_rows[1]
        with self.assertRaises(AssessmentError):
            validate_assessment(wrong, packet)

    def test_unknown_returned_blocks_point_but_bounds_include_it(self) -> None:
        method: HistoricalMethod = {
            "id": "test",
            "queries": [
                {"query_mal_id": i, "status": "ok", "recommendations": []} for i in range(1, 21)
            ],
        }
        method["queries"][0]["recommendations"] = [
            {"mal_id": 22, "rank": 1, "score": 0.8},
            {"mal_id": 23, "rank": 2, "score": 0.7},
        ]
        judgments: list[AssessmentJudgment] = [
            {
                "query_mal_id": 1,
                "candidate_mal_id": candidate_id,
                "relevance": relevance,
                "rationale": "Fixture",
                "evidence": [],
                "same_franchise": same_franchise,
                "franchise_rationale": "Fixture",
            }
            for candidate_id, relevance, same_franchise in (
                (22, 2, True),
                (23, None, None),
            )
        ]
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
