"""Small regression fixtures for the repaired notebook comparator."""
import unittest

from recsys.legacy import (LegacyDataset, LegacyError, build_legacy_vectors,
                           cluster_legacy, legacy_neighbors, legacy_tokens,
                           prepare_legacy_rows)


class LegacyTests(unittest.TestCase):
    def row(self, mal_id, **changes):
        row = {"MAL_ID": str(mal_id), "Name": f"Name {mal_id}", "Score": "8.78",
               "Genres": "Action, Comedy", "English name": "Unknown", "Type": "TV",
               "Episodes": "26", "Aired": "Apr 3, 1998 to Apr 24, 1999",
               "Source": "Unknown", "Duration": "24 min. per ep.",
               "Rating": "PG-13 - Teens 13 or older", "Popularity": "39",
               "Favorites": "61971"}
        row.update(changes)
        return row

    def test_eligibility_and_notebook_scalars(self):
        rows = [self.row(3, Score="Unknown"), self.row(1, **{"English name": "English One"}),
                self.row(2, Name="Unknown", **{"English name": "Second"})]
        synopsis = [{"MAL_ID": "1", "sypnopsis": "No synopsis has been added for this series yet."},
                    {"MAL_ID": "2", "sypnopsis": "Unknown"},
                    {"MAL_ID": "3", "sypnopsis": "A story"}]
        dataset = prepare_legacy_rows(rows, synopsis)
        self.assertEqual(list(dataset.records), [1, 2])
        self.assertEqual(dataset.records[1]["Name"], "English One")
        self.assertEqual(dataset.records[2]["Name"], "Second")
        self.assertEqual(dataset.records[1]["Source"], "Unknown")
        self.assertEqual(dataset.records[1]["Duration"], 1440)
        self.assertEqual(dataset.records[1]["Rating"], 13)
        self.assertEqual(dataset.records[1]["year"], 1998)
        self.assertEqual(dataset.records[1]["season"], "spring")
        self.assertIs(type(dataset.records[1]["Score"]), float)
        self.assertIn("synopsis", " ".join(dataset.records[1]["sypnopsis"].split()))
        self.assertEqual(dataset.exclusions[3], ["Score"])
        tokens = legacy_tokens(dataset.records[1], 2)
        self.assertIn("score", tokens)
        self.assertIn("878", tokens)
        self.assertIn("cluster", tokens)
        self.assertIn("2", tokens)
        self.assertIn("sypnopsis", tokens)
        self.assertIn("action", tokens)
        self.assertIn("comedy", tokens)

    def test_duplicate_id_rejected_and_order_independent(self):
        rows = [self.row(1), self.row(2)]
        syn = [{"MAL_ID": str(i), "sypnopsis": "Story"} for i in (1, 2)]
        self.assertEqual(prepare_legacy_rows(rows, syn).records,
                         prepare_legacy_rows(rows[::-1], syn[::-1]).records)
        with self.assertRaisesRegex(LegacyError, "duplicate MAL_ID"):
            prepare_legacy_rows(rows + [self.row(1)], syn)

    def test_real_kmeans_is_deterministic(self):
        from recsys.legacy import FEATURES
        rows = [self.row(i, Genres=FEATURES[i - 1]) for i in range(1, 22)]
        syn = [{"MAL_ID": str(i), "sypnopsis": "Story"} for i in range(1, 22)]
        dataset = prepare_legacy_rows(rows, syn)
        first, diagnostics = cluster_legacy(dataset)
        second, _ = cluster_legacy(dataset)
        self.assertEqual(first, second)
        self.assertEqual(len(diagnostics["feature_order"]), 41)

    def test_occurrence_sum_oov_and_canceling_zero(self):
        import numpy as np
        dataset = prepare_legacy_rows([self.row(1, Name="X")],
                                      [{"MAL_ID": "1", "sypnopsis": "Story"}])
        label = {1: 0}
        ids, vectors, info = build_legacy_vectors(dataset, label, {})
        self.assertEqual(info["status_counts"]["oov_only"], 1)
        positive = np.zeros(300, dtype=np.float64)
        positive[0] = 1
        negative = np.zeros(300, dtype=np.float64)
        negative[0] = -1
        _, vectors, info = build_legacy_vectors(dataset, label,
                                                {"score": positive, "type": negative})
        self.assertEqual(info["status_counts"]["zero_vector"], 1)
        self.assertEqual(float(vectors[0, 0]), 0.0)
        _, vectors, info = build_legacy_vectors(dataset, label, {"score": positive})
        self.assertEqual(ids, [1])
        self.assertEqual(info["status_counts"]["ready"], 1)
        self.assertEqual(float(vectors[0, 0]), 1.0)

    def test_top99_before_cluster_and_score_year_ties(self):
        import numpy as np
        records = {i: {"Score": 8.0, "year": 2000} for i in range(1, 103)}
        dataset = LegacyDataset(records, {}, {}, {})
        ids = list(records)
        vectors = np.zeros((102, 300), dtype=np.float64)
        vectors[:, 0] = 1.0
        labels = {i: 0 if i in (1, 101, 102) else 1 for i in ids}
        result = legacy_neighbors(dataset, ids, vectors, labels, 1)
        self.assertEqual(result["status"], "no_candidates_after_cluster")
        labels[100] = 0
        result = legacy_neighbors(dataset, ids, vectors, labels, 1)
        self.assertEqual([r["mal_id"] for r in result["recommendations"]], [100])
        self.assertEqual(result["recommendations"][0]["rank"], 1)


if __name__ == "__main__":
    unittest.main()
