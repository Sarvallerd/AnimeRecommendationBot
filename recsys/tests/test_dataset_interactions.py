"""Score-state, grouping, cohort and pseudonym behavior on small histories."""

import csv
import gzip
import hashlib
import tempfile
import traceback
import unittest
from pathlib import Path

from recsys.dataset_interactions import CSV_COLUMNS, build_interactions, user_key
from recsys.dataset_metadata import MetadataResult


def _metadata(n=16):
    entries = {str(i): {"title": str(i), "aliases": [], "genres": [], "score": None,
                        "year": None, "type": None, "episodes": None, "synopsis": None}
               for i in range(1, n + 1)}
    facts = {str(i): {"first_release_date": None, "airing_status": "unknown",
                      "seasonal_year": None} for i in range(1, n + 1)}
    return MetadataResult({"schema_version": 1, "anime": entries},
                          {"schema_version": 1, "anime": facts}, {})


class InteractionTests(unittest.TestCase):
    def run_rows(self, rows, metadata=None, threads=2):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "ratings.csv"
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(CSV_COLUMNS)
                for user, mal, status, score in rows:
                    writer.writerow((user, mal, status, score, "False", "1"))
            output = root / "interactions.tsv.gz"
            result = build_interactions(path, metadata or _metadata(), output,
                                        root / "work", {"memory_limit_mib": 256,
                                        "threads": threads, "spill_limit_mib": 512})
            self.assertEqual((root / "work").stat().st_mode & 0o777, 0o700)
            self.assertEqual((root / "work" / "ratings.duckdb").stat().st_mode & 0o777,
                             0o600)
            return result, output.read_bytes(), gzip.decompress(output.read_bytes())

    def test_all_pair_states_and_mapping_denominator(self):
        rows = [
            ("same", "1", "completed", "8"),
            ("same", "1", "completed", "8"),
            ("mixed", "2", "completed", "9"),
            ("mixed", "2", "watching", "0"),
            ("conflict", "3", "completed", "8"),
            ("conflict", "3", "completed", "9"),
            ("invalid", "4", "completed", "8"),
            ("invalid", "4", "completed", "-1"),
            ("precision", "4", "completed", "8.0000000000000001"),
            ("unrated", "5", "plan_to_watch", "0"),
            ("missing", "6", "completed", "10"),
            ("statuses", "7", "completed", "7"),
            ("statuses", "7", "watching", "7"),
        ]
        result, _, plain = self.run_rows(rows, _metadata(7))
        self.assertEqual(result.qa["counts"]["raw_explicit_pairs"], 6)
        self.assertEqual(result.qa["counts"]["mixed_unset_pairs"], 1)
        self.assertEqual(result.qa["counts"]["conflicting_score_pairs"], 1)
        self.assertEqual(result.qa["counts"]["invalid_state_pairs"], 1)
        self.assertEqual(result.qa["counts"]["invalid_score_rows"], 2)
        self.assertEqual(result.qa["counts"]["status_conflict_pairs"], 1)
        self.assertEqual(result.qa["counts"]["repeated_explicit_triples"], 2)
        self.assertEqual(result.rows, 3)
        self.assertIn(b"\t7\t7\tunknown\n", plain)
        self.assertNotIn(b"mixed", plain)
        self.assertNotIn(b"same", plain)

    def test_14_vs_15_distinct_positive_and_unicode_namespace(self):
        rows = [("Пользователь", str(i), "completed", "8") for i in range(1, 16)]
        rows += [("other", str(i), "completed", "8") for i in range(1, 15)]
        result, first, plain = self.run_rows(rows, threads=1)
        repeated, second, _ = self.run_rows(list(reversed(rows)), threads=2)
        self.assertEqual(first, second)
        self.assertEqual(result.qa["counts"]["positive_users_ge_15"], 1)
        self.assertEqual(result.qa["distributions"]["positive_items_per_user_bins"],
                         {"10": 1, "15": 1})
        self.assertIn(user_key("Пользователь").encode("ascii"), plain)
        self.assertNotIn("Пользователь".encode(), plain)
        self.assertNotEqual(user_key("A"), user_key("a"))
        self.assertEqual(user_key("A"), "u1_" + hashlib.sha256(
            b"arb-user-v1\0neelagiri-2025-v1\0A").hexdigest())

    def test_csv_failure_does_not_expose_private_row(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "ratings.csv"
            source.write_text(",".join(CSV_COLUMNS) + "\n"
                              "private-canary-username,1,completed,8,False,1,extra\n")
            with self.assertRaisesRegex(ValueError, "ratings_csv_invalid") as context:
                build_interactions(source, _metadata(), root / "out.gz", root / "work",
                                   {"memory_limit_mib": 256, "threads": 1,
                                    "spill_limit_mib": 512})
            self.assertNotIn("private-canary-username", str(context.exception))
            self.assertNotIn("private-canary-username", "".join(
                traceback.format_exception(context.exception)))

    def test_exact_csv_header_rejects_renamed_reordered_and_duplicate_columns(self):
        columns = list(CSV_COLUMNS)
        cases = {
            "renamed": ("private-canary-column", *columns[1:]),
            "reordered": (columns[1], columns[0], *columns[2:]),
            "duplicate": (*columns[:-1], columns[0]),
        }
        for label, fields in cases.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / "ratings.csv"
                source.write_bytes((",".join(fields) + "\r\n" +
                                    "private-canary-username,1,completed,8,False,1\r\n"
                                    ).encode("utf-8"))
                with self.assertRaisesRegex(ValueError, "^ratings_csv_invalid$") as context:
                    build_interactions(source, _metadata(), root / "out.gz", root / "work",
                                       {"memory_limit_mib": 256, "threads": 1,
                                        "spill_limit_mib": 512})
                diagnostic = "".join(traceback.format_exception(context.exception))
                self.assertNotIn("private-canary", diagnostic)
                self.assertFalse((root / "out.gz").exists())
                self.assertFalse((root / "work").exists())


if __name__ == "__main__":
    unittest.main()
