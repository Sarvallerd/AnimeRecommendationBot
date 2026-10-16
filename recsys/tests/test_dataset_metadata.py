"""Metadata grammar, ambiguity and first-release rules."""

import csv
import tempfile
import unittest
from pathlib import Path

from recsys.data.metadata import DETAIL_COLUMNS, normalize_metadata


def row(mal, title, **values):
    result = {name: "" for name in DETAIL_COLUMNS}
    result.update(mal_id=str(mal), title=title, **values)
    return [result[name] for name in DETAIL_COLUMNS]


class MetadataTests(unittest.TestCase):
    def run_rows(self, rows):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "details.csv"
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(DETAIL_COLUMNS)
                writer.writerows(rows)
            return normalize_metadata(path)

    def test_field_rules_and_untrusted_lists(self):
        parsed = self.run_rows(
            [
                row(
                    "0001",
                    "",
                    title_japanese="  名前  ",
                    score="8.5",
                    episodes="12.0",
                    genres="[' Drama ', 'Drama', 'Comedy']",
                    start_date="2024-02-29T12:00:00",
                    year="2025.0",
                    status="Finished Airing",
                    synopsis=" A\n\nSource: MAL Rewrite ",
                ),
                row(
                    2,
                    "Two",
                    title_japanese="Two",
                    score="NaN",
                    episodes="12.5",
                    genres="__import__('os').system('false')",
                    start_date="2025-02-29",
                    synopsis=" No synopsis has been added for this series yet. ",
                ),
            ]
        )
        a = parsed.catalog["anime"]["1"]
        self.assertEqual((a["title"], a["aliases"], a["genres"]), ("名前", [], ["Drama", "Comedy"]))
        self.assertEqual((a["score"], a["episodes"], a["year"]), (8.5, 12, 2024))
        self.assertEqual(a["synopsis"], "A\n\nSource: MAL Rewrite")
        self.assertEqual(parsed.facts["anime"]["1"]["seasonal_year"], 2025)
        self.assertIsNone(parsed.catalog["anime"]["2"]["synopsis"])
        self.assertIsNone(parsed.catalog["anime"]["2"]["year"])
        self.assertEqual(parsed.qa["first_release_seasonal_year_disagreement"], 1)
        self.assertEqual(parsed.qa["synopsis_boilerplate"], 1)
        self.assertEqual(parsed.qa["invalid_genres"], 1)

    def test_empty_list_is_valid_and_nonempty_projection_keeps_order(self):
        parsed = self.run_rows(
            [row(1, "One", genres="[]"), row(2, "Two", genres="['Action', 'Drama', 'Action']")]
        )
        self.assertEqual(parsed.qa["empty_genres"], 1)
        self.assertEqual(parsed.qa.get("invalid_genres", 0), 0)
        self.assertEqual(parsed.catalog["anime"]["2"]["genres"], ["Action", "Drama"])

    def test_duplicate_normalized_ids_exclude_all_versions(self):
        parsed = self.run_rows(
            [row("01", "First"), row("1", "Second"), row(2, ""), row(3, "Three")]
        )
        self.assertEqual(set(parsed.catalog["anime"]), {"3"})
        self.assertEqual(parsed.qa["duplicate_mal_ids"], 1)
        self.assertEqual(parsed.qa["missing_title"], 1)

    def test_bad_header_and_extra_cell_fail_without_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "details.csv"
            path.write_text("wrong,header\nprivate-canary,1\n")
            with self.assertRaisesRegex(ValueError, "metadata_schema_invalid") as context:
                normalize_metadata(path)
            self.assertNotIn("private-canary", str(context.exception))


if __name__ == "__main__":
    unittest.main()
