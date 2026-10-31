"""Offline fixtures for the AniList relations snapshot: normalization, merging and the cache."""

from __future__ import annotations

import contextlib
import gzip
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from fetch_anilist_relations import FetchError, group_media, main, normalize  # noqa: E402


def media(
    mal_id: int | None, start: list[int | None], edges: list[tuple[str, int | None, str]]
) -> dict[str, object]:
    year, month, day = start
    return {
        "idMal": mal_id,
        "format": "TV",
        "startDate": {"year": year, "month": month, "day": day},
        "relations": {
            "edges": [
                {"relationType": kind, "node": {"idMal": target, "type": node_type}}
                for kind, target, node_type in edges
            ]
        },
    }


class NormalizeTests(unittest.TestCase):
    def test_keeps_anime_relations_inside_and_outside_the_catalog(self) -> None:
        item = media(
            1,
            [1998, 10, None],
            [
                ("SEQUEL", 2, "ANIME"),
                ("SEQUEL", 2, "ANIME"),
                ("ADAPTATION", 30, "MANGA"),
                ("ALTERNATIVE", 1, "ANIME"),
                ("SIDE_STORY", None, "ANIME"),
                ("SPIN_OFF", 500, "ANIME"),
            ],
        )
        found = group_media([item, media(None, [None] * 3, []), media(9, [None] * 3, [])], {1, 2})
        self.assertEqual(list(found), [1])
        self.assertEqual(
            normalize(1, found[1]),
            {
                "mal_id": 1,
                "format": "TV",
                "start_date": [1998, 10, None],
                "relations": [["SEQUEL", 2], ["SPIN_OFF", 500]],
            },
        )

    def test_merges_entries_that_share_one_mal_id(self) -> None:
        later = media(7, [2013, 5, 23], [])
        earlier = media(7, [2012, 5, 23], [("PARENT", 3, "ANIME")])
        earlier["format"] = "ONA"
        record = normalize(7, group_media([later, earlier], {7})[7])
        self.assertEqual(record["format"], "ONA")
        self.assertEqual(record["start_date"], [2012, 5, 23])
        self.assertEqual(record["relations"], [["PARENT", 3]])


class SnapshotTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.catalog = self.root / "catalog.json"
        self.catalog.write_text(json.dumps({"anime": {"1": {}, "2": {}}}), encoding="utf-8")
        self.cache = self.root / "cache"
        self.cache.mkdir()
        batch = [media(2, [2001, 1, 1], [("PREQUEL", 1, "ANIME")]), media(1, [1999, 1, 1], [])]
        (self.cache / "batch-00000.json").write_text(json.dumps(batch), encoding="utf-8")

    def run_main(self) -> int:
        arguments = ["--catalog", str(self.catalog), "--cache", str(self.cache)]
        with contextlib.redirect_stdout(io.StringIO()):
            return main([*arguments, "--output", str(self.root / "out")])

    def test_builds_a_sorted_reproducible_snapshot_from_the_cache(self) -> None:
        self.assertEqual(self.run_main(), 0)
        raw = (self.root / "out" / "relations.jsonl.gz").read_bytes()
        lines = gzip.decompress(raw).decode().splitlines()
        self.assertEqual([json.loads(line)["mal_id"] for line in lines], [1, 2])
        manifest = json.loads((self.root / "out" / "source-manifest.json").read_bytes())
        self.assertEqual((manifest["found_mal_ids"], manifest["relations"]), (2, 1))
        self.assertEqual(self.run_main(), 0)
        self.assertEqual((self.root / "out" / "relations.jsonl.gz").read_bytes(), raw)

    def test_refuses_a_cache_filled_for_another_catalog(self) -> None:
        self.assertEqual(self.run_main(), 0)
        self.catalog.write_text(json.dumps({"anime": {"1": {}, "3": {}}}), encoding="utf-8")
        with self.assertRaisesRegex(FetchError, "another catalog"):
            self.run_main()


if __name__ == "__main__":
    unittest.main()
