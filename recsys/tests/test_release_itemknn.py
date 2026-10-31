"""ItemKNN bundle derivation: research parity, fallback, provenance and rejected inputs."""

import gzip
import hashlib
import json
import shutil
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.sparse import csr_matrix

from recsys.contracts.bundle import ContractError, check_bundle
from recsys.release.itemknn import (
    ALGORITHM,
    _as_first_entries,
    derive_current,
    derive_itemknn_bundle,
    franchise_groups,
    positive_matrix,
    top_neighbors,
)
from recsys.research.models.collaborative import _fit_knn

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "bundle"
HEADER = "dataset_user_key\tmal_id\trating\twatch_status\n"
# Users liking 1 also like 10; 2 and 3 are liked together; 20 and 99 have no positive rating.
ROWS = [
    ("u1_a", 1, "9", "completed"),
    ("u1_a", 10, "8", "completed"),
    ("u1_a", 20, "7", "completed"),
    ("u1_b", 1, "10", "completed"),
    ("u1_b", 10, "9", "completed"),
    ("u1_b", 11, "8", "completed"),
    ("u1_c", 2, "8", "completed"),
    ("u1_c", 3, "9", "completed"),
    ("u1_c", 99, "", "watching"),
    ("u1_d", 2, "9", "completed"),
    ("u1_d", 3, "10", "dropped"),
    ("u1_d", 12, "8", "completed"),
]


# 1, 10 and 11 form one franchise whose first TV entry is 10; the 2-3 spin-off does not link.
RELATIONS = [
    {"mal_id": 1, "format": "MOVIE", "start_date": [1980, 1, 1], "relations": [["SEQUEL", 10]]},
    {"mal_id": 2, "format": "TV", "start_date": [2001, None, None], "relations": [["SPIN_OFF", 3]]},
    {"mal_id": 10, "format": "TV", "start_date": [1990, 4, 1], "relations": [["PREQUEL", 1]]},
    {"mal_id": 11, "format": "TV", "start_date": [1995, 1, 1], "relations": [["SIDE_STORY", 10]]},
]


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def write_relations(directory: Path, records: list[dict[str, object]]) -> str:
    directory.mkdir()
    lines = "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    raw = gzip.compress(lines.encode("utf-8"), mtime=0)
    (directory / "relations.jsonl.gz").write_bytes(raw)
    return sha(raw)


def compress(records: list[dict[str, object]]) -> bytes:
    return gzip.compress("".join(json.dumps(r) + "\n" for r in records).encode(), mtime=0)


def representative(records: list[dict[str, object]], watchers: list[int]) -> int:
    """Return the MAL ID standing for the franchise of the first record."""
    ids = sorted(int(str(record["mal_id"])) for record in records)
    groups, representatives, _ = franchise_groups(compress(records), ids, watchers)
    return ids[representatives[groups[ids.index(int(str(records[0]["mal_id"])))]]]


def write_dataset(directory: Path, rows: list[tuple[str, int, str, str]]) -> str:
    directory.mkdir()
    body = HEADER + "".join(f"{u}\t{m}\t{r}\t{s}\n" for u, m, r, s in rows)
    raw = gzip.compress(body.encode("utf-8"), mtime=0)
    (directory / "interactions.tsv.gz").write_bytes(raw)
    manifest = json.dumps(
        {
            "files": {
                "interactions.tsv.gz": {"bytes": len(raw), "sha256": sha(raw), "rows": len(rows)}
            }
        }
    ).encode()
    (directory / "dataset-manifest.json").write_bytes(manifest)
    return "sha256:" + sha(manifest)


class FranchiseTests(unittest.TestCase):
    def test_links_only_franchise_relations_and_picks_the_first_tv_entry(self) -> None:
        ids = [1, 2, 3, 10, 11, 12, 20, 99]
        raw = gzip.compress(
            "".join(json.dumps(record) + "\n" for record in RELATIONS).encode(), mtime=0
        )
        groups, representatives, stats = franchise_groups(raw, ids)
        group = {mal_id: groups[column] for column, mal_id in enumerate(ids)}
        self.assertEqual(group[1], group[10])
        self.assertEqual(group[10], group[11])
        self.assertNotEqual(group[2], group[3])
        self.assertEqual(ids[representatives[group[1]]], 10)
        self.assertEqual(ids[representatives[group[2]]], 2)
        self.assertEqual(
            stats,
            {
                "titles_with_relations": 4,
                "franchises": 1,
                "grouped_titles": 3,
                "largest_franchise": 3,
                "crossovers": 0,
            },
        )

    def test_crossovers_stay_alone_and_the_most_watched_first_season_represents(self) -> None:
        ids = [1, 2, 3, 4, 5, 6]
        records = [
            # 1 and 2 are TV series of one story (2 remakes 1); 3 is a TV sequel of 2.
            {"mal_id": 1, "format": "TV", "start_date": [2003, 1, 1], "relations": []},
            {
                "mal_id": 2,
                "format": "TV",
                "start_date": [2009, 1, 1],
                "relations": [["ALTERNATIVE", 1]],
            },
            {
                "mal_id": 3,
                "format": "TV",
                "start_date": [2010, 1, 1],
                "relations": [["PREQUEL", 2]],
            },
            # 5 is a crossover of the 1-2-3 franchise and franchise 4.
            {"mal_id": 4, "format": "TV", "start_date": [1971, 1, 1], "relations": []},
            {
                "mal_id": 5,
                "format": "SPECIAL",
                "start_date": [2009, 1, 1],
                "relations": [["PARENT", 1], ["PARENT", 4]],
            },
        ]
        raw = gzip.compress("".join(json.dumps(r) + "\n" for r in records).encode(), mtime=0)
        groups, representatives, stats = franchise_groups(raw, ids, [10, 50, 90, 5, 1, 0])
        group = {mal_id: groups[column] for column, mal_id in enumerate(ids)}
        self.assertEqual(len({group[1], group[2], group[3]}), 1)
        self.assertEqual(len({group[1], group[4], group[5]}), 3)
        # 3 is the most watched but continues 2, so the more watched first season 2 represents.
        self.assertEqual(ids[representatives[group[1]]], 2)
        self.assertEqual(stats["crossovers"], 1)
        # A TV series nobody watched yet does not displace the earliest watched entry.
        movie = [
            {"mal_id": 1, "format": "MOVIE", "start_date": [1988, 7, 16], "relations": []},
            {
                "mal_id": 2,
                "format": "TV",
                "start_date": [None, None, None],
                "relations": [["ALTERNATIVE", 1]],
            },
        ]
        raw = gzip.compress("".join(json.dumps(r) + "\n" for r in movie).encode(), mtime=0)
        groups, representatives, _ = franchise_groups(raw, [1, 2], [40, 0])
        self.assertEqual(representatives[groups[0]], 0)

    def test_prequels_count_in_release_order(self) -> None:
        # AniList calls the later 2 a prequel of 1, as with Nekomonogatari and Bakemonogatari.
        story = [
            {
                "mal_id": 1,
                "format": "TV",
                "start_date": [2009, 7, 3],
                "relations": [["PREQUEL", 2]],
            },
            {"mal_id": 2, "format": "TV", "start_date": [2012, 12, 31], "relations": []},
        ]
        self.assertEqual(representative(story, [60, 30]), 1)
        # A TV_SHORT first season is a series too, so its TV sequel does not stand for it.
        shorts = [
            {"mal_id": 1, "format": "TV_SHORT", "start_date": [2015, 1, 1], "relations": []},
            {
                "mal_id": 2,
                "format": "TV",
                "start_date": [2016, 1, 1],
                "relations": [["PREQUEL", 1]],
            },
        ]
        self.assertEqual(representative(shorts, [10, 40]), 1)

    def test_without_a_tv_series_the_main_entry_beats_an_earlier_special(self) -> None:
        # Like Promare: the prologue special 2 came out before the film 1; 3 is a later film.
        film = [
            {"mal_id": 1, "format": "MOVIE", "start_date": [2019, 5, 24], "relations": []},
            {"mal_id": 2, "format": "SPECIAL", "start_date": [2019, 5, 3], "relations": []},
            {"mal_id": 3, "format": "MOVIE", "start_date": [2021, 1, 1], "relations": []},
        ]
        film[0]["relations"] = [["PREQUEL", 2], ["SEQUEL", 3]]
        self.assertEqual(representative(film, [90, 20, 95]), 1)
        # A special stands for the franchise only when nothing else was watched.
        self.assertEqual(representative(film, [0, 20, 0]), 2)

    def test_rejects_relations_for_titles_outside_the_catalog(self) -> None:
        raw = gzip.compress(b'{"mal_id": 7, "relations": []}\n', mtime=0)
        with self.assertRaisesRegex(ContractError, "missing from catalog"):
            franchise_groups(raw, [1, 2])


class PositiveMatrixTests(unittest.TestCase):
    def test_watchers_count_list_entries_except_plan_to_watch(self) -> None:
        # One Piece is airing: MAL allows only «watching», so it has no positive at all.
        rows = [
            ("u1", 1, "10", "watching"),
            ("u1", 2, "9", "completed"),
            ("u2", 1, "", "on_hold"),
            ("u2", 2, "", "plan_to_watch"),
            ("u2", 7, "", "plan_to_watch"),
            ("u3", 2, "4", "dropped"),
        ]
        body = HEADER + "".join(f"{u}\t{m}\t{r}\t{s}\n" for u, m, r, s in rows)
        raw = gzip.compress(body.encode(), mtime=0)
        indptr, indices, watchers = positive_matrix(raw, {1: 0, 2: 1}, len(rows))
        self.assertEqual((indptr.tolist(), indices.tolist()), ([0, 1], [1]))
        self.assertEqual(watchers, [2, 2])


class FirstEntryTests(unittest.TestCase):
    def test_rounding_ties_are_ordered_by_the_shown_mal_id(self) -> None:
        ids = [5, 7, 9]
        # 9 is slightly stronger than 5, but both round to 0.123456.
        pairs = [(2, 0.1234564), (0, 0.1234561), (1, 0.05)]
        self.assertEqual(
            _as_first_entries(pairs, ids, [0, 1, 2], [0, 1, 2]),
            [[5, 0.123456], [9, 0.123456], [7, 0.05]],
        )

    def test_ties_at_the_cut_off_follow_the_shown_entry(self) -> None:
        # Item 0 is liked once with 1 and once with 2, so both tie. Column 1 belongs to the
        # franchise shown as 3, column 2 is shown as itself: the smaller shown entry wins.
        indptr = np.array([0, 2, 4], dtype=np.int64)
        indices = np.array([0, 1, 0, 2], dtype=np.int32)
        groups, shown = [0, 1, 2, 1], [0, 3, 2, 3]
        tied = top_neighbors(indptr, indices, 4, limit=1, groups=groups)
        self.assertEqual([column for column, _ in tied[0]], [1])
        chosen = top_neighbors(indptr, indices, 4, limit=1, groups=groups, shown=shown)
        self.assertEqual([column for column, _ in chosen[0]], [2])


class TopNeighborParityTests(unittest.TestCase):
    def test_matches_research_itemknn_top_fifty_exactly(self) -> None:
        rng = np.random.default_rng(11)
        for _ in range(4):
            with tempfile.TemporaryDirectory() as stage:
                items = int(rng.integers(20, 90))
                dense = rng.random((int(rng.integers(50, 300)), items)) < rng.uniform(0.02, 0.3)
                dense = dense[dense.any(axis=1)]
                _, columns = np.nonzero(dense)
                indptr = np.concatenate(([0], np.cumsum(dense.sum(axis=1)))).astype(np.int64)
                indices = columns.astype(np.int32)
                matrix = csr_matrix(
                    (np.ones(len(columns)), columns, indptr), shape=dense.shape, dtype=np.float64
                )
                positive = np.bincount(indices, minlength=items).astype(np.int64)
                _fit_knn(matrix, tuple(range(1, items + 1)), positive, Path(stage))
                pointers, targets, weights = (
                    np.load(Path(stage) / f"{name}.npy")
                    for name in ("indptr", "indices", "weights")
                )
                ours = top_neighbors(indptr, indices, items)
                self.assertEqual(
                    top_neighbors(indptr, indices, items, groups=list(range(items))), ours
                )
                for column in range(items):
                    lo, hi = pointers[column : column + 2]
                    order = np.lexsort((targets[lo:hi], -weights[lo:hi]))[:50]
                    expected = [(int(targets[lo:hi][i]), float(weights[lo:hi][i])) for i in order]
                    self.assertEqual(ours[column], expected)


class DeriveBundleTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.base = self.root / "base"
        shutil.copytree(FIXTURE, self.base, ignore=shutil.ignore_patterns("source.json"))
        self.base_identity, _ = check_bundle(self.base)
        self.dataset = self.root / "dataset"
        self.dataset_identity = write_dataset(self.dataset, ROWS)
        self.store = self.root / "bundles"

    def derive(self) -> tuple[Path, str, int]:
        return derive_itemknn_bundle(
            self.base,
            self.dataset,
            self.store,
            expected_base_identity=self.base_identity,
            expected_dataset_identity=self.dataset_identity,
        )

    def test_itemknn_neighbors_with_glove_fallback_and_provenance(self) -> None:
        path, identity, count = self.derive()
        self.assertEqual(check_bundle(path), (identity, count))
        self.assertEqual(count, 8)
        # Readable by the bot container's own UID, like bundles from `release prepare`.
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o755)
        for name in ("catalog.json", "neighbors.json", "manifest.json"):
            self.assertEqual(stat.S_IMODE((path / name).stat().st_mode), 0o644)
        base_catalog = (self.base / "catalog.json").read_bytes()
        self.assertEqual((path / "catalog.json").read_bytes(), base_catalog)
        written = json.loads((path / "neighbors.json").read_bytes())
        self.assertEqual((written["schema_version"], written["franchises"]), (2, {}))
        neighbors = written["neighbors"]
        glove = json.loads((self.base / "neighbors.json").read_bytes())["neighbors"]
        self.assertEqual([n[0] for n in neighbors["1"]], [10, 11])
        self.assertEqual([n[0] for n in neighbors["10"]], [1, 11])
        self.assertEqual([n[0] for n in neighbors["2"]], [3, 12])
        self.assertEqual([n[0] for n in neighbors["3"]], [2])
        # Neither item has a completed rating >= 8, so the GloVe lists are kept, rounded.
        self.assertEqual(
            neighbors["20"],
            [[n["mal_id"], float(f"{n['similarity']:.6g}")] for n in glove["20"]],
        )
        self.assertEqual(neighbors["99"], [])
        # Two shared fans: c=2, p=2/2 -> 2/2 * 2/12, kept to 6 significant digits.
        self.assertEqual(neighbors["1"][0][1], 0.166667)
        manifest = json.loads((path / "manifest.json").read_bytes())
        self.assertEqual((manifest["schema_version"], ALGORITHM["max_neighbors"]), (2, 50))
        self.assertEqual(manifest["algorithm"], ALGORITHM)
        self.assertEqual(
            [s["name"] for s in manifest["sources"]], ["glove_bundle", "rating_interactions"]
        )
        stats = manifest["sources"][1]["metadata"]
        self.assertEqual((stats["positive_ratings"], stats["users_with_positive"]), (9, 4))
        self.assertEqual(
            (stats["itemknn_items"], stats["fallback_items"], stats["empty_items"]), (6, 1, 1)
        )
        self.assertEqual(self.derive(), (path, identity, count))

    def test_pinned_identity_is_checked_before_publication(self) -> None:
        with self.assertRaisesRegex(ContractError, "differs from the pinned identity"):
            derive_itemknn_bundle(
                self.base,
                self.dataset,
                self.store,
                expected_base_identity=self.base_identity,
                expected_dataset_identity=self.dataset_identity,
                expected_identity="sha256:" + "0" * 64,
            )
        self.assertFalse(self.store.exists())
        path, identity, _ = self.derive()
        self.assertEqual(
            derive_itemknn_bundle(
                self.base,
                self.dataset,
                self.store,
                expected_base_identity=self.base_identity,
                expected_dataset_identity=self.dataset_identity,
                expected_identity=identity,
            )[0],
            path,
        )

    def test_franchises_collapse_to_first_entries_and_skip_the_seed_franchise(self) -> None:
        shutil.rmtree(self.dataset)
        # u1_e links 2 with 11, which belongs to the 1/10/11 franchise.
        rows = [*ROWS, ("u1_e", 2, "9", "completed"), ("u1_e", 11, "9", "completed")]
        self.dataset_identity = write_dataset(self.dataset, rows)
        relations = self.root / "relations"
        relations_sha = write_relations(relations, RELATIONS)
        with self.assertRaisesRegex(ContractError, "unexpected relations snapshot"):
            derive_itemknn_bundle(
                self.base,
                self.dataset,
                self.store,
                expected_base_identity=self.base_identity,
                expected_dataset_identity=self.dataset_identity,
                relations=relations,
                expected_relations_sha256="0" * 64,
            )
        path, _, _ = derive_itemknn_bundle(
            self.base,
            self.dataset,
            self.store,
            expected_base_identity=self.base_identity,
            expected_dataset_identity=self.dataset_identity,
            relations=relations,
            expected_relations_sha256=relations_sha,
        )
        written = json.loads((path / "neighbors.json").read_bytes())
        # Every title that is not its franchise's first TV entry points at it.
        self.assertEqual(written["franchises"], {"1": 10, "11": 10})
        ids = {key: [n[0] for n in value] for key, value in written["neighbors"].items()}
        # 11 is shown as its franchise's first TV entry 10.
        self.assertEqual(ids["2"], [3, 12, 10])
        # 1 has ItemKNN neighbors only inside its own franchise, so GloVe fills in without 10/11.
        self.assertEqual(ids["1"], [12, 2, 3])
        self.assertEqual(ids["10"], [12, 2, 3])
        manifest = json.loads((path / "manifest.json").read_bytes())
        self.assertEqual(
            [s["name"] for s in manifest["sources"]],
            ["anilist_relations", "glove_bundle", "rating_interactions"],
        )
        self.assertEqual(manifest["sources"][0]["sha256"], relations_sha)

    def test_derive_current_uses_the_pinned_release_layout(self) -> None:
        data = self.root / "data"
        lock = {
            "reference": {"bundle_identity": self.base_identity},
            "inputs": {"base": {"identity": self.dataset_identity}},
        }
        with patch("recsys.release.lock.release_info", return_value=lock):
            with self.assertRaisesRegex(ContractError, "run `recsys release prepare` first"):
                derive_current(data)
            shutil.copytree(self.base, data / "bundles" / self.base_identity.replace(":", "-"))
            shutil.copytree(
                self.dataset, data / "datasets" / self.dataset_identity.replace(":", "-")
            )
            with self.assertRaisesRegex(ContractError, "restore the AniList relations snapshot"):
                derive_current(data)
            relations = data / "raw" / "anilist-relations-2026-v1"
            relations.parent.mkdir()
            relations_sha = write_relations(relations, RELATIONS)
            with (
                patch("recsys.release.itemknn.RELATIONS_SHA256", relations_sha),
                patch("recsys.release.itemknn.CURRENT_IDENTITY", "sha256:" + "0" * 64),
                self.assertRaisesRegex(ContractError, "differs from the pinned identity"),
            ):
                derive_current(data)
            expected = derive_itemknn_bundle(
                self.base,
                self.dataset,
                self.root / "elsewhere",
                expected_base_identity=self.base_identity,
                expected_dataset_identity=self.dataset_identity,
                relations=relations,
                expected_relations_sha256=relations_sha,
            )[1]
            with (
                patch("recsys.release.itemknn.RELATIONS_SHA256", relations_sha),
                patch("recsys.release.itemknn.CURRENT_IDENTITY", expected),
            ):
                path, identity, count = derive_current(data)
        self.assertEqual((path.parent, identity, count), (data / "bundles", expected, 8))

    def test_rejects_dataset_manifest_without_interaction_fields(self) -> None:
        manifest = json.loads((self.dataset / "dataset-manifest.json").read_bytes())
        del manifest["files"]["interactions.tsv.gz"]["rows"]
        raw = json.dumps(manifest).encode()
        (self.dataset / "dataset-manifest.json").write_bytes(raw)
        with self.assertRaisesRegex(ContractError, "missing rows"):
            derive_itemknn_bundle(
                self.base,
                self.dataset,
                self.store,
                expected_base_identity=self.base_identity,
                expected_dataset_identity="sha256:" + sha(raw),
            )

    def test_rejects_unexpected_or_changed_inputs(self) -> None:
        with self.assertRaisesRegex(ContractError, "unexpected base bundle"):
            derive_itemknn_bundle(
                self.base,
                self.dataset,
                self.store,
                expected_base_identity="sha256:" + "0" * 64,
                expected_dataset_identity=self.dataset_identity,
            )
        with self.assertRaisesRegex(ContractError, "unexpected dataset"):
            derive_itemknn_bundle(
                self.base,
                self.dataset,
                self.store,
                expected_base_identity=self.base_identity,
                expected_dataset_identity="sha256:" + "0" * 64,
            )
        (self.dataset / "interactions.tsv.gz").write_bytes(gzip.compress(HEADER.encode()))
        with self.assertRaisesRegex(ContractError, "bytes differ"):
            self.derive()

    def test_rejects_unsorted_users_and_unknown_titles(self) -> None:
        for rows, message in (
            ([ROWS[3], ROWS[0]], "sorted by user"),
            ([("u1_a", 7, "9", "completed")], "missing from catalog"),
        ):
            shutil.rmtree(self.dataset)
            self.dataset_identity = write_dataset(self.dataset, rows)
            with self.assertRaisesRegex(ContractError, message):
                self.derive()


if __name__ == "__main__":
    unittest.main()
