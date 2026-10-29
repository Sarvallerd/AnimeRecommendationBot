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
from recsys.release.itemknn import ALGORITHM, derive_current, derive_itemknn_bundle, top_neighbors
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


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


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


class TopNeighborParityTests(unittest.TestCase):
    def test_matches_research_itemknn_top_five_exactly(self) -> None:
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
                for column in range(items):
                    lo, hi = pointers[column : column + 2]
                    order = np.lexsort((targets[lo:hi], -weights[lo:hi]))[:5]
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
        neighbors = json.loads((path / "neighbors.json").read_bytes())["neighbors"]
        glove = json.loads((self.base / "neighbors.json").read_bytes())["neighbors"]
        self.assertEqual([n["mal_id"] for n in neighbors["1"]], [10, 11])
        self.assertEqual([n["mal_id"] for n in neighbors["10"]], [1, 11])
        self.assertEqual([n["mal_id"] for n in neighbors["2"]], [3, 12])
        self.assertEqual([n["mal_id"] for n in neighbors["3"]], [2])
        # Neither item has a completed rating >= 8, so the GloVe lists are kept verbatim.
        self.assertEqual(neighbors["20"], glove["20"])
        self.assertEqual(neighbors["99"], glove["99"])
        # Two shared fans: c=2, p=2/2 -> 2/2 * 2/12.
        self.assertEqual(neighbors["1"][0]["similarity"], 2 / 2 * 2 / 12)
        manifest = json.loads((path / "manifest.json").read_bytes())
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
            with self.assertRaisesRegex(ContractError, "differs from the pinned identity"):
                derive_current(data)
            expected = self.derive()[1]
            with patch("recsys.release.itemknn.CURRENT_IDENTITY", expected):
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
