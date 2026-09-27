"""Small pinned fixtures for the offline GloVe builder and numeric ranking."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from recsys.build import (BuildError, _canonical, build, build_vectors, cosine_neighbors,
                          load_embeddings, tokenize)


def entry(title, aliases=None, genres=None, synopsis=None):
    return {"title": title, "aliases": aliases or [], "genres": genres or [],
            "score": None, "year": None, "type": None, "episodes": None,
            "synopsis": synopsis}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


class BuildTests(unittest.TestCase):
    def test_tokens_and_mean_occurrences(self):
        self.assertEqual(tokenize("IT'S co-op, R2D2 日本語"), ["it's", "co", "op", "r", "d"])
        a = np.zeros(300); a[0] = 2
        b = np.zeros(300); b[1] = 2
        anime = {"10": entry("A a", ["B"], ["A"], "unknown"),
                 "2": entry("日本語", [], [], None), "3": entry("b")}
        ids, vectors, info = build_vectors(anime, {"a": a, "b": b})
        self.assertEqual(ids, [2, 3, 10])
        np.testing.assert_allclose(vectors[2, :2], [3 / np.sqrt(10), 1 / np.sqrt(10)])
        self.assertEqual(info["anime"]["10"]["matched_token_count"], 4)
        self.assertEqual(info["anime"]["2"]["vector_status"], "no_tokens")

    def test_zero_statuses_and_numeric_tie(self):
        x = np.zeros(300); x[0] = 1
        minus = -x
        anime = {"10": entry("x"), "2": entry("x"), "1": entry("x"),
                 "3": entry("minus"), "4": entry("unknown"), "5": entry("!!!"),
                 "6": entry("cancel", ["minus"])}
        ids, vectors, info = build_vectors(anime, {"x": x, "minus": minus, "cancel": x})
        neighbors = cosine_neighbors(ids, vectors, block_size=1)
        self.assertEqual([item["mal_id"] for item in neighbors["10"]], [1, 2])
        self.assertEqual(neighbors["4"], [])
        self.assertEqual(neighbors["5"], [])
        self.assertEqual(neighbors["6"], [])
        self.assertEqual(info["anime"]["6"]["vector_status"], "zero_vector")
        self.assertEqual(neighbors, cosine_neighbors(ids, vectors, block_size=7))

    def test_fifth_tie_and_invalid_numeric_boundary(self):
        ids = [1, 2, 3, 4, 5, 6, 7, 10]
        vectors = np.zeros((len(ids), 300), dtype=np.float64)
        vectors[:, 0] = 1
        neighbors = cosine_neighbors(ids, vectors, block_size=3)
        self.assertEqual([item["mal_id"] for item in neighbors["10"]], [1, 2, 3, 4, 5])
        with self.assertRaises(BuildError):
            cosine_neighbors([1, 1], vectors[:2])
        with self.assertRaises(BuildError):
            cosine_neighbors([1], np.full((1, 300), np.inf))
        with self.assertRaises(BuildError):
            cosine_neighbors([1], vectors[:1], block_size=257)

    def test_overshoot_clamp_and_negative_exclusion(self):
        ids = [1, 2, 3, 4]
        vectors = np.zeros((4, 300), dtype=np.float64)
        vectors[0, 0] = 1
        vectors[1, 0] = 1 + 5e-13
        vectors[2, 0] = 1
        vectors[3, 0] = -1
        result = cosine_neighbors(ids, vectors, block_size=2)
        self.assertEqual(result["1"], [{"mal_id": 2, "similarity": 1.0},
                                       {"mal_id": 3, "similarity": 1.0}])
        self.assertEqual(result["4"], [])
        vectors[1, 0] = 1 + 2e-12
        with self.assertRaises(BuildError):
            cosine_neighbors(ids, vectors)

    def _fixture(self, root, anime=None, glove_lines=None):
        anime = anime or {"2": entry("alpha"), "10": entry("alpha", ["beta"])}
        glove_lines = glove_lines or ["alpha " + " ".join(["1"] + ["0"] * 299),
                                     "beta " + " ".join(["0", "1"] + ["0"] * 298)]
        glove_raw = ("\n".join(glove_lines) + "\n").encode()
        (root / "glove.txt").write_bytes(glove_raw)
        catalog_raw = _canonical({"schema_version": 1, "anime": anime})
        (root / "catalog.json").write_bytes(catalog_raw)
        mal = [{"id": name, "filename": name + ".csv", "size": 1,
                "sha256": sha(name.encode()), "url": "https://example.org/" + name}
               for name in ("mal_anime", "mal_synopsis")]
        registry = {"schema_version": 1, "provenance": {"mal": {"commit": "test", "repository": "https://example.org"},
                    "glove": {"project": "https://example.org", "pretrained_vectors_license": "test"}},
                    "sources": mal + [{"id": "glove_zip", "filename": "glove.zip", "size": 1,
                                      "sha256": sha(b"zip"), "url": "https://example.org/glove.zip"},
                                     {"id": "glove_300d", "filename": "glove.txt", "size": len(glove_raw),
                                      "sha256": sha(glove_raw), "archive_id": "glove_zip", "member": "glove.txt"}]}
        registry_sha = sha(_canonical(registry))
        report = {"schema_version": 1, "normalizer_version": "1", "registry_sha256": registry_sha,
                  "provenance": registry["provenance"], "sources": mal,
                  "files": {"catalog.json": {"sha256": sha(catalog_raw)}},
                  "counts": {"retained": len(anime)}}
        (root / "normalization-report.json").write_bytes(_canonical(report))
        return registry, registry_sha

    def test_end_to_end_deterministic_and_input_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry, registry_sha = self._fixture(root)
            args = (root / "catalog.json", root / "normalization-report.json", root / "glove.txt")
            paths = [build(*args, root / f"out-{size}", block_size=size,
                           registry=registry, registry_sha256=registry_sha) for size in (1, 2)]
            self.assertEqual((paths[0].parent / "neighbors.json").read_bytes(),
                             (paths[1].parent / "neighbors.json").read_bytes())
            same = build(*args, root / "out-1", block_size=1,
                         registry=registry, registry_sha256=registry_sha)
            self.assertEqual(paths[0].read_bytes(), same.read_bytes())
            report = json.loads(paths[0].read_bytes())
            neighbors_raw = (paths[0].parent / "neighbors.json").read_bytes()
            self.assertEqual(report["files"]["neighbors.json"]["sha256"], sha(neighbors_raw))
            self.assertEqual(report["coverage"]["catalog_count"], 2)
            bad = root / "bad.json"
            bad.write_bytes((root / "catalog.json").read_bytes().replace(b'"score":null', b'"score":NaN', 1))
            with self.assertRaises(BuildError):
                build(bad, args[1], args[2], root / "bad-out", registry=registry, registry_sha256=registry_sha)
            self.assertFalse((root / "bad-out").exists())
            bad.write_bytes(b'{"schema_version":1,"schema_version":1,"anime":{}}\n')
            with self.assertRaises(BuildError):
                build(bad, args[1], args[2], root / "bad-out", registry=registry, registry_sha256=registry_sha)

    def test_report_and_raw_hash_mismatch_leave_no_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry, registry_sha = self._fixture(root)
            args = (root / "catalog.json", root / "normalization-report.json", root / "glove.txt")
            report = json.loads(args[1].read_bytes())
            report["registry_sha256"] = sha(b"incorrect")
            args[1].write_bytes(_canonical(report))
            with self.assertRaises(BuildError):
                build(*args, root / "bad-report-out", registry=registry, registry_sha256=registry_sha)
            self.assertFalse((root / "bad-report-out").exists())
            report["registry_sha256"] = registry_sha
            args[1].write_bytes(_canonical(report))
            descriptor = registry["sources"][-1]
            descriptor["sha256"] = sha(b"incorrect")
            with self.assertRaises(BuildError):
                build(*args, root / "bad-glove-out", registry=registry, registry_sha256=registry_sha)
            self.assertFalse((root / "bad-glove-out").exists())

    def test_consumed_glove_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for line in ("alpha 1 2", "alpha " + " ".join(["nan"] + ["0"] * 299),
                         "alpha " + " ".join(["1e9999"] + ["0"] * 299)):
                raw = (line + "\n").encode()
                path = root / "glove.txt"; path.write_bytes(raw)
                with self.assertRaises(BuildError):
                    load_embeddings(path, {"alpha"}, {"size": len(raw), "sha256": sha(raw)})
            valid = "alpha " + " ".join(["0"] * 300)
            raw = (valid + "\n" + valid + "\n").encode()
            path.write_bytes(raw)
            with self.assertRaises(BuildError):
                load_embeddings(path, {"alpha"}, {"size": len(raw), "sha256": sha(raw)})
            with self.assertRaises(BuildError):
                load_embeddings(path, set(), {"size": len(raw), "sha256": sha(b"wrong")})


if __name__ == "__main__":
    unittest.main()
