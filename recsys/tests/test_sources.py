"""Validate the packaged source registry and descriptor safety."""

import copy
import json
import shlex
from pathlib import Path
import unittest

from recsys.sources import (DEFAULT_SNAPSHOT, SNAPSHOT_IDS, SourceError,
                            load_registry, validate_registry)


class SourceTests(unittest.TestCase):
    def assert_documented_download_pins(self, instructions, registry):
        """Read complete shell arguments as text; never execute the documentation."""
        urls = {}
        hashes = {}
        for line in instructions.splitlines():
            if line.startswith("curl "):
                tokens = shlex.split(line)
                self.assertEqual(tokens[:3], ["curl", "--fail", "--location"])
                self.assertEqual(len(tokens), 6)
                self.assertEqual(tokens[4], "-o")
                self.assertTrue(tokens[5].startswith("$AUDIT_DIR/"))
                filename = tokens[5].removeprefix("$AUDIT_DIR/")
                self.assertNotIn("/", filename)
                self.assertNotIn(filename, urls)
                urls[filename] = tokens[3]
            elif line.startswith("printf ") and "| sha256sum" in line:
                command, separator, check = line.partition("|")
                self.assertEqual(separator, "|")
                self.assertEqual(shlex.split(check), ["sha256sum", "--check", "-"])
                tokens = shlex.split(command)
                self.assertEqual(len(tokens), 4)
                self.assertEqual(tokens[0], "printf")
                self.assertTrue(tokens[3].startswith("$AUDIT_DIR/"))
                filename = tokens[3].removeprefix("$AUDIT_DIR/")
                self.assertNotIn("/", filename)
                self.assertNotIn(filename, hashes)
                hashes[filename] = tokens[2]
        expected = {item["filename"]: item for item in registry["sources"]}
        self.assertEqual(set(urls), set(expected))
        self.assertEqual(set(hashes), set(expected))
        for filename, item in expected.items():
            self.assertEqual(urls[filename], item["url"])
            self.assertEqual(hashes[filename], item["sha256"])

    def test_packaged_registry_has_pinned_inputs(self):
        registry, digest = load_registry()
        self.assertEqual(digest, "49bb399935f0ac7ec649fe2f4bd7715444c09c8513704866fa90b9bea29d182f")
        self.assertEqual(DEFAULT_SNAPSHOT, "mal-2020-glove-v1")
        self.assertEqual(SNAPSHOT_IDS, (DEFAULT_SNAPSHOT, "neelagiri-2025-v1"))
        self.assertEqual(
            [(item["filename"], item["size"]) for item in registry["sources"]],
            [
                ("anime.csv", 5662362),
                ("anime_with_synopsis.csv", 7221844),
                ("glove.6B.zip", 862182613),
                ("glove.6B.300d.txt", 1037962819),
            ],
        )
        self.assertEqual(registry["provenance"]["mal"]["commit"],
                         "9a1d7f56482accbde99e24cd7cfcad65e2d24b1f")

    def test_new_packaged_registry_matches_audit_pins(self):
        registry, digest = load_registry("neelagiri-2025-v1")
        self.assertEqual(len(digest), 64)
        self.assertEqual([item["filename"] for item in registry["sources"]],
                         ["details.csv", "ratings.csv"])
        self.assertEqual([item["id"] for item in registry["sources"]],
                         ["neelagiri_details_csv", "neelagiri_ratings_csv"])
        self.assertEqual(registry["provenance"]["dataset"], {
            "snapshot_id": "neelagiri-2025-v1",
            "provider": "Kaggle",
            "slug": "neelagiriaditya/anime-dataset-jan-1917-to-oct-2025",
            "version": 1,
            "page_url": "https://www.kaggle.com/datasets/neelagiriaditya/anime-dataset-jan-1917-to-oct-2025",
            "declared_license": "CC BY-NC-SA 4.0",
            "intake_scope": "noncommercial-research",
            "exact_collection_date": None,
        })
        root = Path(__file__).resolve().parents[1]
        audit = json.loads((root / "reports/dataset-audit-v1/source-audit.json").read_text())
        selected = next(item for item in audit["candidates"] if item["id"] == "neelagiri-2025-v1")
        expected = {item["id"]: item for item in selected["artifacts"]}
        for item, audit_id in zip(registry["sources"],
                                  ("neelagiri_details_csv", "neelagiri_ratings_csv"), strict=True):
            pin = expected[audit_id]
            self.assertEqual((item["url"], item["size"], item["sha256"]),
                             (pin["stable_url"], pin["observed_bytes"], pin["observed_sha256"]))
        instructions = (root / "reports/dataset-audit-v1/README.md").read_text()
        self.assert_documented_download_pins(instructions, registry)

    def test_documented_pins_reject_hash_suffix_and_changed_version(self):
        registry, _ = load_registry("neelagiri-2025-v1")
        root = Path(__file__).resolve().parents[1]
        instructions = (root / "reports/dataset-audit-v1/README.md").read_text()
        first = registry["sources"][0]
        with self.assertRaises(AssertionError):
            self.assert_documented_download_pins(
                instructions.replace(f"'{first['sha256']}'", f"'{first['sha256']}20'", 1),
                registry,
            )
        with self.assertRaises(AssertionError):
            self.assert_documented_download_pins(
                instructions.replace(first["url"],
                                     first["url"].replace("datasetVersionNumber=1",
                                                          "datasetVersionNumber=10"), 1),
                registry,
            )

    def test_unknown_snapshot_rejected_before_resource_lookup(self):
        for snapshot in ("latest", "../source_registry.json", "NEELAGIRI", None):
            with self.subTest(snapshot=snapshot), self.assertRaises(SourceError):
                load_registry(snapshot)

    def test_rejects_unsafe_filenames_and_archive_references(self):
        registry, _ = load_registry()
        for filename in ("../outside", "folder/file", "folder\\file", ".."):
            with self.subTest(filename=filename):
                invalid = copy.deepcopy(registry)
                invalid["sources"][0]["filename"] = filename
                with self.assertRaises(SourceError):
                    validate_registry(invalid)
        invalid = copy.deepcopy(registry)
        invalid["sources"][-1]["archive_id"] = "missing"
        with self.assertRaises(SourceError):
            validate_registry(invalid)


if __name__ == "__main__":
    unittest.main()
