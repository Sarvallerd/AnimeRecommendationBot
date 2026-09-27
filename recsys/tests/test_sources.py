"""Validate the packaged source registry and descriptor safety."""

import copy
import unittest

from recsys.sources import SourceError, load_registry, validate_registry


class SourceTests(unittest.TestCase):
    def test_packaged_registry_has_pinned_inputs(self):
        registry, digest = load_registry()
        self.assertEqual(len(digest), 64)
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
