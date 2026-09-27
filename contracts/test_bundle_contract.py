"""Acceptance checks for the v1 recommendation bundle contract."""

import copy
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from check_bundle import ContractError, check_bundle

FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "bundle"


def encode(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


class BundleContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.bundle = Path(self.temp.name)
        for name in ("catalog.json", "neighbors.json", "manifest.json", "source.json"):
            shutil.copyfile(FIXTURE / name, self.bundle / name)

    def read(self, name):
        return json.loads((self.bundle / name).read_text(encoding="utf-8"))

    def write(self, name, value):
        (self.bundle / name).write_bytes(encode(value))
        if name in ("catalog.json", "neighbors.json"):
            self.rehash(name)

    def rehash(self, name):
        manifest = self.read("manifest.json")
        manifest["files"][name]["sha256"] = hashlib.sha256((self.bundle / name).read_bytes()).hexdigest()
        (self.bundle / "manifest.json").write_bytes(encode(manifest))

    def change(self, name, mutate):
        value = self.read(name)
        mutate(value)
        self.write(name, value)

    def rejected(self, fragment):
        with self.assertRaises(ContractError) as context:
            check_bundle(self.bundle)
        self.assertIn(fragment, str(context.exception))

    def test_fixture_order_source_hash_and_identity(self):
        identity, count = check_bundle(self.bundle)
        manifest = self.read("manifest.json")
        self.assertEqual(count, 8)
        self.assertEqual(identity, "sha256:" + hashlib.sha256((self.bundle / "manifest.json").read_bytes()).hexdigest())
        self.assertEqual(manifest["sources"][0]["sha256"], hashlib.sha256((self.bundle / "source.json").read_bytes()).hexdigest())
        self.assertEqual(set(self.read("catalog.json")["anime"]), {"1", "2", "3", "10", "11", "12", "20", "99"})
        self.assertEqual(self.read("catalog.json")["anime"]["1"]["title"], self.read("catalog.json")["anime"]["2"]["title"])
        self.assertIn("天空の列車", self.read("catalog.json")["anime"]["3"]["aliases"])
        expected = {
            "1": [10, 11, 12, 2, 3], "2": [3, 1, 10, 11, 12],
            "3": [2, 1, 10, 11, 12], "10": [1, 11, 12, 2, 3],
            "11": [1, 10, 12, 2, 3], "12": [1, 10, 11, 2, 3],
            "20": [2, 3], "99": [],
        }
        neighbors = self.read("neighbors.json")["neighbors"]
        self.assertEqual({key: [item["mal_id"] for item in items] for key, items in neighbors.items()}, expected)
        self.assertEqual(self.read("catalog.json")["anime"]["99"]["score"], None)
        self.assertEqual(self.read("catalog.json")["anime"]["99"]["aliases"], [])
        output = subprocess.run([sys.executable, str(Path(__file__).with_name("check_bundle.py")), str(self.bundle)],
                                capture_output=True, text=True, check=False)
        self.assertEqual(output.returncode, 0, output.stderr)
        self.assertEqual(output.stdout.strip(), f"{identity} records=8")

    def test_noncanonical_duplicate_keys_and_invalid_json(self):
        cases = [
            (b'{ "schema_version":1}', "noncanonical"),
            (b'{"schema_version":1,"schema_version":1,"anime":{}}\n', "duplicate JSON key"),
            (b'{"anime":{},"schema_version":NaN}\n', "nonfinite JSON constant"),
            (b'{"anime":{},"schema_version":1e999}\n', "invalid JSON"),
            (b'\xef\xbb\xbf{}\n', "invalid JSON"),
            (b'\xff', "invalid JSON"),
        ]
        for raw, message in cases:
            with self.subTest(raw=raw):
                (self.bundle / "catalog.json").write_bytes(raw)
                self.rehash("catalog.json")
                self.rejected(message)

    def test_structure_and_catalog_values(self):
        mutations = [
            (lambda x: x.update(extra=1), "fields differ"),
            (lambda x: x.update(schema_version=True), "schema_version"),
            (lambda x: x["anime"].update({"01": x["anime"].pop("1")}), "canonical positive decimal"),
            (lambda x: x["anime"].update({"2147483648": x["anime"].pop("1")}), "exceeds"),
            (lambda x: x["anime"]["1"].update(title="  "), "title"),
            (lambda x: x["anime"]["1"].update(aliases=["same", "same"]), "duplicate string"),
            (lambda x: x["anime"]["1"].update(score=11), "score"),
            (lambda x: x["anime"]["1"].update(year=0), "year"),
            (lambda x: x["anime"]["1"].update(episodes=True), "episodes"),
            (lambda x: x["anime"]["1"].update(type=""), "type"),
            (lambda x: x["anime"]["1"].update(unknown=1), "fields differ"),
        ]
        for mutate, message in mutations:
            with self.subTest(message=message, mutate=mutate):
                value = copy.deepcopy(json.loads((FIXTURE / "catalog.json").read_text(encoding="utf-8")))
                mutate(value)
                self.write("catalog.json", value)
                self.rejected(message)

    def test_neighbor_rules(self):
        def first(value):
            return value["neighbors"]["1"]
        mutations = [
            (lambda x: x["neighbors"].pop("99"), "exactly match"),
            (lambda x: x["neighbors"].update({"01": []}), "canonical positive decimal"),
            (lambda x: first(x)[0].update(mal_id=555), "missing from catalog"),
            (lambda x: first(x)[0].update(mal_id=1), "self-neighbor"),
            (lambda x: first(x)[1].update(mal_id=10), "duplicate neighbor"),
            (lambda x: first(x).append({"mal_id": 20, "similarity": 0.1}), "at most 5"),
            (lambda x: first(x)[0].update(similarity=0), "similarity"),
            (lambda x: first(x)[0].update(similarity=-0.1), "similarity"),
            (lambda x: first(x)[0].update(similarity=1.0000000000000002), "similarity"),
            (lambda x: first(x)[0].update(similarity=True), "finite JSON number"),
            (lambda x: first(x).reverse(), "sorted"),
            (lambda x: first(x).__setitem__(slice(0, 2), first(x)[0:2][::-1]), "sorted"),
            (lambda x: first(x)[0].update(extra=1), "fields differ"),
        ]
        for mutate, message in mutations:
            with self.subTest(message=message, mutate=mutate):
                value = copy.deepcopy(json.loads((FIXTURE / "neighbors.json").read_text(encoding="utf-8")))
                mutate(value)
                self.write("neighbors.json", value)
                self.rejected(message)

    def test_manifest_and_hashes(self):
        self.change("manifest.json", lambda x: x["sources"].append(copy.deepcopy(x["sources"][0])))
        self.rejected("sources must be unique")
        self.write("manifest.json", json.loads((FIXTURE / "manifest.json").read_text(encoding="utf-8")))
        self.change("manifest.json", lambda x: x["files"]["catalog.json"].update(sha256="0" * 64))
        self.rejected("SHA-256 mismatch")
        self.write("manifest.json", json.loads((FIXTURE / "manifest.json").read_text(encoding="utf-8")))
        self.change("manifest.json", lambda x: x["sources"][0].update(sha256="A" * 64))
        self.rejected("lowercase SHA-256")
        self.write("manifest.json", json.loads((FIXTURE / "manifest.json").read_text(encoding="utf-8")))
        self.change("manifest.json", lambda x: x["files"].update({"source.json": {"sha256": "0" * 64}}))
        self.rejected("fields differ")

    def test_valid_content_and_provenance_changes_change_identity(self):
        original, _ = check_bundle(self.bundle)
        self.change("catalog.json", lambda x: x["anime"]["1"].update(title="Новое название"))
        changed_content, _ = check_bundle(self.bundle)
        self.assertNotEqual(original, changed_content)
        self.change("manifest.json", lambda x: x["sources"][0]["metadata"].update(note="revised input"))
        changed_provenance, _ = check_bundle(self.bundle)
        self.assertNotEqual(changed_content, changed_provenance)
        self.change("manifest.json", lambda x: x["algorithm"].update(version="2"))
        changed_version, _ = check_bundle(self.bundle)
        self.assertNotEqual(changed_provenance, changed_version)


if __name__ == "__main__":
    unittest.main()
