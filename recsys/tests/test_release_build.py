"""Strict v2 build and v1 runtime export from real small upstream artifacts."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from recsys.bundle import canonical_bytes, check_bundle
from recsys.release import compose_release
from recsys.release_build import ReleaseBuildError, build_release, export_release_bundle
from test_aliases import Fixture


class ReleaseBuildTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.fixture = Fixture(self.tmp.name)
        self.aliases = self.fixture.run()
        self.alias_identity = self.fixture.validate(self.aliases).identity
        manifest = compose_release(self.fixture.base, self.aliases, self.fixture.cache,
                                   self.fixture.root / "release",
                                   expected_base_identity=self.fixture.base_identity,
                                   expected_alias_identity=self.alias_identity,
                                   _base_spec=self.fixture.base_fixture.spec,
                                   _alias_spec=self.fixture.spec)
        self.release = manifest.parent
        self.identity = "sha256:" + hashlib.sha256(manifest.read_bytes()).hexdigest()
        self.glove = self.fixture.root / "glove.txt"
        raw = ("one " + " ".join(["1"] + ["0"] * 299) + "\n"
               "two " + " ".join(["1"] + ["0"] * 299) + "\n").encode()
        self.glove.write_bytes(raw)
        registry = {"schema_version": 1,
                    "provenance": {"glove": {"project": "https://example.invalid",
                                              "pretrained_vectors_license": "PDDL 1.0"}},
                    "sources": [{"id": "glove_zip", "filename": "glove.zip", "size": 1,
                                 "sha256": hashlib.sha256(b"zip").hexdigest(),
                                 "url": "https://example.invalid/glove.zip"},
                                {"id": "glove_300d", "filename": "glove.txt", "size": len(raw),
                                 "sha256": hashlib.sha256(raw).hexdigest(),
                                 "archive_id": "glove_zip", "member": "glove.txt"}]}
        self.registry = canonical_bytes(registry)

    def common(self):
        return dict(expected_release_identity=self.identity, base_dir=self.fixture.base,
                    aliases_dir=self.aliases, alias_data_dir=self.fixture.cache,
                    _base_spec=self.fixture.base_fixture.spec, _alias_spec=self.fixture.spec,
                    _glove_registry_bytes=self.registry)

    def build(self, name, block=256):
        return build_release(self.release, self.glove, self.fixture.root / name,
                             block_size=block, **self.common())

    def export(self, report, name="bundles"):
        return export_release_bundle(self.release, report.parent / "neighbors.json", report,
                                     self.fixture.root / name, **self.common())

    def test_build_export_determinism_and_contract(self):
        a = self.build("a", 256)
        b = self.build("b", 1)
        report_a = json.loads(a.read_bytes())
        report_b = json.loads(b.read_bytes())
        self.assertEqual((a.parent / "neighbors.json").read_bytes(),
                         (b.parent / "neighbors.json").read_bytes())
        neighbors = json.loads((a.parent / "neighbors.json").read_bytes())["neighbors"]
        self.assertEqual([item["mal_id"] for item in neighbors["1"]], [2])
        self.assertEqual([item["mal_id"] for item in neighbors["2"]], [1])
        self.assertEqual({k: v for k, v in report_a.items() if k != "execution"},
                         {k: v for k, v in report_b.items() if k != "execution"})
        self.assertEqual(set(report_a), {"schema_version", "kind", "builder_version",
                         "catalog_release_identity", "algorithm", "sources", "registry_sha256",
                         "inputs", "files", "producer", "execution", "coverage", "anime"})
        self.assertEqual(set(report_a["inputs"]), {"release-manifest.json", "catalog.json",
                                                  "glove.6B.300d.txt"})
        self.assertEqual(set(report_a["producer"]["code_files"]), {"build.py", "release_build.py"})
        first = self.export(a)
        second = self.export(b)
        self.assertNotEqual(first, second)
        self.assertEqual(check_bundle(first)[1], 2)
        self.assertEqual((first / "catalog.json").read_bytes(), (self.release / "catalog.json").read_bytes())
        self.assertEqual((first / "neighbors.json").read_bytes(), (second / "neighbors.json").read_bytes())
        self.assertEqual(self.export(a), first)

    def test_corrupt_glove_and_strict_report(self):
        with self.assertRaisesRegex(ReleaseBuildError, "release_build_integrity_error"):
            build_release(self.release, self.glove, self.fixture.root / "bad",
                          _glove_registry_bytes=b"{}", **{k: v for k, v in self.common().items()
                                                           if k != "_glove_registry_bytes"})
        report = self.build("good")
        original = json.loads(report.read_bytes())
        for change in (
            lambda x: x["algorithm"]["parameters"].update(token_weight=True),
            lambda x: x["execution"].update(requested_block_rows=1.0),
            lambda x: x["coverage"].update(catalog_count=2.0),
            lambda x: x["producer"]["code_files"].update(**{"build.py": "0" * 64}),
            lambda x: x["sources"][0].update(version="forged"),
            lambda x: x["inputs"]["catalog.json"].update(bytes=1),
        ):
            changed = json.loads(canonical_bytes(original))
            change(changed)
            report.write_bytes(canonical_bytes(changed))
            with self.assertRaisesRegex(ReleaseBuildError, "release_build_integrity_error"):
                self.export(report)
            self.assertFalse((self.fixture.root / "bundles").exists())
        report.write_bytes(canonical_bytes(original))
        self.assertEqual(check_bundle(self.export(report))[1], 2)

    def test_each_operation_uses_one_upstream_validation(self):
        import recsys.release as release
        real = release.validate_aliases
        with patch.object(release, "validate_aliases", wraps=real) as call:
            report = self.build("one")
            self.assertEqual(call.call_count, 1)
        with patch.object(release, "validate_aliases", wraps=real) as call:
            self.export(report)
            self.assertEqual(call.call_count, 1)
