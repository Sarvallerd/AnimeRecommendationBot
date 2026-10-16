"""Real small dataset and alias producers feed the immutable catalog release."""

import hashlib
import json
import tempfile
import unittest
from unittest.mock import patch

from test_aliases import Fixture, record

from recsys.contracts.bundle import canonical_bytes
from recsys.release.catalog import ReleaseError, compose_release, validate_release


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.fixture = Fixture(self.tmp.name)
        self.aliases = self.fixture.run()
        self.alias_identity = self.fixture.validate(self.aliases).identity

    def compose(self, name="release"):
        return compose_release(
            self.fixture.base,
            self.aliases,
            self.fixture.cache,
            self.fixture.root / name,
            expected_base_identity=self.fixture.base_identity,
            expected_alias_identity=self.alias_identity,
            _base_spec=self.fixture.base_fixture.spec,
            _alias_spec=self.fixture.spec,
        )

    def validate(self, directory, identity=None):
        if identity is None:
            identity = (
                "sha256:"
                + hashlib.sha256((directory / "release-manifest.json").read_bytes()).hexdigest()
            )
        return validate_release(
            directory,
            base_dir=self.fixture.base,
            aliases_dir=self.aliases,
            alias_data_dir=self.fixture.cache,
            expected_identity=identity,
            _base_spec=self.fixture.base_fixture.spec,
            _alias_spec=self.fixture.spec,
        )

    def test_composition_order_fields_determinism_and_manifest(self):
        first = self.compose("first").parent
        second = self.compose("second").parent
        self.assertEqual(
            {p.name: p.read_bytes() for p in first.iterdir()},
            {p.name: p.read_bytes() for p in second.iterdir()},
        )
        catalog = json.loads((first / "catalog.json").read_bytes())["anime"]
        self.assertEqual(set(catalog), set(self.fixture.base_catalog))
        self.assertEqual(catalog["1"]["aliases"][:2], ["One", "Alternate One"])
        for key, entry in catalog.items():
            self.assertEqual(
                {k: v for k, v in entry.items() if k != "aliases"},
                {k: v for k, v in self.fixture.base_catalog[key].items() if k != "aliases"},
            )
        manifest = json.loads((first / "release-manifest.json").read_bytes())
        self.assertEqual(
            set(manifest),
            {
                "schema_version",
                "kind",
                "normalizer",
                "policy",
                "inputs",
                "producer",
                "counts",
                "files",
            },
        )
        self.assertEqual(
            set(manifest["counts"]),
            {
                "catalog_count",
                "enrichment_items",
                "catalog_aliases_changed_items",
                "first_alias_changed_items",
                "alias_strings_before",
                "alias_strings_after",
            },
        )
        self.assertEqual(set(manifest["inputs"]), {"base", "aliases"})
        self.assertEqual(self.validate(first).catalog_count, 2)
        self.assertEqual(
            {p.name for p in first.iterdir()}, {"catalog.json", "release-manifest.json"}
        )
        self.assertEqual(first.stat().st_mode & 0o777, 0o700)
        self.assertTrue(all(p.stat().st_mode & 0o777 == 0o600 for p in first.iterdir()))

    def test_wrong_identity_rehashed_overlay_and_typed_binding(self):
        output = self.compose().parent
        original = self.validate(output).identity
        with self.assertRaisesRegex(ReleaseError, "release_integrity_error"):
            self.validate(output, "sha256:" + "0" * 64)
        manifest_path = output / "release-manifest.json"
        original_manifest = json.loads(manifest_path.read_bytes())
        for bad in (True, 2.0, 3):
            changed = json.loads(canonical_bytes(original_manifest))
            changed["counts"]["catalog_count"] = bad
            manifest_path.write_bytes(canonical_bytes(changed))
            with self.assertRaisesRegex(ReleaseError, "release_integrity_error"):
                self.validate(output)
        manifest_path.write_bytes(canonical_bytes(original_manifest))
        catalog_path = output / "catalog.json"
        catalog = json.loads(catalog_path.read_bytes())
        catalog["anime"]["1"]["aliases"][0] = "Forged"
        catalog_path.write_bytes(canonical_bytes(catalog))
        manifest = json.loads(manifest_path.read_bytes())
        raw = catalog_path.read_bytes()
        manifest["files"]["catalog.json"] = {
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
        manifest_path.write_bytes(canonical_bytes(manifest))
        with self.assertRaisesRegex(ReleaseError, "release_integrity_error"):
            self.validate(output)
        self.assertNotEqual(
            "sha256:" + hashlib.sha256(manifest_path.read_bytes()).hexdigest(), original
        )

    def test_policy_schema_and_historical_producer_fingerprint(self):
        output = self.compose().parent
        manifest_path = output / "release-manifest.json"
        original = json.loads(manifest_path.read_bytes())
        for mutate in (
            lambda x: x.update(schema_version=True),
            lambda x: x.update(policy="forged"),
            lambda x: x["normalizer"].update(version=1),
        ):
            changed = json.loads(canonical_bytes(original))
            mutate(changed)
            manifest_path.write_bytes(canonical_bytes(changed))
            with self.assertRaisesRegex(ReleaseError, "release_integrity_error"):
                self.validate(output)
        historical = json.loads(canonical_bytes(original))
        historical["normalizer"]["code_files_sha256"]["release/catalog.py"] = "0" * 64
        historical["producer"]["python_version"] = "3.12.3"
        manifest_path.write_bytes(canonical_bytes(historical))
        self.assertEqual(self.validate(output).catalog_count, 2)

    def test_one_upstream_call_and_consumed_mutation(self):
        import recsys.release.catalog as release

        real = release.validate_aliases
        calls = []

        def validating(*args, **kwargs):
            calls.append(1)
            result = real(*args, **kwargs)
            if len(calls) == 1:
                (self.fixture.base / "catalog.json").write_bytes(b"{}\n")
            return result

        with patch.object(release, "validate_aliases", side_effect=validating):
            with self.assertRaisesRegex(ReleaseError, "release_integrity_error"):
                self.compose()
        self.assertEqual(len(calls), 1)
        self.assertFalse((self.fixture.root / "release").exists())

    def test_publication_safety(self):
        self.compose()
        with self.assertRaisesRegex(ReleaseError, "release_integrity_error"):
            self.compose()
        self.assertFalse(any(p.name.startswith(".release.") for p in self.fixture.root.iterdir()))
        (self.fixture.root / "link").symlink_to(self.fixture.base, target_is_directory=True)
        with self.assertRaisesRegex(ReleaseError, "release_integrity_error"):
            compose_release(
                self.fixture.base,
                self.aliases,
                self.fixture.cache,
                self.fixture.root / "link",
                expected_base_identity=self.fixture.base_identity,
                expected_alias_identity=self.alias_identity,
                _base_spec=self.fixture.base_fixture.spec,
                _alias_spec=self.fixture.spec,
            )

    def test_unenriched_item_keeps_exact_base_entry(self):
        self.fixture.update([record("1", "Alternate One", "One")])
        aliases = self.fixture.run("one-alias")
        identity = self.fixture.validate(aliases).identity
        manifest = compose_release(
            self.fixture.base,
            aliases,
            self.fixture.cache,
            self.fixture.root / "one-release",
            expected_base_identity=self.fixture.base_identity,
            expected_alias_identity=identity,
            _base_spec=self.fixture.base_fixture.spec,
            _alias_spec=self.fixture.spec,
        )
        catalog = json.loads((manifest.parent / "catalog.json").read_bytes())["anime"]
        self.assertEqual(catalog["2"], self.fixture.base_catalog["2"])
