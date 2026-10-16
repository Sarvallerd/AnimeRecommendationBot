"""Frozen original-0.1 producer bytes remain readable but cannot be relabeled."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from recsys.contracts.bundle import canonical_bytes, check_bundle
from recsys.contracts.provenance import LEGACY_V1_SOURCE_SHA256
from recsys.data.aliases import AliasError, AliasSpec, validate_aliases
from recsys.data.dataset import DatasetError, DatasetSpec, validate_dataset
from recsys.release.catalog import ReleaseError, validate_release
from recsys.release.prepare import PrepareError, prepare_release

FROZEN = Path(__file__).with_name("fixtures") / "layout_v1"


class LegacyLayoutBridgeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        shutil.copytree(FROZEN / "data", self.root / "data")
        specs = FROZEN / "specs"
        self.base_spec = DatasetSpec(
            "neelagiri-2025-v1",
            (specs / "base-registry.json").read_bytes(),
            "neelagiri_2025_v1_expectations.json",
            (specs / "base-expectations.json").read_bytes(),
        )
        self.alias_spec = AliasSpec(
            "nafiul-2026-v1",
            (specs / "alias-registry.json").read_bytes(),
            "nafiul_2026_v1_alias_expectations.json",
            (specs / "alias-expectations.json").read_bytes(),
        )
        self.glove_registry = (specs / "glove-registry.json").read_bytes()
        self.lock_bytes = (FROZEN / "test-lock.json").read_bytes()
        self.lock = json.loads(self.lock_bytes)
        self.base = (
            self.root / "data/datasets" / self.lock["inputs"]["base"]["identity"].replace(":", "-")
        )
        self.aliases = (
            self.root
            / "data/aliases"
            / self.lock["inputs"]["aliases"]["identity"].replace(":", "-")
        )
        self.release = (
            self.root
            / "data/releases"
            / self.lock["inputs"]["catalog_release"]["identity"].replace(":", "-")
        )
        self.bundle = (
            self.root / "data/bundles" / self.lock["reference"]["bundle_identity"].replace(":", "-")
        )
        self.alias_data = self.root / "data/raw/nafiul-2026-v1"

    def alias_summary(self):
        return validate_aliases(
            self.aliases,
            data_dir=self.alias_data,
            base_dir=self.base,
            expected_base_identity=self.lock["inputs"]["base"]["identity"],
            _spec=self.alias_spec,
            _base_spec=self.base_spec,
        )

    def release_summary(self):
        return validate_release(
            self.release,
            base_dir=self.base,
            aliases_dir=self.aliases,
            alias_data_dir=self.alias_data,
            expected_identity=self.lock["inputs"]["catalog_release"]["identity"],
            _base_spec=self.base_spec,
            _alias_spec=self.alias_spec,
        )

    def test_frozen_original_producer_reuses_cached_bundle(self) -> None:
        receipt = json.loads((FROZEN / "generator-receipt.json").read_bytes())
        self.assertEqual(receipt["package_version"], "0.1.0")
        for name, digest in LEGACY_V1_SOURCE_SHA256.items():
            self.assertEqual(receipt["installed_source_files"][name]["sha256"], digest)
        self.assertEqual(
            validate_dataset(self.base, _spec=self.base_spec).identity,
            self.lock["inputs"]["base"]["identity"],
        )
        self.assertEqual(self.alias_summary().identity, self.lock["inputs"]["aliases"]["identity"])
        self.assertEqual(self.release_summary().catalog_count, 2)
        checked, count = check_bundle(self.bundle)
        self.assertEqual(count, 2)
        self.assertEqual(checked, self.lock["reference"]["bundle_identity"])
        self.assertEqual(
            prepare_release(
                self.root / "data",
                _lock_bytes=self.lock_bytes,
                _base_spec=self.base_spec,
                _alias_spec=self.alias_spec,
                _glove_registry_bytes=self.glove_registry,
            )[1],
            checked,
        )

    def test_missing_original_producer_fails_before_writes(self) -> None:
        shutil.rmtree(self.bundle)
        before = {
            p.relative_to(self.root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in self.root.rglob("*")
            if p.is_file()
        }
        with self.assertRaisesRegex(PrepareError, "release_reference_producer_required"):
            prepare_release(
                self.root / "data",
                _lock_bytes=self.lock_bytes,
                _base_spec=self.base_spec,
                _alias_spec=self.alias_spec,
                _glove_registry_bytes=self.glove_registry,
            )
        after = {
            p.relative_to(self.root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in self.root.rglob("*")
            if p.is_file()
        }
        self.assertEqual(after, before)
        self.assertFalse((self.root / "data/work").exists())

    def test_v1_dataset_source_keys_hash_version_and_package_are_authoritative(self) -> None:
        path = self.base / "dataset-manifest.json"
        original = json.loads(path.read_bytes())
        for mutate in (
            lambda m: m["normalizer"]["code_files_sha256"].update({"dataset.py": "0" * 64}),
            lambda m: m["normalizer"]["code_files_sha256"].update({"data/dataset.py": "0" * 64}),
            lambda m: m["normalizer"].update(version="2"),
            lambda m: m["producer"].update(package_version="0.2.0"),
        ):
            with self.subTest(mutate=mutate.__code__.co_firstlineno):
                altered = json.loads(canonical_bytes(original))
                mutate(altered)
                path.write_bytes(canonical_bytes(altered))
                with self.assertRaisesRegex(DatasetError, "dataset_integrity_error"):
                    validate_dataset(self.base, _spec=self.base_spec)
        path.write_bytes(canonical_bytes(original))

    def test_v1_lock_rejects_rewritten_source_hash_key_and_package(self) -> None:
        for mutate in (
            lambda lock: lock["reference"]["producer"]["code_files_sha256"].update(
                {"build.py": "0" * 64}
            ),
            lambda lock: lock["reference"]["producer"]["code_files_sha256"].update(
                {"release/build.py": "0" * 64}
            ),
            lambda lock: lock["reference"]["producer"].update(package_version="0.2.0"),
        ):
            with self.subTest(mutate=mutate.__code__.co_firstlineno):
                altered = json.loads(self.lock_bytes)
                mutate(altered)
                with self.assertRaisesRegex(PrepareError, "release_lock_invalid"):
                    prepare_release(
                        self.root / "data",
                        _lock_bytes=canonical_bytes(altered),
                        _base_spec=self.base_spec,
                        _alias_spec=self.alias_spec,
                        _glove_registry_bytes=self.glove_registry,
                    )

    def test_v1_alias_and_release_source_hashes_are_authoritative(self) -> None:
        for path, key, validate, error in (
            (self.aliases / "alias-manifest.json", "aliases.py", self.alias_summary, AliasError),
            (
                self.release / "release-manifest.json",
                "release.py",
                self.release_summary,
                ReleaseError,
            ),
        ):
            original = json.loads(path.read_bytes())
            for mutator in (
                lambda m: m["normalizer"]["code_files_sha256"].update({key: "0" * 64}),
                lambda m: m["normalizer"].update(version="2"),
                lambda m: m["producer"].update(package_version="0.2.0"),
            ):
                with self.subTest(path=path.name, mutation=mutator.__code__.co_firstlineno):
                    altered = json.loads(canonical_bytes(original))
                    mutator(altered)
                    path.write_bytes(canonical_bytes(altered))
                    with self.assertRaises(error):
                        validate()
            path.write_bytes(canonical_bytes(original))


if __name__ == "__main__":
    unittest.main()
