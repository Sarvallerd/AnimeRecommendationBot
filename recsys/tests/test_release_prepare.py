"""Small verified inputs exercise the same locked prepare checks as production."""

import copy
import hashlib
import json
import platform
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from test_aliases import Fixture

from recsys import __version__
from recsys.contracts.bundle import canonical_bytes
from recsys.contracts.provenance import SOURCE_LAYOUT, source_fingerprints
from recsys.release.build import build_release, export_release_bundle
from recsys.release.catalog import compose_release
from recsys.release.lock import _lock, release_info
from recsys.release.prepare import PrepareError, prepare_release


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


class PrepareTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.data = Path(self.temp.name) / "data"
        for name in ("raw", "datasets", "aliases", "releases", "work", "bundles"):
            (self.data / name).mkdir(parents=True)
        fixture_root = Path(self.temp.name) / "fixture"
        fixture_root.mkdir()
        fixture = Fixture(fixture_root)
        aliases = fixture.run()
        alias_identity = fixture.validate(aliases).identity
        self.base = self.data / "datasets" / fixture.base_identity.replace(":", "-")
        self.aliases = self.data / "aliases" / alias_identity.replace(":", "-")
        self.nafiul = self.data / "raw" / "nafiul-2026-v1"
        shutil.copytree(fixture.base, self.base)
        shutil.copytree(aliases, self.aliases)
        shutil.copytree(fixture.cache, self.nafiul)
        self.base_spec = fixture.base_fixture.spec
        self.alias_spec = fixture.spec
        glove_raw = (
            "one "
            + " ".join(["1"] + ["0"] * 299)
            + "\n"
            + "two "
            + " ".join(["0", "1"] + ["0"] * 298)
            + "\n"
        ).encode()
        glove_dir = self.data / "raw" / "glove-6b-300d-v1"
        glove_dir.mkdir()
        self.glove = glove_dir / "glove.6B.300d.txt"
        self.glove.write_bytes(glove_raw)
        registry = {
            "schema_version": 1,
            "provenance": {
                "glove": {
                    "project": "https://example.invalid",
                    "pretrained_vectors_license": "fixture-only",
                }
            },
            "sources": [
                {
                    "id": "glove_zip",
                    "filename": "glove.zip",
                    "size": 1,
                    "sha256": sha(b"z"),
                    "url": "https://example.invalid/glove.zip",
                },
                {
                    "id": "glove_300d",
                    "filename": "glove.6B.300d.txt",
                    "size": len(glove_raw),
                    "sha256": sha(glove_raw),
                    "archive_id": "glove_zip",
                    "member": "glove.6B.300d.txt",
                },
            ],
        }
        self.glove_registry = canonical_bytes(registry)
        temporary_release = self.data / "work" / "reference-release"
        manifest = compose_release(
            self.base,
            self.aliases,
            self.nafiul,
            temporary_release,
            expected_base_identity=fixture.base_identity,
            expected_alias_identity=alias_identity,
            _base_spec=self.base_spec,
            _alias_spec=self.alias_spec,
        )
        release_identity = "sha256:" + sha(manifest.read_bytes())
        self.release = self.data / "releases" / release_identity.replace(":", "-")
        temporary_release.rename(self.release)
        common = dict(
            expected_release_identity=release_identity,
            base_dir=self.base,
            aliases_dir=self.aliases,
            alias_data_dir=self.nafiul,
            _base_spec=self.base_spec,
            _alias_spec=self.alias_spec,
            _glove_registry_bytes=self.glove_registry,
        )
        report_path = build_release(
            self.release,
            self.glove,
            self.data / "work" / "reference-build",
            block_size=256,
            **common,
        )
        candidate = export_release_bundle(
            self.release,
            report_path.parent / "neighbors.json",
            report_path,
            self.data / "work" / "reference-store",
            **common,
        )
        report = json.loads(report_path.read_bytes())
        catalog = json.loads((self.release / "catalog.json").read_bytes())["anime"]
        neighbors = json.loads((candidate / "neighbors.json").read_bytes())["neighbors"]
        nonready = report["coverage"]["zero_vector_ids"]
        self.assertEqual(nonready, [])
        self.lock = {
            "schema_version": 2,
            "kind": "arb-release-lock",
            "release_id": "fixture-v1",
            "inputs": {
                "base": {"snapshot_id": "neelagiri-2025-v1", "identity": fixture.base_identity},
                "aliases": {"snapshot_id": "nafiul-2026-v1", "identity": alias_identity},
                "catalog_release": {
                    "identity": release_identity,
                    "catalog_sha256": sha((self.release / "catalog.json").read_bytes()),
                    "catalog_count": 2,
                },
                "embeddings": {
                    "snapshot_id": "glove-6b-300d-v1",
                    "registry_sha256": sha(self.glove_registry),
                    "text_size": len(glove_raw),
                    "text_sha256": sha(glove_raw),
                },
                "algorithm": report["algorithm"],
            },
            "reference": {
                "block_size": 256,
                "build_report_sha256": sha(report_path.read_bytes()),
                "bundle_identity": "sha256:" + sha((candidate / "manifest.json").read_bytes()),
                "catalog_sha256": report["files"]["catalog.json"]["sha256"],
                "neighbors_sha256": report["files"]["neighbors.json"]["sha256"],
                "producer": {
                    "source_commit": "0" * 40,
                    "package_version": __version__,
                    "python_version": platform.python_version(),
                    "numpy_version": np.__version__,
                    "source_layout": SOURCE_LAYOUT,
                    "code_files_sha256": source_fingerprints(),
                },
            },
            "semantic_expectations": {
                "catalog_count": 2,
                "vector_status_counts": report["coverage"]["vector_status_counts"],
                "zero_vector_ids": nonready,
                "neighbor_count_histogram": report["coverage"]["neighbor_count_histogram"],
                "smoke_seed": {
                    "mal_id": 1,
                    "query": catalog["1"]["title"],
                    "neighbor_ids": [e["mal_id"] for e in neighbors["1"]],
                },
                "empty_seed": None,
            },
        }
        self.lock_bytes = canonical_bytes(self.lock)
        self.common = dict(
            _lock_bytes=self.lock_bytes,
            _base_spec=self.base_spec,
            _alias_spec=self.alias_spec,
            _glove_registry_bytes=self.glove_registry,
        )
        self.candidate = candidate

    def test_current_alias_and_lock_schema(self):
        self.assertEqual(release_info()["release_id"], "neelagiri-2025-nafiul-2026-v1")
        self.assertEqual(_lock(self.lock_bytes), self.lock)
        with self.assertRaisesRegex(PrepareError, "unknown_release"):
            release_info("latest")

    def test_lock_schema_mutations_fail_before_input_access(self):
        outside = Path(self.temp.name) / "outside-private"
        outside.mkdir()
        sentinel = outside / "sentinel"
        sentinel.write_bytes(b"unchanged")
        cases = [
            (("schema_version",), True),
            (("kind",), 1),
            (("release_id",), "../outside"),
            (("release_id",), "/tmp/outside"),
            (("inputs",), []),
            (("inputs", "future"), 1),
            (("inputs", "base"), []),
            (("inputs", "base", "snapshot_id"), "../outside"),
            (("inputs", "base", "snapshot_id"), "wrong-snapshot"),
            (("inputs", "base", "identity"), None),
            (("inputs", "base", "identity"), "sha256:" + "A" * 64),
            (("inputs", "aliases", "snapshot_id"), str(outside)),
            (("inputs", "aliases", "snapshot_id"), "../../outside-private"),
            (("inputs", "embeddings", "snapshot_id"), str(outside)),
            (("inputs", "embeddings", "registry_sha256"), True),
            (("inputs", "embeddings", "text_sha256"), "x" * 64),
            (("inputs", "embeddings", "text_size"), False),
            (("inputs", "catalog_release", "catalog_sha256"), 1),
            (("inputs", "catalog_release", "catalog_count"), True),
            (("inputs", "algorithm", "future"), "bad"),
            (("inputs", "algorithm", "version"), "future-v2"),
            (("inputs", "algorithm", "parameters", "token_weight"), True),
            (("reference",), []),
            (("reference", "block_size"), True),
            (("reference", "block_size"), 257),
            (("reference", "neighbors_sha256"), 123),
            (("reference", "neighbors_sha256"), "A" * 64),
            (("reference", "build_report_sha256"), "0" * 63),
            (("reference", "bundle_identity"), "sha256:" + "g" * 64),
            (("reference", "producer"), []),
            (("reference", "producer", "source_commit"), 123),
            (("reference", "producer", "source_commit"), "A" * 40),
            (("reference", "producer", "package_version"), "fixture"),
            (("reference", "producer", "python_version"), 3.12),
            (("reference", "producer", "numpy_version"), "2.5"),
            (("reference", "producer", "code_files_sha256"), []),
            (("reference", "producer", "code_files_sha256", "future.py"), "0" * 64),
            (("reference", "producer", "code_files_sha256", "build.py"), None),
            (("semantic_expectations",), []),
            (("semantic_expectations", "catalog_count"), False),
            (("semantic_expectations", "vector_status_counts"), []),
            (("semantic_expectations", "vector_status_counts", "future"), 0),
            (("semantic_expectations", "vector_status_counts", "ready"), True),
            (("semantic_expectations", "zero_vector_ids"), "1"),
            (("semantic_expectations", "zero_vector_ids"), [True]),
            (("semantic_expectations", "neighbor_count_histogram"), []),
            (("semantic_expectations", "neighbor_count_histogram", "6"), 0),
            (("semantic_expectations", "neighbor_count_histogram", "0"), True),
            (("semantic_expectations", "smoke_seed"), []),
            (("semantic_expectations", "smoke_seed", "mal_id"), False),
            (("semantic_expectations", "smoke_seed", "query"), " "),
            (("semantic_expectations", "smoke_seed", "neighbor_ids"), "2"),
            (("semantic_expectations", "smoke_seed", "neighbor_ids"), [1]),
            (("semantic_expectations", "smoke_seed", "neighbor_ids"), [2, 2]),
            (("semantic_expectations", "empty_seed"), True),
            (("semantic_expectations", "empty_seed"), {"mal_id": True, "query": "Two"}),
        ]
        for path, replacement in cases:
            with self.subTest(path=path, value=replacement):
                changed = copy.deepcopy(self.lock)
                parent = changed
                for key in path[:-1]:
                    parent = parent[key]
                parent[path[-1]] = replacement
                raw = canonical_bytes(changed)
                with patch.object(Path, "read_bytes", return_value=b"") as source_read:
                    with patch(
                        "recsys.release.prepare.tempfile.mkdtemp",
                        side_effect=AssertionError("publication"),
                    ) as work_create:
                        with patch.object(
                            Path, "mkdir", side_effect=AssertionError("mkdir")
                        ) as mkdir:
                            with self.assertRaises(PrepareError) as failure:
                                prepare_release(
                                    self.data,
                                    _lock_bytes=raw,
                                    _base_spec=self.base_spec,
                                    _alias_spec=self.alias_spec,
                                    _glove_registry_bytes=self.glove_registry,
                                )
                source_read.assert_not_called()
                work_create.assert_not_called()
                mkdir.assert_not_called()
                self.assertEqual(str(failure.exception), "release_lock_invalid")
                self.assertEqual(sentinel.read_bytes(), b"unchanged")
                self.assertEqual(list((self.data / "bundles").iterdir()), [])

        for raw in (b'{"secret":1,"secret":2}\n', b'{"secret":"private"}\n', b"[]\n"):
            with self.subTest(raw=raw):
                with self.assertRaises(PrepareError) as failure:
                    _lock(raw)
                self.assertEqual(str(failure.exception), "release_lock_invalid")

        for nonready in ([1, 1], [2, 1], [True, 2]):
            with self.subTest(nonready=nonready):
                changed = copy.deepcopy(self.lock)
                semantics = changed["semantic_expectations"]
                semantics["vector_status_counts"] = {
                    "ready": 0,
                    "no_tokens": 2,
                    "oov_only": 0,
                    "zero_vector": 0,
                }
                semantics["zero_vector_ids"] = nonready
                semantics["neighbor_count_histogram"] = {
                    "0": 2,
                    "1": 0,
                    "2": 0,
                    "3": 0,
                    "4": 0,
                    "5": 0,
                }
                with self.assertRaises(PrepareError) as failure:
                    _lock(canonical_bytes(changed))
                self.assertEqual(str(failure.exception), "release_lock_invalid")

    def test_wrong_semantic_never_publishes(self):
        bad = json.loads(self.lock_bytes)
        bad["semantic_expectations"]["smoke_seed"]["query"] = "Two"
        with self.assertRaisesRegex(PrepareError, "release_semantic_mismatch"):
            prepare_release(
                self.data,
                _lock_bytes=canonical_bytes(bad),
                _base_spec=self.base_spec,
                _alias_spec=self.alias_spec,
                _glove_registry_bytes=self.glove_registry,
            )
        self.assertEqual(list((self.data / "bundles").iterdir()), [])

    def test_fresh_prepare_and_repeat_are_immutable(self):
        first, identity, count = prepare_release(self.data, **self.common)
        self.assertEqual((identity, count), (self.lock["reference"]["bundle_identity"], 2))
        saved = {
            name: (first / name).read_bytes()
            for name in ("catalog.json", "neighbors.json", "manifest.json")
        }
        second, _, _ = prepare_release(self.data, **self.common)
        self.assertEqual(first, second)
        self.assertEqual(saved, {name: (second / name).read_bytes() for name in saved})

    def test_missing_release_is_verified_before_publication(self):
        shutil.rmtree(self.release)
        prepared, identity, count = prepare_release(self.data, **self.common)
        self.assertTrue(self.release.is_dir())
        self.assertEqual(
            (prepared.name, identity, count),
            (
                self.lock["reference"]["bundle_identity"].replace(":", "-"),
                self.lock["reference"]["bundle_identity"],
                2,
            ),
        )

    def test_missing_frozen_prerequisite(self):
        shutil.rmtree(self.base)
        with self.assertRaisesRegex(PrepareError, "release_prerequisite_missing"):
            prepare_release(self.data, **self.common)
        self.assertEqual(list((self.data / "bundles").iterdir()), [])

    def test_prepare_uses_one_own_upstream_validation(self):
        import recsys.release.catalog as release_module

        original = release_module._verified_release
        with patch.object(release_module, "_verified_release", wraps=original) as observed:
            prepare_release(self.data, **self.common)
        self.assertEqual(observed.call_count, 1)

    def test_legacy_missing_cache_returns_fixed_error(self):
        (self.data / "raw" / "mal-2020-glove-v1").mkdir()
        with self.assertRaisesRegex(PrepareError, "release_prepare_integrity_error"):
            prepare_release(self.data, "mal-2020-glove-v1")

    def test_producer_mismatch_precedes_build(self):
        bad = json.loads(self.lock_bytes)
        bad["reference"]["producer"]["code_files_sha256"]["release/build.py"] = "0" * 64
        with self.assertRaisesRegex(PrepareError, "release_producer_mismatch"):
            prepare_release(
                self.data,
                _lock_bytes=canonical_bytes(bad),
                _base_spec=self.base_spec,
                _alias_spec=self.alias_spec,
                _glove_registry_bytes=self.glove_registry,
            )
        self.assertEqual(list((self.data / "bundles").iterdir()), [])

    def test_reuse_uses_historical_producer_fingerprint(self):
        target = self.data / "bundles" / self.lock["reference"]["bundle_identity"].replace(":", "-")
        shutil.copytree(self.candidate, target)
        with patch("recsys.release.prepare._producer", side_effect=AssertionError("rebuild")):
            path, identity, count = prepare_release(self.data, **self.common)
        self.assertEqual(
            (path, identity, count), (target, self.lock["reference"]["bundle_identity"], 2)
        )
        self.assertEqual(target.stat().st_mode & 0o777, 0o755)
        self.assertTrue(
            all(
                (target / name).stat().st_mode & 0o777 == 0o644
                for name in ("catalog.json", "neighbors.json", "manifest.json")
            )
        )

    def test_symlink_tree_rejected_before_mutation(self):
        (self.data / "bundles").rmdir()
        (self.data / "bundles").symlink_to(self.base, target_is_directory=True)
        with self.assertRaisesRegex(PrepareError, "unsafe_release_path"):
            prepare_release(self.data, **self.common)
