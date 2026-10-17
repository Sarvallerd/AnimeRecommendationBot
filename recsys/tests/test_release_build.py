"""Strict v2 build and v1 runtime export from real small upstream artifacts."""

import hashlib
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import TypedDict
from unittest.mock import patch

from test_aliases import Fixture

from recsys.contracts.bundle import canonical_bytes, check_bundle
from recsys.contracts.provenance import source_fingerprints
from recsys.data.aliases import AliasSpec
from recsys.data.dataset import DatasetSpec
from recsys.release.build import ReleaseBuildError, build_release, export_release_bundle
from recsys.release.catalog import compose_release


class _BuildOptions(TypedDict):
    expected_release_identity: str
    base_dir: Path
    aliases_dir: Path
    alias_data_dir: Path
    _base_spec: DatasetSpec
    _alias_spec: AliasSpec
    _glove_registry_bytes: bytes


class ReleaseBuildTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.fixture = Fixture(self.tmp.name)
        self.aliases = self.fixture.run()
        self.alias_identity = self.fixture.validate(self.aliases).identity
        manifest = compose_release(
            self.fixture.base,
            self.aliases,
            self.fixture.cache,
            self.fixture.root / "release",
            expected_base_identity=self.fixture.base_identity,
            expected_alias_identity=self.alias_identity,
            _base_spec=self.fixture.base_fixture.spec,
            _alias_spec=self.fixture.spec,
        )
        self.release = manifest.parent
        self.identity = "sha256:" + hashlib.sha256(manifest.read_bytes()).hexdigest()
        self.glove = self.fixture.root / "glove.txt"
        raw = (
            "one " + " ".join(["1"] + ["0"] * 299) + "\ntwo " + " ".join(["1"] + ["0"] * 299) + "\n"
        ).encode()
        self.glove.write_bytes(raw)
        registry = {
            "schema_version": 1,
            "provenance": {
                "glove": {
                    "project": "https://example.invalid",
                    "pretrained_vectors_license": "PDDL 1.0",
                }
            },
            "sources": [
                {
                    "id": "glove_zip",
                    "filename": "glove.zip",
                    "size": 1,
                    "sha256": hashlib.sha256(b"zip").hexdigest(),
                    "url": "https://example.invalid/glove.zip",
                },
                {
                    "id": "glove_300d",
                    "filename": "glove.txt",
                    "size": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "archive_id": "glove_zip",
                    "member": "glove.txt",
                },
            ],
        }
        self.registry = canonical_bytes(registry)

    def common(self) -> _BuildOptions:
        return dict(
            expected_release_identity=self.identity,
            base_dir=self.fixture.base,
            aliases_dir=self.aliases,
            alias_data_dir=self.fixture.cache,
            _base_spec=self.fixture.base_fixture.spec,
            _alias_spec=self.fixture.spec,
            _glove_registry_bytes=self.registry,
        )

    def build(self, name: str, block: int = 256) -> Path:
        return build_release(
            self.release, self.glove, self.fixture.root / name, block_size=block, **self.common()
        )

    def export(self, report: Path, name: str = "bundles") -> Path:
        return export_release_bundle(
            self.release,
            report.parent / "neighbors.json",
            report,
            self.fixture.root / name,
            **self.common(),
        )

    def test_build_export_determinism_and_contract(self) -> None:
        a = self.build("a", 256)
        b = self.build("b", 1)
        report_a = json.loads(a.read_bytes())
        report_b = json.loads(b.read_bytes())
        self.assertEqual(
            (a.parent / "neighbors.json").read_bytes(), (b.parent / "neighbors.json").read_bytes()
        )
        neighbors = json.loads((a.parent / "neighbors.json").read_bytes())["neighbors"]
        self.assertEqual([item["mal_id"] for item in neighbors["1"]], [2])
        self.assertEqual([item["mal_id"] for item in neighbors["2"]], [1])
        self.assertEqual(
            {k: v for k, v in report_a.items() if k != "execution"},
            {k: v for k, v in report_b.items() if k != "execution"},
        )
        self.assertEqual(
            set(report_a),
            {
                "schema_version",
                "kind",
                "builder_version",
                "catalog_release_identity",
                "algorithm",
                "sources",
                "registry_sha256",
                "inputs",
                "files",
                "producer",
                "execution",
                "coverage",
                "anime",
            },
        )
        self.assertEqual(
            set(report_a["inputs"]), {"release-manifest.json", "catalog.json", "glove.6B.300d.txt"}
        )
        self.assertEqual(set(report_a["producer"]["code_files"]), set(source_fingerprints()))
        first = self.export(a)
        second = self.export(b)
        self.assertNotEqual(first, second)
        self.assertEqual(check_bundle(first)[1], 2)
        self.assertEqual(
            (first / "catalog.json").read_bytes(), (self.release / "catalog.json").read_bytes()
        )
        self.assertEqual(
            (first / "neighbors.json").read_bytes(), (second / "neighbors.json").read_bytes()
        )
        self.assertEqual(self.export(a), first)

    def test_corrupt_glove_and_strict_report(self) -> None:
        with self.assertRaisesRegex(ReleaseBuildError, "release_build_integrity_error"):
            build_release(
                self.release,
                self.glove,
                self.fixture.root / "bad",
                _glove_registry_bytes=b"{}",
                expected_release_identity=self.identity,
                base_dir=self.fixture.base,
                aliases_dir=self.aliases,
                alias_data_dir=self.fixture.cache,
                _base_spec=self.fixture.base_fixture.spec,
                _alias_spec=self.fixture.spec,
            )
        report = self.build("good")
        original = json.loads(report.read_bytes())
        for path, replacement in (
            (("algorithm", "parameters", "token_weight"), True),
            (("execution", "requested_block_rows"), 1.0),
            (("coverage", "catalog_count"), 2.0),
            (("coverage", "neighbor_count_histogram"), []),
            (("producer",), []),
            (("producer", "code_files", "build.py"), "0" * 64),
            (("anime", "1"), []),
            (("sources", 0, "version"), "forged"),
            (("inputs", "catalog.json", "bytes"), 1),
        ):
            changed = json.loads(canonical_bytes(original))
            target = changed
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = replacement
            report.write_bytes(canonical_bytes(changed))
            with self.assertRaisesRegex(ReleaseBuildError, "release_build_integrity_error"):
                self.export(report)
            self.assertFalse((self.fixture.root / "bundles").exists())
        report.write_bytes(canonical_bytes(original))
        self.assertEqual(check_bundle(self.export(report))[1], 2)

    def test_each_operation_uses_one_upstream_validation(self) -> None:
        import recsys.release.catalog as release

        real = release.validate_aliases
        with patch.object(release, "validate_aliases", wraps=real) as call:
            report = self.build("one")
            self.assertEqual(call.call_count, 1)
        with patch.object(release, "validate_aliases", wraps=real) as call:
            self.export(report)
            self.assertEqual(call.call_count, 1)

    def test_export_store_separation_precedes_validation_and_writes(self) -> None:
        report = self.build("build")
        neighbor_dir = self.fixture.root / "other-neighbors"
        neighbor_dir.mkdir()
        neighbor_path = neighbor_dir / "neighbors.json"
        shutil.copyfile(report.parent / "neighbors.json", neighbor_path)
        inputs = (
            self.release,
            self.fixture.base,
            self.aliases,
            self.fixture.cache,
            report.parent,
            neighbor_dir,
        )
        forbidden = [
            *inputs,
            self.fixture.root,
            self.release / "nested-store",
            self.fixture.base / "nested-store",
            self.aliases / "nested-store",
            self.fixture.cache / "nested-store",
            report.parent / "nested-store",
            neighbor_dir / "nested-store",
        ]
        for target in forbidden:
            with self.subTest(target=target):
                before = {
                    directory: {entry.name for entry in directory.iterdir()}
                    for directory in (self.fixture.root, *inputs)
                }
                with patch(
                    "recsys.release.build._verified",
                    side_effect=AssertionError("overlapping store reached full validation"),
                ):
                    with self.assertRaisesRegex(ReleaseBuildError, "release_build_integrity_error"):
                        export_release_bundle(
                            self.release, neighbor_path, report, target, **self.common()
                        )
                self.assertEqual(
                    before,
                    {
                        directory: {entry.name for entry in directory.iterdir()}
                        for directory in (self.fixture.root, *inputs)
                    },
                )
                self.assertFalse((target / ".publication.lock").exists())

        special = self.fixture.root / "special-store"
        os.mkfifo(special)
        linked = self.fixture.root / "linked-store"
        linked.symlink_to(self.fixture.root / "separate-store", target_is_directory=True)
        parent_link = self.fixture.root / "linked-parent"
        parent_link.symlink_to(self.fixture.root, target_is_directory=True)
        for target in (special, linked, parent_link / "nested-store"):
            with self.subTest(target=target):
                with patch(
                    "recsys.release.build._verified",
                    side_effect=AssertionError("unsafe store reached full validation"),
                ):
                    with self.assertRaisesRegex(ReleaseBuildError, "release_build_integrity_error"):
                        export_release_bundle(
                            self.release, neighbor_path, report, target, **self.common()
                        )
                self.assertFalse((self.fixture.root / ".publication.lock").exists())

        with tempfile.TemporaryDirectory() as other_root:
            outside = Path(other_root)
            report_dir = outside / "report-input"
            neighbor_dir = outside / "neighbor-input"
            report_dir.mkdir()
            neighbor_dir.mkdir()
            outside_report = report_dir / "build-report.json"
            outside_neighbor = neighbor_dir / "neighbors.json"
            shutil.copyfile(report, outside_report)
            shutil.copyfile(neighbor_path, outside_neighbor)
            outside_before = {entry.name for entry in outside.iterdir()}
            with patch(
                "recsys.release.build._verified",
                side_effect=AssertionError("build-parent ancestor reached full validation"),
            ):
                with self.assertRaisesRegex(ReleaseBuildError, "release_build_integrity_error"):
                    export_release_bundle(
                        self.release, outside_neighbor, outside_report, outside, **self.common()
                    )
            self.assertEqual({entry.name for entry in outside.iterdir()}, outside_before)
            self.assertFalse((outside / ".publication.lock").exists())

        separate = self.fixture.root / "separate-store"
        first = export_release_bundle(
            self.release, neighbor_path, report, separate, **self.common()
        )
        self.assertEqual(check_bundle(first)[1], 2)
        self.assertEqual(
            export_release_bundle(self.release, neighbor_path, report, separate, **self.common()),
            first,
        )
