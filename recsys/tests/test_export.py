"""Small local inputs for provenance checks and immutable publication."""

import hashlib
import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable
from unittest.mock import patch

from recsys import __version__
from recsys.content.glove import _canonical, build
from recsys.contracts.bundle import AnimeRecord, ContractError, check_bundle
from recsys.data.sources import SourceDescriptor, SourceRegistry
from recsys.release.export import ALGORITHM, export_bundle


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _object(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise AssertionError("expected JSON object")
    return value


class ExportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.registry, self.registry_sha = self._fixture(self.root)
        self.catalog = self.root / "catalog.json"
        self.neighbors = self.root / "built" / "neighbors.json"
        self.report = build(
            self.catalog,
            self.root / "normalization-report.json",
            self.root / "glove.txt",
            self.root / "built",
            block_size=1,
            registry=self.registry,
            registry_sha256=self.registry_sha,
        )
        self.normalization = self.root / "normalization-report.json"
        self.inputs = (self.catalog, self.neighbors, self.report, self.normalization)

    @staticmethod
    def _entry(
        title: str,
        aliases: list[str] | None = None,
        genres: list[str] | None = None,
        synopsis: str | None = None,
    ) -> AnimeRecord:
        return {
            "title": title,
            "aliases": aliases or [],
            "genres": genres or [],
            "score": None,
            "year": None,
            "type": None,
            "episodes": None,
            "synopsis": synopsis,
        }

    def _fixture(
        self,
        root: Path,
        anime: dict[str, AnimeRecord] | None = None,
        glove_lines: list[str] | None = None,
    ) -> tuple[SourceRegistry, str]:
        anime = anime or {"2": self._entry("alpha"), "10": self._entry("alpha", ["beta"])}
        glove_lines = glove_lines or [
            "alpha " + " ".join(["1"] + ["0"] * 299),
            "beta " + " ".join(["0", "1"] + ["0"] * 298),
        ]
        glove_raw = ("\n".join(glove_lines) + "\n").encode()
        (root / "glove.txt").write_bytes(glove_raw)
        catalog_raw = _canonical({"schema_version": 1, "anime": anime})
        (root / "catalog.json").write_bytes(catalog_raw)
        mal: list[SourceDescriptor] = [
            {
                "id": name,
                "filename": name + ".csv",
                "size": 1,
                "sha256": sha(name.encode()),
                "url": "https://example.org/" + name,
            }
            for name in ("mal_anime", "mal_synopsis")
        ]
        registry: SourceRegistry = {
            "schema_version": 1,
            "provenance": {
                "mal": {"commit": "test", "repository": "https://example.org"},
                "glove": {"project": "https://example.org", "pretrained_vectors_license": "test"},
            },
            "sources": mal
            + [
                {
                    "id": "glove_zip",
                    "filename": "glove.zip",
                    "size": 1,
                    "sha256": sha(b"zip"),
                    "url": "https://example.org/glove.zip",
                },
                {
                    "id": "glove_300d",
                    "filename": "glove.txt",
                    "size": len(glove_raw),
                    "sha256": sha(glove_raw),
                    "archive_id": "glove_zip",
                    "member": "glove.txt",
                },
            ],
        }
        registry_sha = sha(_canonical(registry))
        report = {
            "schema_version": 1,
            "normalizer_version": "1",
            "registry_sha256": registry_sha,
            "provenance": registry["provenance"],
            "sources": mal,
            "files": {"catalog.json": {"sha256": sha(catalog_raw)}},
            "counts": {"retained": len(anime)},
        }
        (root / "normalization-report.json").write_bytes(_canonical(report))
        return registry, registry_sha

    def export(self, store: str = "store") -> Path:
        return export_bundle(
            *self.inputs,
            self.root / store,
            registry=self.registry,
            registry_sha256=self.registry_sha,
        )

    def mutate(self, path: Path, change: Callable[[dict[str, object]], object]) -> None:
        value = json.loads(path.read_bytes())
        change(_object(value))
        path.write_bytes(_canonical(value))

    def test_pinned_profile_matches_actual_builder(self) -> None:
        self.assertEqual(json.loads(self.report.read_bytes())["algorithm"], ALGORITHM)
        self.assertEqual(
            json.loads(self.report.read_bytes())["producer"]["package_version"], __version__
        )

    def test_legacy_export_rejects_release_report_shape(self) -> None:
        self.mutate(
            self.report,
            lambda value: value.update(
                schema_version=2, kind="arb-release-build", builder_version="arb031-v1"
            ),
        )
        with self.assertRaises(ContractError):
            self.export()

    def test_determinism_byte_preservation_and_existing_target(self) -> None:
        first = self.export("a")
        second = self.export("b")
        self.assertEqual(first.name, second.name)
        self.assertEqual(
            {p.name for p in first.iterdir()}, {"catalog.json", "neighbors.json", "manifest.json"}
        )
        for name in ("catalog.json", "neighbors.json", "manifest.json"):
            self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())
        self.assertEqual((first / "catalog.json").read_bytes(), self.catalog.read_bytes())
        self.assertEqual((first / "neighbors.json").read_bytes(), self.neighbors.read_bytes())
        identity, count = check_bundle(first)
        self.assertEqual(first.name, identity.replace(":", "-"))
        self.assertEqual(count, 2)
        inode = first.stat().st_ino
        self.assertEqual(self.export("a"), first)
        self.assertEqual(first.stat().st_ino, inode)
        manifest = json.loads((first / "manifest.json").read_bytes())
        self.assertEqual(len(manifest["sources"]), 8)
        self.assertEqual(manifest["sources"][0]["name"], "build_report")

    def test_provenance_change_changes_identity_with_same_artifacts(self) -> None:
        first = self.export("a")
        self.mutate(
            self.report,
            lambda value: _object(value["execution"]).update(
                requested_block_rows=2, effective_block_rows=2
            ),
        )
        second = self.export("a")
        self.assertNotEqual(first, second)
        self.assertEqual(
            (first / "neighbors.json").read_bytes(), (second / "neighbors.json").read_bytes()
        )

    def test_report_corruption_rejected_before_store(self) -> None:
        mutations = [
            lambda x: x["files"]["neighbors.json"].update(sha256=sha(b"wrong")),
            lambda x: x["inputs"]["normalization-report.json"].update(sha256=sha(b"wrong")),
            lambda x: x.update(registry_sha256=sha(b"wrong")),
            lambda x: x["sources"][0]["metadata"].update(size=0),
            lambda x: x.update(builder_version="future"),
            lambda x: x["producer"]["code_files"].update(**{"build.py": sha(b"wrong")}),
            lambda x: x["algorithm"]["parameters"].update(dimension=299),
            lambda x: x["coverage"].update(token_occurrences=0),
            lambda x: x["anime"]["2"].update(matched_token_count=999),
            lambda x: x["anime"]["2"].update(vector_status="oov_only"),
            lambda x: x["anime"]["2"].update(neighbor_count=0),
        ]
        original = self.report.read_bytes()
        for i, mutation in enumerate(mutations):
            with self.subTest(i=i):
                self.report.write_bytes(original)
                self.mutate(self.report, mutation)
                with self.assertRaises(ContractError):
                    self.export(f"bad-{i}")
                self.assertFalse((self.root / f"bad-{i}").exists())
        self.report.write_bytes(original)

    def test_corrupt_catalog_neighbor_and_normalization_rejected(self) -> None:
        for path, edit in (
            (self.catalog, b"bad"),
            (self.neighbors, b"bad"),
            (self.normalization, b"bad"),
        ):
            with self.subTest(path=path):
                original = path.read_bytes()
                path.write_bytes(edit)
                with self.assertRaises(ContractError):
                    self.export(f"bad-{path.name}")
                self.assertFalse((self.root / f"bad-{path.name}").exists())
                path.write_bytes(original)

    def test_existing_conflicts_are_untouched(self) -> None:
        valid = self.export("source")
        for kind in ("incomplete", "extra", "symlink", "file"):
            store = self.root / kind
            store.mkdir()
            target = store / valid.name
            if kind == "symlink":
                target.symlink_to(valid, target_is_directory=True)
            elif kind == "file":
                target.write_bytes(b"keep")
            else:
                target.mkdir()
                (target / "catalog.json").write_bytes(b"keep")
                if kind == "extra":
                    for name in ("neighbors.json", "manifest.json"):
                        (target / name).write_bytes((valid / name).read_bytes())
                    (target / "extra").write_bytes(b"keep")
            with self.subTest(kind=kind), self.assertRaises(ContractError):
                self.export(kind)
            self.assertTrue(target.exists() or target.is_symlink())

    def test_stage_validation_failure_leaves_no_published_directory(self) -> None:
        store = self.root / "failed"
        with patch(
            "recsys.release.export.check_bundle", side_effect=ContractError("staged failure")
        ):
            with self.assertRaisesRegex(ContractError, "staged failure"):
                self.export("failed")
        self.assertEqual([p.name for p in store.iterdir()], [".publication.lock"])

    def test_staged_write_failure_and_symlink_lock_are_safe(self) -> None:
        store = self.root / "write-failed"
        with patch("recsys.release.export.os.fsync", side_effect=OSError("sync failed")):
            with self.assertRaisesRegex(OSError, "sync failed"):
                self.export("write-failed")
        self.assertEqual([p.name for p in store.iterdir()], [".publication.lock"])
        lock_store = self.root / "unsafe-lock"
        lock_store.mkdir()
        (lock_store / ".publication.lock").symlink_to(self.report)
        with self.assertRaises(OSError):
            self.export("unsafe-lock")
        self.assertEqual([p.name for p in lock_store.iterdir()], [".publication.lock"])

    def test_cooperating_concurrent_exports(self) -> None:
        with ThreadPoolExecutor(max_workers=3) as pool:
            paths = list(pool.map(lambda _: self.export("concurrent"), range(3)))
        self.assertEqual(paths, [paths[0]] * 3)
        self.assertEqual(check_bundle(paths[0])[1], 2)
