"""Validate the packaged source registry and descriptor safety."""

import copy
import json
import unittest
from pathlib import Path
from typing import cast

from recsys.data.sources import (
    DEFAULT_SNAPSHOT,
    SNAPSHOT_IDS,
    SourceError,
    SourceRegistry,
    load_registry,
    validate_registry,
)


def _object(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise AssertionError("expected source-audit object with string keys")
    return cast("dict[str, object]", value)


def _items(value: object) -> list[object]:
    if not isinstance(value, list):
        raise AssertionError("expected source-audit list")
    return cast("list[object]", value)


class SourceTests(unittest.TestCase):
    def assert_registry_matches_audit(
        self, registry: SourceRegistry, candidate: dict[str, object]
    ) -> None:
        """Compare selected source pins with the retained source audit."""
        artifacts = [_object(item) for item in _items(candidate["artifacts"])]
        artifact_ids: list[str] = []
        for artifact in artifacts:
            artifact_id = artifact.get("id")
            if not isinstance(artifact_id, str) or not artifact_id:
                raise AssertionError("audit artifact id must be a nonempty string")
            self.assertNotIn(artifact_id, artifact_ids)
            artifact_ids.append(artifact_id)
        by_id = dict(zip(artifact_ids, artifacts, strict=True))
        for source in registry["sources"]:
            self.assertIn(source["id"], by_id)
            artifact = by_id[source["id"]]
            self.assertEqual(
                (source["url"], source["size"], source["sha256"]),
                (
                    artifact["stable_url"],
                    artifact["observed_bytes"],
                    artifact["observed_sha256"],
                ),
            )

    def test_packaged_registry_has_pinned_inputs(self) -> None:
        registry, digest = load_registry()
        self.assertEqual(digest, "49bb399935f0ac7ec649fe2f4bd7715444c09c8513704866fa90b9bea29d182f")
        self.assertEqual(DEFAULT_SNAPSHOT, "mal-2020-glove-v1")
        self.assertEqual(
            SNAPSHOT_IDS,
            (DEFAULT_SNAPSHOT, "neelagiri-2025-v1", "nafiul-2026-v1", "glove-6b-300d-v1"),
        )
        self.assertEqual(
            [(item["filename"], item["size"]) for item in registry["sources"]],
            [
                ("anime.csv", 5662362),
                ("anime_with_synopsis.csv", 7221844),
                ("glove.6B.zip", 862182613),
                ("glove.6B.300d.txt", 1037962819),
            ],
        )
        self.assertEqual(
            _object(registry["provenance"]["mal"])["commit"],
            "9a1d7f56482accbde99e24cd7cfcad65e2d24b1f",
        )

    def test_glove_selector_is_exact_legacy_projection(self) -> None:
        legacy, _ = load_registry()
        selected, _ = load_registry("glove-6b-300d-v1")
        self.assertEqual(
            selected,
            {
                "schema_version": 1,
                "provenance": {"glove": legacy["provenance"]["glove"]},
                "sources": legacy["sources"][2:],
            },
        )

    def test_new_packaged_registry_matches_audit_pins(self) -> None:
        registry, digest = load_registry("neelagiri-2025-v1")
        self.assertEqual(len(digest), 64)
        self.assertEqual(
            [item["filename"] for item in registry["sources"]], ["details.csv", "ratings.csv"]
        )
        self.assertEqual(
            [item["id"] for item in registry["sources"]],
            ["neelagiri_details_csv", "neelagiri_ratings_csv"],
        )
        self.assertEqual(
            registry["provenance"]["dataset"],
            {
                "snapshot_id": "neelagiri-2025-v1",
                "provider": "Kaggle",
                "slug": "neelagiriaditya/anime-dataset-jan-1917-to-oct-2025",
                "version": 1,
                "page_url": "https://www.kaggle.com/datasets/neelagiriaditya/anime-dataset-jan-1917-to-oct-2025",
                "declared_license": "CC BY-NC-SA 4.0",
                "intake_scope": "noncommercial-research",
                "exact_collection_date": None,
            },
        )
        root = Path(__file__).resolve().parents[1]
        audit = _object(
            json.loads((root / "reports/dataset-audit-v1/source-audit.json").read_text())
        )
        selected = next(
            _object(item)
            for item in _items(audit["candidates"])
            if _object(item)["id"] == "neelagiri-2025-v1"
        )
        self.assert_registry_matches_audit(registry, selected)

    def test_nafiul_metadata_only_registry_and_unchanged_legacy_pin(self) -> None:
        registry, digest = load_registry("nafiul-2026-v1")
        self.assertEqual(digest, "56383b6fca667d1352ff9df7034b2d01415ef3a08aad1a82734f537defb26879")
        self.assertEqual(
            registry["sources"],
            [
                {
                    "id": "nafiul_anime_csv",
                    "filename": "anime_info.csv",
                    "size": 22314208,
                    "sha256": "e833476c0da3eadd4c033889e1e8defb2808baf95b0ae38b55009f45d5d72c35",
                    "url": "https://www.kaggle.com/api/v1/datasets/download/nafiulislam490/"
                    "comprehensive-myanimelist-mal-dataset-2026/anime_info.csv?datasetVersionNumber=1",
                }
            ],
        )
        self.assertEqual(
            _object(registry["provenance"]["dataset"])["artifact_scope"], "anime-info-only"
        )
        self.assertEqual(_object(registry["provenance"]["dataset"])["exact_collection_date"], None)
        self.assertEqual(
            load_registry()[1], "49bb399935f0ac7ec649fe2f4bd7715444c09c8513704866fa90b9bea29d182f"
        )
        root = Path(__file__).resolve().parents[1]
        audit = _object(
            json.loads((root / "reports/dataset-audit-v1/source-audit.json").read_text())
        )
        selected = next(
            _object(item)
            for item in _items(audit["candidates"])
            if _object(item)["id"] == "nafiul-2026-v1"
        )
        self.assert_registry_matches_audit(registry, selected)

    def test_audit_pins_reject_changed_hash_version_size_and_duplicate_id(self) -> None:
        registry, _ = load_registry("neelagiri-2025-v1")
        root = Path(__file__).resolve().parents[1]
        audit = _object(
            json.loads((root / "reports/dataset-audit-v1/source-audit.json").read_text())
        )
        selected = next(
            _object(item)
            for item in _items(audit["candidates"])
            if _object(item)["id"] == "neelagiri-2025-v1"
        )
        first = registry["sources"][0]
        for field, value in (
            ("observed_sha256", f"{first['sha256']}20"),
            (
                "stable_url",
                first["url"].replace("datasetVersionNumber=1", "datasetVersionNumber=10"),
            ),
            ("observed_bytes", first["size"] + 1),
            ("observed_bytes", first["size"] - 1),
        ):
            with self.subTest(field=field, value=value):
                candidate = copy.deepcopy(selected)
                artifact = _object(_items(candidate["artifacts"])[0])
                artifact[field] = value
                with self.assertRaises(AssertionError):
                    self.assert_registry_matches_audit(registry, candidate)

        for duplicate_id in (registry["sources"][0]["id"], None):
            with self.subTest(duplicate_id=duplicate_id):
                candidate = copy.deepcopy(selected)
                artifacts = _items(candidate["artifacts"])
                _object(artifacts[1])["id"] = duplicate_id
                with self.assertRaises(AssertionError):
                    self.assert_registry_matches_audit(registry, candidate)

    def test_unknown_snapshot_rejected_before_resource_lookup(self) -> None:
        for snapshot in ("latest", "../source_registry.json", "NEELAGIRI"):
            with self.subTest(snapshot=snapshot), self.assertRaises(SourceError):
                load_registry(snapshot)
        with self.subTest(snapshot=None), self.assertRaises(SourceError):
            load_registry(None)  # type: ignore[arg-type]  # intentional invalid runtime input

    def test_rejects_unsafe_filenames_and_archive_references(self) -> None:
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
