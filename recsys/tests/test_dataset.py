"""Small end-to-end research dataset fixtures and integrity checks."""

import csv
import gzip
import hashlib
import io
import json
import tempfile
import unittest
from importlib import resources
from pathlib import Path

from recsys.bundle import canonical_bytes
from recsys.dataset import (
    DatasetError,
    DatasetSpec,
    _validate_interactions,
    normalize_dataset,
    validate_dataset,
)
from recsys.dataset_interactions import CSV_COLUMNS, HEADER
from recsys.dataset_metadata import DETAIL_COLUMNS


def _write_csv(path, fields, records):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        writer.writerows(records)


def _detail(mal, title, date, status, year):
    row = {field: "" for field in DETAIL_COLUMNS}
    row.update(
        mal_id=str(mal),
        title=title,
        title_japanese=title + " JP",
        start_date=date,
        status=status,
        year=str(year),
        genres="['Drama']",
        score="8.0",
        episodes="12.0",
        synopsis="A story.",
        type="TV",
    )
    return [row[field] for field in DETAIL_COLUMNS]


def _gzip(raw):
    target = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=target, compresslevel=6, mtime=0) as stream:
        stream.write(raw)
    return target.getvalue()


class Fixture:
    def __init__(self, root):
        self.root = Path(root)
        self.cache = self.root / "cache"
        self.cache.mkdir()
        _write_csv(
            self.cache / "details.csv",
            DETAIL_COLUMNS,
            [
                _detail(1, "One", "2025-01-01", "Finished Airing", 2025),
                _detail(2, "Two", "2024-02-01", "Currently Airing", 2024),
            ],
        )
        self.ratings = [
            ["A", "1", "completed", "8", "False", "1"],
            ["A", "1", "completed", "8", "False", "1"],
            ["A", "2", "watching", "9", "False", "1"],
            ["B", "2", "dropped", "7", "False", "1"],
            ["C", "2", "plan_to_watch", "0", "False", "1"],
            ["", "1", "completed", "6", "False", "1"],
            ["A", "2", "plan_to_watch", "0", "False", "1"],
        ]
        _write_csv(self.cache / "ratings.csv", CSV_COLUMNS, self.ratings)
        sources = []
        for name, source_id in (
            ("details.csv", "neelagiri_details_csv"),
            ("ratings.csv", "neelagiri_ratings_csv"),
        ):
            raw = (self.cache / name).read_bytes()
            sources.append(
                {
                    "id": source_id,
                    "filename": name,
                    "size": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "url": f"https://example.invalid/{name}",
                }
            )
        registry = {
            "schema_version": 1,
            "provenance": {"dataset": {"snapshot_id": "neelagiri-2025-v1"}},
            "sources": sources,
        }
        registry_bytes = canonical_bytes(registry)
        manifest = {
            "schema_version": 1,
            "registry_sha256": hashlib.sha256(registry_bytes).hexdigest(),
            "provenance": registry["provenance"],
            "verified": [
                {k: source[k] for k in ("filename", "size", "sha256")} for source in sources
            ],
        }
        (self.cache / "source-manifest.json").write_bytes(canonical_bytes(manifest))
        counts = {
            key: 0
            for key in json.loads(
                resources.files("recsys")
                .joinpath("neelagiri_2025_v1_expectations.json")
                .read_text()
            )["counts"]
        }
        counts.update(
            metadata_rows=2,
            catalog_items=2,
            raw_rows=7,
            blank_user_rows=1,
            zero_or_missing_rows=2,
            valid_explicit_rows=4,
            raw_explicit_pairs=3,
            repeated_explicit_triples=1,
            raw_mapped_pairs=3,
            accepted_pairs=2,
            accepted_users=2,
            accepted_rated_mal_ids=2,
            completed_only_pairs=1,
            completed_only_users=1,
            recent_2024_plus=2,
            recent_2025_plus=1,
            mixed_unset_pairs=1,
        )
        distributions = {
            "raw_status_histogram": {
                "completed": 3,
                "watching": 1,
                "dropped": 1,
                "plan_to_watch": 2,
            },
            "accepted_status_histogram": {"completed": 1, "dropped": 1},
            "accepted_score_histogram": {"8": 1, "7": 1},
            "rated_items_per_user_bins": {"0": 2},
            "positive_items_per_user_bins": {"0": 2},
        }
        expectations = {
            "schema_version": 1,
            "snapshot_id": "neelagiri-2025-v1",
            "source_sha256": {s["filename"]: s["sha256"] for s in sources},
            "counts": counts,
            "distributions": distributions,
        }
        self.spec = DatasetSpec(
            "neelagiri-2025-v1",
            registry_bytes,
            "neelagiri_2025_v1_expectations.json",
            canonical_bytes(expectations),
        )

    def run(self, name="output", *, threads=2):
        output = self.root / name
        normalize_dataset(
            "neelagiri-2025-v1",
            self.cache,
            output,
            scratch_dir=self.root / "scratch",
            threads=threads,
            _spec=self.spec,
        )
        return output


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.fixture = Fixture(self.tmp.name)

    def test_full_fixture_determinism_and_stream_validation(self):
        first = self.fixture.run("first", threads=1)
        second = self.fixture.run("second", threads=2)
        self.assertEqual(
            {p.name: p.read_bytes() for p in first.iterdir()},
            {p.name: p.read_bytes() for p in second.iterdir()},
        )
        summary = validate_dataset(first, _spec=self.fixture.spec)
        self.assertEqual((summary.catalog_items, summary.interactions, summary.users), (2, 2, 2))
        qa = json.loads((first / "dataset-qa.json").read_bytes())
        self.assertEqual(qa["counts"]["raw_nonblank_users"], 3)
        self.assertEqual(qa["counts"]["raw_explicit_mal_ids"], 2)
        self.assertEqual(qa["counts"]["unrated_or_invalid_only_pairs"], 1)
        self.assertEqual(
            qa["distributions"]["raw_score_histogram"], {"6": 1, "7": 1, "8": 2, "9": 1, "unset": 2}
        )
        self.assertEqual(len(list(first.iterdir())), 5)
        self.assertEqual(first.stat().st_mode & 0o777, 0o700)
        self.assertTrue(all((path.stat().st_mode & 0o777) == 0o600 for path in first.iterdir()))
        self.assertEqual(list((self.fixture.root / "scratch").iterdir()), [])
        with self.assertRaisesRegex(DatasetError, "dataset_integrity_error"):
            self.fixture.run("first")

    def test_permuted_source_rows_preserve_interactions_but_change_identity(self):
        first = self.fixture.run("first")
        alternative_root = self.fixture.root / "alternative"
        alternative_root.mkdir()
        alternative = Fixture(alternative_root)
        _write_csv(
            alternative.cache / "ratings.csv", CSV_COLUMNS, list(reversed(alternative.ratings))
        )
        ratings_raw = (alternative.cache / "ratings.csv").read_bytes()
        registry = json.loads(alternative.spec.registry_bytes)
        source = next(item for item in registry["sources"] if item["filename"] == "ratings.csv")
        source["size"] = len(ratings_raw)
        source["sha256"] = hashlib.sha256(ratings_raw).hexdigest()
        registry_bytes = canonical_bytes(registry)
        cache_manifest = json.loads((alternative.cache / "source-manifest.json").read_bytes())
        cache_manifest["registry_sha256"] = hashlib.sha256(registry_bytes).hexdigest()
        cache_manifest["verified"][1]["size"] = source["size"]
        cache_manifest["verified"][1]["sha256"] = source["sha256"]
        (alternative.cache / "source-manifest.json").write_bytes(canonical_bytes(cache_manifest))
        expected = json.loads(alternative.spec.expectations_bytes)
        expected["source_sha256"]["ratings.csv"] = source["sha256"]
        alternative.spec = DatasetSpec(
            "neelagiri-2025-v1",
            registry_bytes,
            "neelagiri_2025_v1_expectations.json",
            canonical_bytes(expected),
        )
        second = alternative.run("second")
        self.assertEqual(
            (first / "interactions.tsv.gz").read_bytes(),
            (second / "interactions.tsv.gz").read_bytes(),
        )
        self.assertNotEqual(
            validate_dataset(first, _spec=self.fixture.spec).identity,
            validate_dataset(second, _spec=alternative.spec).identity,
        )

    def test_pinned_expectations_match_source_audit(self):
        repository = Path(__file__).parents[2]
        report = json.loads(
            (
                repository / "recsys" / "reports" / "dataset-audit-v1" / "source-audit.json"
            ).read_bytes()
        )
        audit = next(item for item in report["candidates"] if item["id"] == "neelagiri-2025-v1")
        pinned = json.loads(
            resources.files("recsys").joinpath("neelagiri_2025_v1_expectations.json").read_bytes()
        )
        pairs = {
            "raw_rows": "raw_interaction_rows",
            "valid_explicit_rows": "valid_rated_rows",
            "raw_explicit_pairs": "distinct_valid_rated_source_pairs",
            "raw_mapped_pairs": "unambiguously_joined_source_pairs",
            "accepted_pairs": "accepted_distinct_user_mal_pairs",
            "accepted_users": "accepted_rated_users",
            "positive_users_ge_15": "accepted_positive_users_ge_15",
            "accepted_rated_mal_ids": "accepted_rated_mal_ids",
            "recent_2024_plus": "accepted_rated_2024_plus",
            "recent_2025_plus": "accepted_rated_2025_plus",
        }
        for key, audit_key in pairs.items():
            self.assertEqual(pinned["counts"][key], audit["counts"][audit_key])
        self.assertEqual(
            pinned["distributions"]["raw_status_histogram"],
            audit["distributions"]["status_histogram"],
        )
        self.assertEqual(
            pinned["distributions"]["accepted_status_histogram"],
            audit["distributions"]["accepted_status_histogram"],
        )
        self.assertEqual(
            pinned["distributions"]["rated_items_per_user_bins"],
            audit["distributions"]["rated_items_per_user_bins"],
        )

    def test_corrupt_cache_preserves_target_absence(self):
        with (self.fixture.cache / "ratings.csv").open("ab") as stream:
            stream.write(b"canary-private-username")
        with self.assertRaisesRegex(DatasetError, "input_hash_mismatch"):
            self.fixture.run()
        self.assertFalse((self.fixture.root / "output").exists())

    def test_foreign_source_manifest_preserved(self):
        source_manifest = self.fixture.cache / "source-manifest.json"
        changed = json.loads(source_manifest.read_bytes())
        changed["registry_sha256"] = "0" * 64
        raw = canonical_bytes(changed)
        source_manifest.write_bytes(raw)
        with self.assertRaisesRegex(DatasetError, "input_hash_mismatch"):
            self.fixture.run()
        self.assertEqual(source_manifest.read_bytes(), raw)
        self.assertFalse((self.fixture.root / "output").exists())

    def test_fixture_qa_mismatch_cannot_publish(self):
        expected = json.loads(self.fixture.spec.expectations_bytes)
        expected["counts"]["accepted_users"] += 1
        changed = DatasetSpec(
            self.fixture.spec.snapshot_id,
            self.fixture.spec.registry_bytes,
            self.fixture.spec.expectations_resource,
            canonical_bytes(expected),
        )
        with self.assertRaisesRegex(DatasetError, "qa_mismatch"):
            normalize_dataset(
                "neelagiri-2025-v1",
                self.fixture.cache,
                self.fixture.root / "output",
                scratch_dir=self.fixture.root / "scratch",
                _spec=changed,
            )
        self.assertFalse((self.fixture.root / "output").exists())
        self.assertEqual(list((self.fixture.root / "scratch").iterdir()), [])

    def test_frozen_provenance_and_trusted_identity(self):
        output = self.fixture.run()
        manifest_path = output / "dataset-manifest.json"
        manifest = json.loads(manifest_path.read_bytes())
        original = validate_dataset(output, _spec=self.fixture.spec).identity
        manifest["normalizer"]["code_files_sha256"]["dataset.py"] = "0" * 64
        manifest["producer"]["duckdb_version"] = "1.5.0"
        manifest_path.write_bytes(canonical_bytes(manifest))
        self.assertNotEqual(validate_dataset(output, _spec=self.fixture.spec).identity, original)
        with self.assertRaisesRegex(DatasetError, "dataset_integrity_error"):
            validate_dataset(output, expected_identity=original, _spec=self.fixture.spec)
        manifest["parameters"]["positive_threshold"] = 9
        manifest_path.write_bytes(canonical_bytes(manifest))
        with self.assertRaisesRegex(DatasetError, "dataset_integrity_error"):
            validate_dataset(output, _spec=self.fixture.spec)

    def test_qa_tampering_and_truncated_gzip(self):
        output = self.fixture.run()
        manifest_path = output / "dataset-manifest.json"
        qa_path = output / "dataset-qa.json"
        qa = json.loads(qa_path.read_bytes())
        qa["counts"]["accepted_users"] = 3
        qa_path.write_bytes(canonical_bytes(qa))
        manifest = json.loads(manifest_path.read_bytes())
        raw = qa_path.read_bytes()
        manifest["files"]["dataset-qa.json"] = {
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
        manifest_path.write_bytes(canonical_bytes(manifest))
        with self.assertRaisesRegex(DatasetError, "dataset_integrity_error"):
            validate_dataset(output, _spec=self.fixture.spec)
        qa["counts"]["accepted_users"] = 2
        qa_path.write_bytes(canonical_bytes(qa))
        compressed = output / "interactions.tsv.gz"
        compressed.write_bytes(compressed.read_bytes()[:-4])
        with self.assertRaisesRegex(DatasetError, "dataset_integrity_error"):
            validate_dataset(output, _spec=self.fixture.spec)

    def test_invalid_fixture_expectations_cannot_bypass_pins(self):
        resource = json.loads(self.fixture.spec.expectations_bytes)
        del resource["counts"]["accepted_pairs"]
        bad = DatasetSpec(
            self.fixture.spec.snapshot_id,
            self.fixture.spec.registry_bytes,
            self.fixture.spec.expectations_resource,
            canonical_bytes(resource),
        )
        with self.assertRaisesRegex(DatasetError, "dataset_integrity_error"):
            normalize_dataset(
                "neelagiri-2025-v1",
                self.fixture.cache,
                self.fixture.root / "bad",
                scratch_dir=self.fixture.root / "scratch",
                _spec=bad,
            )
        self.assertFalse((self.fixture.root / "bad").exists())
        with self.assertRaisesRegex(DatasetError, "dataset_integrity_error"):
            normalize_dataset(
                "other-snapshot",
                self.fixture.cache,
                self.fixture.root / "other",
                scratch_dir=self.fixture.root / "scratch",
                _spec=self.fixture.spec,
            )

    def test_sorted_unique_pairs_and_single_gzip_member(self):
        output = self.fixture.run()
        compressed = output / "interactions.tsv.gz"
        plain = gzip.decompress(compressed.read_bytes())
        lines = plain.splitlines(keepends=True)
        for changed in (
            b"".join((lines[0], lines[2], lines[1])),
            b"".join((lines[0], lines[1], lines[1])),
        ):
            compressed.write_bytes(_gzip(changed))
            self._refresh_interactions_manifest(output, changed)
            with self.assertRaisesRegex(DatasetError, "dataset_integrity_error"):
                validate_dataset(output, _spec=self.fixture.spec)
        compressed.write_bytes(_gzip(plain) * 2)
        self._refresh_interactions_manifest(output, plain * 2)
        with self.assertRaisesRegex(DatasetError, "dataset_integrity_error"):
            validate_dataset(output, _spec=self.fixture.spec)

    def test_streaming_validator_handles_high_compression_ratio(self):
        key = b"u1_" + b"0" * 64
        raw = HEADER + b"".join(
            key + b"\t" + str(mal).encode("ascii") + b"\t8\tcompleted\n"
            for mal in range(1, 250_001)
        )
        self.assertGreater(len(raw), 16 * 1024 * 1024)
        path = self.fixture.root / "high-compression.gz"
        compressed = _gzip(raw)
        self.assertLess(len(compressed), 1024 * 1024)
        path.write_bytes(compressed)
        entry = {
            "rows": 250_000,
            "uncompressed_bytes": len(raw),
            "canonical_sha256": hashlib.sha256(raw).hexdigest(),
        }
        derived, rated_ids = _validate_interactions(
            path, dict.fromkeys(map(str, range(1, 250_001))), entry
        )
        self.assertEqual(derived["counts"]["accepted_pairs"], 250_000)
        self.assertEqual(derived["counts"]["accepted_users"], 1)
        self.assertEqual(len(rated_ids), 250_000)

    @staticmethod
    def _refresh_interactions_manifest(output, plain):
        manifest_path = output / "dataset-manifest.json"
        manifest = json.loads(manifest_path.read_bytes())
        raw = (output / "interactions.tsv.gz").read_bytes()
        manifest["files"]["interactions.tsv.gz"] = {
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "rows": len(plain.splitlines()) - 1,
            "uncompressed_bytes": len(plain),
            "canonical_sha256": hashlib.sha256(plain).hexdigest(),
        }
        manifest_path.write_bytes(canonical_bytes(manifest))


if __name__ == "__main__":
    unittest.main()
