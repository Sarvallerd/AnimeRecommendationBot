"""Small alias fixtures exercise source parsing, base binding, and publication."""

import csv
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from test_dataset import Fixture as DatasetFixture

from recsys.contracts.bundle import canonical_bytes
from recsys.data.aliases import (
    CSV_COLUMNS,
    AliasError,
    AliasSpec,
    _derive,
    _rows,
    normalize_aliases,
    validate_aliases,
)
from recsys.data.dataset import validate_dataset


def csv_bytes(rows):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(CSV_COLUMNS)
    for mal_id, main, english, url, synonyms in rows:
        row = {name: "" for name in CSV_COLUMNS}
        row.update(
            mal_id=mal_id, title=main, title_english=english, mal_url=url, title_synonyms=synonyms
        )
        writer.writerow([row[name] for name in CSV_COLUMNS])
    return stream.getvalue().encode("utf-8")


def record(mal_id, main, english="", *, url=None, synonyms=""):
    if url is None:
        url = f"https://myanimelist.net/anime/{mal_id}/slug"
    return (mal_id, main, english, url, synonyms)


class Fixture:
    def __init__(self, root):
        self.root = Path(root)
        self.base_fixture = DatasetFixture(self.root)
        self.base = self.base_fixture.run("base")
        self.base_identity = validate_dataset(self.base, _spec=self.base_fixture.spec).identity
        self.base_catalog = json.loads((self.base / "catalog.json").read_bytes())["anime"]
        self.cache = self.root / "alias-cache"
        self.cache.mkdir()
        self.rows = [
            record("1", "Alternate One", "One", synonyms="Ignored"),
            record("2", "Two", "Two JP"),
        ]
        self.update()

    def update(self, rows=None):
        if rows is not None:
            self.rows = rows
        raw = csv_bytes(self.rows)
        (self.cache / "anime_info.csv").write_bytes(raw)
        source = {
            "id": "nafiul_anime_csv",
            "filename": "anime_info.csv",
            "size": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "url": "https://example.invalid/data.csv",
        }
        registry = {
            "schema_version": 1,
            "provenance": {"dataset": {"snapshot_id": "nafiul-2026-v1", "version": 1}},
            "sources": [source],
        }
        registry_bytes = canonical_bytes(registry)
        manifest = {
            "schema_version": 1,
            "registry_sha256": hashlib.sha256(registry_bytes).hexdigest(),
            "provenance": registry["provenance"],
            "verified": [{key: source[key] for key in ("filename", "size", "sha256")}],
        }
        (self.cache / "source-manifest.json").write_bytes(canonical_bytes(manifest))
        _, qa = _derive(raw, self.base_catalog)
        expected = {
            "schema_version": 1,
            "snapshot_id": "nafiul-2026-v1",
            "source_sha256": source["sha256"],
            "base_identity": self.base_identity,
            "counts": qa["counts"],
            "fields": qa["fields"],
        }
        self.spec = AliasSpec(
            "nafiul-2026-v1",
            registry_bytes,
            "nafiul_2026_v1_alias_expectations.json",
            canonical_bytes(expected),
        )

    def run(self, name="aliases"):
        output = self.root / name
        normalize_aliases(
            self.cache,
            self.base,
            output,
            snapshot="nafiul-2026-v1",
            expected_base_identity=self.base_identity,
            _spec=self.spec,
            _base_spec=self.base_fixture.spec,
        )
        return output

    def validate(self, output, expected_identity=None):
        return validate_aliases(
            output,
            data_dir=self.cache,
            base_dir=self.base,
            expected_base_identity=self.base_identity,
            expected_identity=expected_identity,
            _spec=self.spec,
            _base_spec=self.base_fixture.spec,
        )


class AliasTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.fixture = Fixture(self.tmp.name)

    def test_deterministic_fixture_and_preferred_english_even_when_equal_to_base(self):
        a = self.fixture.run("a")
        b = self.fixture.run("b")
        self.assertEqual(
            {file.name: file.read_bytes() for file in a.iterdir()},
            {file.name: file.read_bytes() for file in b.iterdir()},
        )
        summary = self.fixture.validate(a)
        self.assertEqual(
            (summary.items, summary.aliases, summary.english, summary.main), (2, 3, 2, 1)
        )
        manifest = json.loads((a / "alias-manifest.json").read_bytes())
        self.assertEqual(manifest["base"]["catalog_items"], 2)
        qa = json.loads((a / "alias-qa.json").read_bytes())
        self.assertEqual(
            {
                key: qa["counts"][key]
                for key in (
                    "source_rows",
                    "eligible_overlap",
                    "emitted_items",
                    "emitted_strings",
                    "accepted_english_eq_base_primary",
                    "accepted_english_eq_base_alias",
                    "redundant_main",
                    "ignored_nonempty_synonyms_eligible",
                )
            },
            {
                "source_rows": 2,
                "eligible_overlap": 2,
                "emitted_items": 2,
                "emitted_strings": 3,
                "accepted_english_eq_base_primary": 1,
                "accepted_english_eq_base_alias": 1,
                "redundant_main": 1,
                "ignored_nonempty_synonyms_eligible": 1,
            },
        )
        result = json.loads((a / "aliases.json").read_bytes())["aliases"]
        self.assertEqual(
            result["1"],
            [
                {"source_field": "title_english", "text": "One"},
                {"source_field": "title", "text": "Alternate One"},
            ],
        )
        self.assertEqual(result["2"], [{"source_field": "title_english", "text": "Two JP"}])
        self.assertEqual(len(list(a.iterdir())), 3)
        self.assertEqual(a.stat().st_mode & 0o777, 0o700)
        self.assertTrue(all(path.stat().st_mode & 0o777 == 0o600 for path in a.iterdir()))
        self.assertEqual(self.fixture.validate(b).identity, summary.identity)
        self.assertEqual(
            json.loads((self.fixture.base / "catalog.json").read_bytes())["anime"],
            self.fixture.base_catalog,
        )

    def test_duplicate_ids_quarantine_all_copies_and_foreign_ids_do_not_emit(self):
        rows = [
            record("1", "First", "English One"),
            record("1", "Second", "Other"),
            record("2", "Two", "Two English"),
            record("999", "Foreign", "Outside"),
        ]
        self.fixture.update(rows)
        result = self.fixture.run()
        aliases = json.loads((result / "aliases.json").read_bytes())["aliases"]
        qa = json.loads((result / "alias-qa.json").read_bytes())["counts"]
        self.assertEqual(list(aliases), ["2"])
        self.assertEqual(
            (
                qa["duplicate_id_groups"],
                qa["duplicate_id_rows"],
                qa["duplicate_overlap_groups"],
                qa["unique_outside"],
            ),
            (1, 2, 1, 1),
        )
        self.assertEqual(qa["ignored_nonempty_synonyms_source"], 0)

    def test_canonical_ids_and_exact_url_reject_only_affected_rows(self):
        rows = [
            record("1", "One Alternative", "EN", url="http://myanimelist.net/anime/1"),
            record("2", "Two Alternative", "EN2", url="https://myanimelist.net:443/anime/2"),
            record("01", "Noncanonical", "English"),
            record("3", "Unknown", "English", url="https://myanimelist.net/anime/3?x=1"),
        ]
        _, qa = _derive(csv_bytes(rows), self.fixture.base_catalog)
        self.assertEqual(
            (
                qa["counts"]["invalid_id_rows"],
                qa["counts"]["invalid_url_rows"],
                qa["counts"]["invalid_url_overlap_rows"],
                qa["counts"]["emitted_items"],
            ),
            (1, 3, 2, 0),
        )
        for value in ("0", "01", "+1", " 1", "1.0", "2147483648", "١", "1\u200b"):
            with self.subTest(value=value):
                _, qa = _derive(
                    csv_bytes([record(value, "Candidate", "English")]), self.fixture.base_catalog
                )
                self.assertEqual(qa["counts"]["invalid_id_rows"], 1)

    def test_field_quarantine_nfc_and_no_transliteration(self):
        rows = [
            record("1", "Cafe\u0301", "Unknown"),
            record("2", "Two\u200b", "Γάμμα", synonyms="Novel|English"),
        ]
        aliases, qa = _derive(csv_bytes(rows), self.fixture.base_catalog)
        self.assertEqual(aliases["aliases"]["1"], [{"source_field": "title", "text": "Café"}])
        self.assertEqual(
            aliases["aliases"]["2"], [{"source_field": "title_english", "text": "Γάμμα"}]
        )
        self.assertEqual(qa["fields"]["eligible_overlap"]["title"]["rejected_Cf"], 1)
        self.assertEqual(qa["fields"]["eligible_overlap"]["title"]["nfc_changed"], 1)
        self.assertEqual(qa["fields"]["eligible_overlap"]["title_english"]["sentinel"], 1)
        self.assertEqual(qa["counts"]["emitted_english_not_latin_eligible"], 1)
        self.assertEqual(qa["counts"]["ignored_nonempty_synonyms_eligible"], 1)
        for bad in ("\ufffd", "Bad\nText", "Hidden\u200b", "\ud800"):
            # Surrogates cannot be serialized into a valid UTF-8 source; the first
            # three exercise independent field quarantine in the derived rows.
            if bad == "\ud800":
                continue
            aliases, qa = _derive(csv_bytes([record("1", "One", bad)]), self.fixture.base_catalog)
            self.assertNotIn("1", aliases["aliases"])
            self.assertEqual(qa["fields"]["eligible_overlap"]["title_english"]["accepted"], 0)

    def test_bad_csv_header_width_and_utf8(self):
        raw = csv_bytes(self.fixture.rows)
        for changed in (
            raw.replace(b"mal_id", b"other_", 1),
            raw.replace(b"title,title_english", b"title,title", 1),
            raw + b"1,too-short\n",
            raw + b"\xff",
        ):
            with (
                self.subTest(changed=changed[-12:]),
                self.assertRaisesRegex(AliasError, "alias_csv_invalid"),
            ):
                _rows(changed)

    def test_mismatched_fixture_expectations_and_source_manifest(self):
        expectations = json.loads(self.fixture.spec.expectations_bytes)
        expectations["counts"]["emitted_items"] += 1
        altered = AliasSpec(
            self.fixture.spec.snapshot_id,
            self.fixture.spec.registry_bytes,
            self.fixture.spec.expectations_resource,
            canonical_bytes(expectations),
        )
        with self.assertRaisesRegex(AliasError, "alias_qa_mismatch"):
            normalize_aliases(
                self.fixture.cache,
                self.fixture.base,
                self.fixture.root / "wrong",
                snapshot="nafiul-2026-v1",
                expected_base_identity=self.fixture.base_identity,
                _spec=altered,
                _base_spec=self.fixture.base_fixture.spec,
            )
        self.assertFalse((self.fixture.root / "wrong").exists())
        source = self.fixture.cache / "anime_info.csv"
        source.write_bytes(source.read_bytes() + b"\n")
        with self.assertRaisesRegex(AliasError, "input_hash_mismatch"):
            self.fixture.run()

    def test_source_provenance_and_base_catalog_tampering(self):
        cache_manifest = self.fixture.cache / "source-manifest.json"
        original = cache_manifest.read_bytes()
        cache_manifest.write_bytes(
            original.replace(b'"schema_version":1', b'"schema_version":1,"schema_version":1')
        )
        with self.assertRaisesRegex(AliasError, "input_hash_mismatch"):
            self.fixture.run()
        cache_manifest.write_bytes(original)
        source_manifest = json.loads(original)
        source_manifest["provenance"]["dataset"]["snapshot_id"] = "foreign"
        cache_manifest.write_bytes(canonical_bytes(source_manifest))
        with self.assertRaisesRegex(AliasError, "input_hash_mismatch"):
            self.fixture.run()
        cache_manifest.write_bytes(original)
        base_catalog = self.fixture.base / "catalog.json"
        base_catalog.write_bytes(base_catalog.read_bytes() + b"\n")
        with self.assertRaisesRegex(AliasError, "base_integrity_error"):
            self.fixture.run()

    def test_rehashed_output_tampering_and_historical_producer_fingerprint(self):
        output = self.fixture.run()
        original = self.fixture.validate(output).identity
        alias_path = output / "aliases.json"
        manifest_path = output / "alias-manifest.json"
        aliases = json.loads(alias_path.read_bytes())
        aliases["aliases"]["1"][0]["text"] = "Forged"
        alias_path.write_bytes(canonical_bytes(aliases))
        manifest = json.loads(manifest_path.read_bytes())
        changed = alias_path.read_bytes()
        manifest["files"]["aliases.json"] = {
            "bytes": len(changed),
            "sha256": hashlib.sha256(changed).hexdigest(),
        }
        manifest_path.write_bytes(canonical_bytes(manifest))
        with self.assertRaisesRegex(AliasError, "alias_integrity_error"):
            self.fixture.validate(output)
        aliases["aliases"]["1"][0]["text"] = "One"
        alias_path.write_bytes(canonical_bytes(aliases))
        corrected = alias_path.read_bytes()
        manifest["files"]["aliases.json"] = {
            "bytes": len(corrected),
            "sha256": hashlib.sha256(corrected).hexdigest(),
        }
        manifest["normalizer"]["code_files_sha256"]["data/aliases.py"] = "0" * 64
        manifest_path.write_bytes(canonical_bytes(manifest))
        self.assertNotEqual(self.fixture.validate(output).identity, original)
        with self.assertRaisesRegex(AliasError, "alias_integrity_error"):
            self.fixture.validate(output, expected_identity=original)

    def test_manifest_immutable_json_blocks_reject_equal_python_numeric_values(self):
        output = self.fixture.run()
        path = output / "alias-manifest.json"
        original = json.loads(path.read_bytes())

        def changed(mutator):
            document = json.loads(canonical_bytes(original))
            mutator(document)
            path.write_bytes(canonical_bytes(document))
            with self.assertRaisesRegex(AliasError, "alias_integrity_error"):
                self.fixture.validate(output)

        changed(lambda document: document.__setitem__("schema_version", True))
        changed(
            lambda document: document["base"]["catalog"].__setitem__(
                "bytes", float(original["base"]["catalog"]["bytes"])
            )
        )
        for bad_count in (True, 2.0, 3):
            with self.subTest(bad_count=bad_count):
                changed(lambda document: document["base"].__setitem__("catalog_items", bad_count))
        changed(
            lambda document: document["sources"][0].__setitem__(
                "size", float(original["sources"][0]["size"])
            )
        )
        changed(lambda document: document["provenance"]["dataset"].__setitem__("version", True))
        path.write_bytes(canonical_bytes(original))
        self.assertEqual(self.fixture.validate(output).base_identity, self.fixture.base_identity)

    def test_source_manifest_nested_equal_numeric_tampering_is_rejected(self):
        path = self.fixture.cache / "source-manifest.json"
        original = json.loads(path.read_bytes())
        for nested in ("verified", "provenance"):
            document = json.loads(canonical_bytes(original))
            if nested == "verified":
                document["verified"][0]["size"] = float(document["verified"][0]["size"])
            else:
                document["provenance"]["dataset"]["version"] = True
            path.write_bytes(canonical_bytes(document))
            with (
                self.subTest(nested=nested),
                self.assertRaisesRegex(AliasError, "input_hash_mismatch"),
            ):
                self.fixture.run()
        path.write_bytes(canonical_bytes(original))

    def test_fixture_registry_version_requires_exact_integer_type(self):
        registry = json.loads(self.fixture.spec.registry_bytes)
        registry["provenance"]["dataset"]["version"] = True
        altered = AliasSpec(
            self.fixture.spec.snapshot_id,
            canonical_bytes(registry),
            self.fixture.spec.expectations_resource,
            self.fixture.spec.expectations_bytes,
        )
        with self.assertRaisesRegex(AliasError, "alias_integrity_error"):
            normalize_aliases(
                self.fixture.cache,
                self.fixture.base,
                self.fixture.root / "invalid",
                snapshot="nafiul-2026-v1",
                expected_base_identity=self.fixture.base_identity,
                _spec=altered,
                _base_spec=self.fixture.base_fixture.spec,
            )
        self.assertFalse((self.fixture.root / "invalid").exists())

    def test_base_identity_collision_and_permissions(self):
        with self.assertRaisesRegex(AliasError, "base_integrity_error"):
            normalize_aliases(
                self.fixture.cache,
                self.fixture.base,
                self.fixture.root / "bad",
                snapshot="nafiul-2026-v1",
                expected_base_identity="sha256:" + "0" * 64,
                _spec=self.fixture.spec,
                _base_spec=self.fixture.base_fixture.spec,
            )
        with self.assertRaisesRegex(AliasError, "alias_integrity_error"):
            self.fixture.run(name="alias-cache")
        result = self.fixture.run()
        with self.assertRaisesRegex(AliasError, "alias_integrity_error"):
            self.fixture.run()
        (result / "extra").write_text("foreign")
        with self.assertRaisesRegex(AliasError, "alias_integrity_error"):
            self.fixture.validate(result)


if __name__ == "__main__":
    unittest.main()
