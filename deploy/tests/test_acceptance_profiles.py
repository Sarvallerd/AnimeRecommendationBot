"""Tiny strict-profile fixtures; no full inputs or private evidence."""

import copy
import importlib.util
import json
import pathlib
import shutil
import tempfile
import unittest
from typing import cast

from deploy.tests.acceptance_types import AcceptanceProfile, ProfileModule

SCRIPT = pathlib.Path(__file__).with_name("acceptance-profiles.py")
SPEC = importlib.util.spec_from_file_location("arb033_profile_tests", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"cannot load acceptance profiles from {SCRIPT}")
loaded = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(loaded)
for export in ("ProfileError", "BundleSummary", "load_profile", "check_profile_bundle", "digest"):
    if not callable(getattr(loaded, export, None)):
        raise ImportError(f"acceptance profiles missing {export}")
p = cast("ProfileModule", loaded)
FIXTURE = pathlib.Path(__file__).resolve().parents[2] / "tests/fixtures/bundle"


def encoded(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()


def fixture_profile() -> AcceptanceProfile:
    lock = copy.deepcopy(p.load_profile(p.NEW_RELEASE).lock)
    if lock is None:
        raise AssertionError("current acceptance profile requires a lock")
    lock["semantic_expectations"] = {
        "catalog_count": 8,
        "neighbor_count_histogram": {"0": 1, "1": 0, "2": 1, "3": 0, "4": 0, "5": 6},
        "smoke_seed": {"mal_id": 1, "query": "Общее название", "neighbor_ids": [10, 11, 12, 2, 3]},
        "empty_seed": {"mal_id": 99, "query": "Тихий ноль"},
        "vector_status_counts": {"ready": 7, "oov_only": 1, "no_tokens": 0, "zero_vector": 0},
        "zero_vector_ids": [99],
    }
    reference = lock["reference"]
    if type(reference) is not dict:
        raise AssertionError("current acceptance lock requires reference fields")
    reference["bundle_identity"] = "sha256:" + p.digest((FIXTURE / "manifest.json").read_bytes())
    reference["catalog_sha256"] = p.digest((FIXTURE / "catalog.json").read_bytes())
    reference["neighbors_sha256"] = p.digest((FIXTURE / "neighbors.json").read_bytes())
    lock_raw = encoded(lock)
    profile = json.loads((p.PROFILE_DIR / p.PROFILE_FILES[p.NEW_RELEASE]).read_bytes())
    profile["expectations"]["release_lock_sha256"] = p.digest(lock_raw)
    return p.load_profile(p.NEW_RELEASE, _profile_bytes=encoded(profile), _lock_bytes=lock_raw)


class ProfileTests(unittest.TestCase):
    def test_committed_pins_and_fixture_semantics(self) -> None:
        actual = p.load_profile(p.NEW_RELEASE)
        self.assertEqual(actual.lock_sha256, p.TRUSTED_LOCK_SHA)
        self.assertEqual(actual.expectations["catalog_count"], 28955)
        self.assertEqual(p.load_profile(p.LEGACY_RELEASE).expectations["catalog_count"], 17562)
        profile = fixture_profile()
        if profile.reference is None:
            raise AssertionError("current acceptance profile requires a reference")
        with tempfile.TemporaryDirectory() as scratch:
            bundle = pathlib.Path(scratch) / "bundle"
            bundle.mkdir()
            for name in ("catalog.json", "neighbors.json", "manifest.json"):
                shutil.copyfile(FIXTURE / name, bundle / name)
            summary = p.check_profile_bundle(
                bundle, profile, expected_identity=profile.reference["bundle_identity"]
            )
        self.assertEqual(summary.count, 8)
        self.assertEqual(summary.histogram, profile.expectations["neighbor_count_histogram"])
        self.assertEqual([r["mal_id"] for r in summary.neighbors["1"]], [10, 11, 12, 2, 3])

    def test_selector_lock_schema_and_json_guards(self) -> None:
        with self.assertRaisesRegex(p.ProfileError, "UNKNOWN_RELEASE"):
            p.load_profile("current")
        with self.assertRaisesRegex(p.ProfileError, "UNKNOWN_RELEASE"):
            p.load_profile("../../private")
        raw = (p.PROFILE_DIR / p.PROFILE_FILES[p.NEW_RELEASE]).read_bytes()
        for altered, code in (
            (raw.replace(b'"schema_version":1', b'"schema_version":true'), "INVALID_PROFILE"),
            (raw.replace(b'"release-lock-v1"', b'"other"'), "INVALID_PROFILE"),
            (raw + b" ", "NONCANONICAL_JSON"),
            (raw.replace(b'"kind":', b'"kind":"duplicate","kind":'), "DUPLICATE_JSON_KEY"),
        ):
            with self.subTest(code=code), self.assertRaises(p.ProfileError) as caught:
                p.load_profile(p.NEW_RELEASE, _profile_bytes=altered)
            self.assertEqual(caught.exception.code, code)
        with self.assertRaisesRegex(p.ProfileError, "LOCK_SHA_MISMATCH"):
            p.load_profile(p.NEW_RELEASE, _lock_bytes=b"{}\n")

    def test_bundle_recalculation_and_unsafe_paths(self) -> None:
        profile = fixture_profile()
        with tempfile.TemporaryDirectory() as scratch:
            bundle = pathlib.Path(scratch) / "bundle"
            bundle.mkdir()
            for name in ("catalog.json", "neighbors.json", "manifest.json"):
                shutil.copyfile(FIXTURE / name, bundle / name)
            with self.assertRaisesRegex(p.ProfileError, "BUNDLE_IDENTITY_MISMATCH"):
                p.check_profile_bundle(bundle, profile, expected_identity="sha256:" + "f" * 64)
            catalog = json.loads((bundle / "catalog.json").read_bytes())
            catalog["anime"]["1"]["title"] = "Wrong"
            (bundle / "catalog.json").write_bytes(encoded(catalog))
            manifest = json.loads((bundle / "manifest.json").read_bytes())
            manifest["files"]["catalog.json"]["sha256"] = p.digest(
                (bundle / "catalog.json").read_bytes()
            )
            (bundle / "manifest.json").write_bytes(encoded(manifest))
            with self.assertRaisesRegex(p.ProfileError, "BUNDLE_SMOKE_MISMATCH"):
                p.check_profile_bundle(
                    bundle,
                    profile,
                    expected_identity="sha256:" + p.digest((bundle / "manifest.json").read_bytes()),
                )
            (bundle / "catalog.json").unlink()
            (bundle / "catalog.json").symlink_to(FIXTURE / "catalog.json")
            with self.assertRaisesRegex(p.ProfileError, "UNSAFE_INPUT"):
                p.check_profile_bundle(
                    bundle,
                    profile,
                    expected_identity="sha256:" + p.digest((bundle / "manifest.json").read_bytes()),
                )


if __name__ == "__main__":
    unittest.main()
