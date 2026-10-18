"""The reviewed package migration remains independently recoverable from Git."""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

from recsys.contracts.provenance import SOURCE_FILES, SOURCE_LAYOUT

ROOT = Path(__file__).resolve().parents[2]
RECORD = ROOT / "docs/migrations/arb054-recsys-layout-v2.json"
TRAINING_RECORD = ROOT / "docs/migrations/arb058-training-artifacts-v1.json"
RETIREMENT_RECORD = ROOT / "docs/migrations/arb059-cupy-scoring-retirement.json"
NATIVE_RECORD = ROOT / "docs/migrations/arb060-native-pytorch-training-v1.json"
NATIVE_CAMPAIGN_RECORD = ROOT / "docs/migrations/arb062-native-validation-campaign-v2.json"
LEGACY_RECORD_SHA256 = "27dd3f274a4cee02b0d6e7cf43fa9f4fcfbfe755e04d0c43d68d6c319ced5016"
TRAINING_RECORD_SHA256 = "9a891c85f97340713be41319aeed559285c8cab54363fb8ede1f94ad76dff221"
LOCK = ROOT / "src/recsys/resources/releases/release_neelagiri_2025_nafiul_2026_v1.json"
SOURCE_BASELINE = "1441573fbbbffc4693c7e1100aaae4458d9e88d4"
TASK_BASELINE = "b385bfae9ad128f257784aec61e541296b648e31"
COMPARATOR_BASELINE = "04588dcba21898210918f44999fbccf0068b468f"
LOCK_SHA256 = "46f3ffd60239675b802026025016ee6719cb0525325e088fbfa248012db3de9b"


class MigrationRecordTests(unittest.TestCase):
    def test_training_record_preserves_historical_authority(self) -> None:
        self.assertEqual(hashlib.sha256(RECORD.read_bytes()).hexdigest(), LEGACY_RECORD_SHA256)
        legacy = json.loads(RECORD.read_bytes())
        current = json.loads(TRAINING_RECORD.read_bytes())
        self.assertEqual(current["migration_id"], "arb058-training-artifacts-v1")
        self.assertEqual(
            current["task_baseline_commit"], "fa983eaeeeeb41b41ac893b7de9827a40c25d27d"
        )
        self.assertEqual(
            (current["package_version_from"], current["package_version_to"]),
            ("0.2.0", "0.2.1"),
        )
        self.assertEqual(current["source_layout"], SOURCE_LAYOUT)
        self.assertEqual(
            current["source_inventory"],
            json.loads(RETIREMENT_RECORD.read_bytes())["previous_source_inventory"],
        )
        self.assertEqual(current["previous_artifact_versions"], legacy["artifact_versions"])
        self.assertEqual(
            current["preserved_resource_sha256"],
            {item["to"]: item["sha256"] for item in legacy["resource_moves"]},
        )
        self.assertEqual(
            current["acceptance"],
            {"full_ml02": False, "model_quality": False, "release_accepted": False},
        )

    def test_durable_layout_recovery_and_release_authority(self) -> None:
        record = json.loads(RECORD.read_bytes())
        self.assertEqual(record["schema_version"], 1)
        self.assertEqual(record["kind"], "arb-python-package-migration")
        self.assertEqual(record["migration_id"], SOURCE_LAYOUT)
        self.assertEqual(record["layout"], SOURCE_LAYOUT)
        self.assertEqual(record["baseline_commit"], SOURCE_BASELINE)
        self.assertEqual(record["source_layout_baseline_commit"], SOURCE_BASELINE)
        self.assertEqual(record["task_baseline_commit"], TASK_BASELINE)
        self.assertEqual(record["original_producer_recovery_commit"], SOURCE_BASELINE)
        self.assertEqual(record["historical_comparator_recovery_commit"], COMPARATOR_BASELINE)
        self.assertEqual(record["package_version_from"], "0.1.0")
        self.assertEqual(record["package_version_to"], "0.2.0")
        self.assertEqual(
            record["canonical_notion_url"],
            "https://app.notion.com/p/3fb3908d6b1f8136a1dfff3c4a12108e",
        )
        self.assertEqual(
            record["task_notion_url"],
            "https://app.notion.com/p/3fb3908d6b1f81acade0cc2247d1980a",
        )
        added = {
            "research/training/__init__.py",
            "research/training/artifacts.py",
            "research/training/spec.py",
        }
        self.assertTrue(added.isdisjoint(record["source_inventory"]))
        self.assertEqual(
            set(record["source_inventory"]) | added,
            set(json.loads(TRAINING_RECORD.read_bytes())["source_inventory"]),
        )
        self.assertEqual(len(record["module_moves"]), 33)
        self.assertEqual(len(record["resource_moves"]), 11)
        for item in record["resource_moves"]:
            with self.subTest(resource=item["to"]):
                raw = (ROOT / item["to"]).read_bytes()
                self.assertEqual(len(raw), item["bytes"])
                self.assertEqual(hashlib.sha256(raw).hexdigest(), item["sha256"])

        raw = LOCK.read_bytes()
        lock = json.loads(raw)
        authority = record["current_release_authority"]
        self.assertEqual(hashlib.sha256(raw).hexdigest(), LOCK_SHA256)
        self.assertEqual(authority["selector"], "current")
        self.assertEqual(authority["named_lock_path"], LOCK.relative_to(ROOT).as_posix())
        self.assertEqual(authority["named_lock_sha256"], LOCK_SHA256)
        self.assertEqual(authority["release_id"], lock["release_id"])
        self.assertEqual(authority["lock_schema_version"], 1)
        self.assertEqual(
            authority["catalog_release_identity"],
            lock["inputs"]["catalog_release"]["identity"],
        )
        self.assertEqual(
            authority["reference_bundle_identity"], lock["reference"]["bundle_identity"]
        )
        self.assertEqual(
            authority["reference_producer_source_commit"],
            lock["reference"]["producer"]["source_commit"],
        )
        self.assertEqual(authority["reference_producer_package_version"], "0.1.0")
        self.assertEqual(authority["runtime_bundle_schema_version"], 1)
        self.assertIs(authority["new_layout_full_catalog_accepted"], False)

    def test_artifact_version_matrix_has_all_boundaries(self) -> None:
        versions = json.loads(RECORD.read_bytes())["artifact_versions"]
        self.assertEqual(
            set(versions),
            {
                "runtime_bundle",
                "dataset_manifest",
                "alias_manifest",
                "catalog_release_manifest",
                "mal_build_report",
                "composite_build_report",
                "named_release_lock",
                "research_preparation",
                "research_model",
                "research_campaign_plan",
                "historical_comparison",
                "predictions_and_evaluation",
            },
        )
        for name in ("dataset_manifest", "alias_manifest", "catalog_release_manifest"):
            with self.subTest(artifact=name):
                self.assertEqual(versions[name]["schema_version"], 1)
                self.assertEqual(versions[name]["new_normalizer_version"], "2")
                self.assertEqual(versions[name]["legacy_normalizer_version"], "1")
        for name in ("research_preparation", "research_model", "research_campaign_plan"):
            with self.subTest(artifact=name):
                self.assertEqual(versions[name]["new_schema_version"], 2)
                self.assertEqual(versions[name]["legacy_schema_version"], 1)
                self.assertTrue(versions[name]["legacy_handling"].startswith("reject-before-"))
        self.assertEqual(versions["runtime_bundle"]["new_schema_version"], 1)
        self.assertEqual(versions["composite_build_report"]["new_schema_version"], 3)
        self.assertEqual(
            versions["named_release_lock"]["new_schema_authority"], "tiny-fixture-only"
        )

    def test_retirement_record_chains_frozen_layout_and_training(self) -> None:
        self.assertEqual(hashlib.sha256(RECORD.read_bytes()).hexdigest(), LEGACY_RECORD_SHA256)
        self.assertEqual(
            hashlib.sha256(TRAINING_RECORD.read_bytes()).hexdigest(), TRAINING_RECORD_SHA256
        )
        legacy = json.loads(RECORD.read_bytes())
        training = json.loads(TRAINING_RECORD.read_bytes())
        retired = json.loads(RETIREMENT_RECORD.read_bytes())
        self.assertEqual(
            set(legacy["source_inventory"]) | set(training["added_sources"]),
            set(training["source_inventory"]),
        )
        self.assertEqual(
            set(training["source_inventory"]) - set(retired["removed_sources"]),
            set(retired["source_inventory"]),
        )
        self.assertEqual(len(retired["source_inventory"]), 75)
        self.assertEqual(retired["previous_source_inventory"], training["source_inventory"])
        self.assertEqual(retired["removed_sources"], ["research/gpu.py"])
        self.assertEqual(
            retired["task_baseline_commit"], "20757caeaf102b1f2137abede43c7d472e8778e2"
        )
        self.assertEqual(
            retired["previous_migration_sha256"],
            {
                str(RECORD.relative_to(ROOT)): LEGACY_RECORD_SHA256,
                str(TRAINING_RECORD.relative_to(ROOT)): TRAINING_RECORD_SHA256,
            },
        )
        self.assertEqual(retired["source_layout"], SOURCE_LAYOUT)
        self.assertEqual(retired["package_version_from"], "0.2.1")
        self.assertEqual(retired["package_version_to"], "0.3.0")
        self.assertEqual(
            retired["historical_gpu_scoring_recovery_commit"],
            "fa983eaeeeeb41b41ac893b7de9827a40c25d27d",
        )
        self.assertEqual(retired["original_producer_recovery_commit"], SOURCE_BASELINE)
        self.assertEqual(
            retired["artifact_versions"],
            {
                "research_preparation": 2,
                "research_model": 2,
                "research_campaign_plan": 3,
                "research_campaign_report": 2,
                "research_campaign_envelope": 1,
                "research_training_config": 1,
                "research_training_artifact": 1,
            },
        )
        self.assertEqual(
            retired["preserved_resource_sha256"], training["preserved_resource_sha256"]
        )
        for name, digest in retired["preserved_resource_sha256"].items():
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), digest)
        self.assertEqual(
            retired["acceptance"],
            {"full_ml02": False, "model_quality": False, "release_accepted": False},
        )

    def test_native_training_migration_preserves_source_and_resource_history(self) -> None:
        retired = json.loads(RETIREMENT_RECORD.read_bytes())
        current = json.loads(NATIVE_RECORD.read_bytes())
        self.assertEqual(current["migration_id"], "arb060-native-pytorch-training-v1")
        self.assertEqual(
            current["task_baseline_commit"], "18297ed7db59b66f6b4a63a179ea8daacdea0490"
        )
        self.assertEqual(current["previous_source_inventory"], retired["source_inventory"])
        self.assertEqual(len(current["previous_source_inventory"]), 75)
        self.assertEqual(len(current["added_sources"]), 9)
        self.assertEqual(len(current["source_inventory"]), 84)
        self.assertEqual(
            sorted(set(current["source_inventory"]) - set(current["previous_source_inventory"])),
            current["added_sources"],
        )
        for name, digest in current["previous_migration_sha256"].items():
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), digest)
        self.assertEqual(current["preserved_resource_sha256"], retired["preserved_resource_sha256"])
        for name, digest in current["preserved_resource_sha256"].items():
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), digest)
        resource = current["candidate_resource"]
        raw = (ROOT / resource["path"]).read_bytes()
        self.assertEqual(
            (len(raw), hashlib.sha256(raw).hexdigest()), (resource["bytes"], resource["sha256"])
        )
        self.assertEqual(current["acceptance"], retired["acceptance"])

    def test_native_validation_campaign_migration_adds_versioned_contract(self) -> None:
        previous = json.loads(NATIVE_RECORD.read_bytes())
        current = json.loads(NATIVE_CAMPAIGN_RECORD.read_bytes())
        self.assertEqual(current["migration_id"], "arb062-native-validation-campaign-v2")
        self.assertEqual(
            current["task_baseline_commit"], "c00440d477a32980e1c780c3990dd4a61fa4c4b2"
        )
        self.assertEqual(current["source_layout"], SOURCE_LAYOUT)
        self.assertEqual(
            (current["package_version_from"], current["package_version_to"]), ("0.4.0", "0.5.0")
        )
        self.assertEqual(current["previous_source_inventory"], previous["source_inventory"])
        self.assertEqual(len(current["previous_source_inventory"]), 84)
        self.assertEqual(current["source_inventory"], list(SOURCE_FILES))
        self.assertEqual(len(SOURCE_FILES), 89)
        self.assertEqual(
            sorted(set(SOURCE_FILES) - set(previous["source_inventory"])),
            current["added_sources"],
        )
        for path, digest in current["previous_migration_sha256"].items():
            self.assertEqual(hashlib.sha256((ROOT / path).read_bytes()).hexdigest(), digest)
        for path, digest in current["preserved_resource_sha256"].items():
            self.assertEqual(hashlib.sha256((ROOT / path).read_bytes()).hexdigest(), digest)
        resource = current["new_campaign_resource"]
        raw = (ROOT / resource["path"]).read_bytes()
        self.assertEqual((len(raw), hashlib.sha256(raw).hexdigest()), (1596, resource["sha256"]))
        self.assertEqual(
            resource["sha256"], "4c395e475dc23e1c3faebb0ed0458fceb1bcc3c5d2af43318d0394acd729a13d"
        )
        self.assertEqual(current["artifact_versions"]["legacy_campaign_plan"], 3)
        self.assertEqual(current["artifact_versions"]["native_campaign_plan"], 4)
        self.assertEqual(current["artifact_versions"]["native_campaign_report"], 3)
        self.assertIs(current["acceptance"]["full_ml02"], False)


if __name__ == "__main__":
    unittest.main()
