"""The reviewed package migration remains independently recoverable from Git."""

from __future__ import annotations

import hashlib
import json
import tomllib
import unittest
from pathlib import Path

from recsys import __version__
from recsys.contracts.bundle import ALGORITHMS
from recsys.contracts.provenance import SOURCE_FILES, SOURCE_LAYOUT
from recsys.release.itemknn import ALGORITHM as ITEMKNN_ALGORITHM
from recsys.release.itemknn import CURRENT_IDENTITY as ITEMKNN_IDENTITY
from recsys.release.itemknn import RELATIONS_SHA256, RELATIONS_SNAPSHOT

ROOT = Path(__file__).resolve().parents[2]
RECORD = ROOT / "docs/migrations/arb054-recsys-layout-v2.json"
TRAINING_RECORD = ROOT / "docs/migrations/arb058-training-artifacts-v1.json"
RETIREMENT_RECORD = ROOT / "docs/migrations/arb059-cupy-scoring-retirement.json"
NATIVE_RECORD = ROOT / "docs/migrations/arb060-native-pytorch-training-v1.json"
NATIVE_CAMPAIGN_RECORD = ROOT / "docs/migrations/arb062-native-validation-campaign-v2.json"
NATIVE_REPORT_CAP_RECORD = ROOT / "docs/migrations/arb063-native-campaign-report-cap-v1.json"
BOUNDED_EVALUATION_RECORD = ROOT / "docs/migrations/arb064-bounded-evaluation-v1.json"
BINARY_PREDICTION_RECORD = ROOT / "docs/migrations/arb065-binary-predictions-v1.json"
ITEMKNN_RECORD = ROOT / "docs/migrations/arb069-itemknn-bundle-v1.json"
FRANCHISE_RECORD = ROOT / "docs/migrations/arb069-franchises-v1.json"
BUNDLE_V2_RECORD = ROOT / "docs/migrations/arb069-bundle-v2-v1.json"
LEGACY_RECORD_SHA256 = "27dd3f274a4cee02b0d6e7cf43fa9f4fcfbfe755e04d0c43d68d6c319ced5016"
TRAINING_RECORD_SHA256 = "9a891c85f97340713be41319aeed559285c8cab54363fb8ede1f94ad76dff221"
LOCK = ROOT / "src/recsys/resources/releases/release_neelagiri_2025_nafiul_2026_v1.json"
SOURCE_BASELINE = "1441573fbbbffc4693c7e1100aaae4458d9e88d4"
TASK_BASELINE = "b385bfae9ad128f257784aec61e541296b648e31"
COMPARATOR_BASELINE = "04588dcba21898210918f44999fbccf0068b468f"
LOCK_SHA256 = "46f3ffd60239675b802026025016ee6719cb0525325e088fbfa248012db3de9b"
F2_EVALUATOR_SHA256 = "09de66b5fd21c9daef396684aadb8e962c91f94bf79f5797c49cb03999a81bfb"
F2_NATIVE_REPORT_CAP_SHA256 = "03fa9f95e786437fe139d7877ded095dcaab816bb7e31626f495f9c2b20345ba"


class MigrationRecordTests(unittest.TestCase):
    def test_binary_prediction_migration_keeps_historical_bytes_and_pins_new_contract(self) -> None:
        previous = json.loads(BOUNDED_EVALUATION_RECORD.read_bytes())
        current = json.loads(BINARY_PREDICTION_RECORD.read_bytes())
        later = json.loads(ITEMKNN_RECORD.read_bytes())
        self.assertEqual(current["migration_id"], "arb065-binary-predictions-v1")
        self.assertEqual(
            current["task_baseline_commit"], "a06e369b21f96c5d8ee37765e3116448e1d12b6e"
        )
        self.assertEqual(
            current["historical_native_v2_recovery_commit"], current["task_baseline_commit"]
        )
        self.assertEqual(
            (current["package_version_from"], current["package_version_to"]), ("0.5.2", "0.6.0")
        )
        self.assertEqual(current["previous_source_inventory"], previous["source_inventory"])
        self.assertEqual(current["source_inventory"], later["previous_source_inventory"])
        self.assertEqual(len(current["source_inventory"]), 91)
        self.assertEqual(
            current["added_sources"],
            ["research/score_blocks.py", "resources/campaigns/research_campaign_native_v3.json"],
        )
        self.assertEqual(len(current["preserved_resource_sha256"]), 13)
        self.assertEqual(
            current["preserved_resource_sha256"], previous["preserved_resource_sha256"]
        )
        for path, digest in current["preserved_resource_sha256"].items():
            self.assertEqual(hashlib.sha256((ROOT / path).read_bytes()).hexdigest(), digest)
        self.assertEqual(
            current["previous_migration_sha256"],
            {
                str(BOUNDED_EVALUATION_RECORD.relative_to(ROOT)): hashlib.sha256(
                    BOUNDED_EVALUATION_RECORD.read_bytes()
                ).hexdigest()
            },
        )
        resource = current["new_campaign_resource"]
        raw = (ROOT / resource["path"]).read_bytes()
        self.assertEqual(
            (len(raw), hashlib.sha256(raw).hexdigest()),
            (2123, "7e8096e2065db3e4431d3baaa862be642752ded22c98855b79e5bb846c1418f6"),
        )
        self.assertEqual(current["artifact_versions"]["native_v3_plan"], 5)
        self.assertEqual(current["artifact_versions"]["native_v3_report"], 4)
        self.assertEqual(current["artifact_versions"]["binary_prediction"], 2)
        self.assertEqual(current["artifact_versions"]["binary_score_container"], 1)
        self.assertEqual(current["score_format"], "arb-score-blocks-v1")
        self.assertIs(current["scientific_profile_changed"], False)
        self.assertIs(current["acceptance"]["full_ml02"], False)
        later_changed = later["changed_producer_source_sha256"]
        for path, hashes in current["changed_producer_source_sha256"].items():
            self.assertEqual(set(hashes), {"from", "to"})
            self.assertNotEqual(hashes["from"], hashes["to"])
            expected = (
                later_changed[path]["from"]
                if path in later_changed
                else hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
            )
            self.assertEqual(hashes["to"], expected)
        self.assertEqual(
            current["new_codec_source_sha256"],
            {
                "src/recsys/research/score_blocks.py": hashlib.sha256(
                    (ROOT / "src/recsys/research/score_blocks.py").read_bytes()
                ).hexdigest()
            },
        )

    def test_bundle_v2_migration_changes_the_contract_and_pins_current(self) -> None:
        previous = json.loads(FRANCHISE_RECORD.read_bytes())
        current = json.loads(BUNDLE_V2_RECORD.read_bytes())
        self.assertEqual(
            (current["schema_version"], current["kind"]), (1, "arb-bundle-v2-migration")
        )
        self.assertEqual(current["migration_id"], "arb069-bundle-v2-v1")
        self.assertEqual(
            (current["package_version_from"], current["package_version_to"]),
            (previous["package_version_to"], __version__),
        )
        self.assertEqual(current["source_layout"], SOURCE_LAYOUT)
        self.assertEqual(current["previous_source_inventory"], previous["source_inventory"])
        self.assertEqual(current["source_inventory"], list(SOURCE_FILES))
        self.assertEqual(len(SOURCE_FILES), 92)
        self.assertEqual(current["added_sources"], [])
        self.assertEqual(
            current["previous_migration_sha256"],
            {
                str(FRANCHISE_RECORD.relative_to(ROOT)): hashlib.sha256(
                    FRANCHISE_RECORD.read_bytes()
                ).hexdigest()
            },
        )
        changed = current["changed_producer_source_sha256"]
        self.assertEqual(
            set(changed),
            {
                "src/recsys/__init__.py",
                "src/recsys/contracts/bundle.py",
                "src/recsys/release/itemknn.py",
            },
        )
        for path, hashes in changed.items():
            self.assertEqual(set(hashes), {"from", "to"})
            self.assertNotEqual(hashes["from"], hashes["to"])
            self.assertEqual(hashlib.sha256((ROOT / path).read_bytes()).hexdigest(), hashes["to"])
        self.assertEqual(
            current["preserved_resource_sha256"], previous["preserved_resource_sha256"]
        )
        for path, digest in current["preserved_resource_sha256"].items():
            self.assertEqual(hashlib.sha256((ROOT / path).read_bytes()).hexdigest(), digest)
        self.assertEqual(
            current["relations_snapshot"], {"id": RELATIONS_SNAPSHOT, "sha256": RELATIONS_SHA256}
        )
        self.assertEqual(current["bundle_contract"]["max_neighbors"], {"1": 5, "2": 50})
        derived = current["derived_bundle"]
        self.assertEqual(
            (derived["algorithm_version"], derived["identity"], derived["max_neighbors"]),
            (ITEMKNN_ALGORITHM["version"], ITEMKNN_IDENTITY, ITEMKNN_ALGORITHM["max_neighbors"]),
        )
        self.assertIs(current["bundle_contract_changed"], True)
        for flag in ("release_lock_changed", "research_artifacts_changed"):
            self.assertIs(current[flag], False)

    def test_franchise_migration_changed_the_itemknn_producer(self) -> None:
        previous = json.loads(ITEMKNN_RECORD.read_bytes())
        current = json.loads(FRANCHISE_RECORD.read_bytes())
        later = json.loads(BUNDLE_V2_RECORD.read_bytes())
        self.assertEqual(
            (current["schema_version"], current["kind"]), (1, "arb-itemknn-franchise-migration")
        )
        self.assertEqual(current["migration_id"], "arb069-franchises-v1")
        self.assertEqual(
            (current["package_version_from"], current["package_version_to"]),
            (previous["package_version_to"], "0.8.0"),
        )
        self.assertEqual(current["source_layout"], SOURCE_LAYOUT)
        self.assertEqual(current["previous_source_inventory"], previous["source_inventory"])
        self.assertEqual(current["source_inventory"], later["previous_source_inventory"])
        self.assertEqual(current["added_sources"], [])
        self.assertEqual(
            current["previous_migration_sha256"],
            {
                str(ITEMKNN_RECORD.relative_to(ROOT)): hashlib.sha256(
                    ITEMKNN_RECORD.read_bytes()
                ).hexdigest()
            },
        )
        later_changed = later["changed_producer_source_sha256"]
        for path, hashes in current["changed_producer_source_sha256"].items():
            self.assertEqual(set(hashes), {"from", "to"})
            self.assertNotEqual(hashes["from"], hashes["to"])
            expected = (
                later_changed[path]["from"]
                if path in later_changed
                else hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
            )
            self.assertEqual(hashes["to"], expected)
        self.assertIn("src/recsys/release/itemknn.py", current["changed_producer_source_sha256"])
        self.assertEqual(
            current["preserved_resource_sha256"], previous["preserved_resource_sha256"]
        )
        derived = current["derived_bundle"]
        self.assertEqual(
            (derived["algorithm_version"], derived["identity"]),
            (
                "arb069-v2",
                "sha256:6eec1413efcfdb129cbb51618917972efd01392ba815534e81bbe5fb5336cbf2",
            ),
        )
        self.assertEqual(
            current["relations_snapshot"], {"id": RELATIONS_SNAPSHOT, "sha256": RELATIONS_SHA256}
        )
        for flag in (
            "bundle_contract_changed",
            "release_lock_changed",
            "research_artifacts_changed",
        ):
            self.assertIs(current[flag], False)

    def test_itemknn_bundle_migration_adds_one_producer(self) -> None:
        previous = json.loads(BINARY_PREDICTION_RECORD.read_bytes())
        current = json.loads(ITEMKNN_RECORD.read_bytes())
        later = json.loads(FRANCHISE_RECORD.read_bytes())
        self.assertEqual(
            (current["schema_version"], current["kind"]), (1, "arb-itemknn-bundle-migration")
        )
        self.assertEqual(current["migration_id"], "arb069-itemknn-bundle-v1")
        self.assertEqual(
            current["task_baseline_commit"], "ef8c1db443c2ca98636a91c3827eaad39e7f9e42"
        )
        self.assertEqual(
            (current["package_version_from"], current["package_version_to"]),
            (previous["package_version_to"], "0.7.0"),
        )
        self.assertEqual(current["source_layout"], SOURCE_LAYOUT)
        self.assertEqual(current["previous_source_inventory"], previous["source_inventory"])
        self.assertEqual(current["source_inventory"], later["previous_source_inventory"])
        self.assertEqual(len(current["source_inventory"]), 92)
        self.assertEqual(current["added_sources"], ["release/itemknn.py"])
        self.assertEqual(
            current["previous_migration_sha256"],
            {
                str(BINARY_PREDICTION_RECORD.relative_to(ROOT)): hashlib.sha256(
                    BINARY_PREDICTION_RECORD.read_bytes()
                ).hexdigest()
            },
        )
        # A later migration that changed the file again starts from this one's result.
        later_changes = [
            json.loads(record.read_bytes())["changed_producer_source_sha256"]
            for record in (FRANCHISE_RECORD, BUNDLE_V2_RECORD)
        ]
        produced = current["changed_producer_source_sha256"] | {
            path: {"from": "", "to": digest}
            for path, digest in current["new_source_sha256"].items()
        }
        for path, hashes in produced.items():
            self.assertEqual(set(hashes), {"from", "to"})
            self.assertNotEqual(hashes["from"], hashes["to"])
            expected = next(
                (changes[path]["from"] for changes in later_changes if path in changes),
                hashlib.sha256((ROOT / path).read_bytes()).hexdigest(),
            )
            self.assertEqual(hashes["to"], expected)
        self.assertEqual(len(current["preserved_resource_sha256"]), 14)
        for path, digest in current["preserved_resource_sha256"].items():
            self.assertEqual(hashlib.sha256((ROOT / path).read_bytes()).hexdigest(), digest)
        self.assertEqual(current["bundle_contract"]["manifest_algorithm_names"], list(ALGORITHMS))
        derived = current["derived_bundle"]
        self.assertEqual(
            (derived["algorithm"], derived["algorithm_version"], derived["identity"]),
            (
                ITEMKNN_ALGORITHM["name"],
                "arb069-v1",
                "sha256:a9f4cf27520460a10d9dc2ec18d3d0e556730fbfb61f6e60f666733ce166e2a2",
            ),
        )
        for flag in (
            "release_lock_changed",
            "research_artifacts_changed",
            "scientific_profile_changed",
        ):
            self.assertIs(current[flag], False)

    def test_native_report_cap_migration_preserves_science_and_history(self) -> None:
        previous = json.loads(NATIVE_CAMPAIGN_RECORD.read_bytes())
        current = json.loads(NATIVE_REPORT_CAP_RECORD.read_bytes())
        self.assertEqual(
            (current["schema_version"], current["kind"]),
            (1, "arb-native-campaign-report-cap-migration"),
        )
        self.assertEqual(current["migration_id"], "arb063-native-campaign-report-cap-v1")
        self.assertEqual(current["task_id"], "ARB-063")
        self.assertEqual(
            current["task_baseline_commit"], "fdcd894cfb70c880581f69e2788fb0e605763a03"
        )
        self.assertEqual(current["task_baseline_tree"], "dedbaaaea40bb194eb65727dd017ee2b9e25a7b4")
        self.assertEqual(
            (current["package_version_from"], current["package_version_to"]),
            ("0.5.0", "0.5.1"),
        )
        self.assertEqual(current["source_layout"], SOURCE_LAYOUT)
        self.assertEqual(
            current["source_inventory"],
            json.loads(BOUNDED_EVALUATION_RECORD.read_bytes())["source_inventory"],
        )
        self.assertEqual(len(current["source_inventory"]), 89)
        self.assertEqual(current["artifact_versions"], previous["artifact_versions"])
        self.assertEqual(
            current["native_public_projection_report_limit_bytes"], {"from": 262144, "to": 8388608}
        )
        self.assertIs(current["legacy_limits_changed"], False)
        self.assertIs(current["scientific_profile_changed"], False)
        self.assertEqual(
            current["previous_migration_sha256"],
            {
                str(
                    NATIVE_CAMPAIGN_RECORD.relative_to(ROOT)
                ): "abdd8fb0b72d050c31700853de5e42abfe6cc9c5820434e8a3728343db66e817"
            },
        )
        for path, digest in current["previous_migration_sha256"].items():
            self.assertEqual(hashlib.sha256((ROOT / path).read_bytes()).hexdigest(), digest)
        resources = current["preserved_resource_sha256"]
        self.assertEqual(len(resources), 13)
        self.assertEqual(
            resources,
            previous["preserved_resource_sha256"]
            | {
                previous["new_campaign_resource"]["path"]: previous["new_campaign_resource"][
                    "sha256"
                ]
            },
        )
        for path, digest in resources.items():
            self.assertEqual(hashlib.sha256((ROOT / path).read_bytes()).hexdigest(), digest)
        changed = current["changed_producer_source_sha256"]
        self.assertEqual(
            set(changed),
            {
                "src/recsys/__init__.py",
                "src/recsys/research/campaign/common.py",
                "src/recsys/research/campaign/verification.py",
                "src/recsys/research/campaign/runner.py",
                "src/recsys/research/campaign/report.py",
            },
        )
        for path, digests in changed.items():
            self.assertEqual(set(digests), {"from", "to"})
            self.assertNotEqual(digests["from"], digests["to"])
            if path == "src/recsys/__init__.py":
                self.assertEqual(
                    digests["to"],
                    json.loads(BOUNDED_EVALUATION_RECORD.read_bytes())[
                        "changed_producer_source_sha256"
                    ][path]["from"],
                )
            else:
                self.assertEqual(len(digests["to"]), 64)
        self.assertEqual(
            current["acceptance"],
            {
                "full_ml02": False,
                "model_quality": False,
                "test_labels_opened": False,
                "selected_n": False,
                "release_accepted": False,
            },
        )

    def test_bounded_evaluation_migration_preserves_artifacts_and_pins(self) -> None:
        previous = json.loads(NATIVE_REPORT_CAP_RECORD.read_bytes())
        current = json.loads(BOUNDED_EVALUATION_RECORD.read_bytes())
        self.assertEqual(
            hashlib.sha256(NATIVE_REPORT_CAP_RECORD.read_bytes()).hexdigest(),
            F2_NATIVE_REPORT_CAP_SHA256,
        )
        self.assertEqual(
            (current["schema_version"], current["kind"]), (1, "arb-bounded-evaluation-migration")
        )
        self.assertEqual(current["migration_id"], "arb064-bounded-evaluation-v1")
        self.assertEqual(current["task_id"], "ARB-064")
        self.assertEqual(
            current["task_notion_url"], "https://app.notion.com/p/3fd3908d6b1f81a1a338ef58888149a6"
        )
        self.assertEqual(
            current["task_baseline_commit"], "b61881af16550a44cde6e1b54d98b522052e0e53"
        )
        self.assertEqual(current["task_baseline_tree"], "505688886fecd356b70ec41e1cda8c1058a17c6e")
        self.assertEqual(
            (current["package_version_from"], current["package_version_to"]),
            (previous["package_version_to"], "0.5.2"),
        )
        self.assertEqual(current["source_layout"], SOURCE_LAYOUT)
        self.assertEqual(current["source_inventory"], previous["source_inventory"])
        self.assertEqual(len(current["source_inventory"]), 89)
        self.assertEqual(current["artifact_versions"], previous["artifact_versions"])
        self.assertEqual(
            current["preserved_resource_sha256"], previous["preserved_resource_sha256"]
        )
        self.assertEqual(len(current["preserved_resource_sha256"]), 13)
        for path, digest in current["preserved_resource_sha256"].items():
            self.assertEqual(hashlib.sha256((ROOT / path).read_bytes()).hexdigest(), digest)
        self.assertEqual(
            current["previous_migration_sha256"],
            {str(NATIVE_REPORT_CAP_RECORD.relative_to(ROOT)): F2_NATIVE_REPORT_CAP_SHA256},
        )
        self.assertEqual(current["native_public_projection_report_limit_bytes"], 8_388_608)
        self.assertEqual(current["legacy_public_projection_report_limit_bytes"], 262_144)
        for name in (
            "scientific_profile_changed",
            "resource_limits_changed",
            "score_encoding_changed",
            "evaluation_public_schema_changed",
        ):
            self.assertIs(current[name], False)
        self.assertEqual(current["acceptance"], previous["acceptance"])
        changed = current["changed_producer_source_sha256"]
        self.assertEqual(
            set(changed), {"src/recsys/__init__.py", "src/recsys/research/evaluation/evaluate.py"}
        )
        self.assertEqual(
            changed["src/recsys/__init__.py"]["from"],
            previous["changed_producer_source_sha256"]["src/recsys/__init__.py"]["to"],
        )
        self.assertEqual(
            changed["src/recsys/research/evaluation/evaluate.py"]["from"], F2_EVALUATOR_SHA256
        )
        for path, digests in changed.items():
            self.assertEqual(set(digests), {"from", "to"})
            self.assertNotEqual(digests["from"], digests["to"])
            self.assertEqual(len(digests["to"]), 64)
        lock_packages = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))["package"]
        third_party = [
            item for item in lock_packages if item["name"] != "anime-recommendation-recsys"
        ]
        self.assertEqual((len(lock_packages), len(third_party)), (143, 142))
        self.assertEqual(
            hashlib.sha256(
                json.dumps(third_party, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "82f851a738a18df5dd2142d12dbaa1c6701e6dd0b7a7d53e760951cb5fb96625",
        )
        self.assertEqual(
            next(
                item["version"]
                for item in lock_packages
                if item["name"] == "anime-recommendation-recsys"
            ),
            __version__,
        )

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
        self.assertEqual(
            current["source_inventory"],
            json.loads(NATIVE_REPORT_CAP_RECORD.read_bytes())["source_inventory"],
        )
        self.assertEqual(len(current["source_inventory"]), 89)
        self.assertEqual(
            sorted(set(current["source_inventory"]) - set(previous["source_inventory"])),
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
