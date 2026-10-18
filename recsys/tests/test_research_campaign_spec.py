"""Immutable native campaign resource and its closed scientific arm set."""

from __future__ import annotations

import copy
import hashlib
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from typing import cast
from unittest.mock import patch

from research_model_support import ModelFixture, patch_release_info
from test_research_model_content import _glove

from recsys.contracts.resources import resource_bytes
from recsys.research.campaign.common import VARIANTS
from recsys.research.campaign.plan import _expected_nodes, _plan, _validate_plan_structure
from recsys.research.campaign.spec import (
    NATIVE_RESOURCE_SHA256,
    NATIVE_VARIANT,
    campaign_variants,
    native_campaign_spec,
)
from recsys.research.protocol import ResearchError
from recsys.research.training.candidate import candidate_identity, load_candidate
from recsys.research.training.spec import DeviceSpec


class CampaignSpecTests(unittest.TestCase):
    def test_legacy_rejects_native_only_controls_before_scientific_inputs(self) -> None:
        with patch(
            "recsys.research.models.inputs.read_scoring_inputs",
            side_effect=AssertionError("scientific input opened"),
        ):
            with self.assertRaisesRegex(ResearchError, "research_campaign_profile_unsupported"):
                _plan(
                    Path("/absent/prepared"),
                    "sha256:" + "0" * 64,
                    "a" * 40,
                    Path("/absent/glove"),
                    2,
                    datetime.now(timezone.utc),
                    native_train_device="cpu",
                )

    def test_native_plan_has_strict_eight_arm_cpu_fixture_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ModelFixture(temporary)
            glove = Path(temporary) / "glove.txt"
            fixture.lock["inputs"]["embeddings"] = _glove(glove)
            prepared, identity = fixture.make_prepared()
            output = Path(temporary) / "campaign"
            output.mkdir()
            with patch_release_info(fixture.lock):
                plan = _plan(
                    prepared,
                    identity,
                    "a" * 40,
                    glove,
                    2,
                    datetime.now(timezone.utc),
                    campaign_name="ml02-native-v2",
                    native_train_device="cpu",
                    physical_storage_root=Path(temporary),
                    output_root=output,
                )
            _validate_plan_structure(plan)
            self.assertEqual(plan["schema_version"], 4)
            self.assertEqual(plan["execution_config_count"], 8)
            self.assertEqual(len(_expected_nodes(plan)), 32)
            self.assertEqual(plan["variants"][-1]["variant_id"], NATIVE_VARIANT)
            self.assertEqual(plan["variants"][-1]["train_device"], {"kind": "cpu", "index": None})
            self.assertEqual(
                plan["variants"][-1]["inference_device"], {"kind": "cpu", "index": None}
            )
            self.assertEqual(plan["variants"][-1]["training_spec"]["parameters"]["epochs"], 3)
            self.assertEqual(plan["storage"]["physical_storage_root"], temporary)
            for field, changed in (
                ("execution_config_count", 7),
                ("candidate_resource_identity", "sha256:" + "0" * 64),
                ("deadline_utc", plan["created_utc"]),
            ):
                with self.subTest(field=field):
                    malformed = copy.deepcopy(plan)
                    cast("dict[str, object]", malformed)[field] = changed
                    with self.assertRaisesRegex(ResearchError, "research_campaign_plan_mismatch"):
                        _validate_plan_structure(malformed)
            for mutation in ("order", "epochs", "inference", "framework", "storage"):
                malformed = copy.deepcopy(plan)
                if mutation == "order":
                    malformed["variants"].reverse()
                elif mutation == "epochs":
                    malformed["variants"][-1]["training_spec"]["parameters"]["epochs"] = 4
                elif mutation == "inference":
                    malformed["variants"][-1]["inference_device"] = {"kind": "cuda", "index": 0}
                elif mutation == "framework":
                    malformed["producer"]["package_versions"]["torch"] = "2.12.0+cpu"
                else:
                    malformed["storage"]["physical_filesystem_device"] = True
                with self.assertRaisesRegex(ResearchError, "research_campaign_plan_mismatch"):
                    _validate_plan_structure(malformed)

    def test_native_resource_binds_unchanged_parents_and_eight_arms(self) -> None:
        raw = resource_bytes("campaigns", "research_campaign_native_v2.json")
        self.assertEqual(len(raw), 1596)
        self.assertEqual(hashlib.sha256(raw).hexdigest(), NATIVE_RESOURCE_SHA256)
        spec, identity = native_campaign_spec()
        self.assertEqual(identity, "sha256:" + NATIVE_RESOURCE_SHA256)
        self.assertEqual(tuple(spec["arms"]), (*VARIANTS, NATIVE_VARIANT))
        self.assertEqual(campaign_variants("ml02-cpu-v1"), VARIANTS)
        self.assertEqual(campaign_variants("ml02-native-v2"), (*VARIANTS, NATIVE_VARIANT))
        self.assertEqual(spec["candidate_resource_identity"], candidate_identity())
        self.assertEqual(
            spec["baseline_campaign_sha256"],
            "87dabe5ae826b2134ddb18de3beba100d9f7764873f2d91e9e4ba06ec4ea7bfd",
        )
        candidate = load_candidate(
            train_device=DeviceSpec("cuda", 0), inference_device=DeviceSpec("cpu", None)
        )
        self.assertEqual(len(candidate.parameters), 32)
        self.assertEqual(spec["execution_profile"]["score_batch_rows"], 128)
        self.assertEqual(spec["storage_policy"]["reserve_bytes"], 20 * 1024**3)

    def test_raw_tamper_fails_before_consumer_use(self) -> None:
        raw = resource_bytes("campaigns", "research_campaign_native_v2.json")
        with patch("recsys.research.campaign.spec.resource_bytes", return_value=raw + b" "):
            with self.assertRaisesRegex(ResearchError, "research_campaign_spec_invalid"):
                native_campaign_spec()


if __name__ == "__main__":
    unittest.main()
