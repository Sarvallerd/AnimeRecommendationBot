"""Native campaign keeps a conditional host-volume reserve during emission."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from recsys.research.campaign.storage import (
    GIB,
    STOP_BYTES,
    StorageSnapshot,
    admission_bytes,
    check_binding,
    prediction_bounds,
    require_storage,
    score_write_guard,
)
from recsys.research.protocol import ResearchError
from recsys.research.score_blocks import ScoreDimensions


class CampaignStorageTests(unittest.TestCase):
    def test_conditional_admission_and_both_volumes(self) -> None:
        variants = ("popularity", "a", "b", "c", "d", "e", "f", "g")
        bounds = prediction_bounds(ScoreDimensions(28_955, 45_938, 0), variants)
        self.assertEqual(bounds["popularity"], 1_906_586)
        self.assertEqual(bounds["a"], 10_809_430_482)
        required = admission_bytes(set(variants), bounds)
        self.assertEqual(required, 110_027_658_328)
        self.assertEqual(admission_bytes(set(), bounds), 32 * GIB)
        require_storage(
            StorageSnapshot(required, required),
            required_bytes=required,
        )
        for snapshot in (
            StorageSnapshot(required - 1, required),
            StorageSnapshot(required, required - 1),
        ):
            with self.assertRaisesRegex(ResearchError, "research_campaign_storage_low"):
                require_storage(snapshot, required_bytes=required)

    def test_emit_guard_checks_next_bytes_and_mount_binding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            physical = output / "physical"
            physical.mkdir()
            device = output.stat().st_dev
            with patch(
                "recsys.research.campaign.storage._filesystem",
                side_effect=[
                    (device, STOP_BYTES + 2),
                    (device, STOP_BYTES + 2),
                    (device, STOP_BYTES + 2),
                    (device, STOP_BYTES + 2),
                    (device, STOP_BYTES + 1),
                    (device, STOP_BYTES + 2),
                ],
            ):
                guard = score_write_guard(output, physical)
                guard(2)
                with self.assertRaisesRegex(ResearchError, "research_campaign_storage_low"):
                    guard(2)
            with patch(
                "recsys.research.campaign.storage._filesystem",
                side_effect=[(device + 1, STOP_BYTES), (device, STOP_BYTES)],
            ):
                with self.assertRaisesRegex(
                    ResearchError, "research_campaign_storage_binding_changed"
                ):
                    check_binding(output, physical, output_device=device, physical_device=device)
