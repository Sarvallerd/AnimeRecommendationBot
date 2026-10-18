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
    require_storage,
    score_write_guard,
)
from recsys.research.protocol import ResearchError


class CampaignStorageTests(unittest.TestCase):
    def test_conditional_admission_and_both_volumes(self) -> None:
        self.assertEqual(admission_bytes(8), 192 * GIB)
        self.assertEqual(admission_bytes(3), 92 * GIB)
        self.assertEqual(admission_bytes(0), 32 * GIB)
        require_storage(
            StorageSnapshot(output_available_bytes=192 * GIB, physical_available_bytes=192 * GIB),
            required_bytes=admission_bytes(8),
        )
        for snapshot in (
            StorageSnapshot(192 * GIB - 1, 192 * GIB),
            StorageSnapshot(192 * GIB, 192 * GIB - 1),
        ):
            with self.assertRaisesRegex(ResearchError, "research_campaign_storage_low"):
                require_storage(snapshot, required_bytes=admission_bytes(8))

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
