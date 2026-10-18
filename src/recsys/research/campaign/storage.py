"""Conditional physical-space admission and bounded score emission checks."""

from __future__ import annotations

import os
import stat
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from recsys.research.campaign.common import _fail

GIB = 1024**3
SCORE_ESTIMATE_BYTES = 20 * GIB
OTHER_ESTIMATE_BYTES = 8 * GIB
RESERVE_BYTES = 20 * GIB
STOP_MARGIN_BYTES = 4 * GIB
STOP_BYTES = RESERVE_BYTES + STOP_MARGIN_BYTES


@dataclass(frozen=True, slots=True)
class StorageSnapshot:
    output_available_bytes: int
    physical_available_bytes: int


def _filesystem(path: Path) -> tuple[int, int]:
    """Read the real filesystem containing a directory; refuse link substitution."""
    if not path.is_absolute():
        _fail("research_campaign_storage_root_invalid")
    try:
        details = path.lstat()
        if not stat.S_ISDIR(details.st_mode):
            _fail("research_campaign_storage_root_invalid")
        filesystem = os.statvfs(path)
        available = filesystem.f_bavail * filesystem.f_frsize
        if available < 0:
            _fail("research_campaign_storage_probe_failed")
        return details.st_dev, available
    except OSError:
        _fail("research_campaign_storage_probe_failed")


def storage_snapshot(output_root: Path, physical_storage_root: Path) -> StorageSnapshot:
    """Measure both output and physical host volumes on every admission check."""
    _, output_available = _filesystem(output_root)
    _, physical_available = _filesystem(physical_storage_root)
    return StorageSnapshot(output_available, physical_available)


def require_storage(snapshot: StorageSnapshot, *, required_bytes: int) -> None:
    if type(required_bytes) is not int or required_bytes < STOP_BYTES:
        _fail("research_campaign_storage_probe_failed")
    if (
        snapshot.output_available_bytes < required_bytes
        or snapshot.physical_available_bytes < required_bytes
    ):
        _fail("research_campaign_storage_low")


def admission_bytes(remaining_prediction_arms: int) -> int:
    if type(remaining_prediction_arms) is not int or not 0 <= remaining_prediction_arms <= 8:
        _fail("research_campaign_storage_probe_failed")
    return remaining_prediction_arms * SCORE_ESTIMATE_BYTES + OTHER_ESTIMATE_BYTES + STOP_BYTES


def check_binding(
    output_root: Path,
    physical_storage_root: Path,
    *,
    output_device: int,
    physical_device: int,
) -> StorageSnapshot:
    observed_output_device, output_available = _filesystem(output_root)
    observed_physical_device, physical_available = _filesystem(physical_storage_root)
    if observed_output_device != output_device or observed_physical_device != physical_device:
        _fail("research_campaign_storage_binding_changed")
    return StorageSnapshot(output_available, physical_available)


def score_write_guard(output_root: Path, physical_storage_root: Path) -> Callable[[int], None]:
    """Check both volumes before each bounded compressed emit."""
    output_device, _ = _filesystem(output_root)
    physical_device, _ = _filesystem(physical_storage_root)

    def guard(next_emit_bytes: int) -> None:
        if type(next_emit_bytes) is not int or next_emit_bytes < 0:
            _fail("research_campaign_storage_probe_failed")
        snapshot = check_binding(
            output_root,
            physical_storage_root,
            output_device=output_device,
            physical_device=physical_device,
        )
        require_storage(snapshot, required_bytes=STOP_BYTES + next_emit_bytes)

    return guard
