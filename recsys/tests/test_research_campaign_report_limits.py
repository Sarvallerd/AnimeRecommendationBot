"""Native public-report capacity and bounded private-file transport."""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

from recsys.research.campaign.common import (
    NATIVE_PUBLIC_REPORT_MAX_BYTES,
    _canonical,
    _native_public_bytes,
    _private_file,
    _read_native_public,
)
from recsys.research.protocol import ResearchError


def synthetic_groups(
    *, include_short: bool = True, include_slices: bool = True
) -> list[dict[str, object]]:
    """All query cells and conservative overcomplete public slices, without private rows."""
    value = 1.2345678901234567e-200 if include_slices else 0.0
    aggregate = {
        "users": 1000,
        "queries": 1000,
        "metrics": {
            "ndcg5": value,
            "ndcg10": value,
            "recall5": value,
            "recall10": value,
            "hit5": value,
            "hit10": value,
        },
        "genre_diversity": value,
        "valid_pair_coverage": value,
        "recent_2024_seed_share": value,
    }
    slices: dict[str, dict[str, dict[str, object]]] = {"history_bin": {}, "seed_band": {}}
    if include_slices:
        segment = {**aggregate, "users": 20, "queries": 20}
        slices = {
            "history_bin": {
                name: segment for name in ("lt20", "20-49", "50-99", "100-199", "200plus")
            },
            "seed_band": {name: segment for name in ("none", "cold", "head", "tail", "mixed")},
        }
    groups: list[dict[str, object]] = []
    cells = [("popularity", 0)] + [
        (order, n)
        for order in ("random-liked-1", "random-liked-2", "random-liked-3", "top-rated")
        for n in range(1, 11)
    ]
    for cohort in ("main", "short") if include_short else ("main",):
        for mask in ("controlled", "product"):
            for order, n in cells:
                groups.append(
                    {
                        "split": "validation",
                        "cohort": cohort,
                        "mask": mask,
                        "order": order,
                        "n": n,
                        **(
                            {**aggregate, "users": 250, "queries": 250}
                            if cohort == "short"
                            else aggregate
                        ),
                        "coverage": value,
                        "empty_rate": value,
                        "short_rate": value,
                        "known_history_repeat_rate": value,
                        "completed_repeat_rate": value,
                        "slices": slices,
                    }
                )
    return groups


class NativePublicReportLimitTests(unittest.TestCase):
    def test_fifo_is_rejected_without_blocking_open(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "projection.json"
            os.mkfifo(path, 0o600)
            os.chmod(path, 0o600)
            root = Path(__file__).resolve().parents[2]
            child = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "from pathlib import Path\n"
                    "import sys\n"
                    "from recsys.research.campaign.common import _read_native_public\n"
                    "from recsys.research.protocol import ResearchError\n"
                    "try:\n"
                    "    _read_native_public(Path(sys.argv[1]))\n"
                    "except ResearchError as error:\n"
                    "    print(str(error))\n"
                    "else:\n"
                    "    raise AssertionError('FIFO unexpectedly accepted')\n",
                    str(path),
                ],
                env={**os.environ, "PYTHONPATH": str(root / "src")},
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
            self.assertEqual(child.returncode, 0, child.stderr)
            self.assertEqual(child.stdout.strip(), "research_campaign_report_invalid")

    def test_full_protocol_shape_and_main_only_lower_bound(self) -> None:
        groups = synthetic_groups()
        self.assertEqual(len(groups), 2 * 2 * 41)
        self.assertEqual(len(synthetic_groups(include_short=False, include_slices=False)), 2 * 41)
        self.assertEqual(len(groups) * 8, 1312)
        self.assertEqual({row["n"] for row in groups}, set(range(11)))
        self.assertEqual({row["cohort"] for row in groups}, {"main", "short"})
        self.assertEqual({row["mask"] for row in groups}, {"controlled", "product"})
        main_only = _canonical(
            {
                "variants": [
                    {"checked_groups": synthetic_groups(include_short=False, include_slices=False)}
                    for _ in range(8)
                ]
            }
        )
        self.assertGreater(len(main_only), 256 * 1024)
        self.assertEqual(len(main_only), 284415)
        first_slices = cast("dict[str, dict[str, dict[str, object]]]", groups[0]["slices"])
        self.assertEqual(
            set(first_slices["history_bin"]),
            {"lt20", "20-49", "50-99", "100-199", "200plus"},
        )
        self.assertEqual(
            set(first_slices["seed_band"]),
            {"none", "cold", "head", "tail", "mixed"},
        )
        self.assertTrue(
            all(
                set(cast("dict[str, dict[str, dict[str, object]]]", group["slices"])["history_bin"])
                == set(first_slices["history_bin"])
                and set(
                    cast("dict[str, dict[str, dict[str, object]]]", group["slices"])["seed_band"]
                )
                == set(first_slices["seed_band"])
                for group in groups
            )
        )
        full = _canonical({"variants": [{"checked_groups": groups} for _ in range(8)]})
        self.assertGreater(len(full), 5_000_000)
        self.assertLess(len(full), NATIVE_PUBLIC_REPORT_MAX_BYTES)
        self.assertEqual(NATIVE_PUBLIC_REPORT_MAX_BYTES, 8_388_608)
        self.assertEqual(
            json.loads(full), {"variants": [{"checked_groups": groups} for _ in range(8)]}
        )

    def test_exact_limit_and_one_byte_over(self) -> None:
        overhead = len(_canonical({"payload": ""}))
        at_cap = {"payload": "a" * (NATIVE_PUBLIC_REPORT_MAX_BYTES - overhead)}
        raw = _native_public_bytes(at_cap)
        self.assertEqual(len(raw), NATIVE_PUBLIC_REPORT_MAX_BYTES)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "campaign-report.json"
            _private_file(path, raw)
            self.assertEqual(_read_native_public(path), (at_cap, raw))
        with self.assertRaisesRegex(ResearchError, "^research_campaign_report_invalid$"):
            _native_public_bytes({"payload": at_cap["payload"] + "a"})

    def test_reader_rejects_oversize_before_read_and_growth_after_fstat(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "projection.json"
            raw = _canonical({"payload": "a" * NATIVE_PUBLIC_REPORT_MAX_BYTES})
            _private_file(path, raw)
            with self.assertRaisesRegex(ResearchError, "^research_campaign_report_invalid$"):
                _read_native_public(path)
            original = os.fstat

            def stale_size(fd: int) -> SimpleNamespace:
                info = original(fd)
                return SimpleNamespace(st_mode=info.st_mode, st_uid=info.st_uid, st_size=1)

            with patch("recsys.research.campaign.common.os.fstat", side_effect=stale_size):
                with self.assertRaisesRegex(ResearchError, "^research_campaign_report_invalid$"):
                    _read_native_public(path)

    def test_reader_rejects_symlink_mode_unknown_name_and_noncanonical(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "campaign-report.json"
            raw = _native_public_bytes({"status": "complete"})
            _private_file(path, raw)
            self.assertEqual(_read_native_public(path), ({"status": "complete"}, raw))
            alias = root / "projection.json"
            alias.symlink_to(path)
            for target in (alias, root / "other.json"):
                with self.assertRaisesRegex(ResearchError, "^research_campaign_report_invalid$"):
                    _read_native_public(target)
            path.chmod(0o644)
            with self.assertRaisesRegex(ResearchError, "^research_campaign_report_invalid$"):
                _read_native_public(path)
            path.chmod(stat.S_IRUSR | stat.S_IWUSR)
            path.write_bytes(b'{"status": "complete"}\n')
            with self.assertRaisesRegex(ResearchError, "^research_campaign_report_invalid$"):
                _read_native_public(path)


if __name__ == "__main__":
    unittest.main()
