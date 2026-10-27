"""Lossless score container and adversarial boundary tests."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
import zlib
from pathlib import Path

import numpy as np

from recsys.research.models.types import ScoreVector
from recsys.research.predict import _checked_vector
from recsys.research.protocol import ResearchError
from recsys.research.score_blocks import (
    BLOCK,
    FILE,
    MAGIC,
    BlockScoreWriter,
    DecodedScoreRow,
    EncodedVector,
    ScoreDimensions,
    check_block_descriptor,
    iter_score_rows,
    require_query_binding,
    score_upper_bound,
)


class ScoreBlockTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "scores.bin"
        self.ids = (1, 4, 9)
        self.binding = "a" * 64
        self.popularity = (3, 0, 11)

    def _read(self, descriptor: object) -> list[DecodedScoreRow]:
        return list(
            iter_score_rows(
                self.path,
                descriptor,
                expected_catalog_ids=self.ids,
                expected_binding_sha256=self.binding,
                expected_popularity=self.popularity,
                popularity_arm=False,
            )
        )

    def _write(self) -> dict[str, object]:
        writer = BlockScoreWriter(
            self.path,
            catalog_ids=self.ids,
            dimensions=ScoreDimensions(3, 3, 1),
            popularity=self.popularity,
            popularity_arm=False,
            binding_sha256=self.binding,
        )
        writer.write_query("q1_" + "1" * 64, 0, None)
        writer.write_query(
            "q1_" + "2" * 64,
            1,
            EncodedVector(
                np.array([-(2**63), 0, 2**63 - 1], dtype=np.int64),
                np.array([True, True, True], dtype=np.bool_),
            ),
        )
        writer.write_query(
            "q1_" + "3" * 64,
            2,
            EncodedVector(
                np.array([0, 0, -7], dtype=np.int64),
                np.array([False, True, True], dtype=np.bool_),
            ),
        )
        return dict(writer.close())

    def test_lossless_signed_endpoints_null_zero_and_deterministic_bytes(self) -> None:
        descriptor = self._write()
        raw = self.path.read_bytes()
        self.assertEqual(raw[:8], MAGIC)
        self.assertEqual(len(raw), descriptor["bytes"])
        self.assertLessEqual(
            len(raw), score_upper_bound(ScoreDimensions(3, 3, 1), popularity_arm=False)
        )
        rows = self._read(descriptor)
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0].mode, "popularity")
        self.assertEqual(rows[0].values.tolist(), [3, 0, 11])
        self.assertEqual(rows[1].values.tolist(), [-(2**63), 0, 2**63 - 1])
        self.assertEqual(rows[2].available.tolist(), [False, True, True])
        self.assertEqual(rows[2].values.tolist(), [0, 0, -7])
        require_query_binding(rows[0], query_id="q1_" + "1" * 64, n=0, popularity_arm=False)
        with self.assertRaisesRegex(ResearchError, "research_prediction_invalid"):
            require_query_binding(rows[1], query_id="q1_" + "2" * 64, n=0, popularity_arm=False)
        self.path.unlink()
        rewritten_descriptor = self._write()
        self.assertEqual((raw, descriptor), (self.path.read_bytes(), rewritten_descriptor))

    def test_truncation_trailing_header_binding_and_descriptor_rejected(self) -> None:
        descriptor = self._write()
        original = self.path.read_bytes()
        for name, altered in (
            ("magic", b"X" + original[1:]),
            ("version", original[:8] + b"\x02\x00" + original[10:]),
            (
                "catalog",
                original[: FILE.size + BLOCK.size]
                + b"\xff"
                + original[FILE.size + BLOCK.size + 1 :],
            ),
            ("truncated", original[:-1]),
            ("trailing", original + b"x"),
        ):
            with self.subTest(name=name):
                self.path.write_bytes(altered)
                with self.assertRaises(ResearchError):
                    self._read(descriptor)
        self.path.write_bytes(original)
        for key, changed in (
            ("sha256", "0" * 64),
            ("decoded_sha256", "0" * 64),
            ("dense_rows", 0),
            ("query_block_count", 2),
            ("bytes", len(original) + 1),
        ):
            with self.subTest(key=key):
                with self.assertRaises(ResearchError):
                    self._read({**descriptor, key: changed})
        with self.assertRaises(ResearchError):
            self._read({**descriptor, "extra": 1})
        with self.assertRaises(ResearchError):
            list(
                iter_score_rows(
                    self.path,
                    descriptor,
                    expected_catalog_ids=self.ids,
                    expected_binding_sha256="b" * 64,
                    expected_popularity=self.popularity,
                    popularity_arm=False,
                )
            )
        with self.assertRaises(ResearchError):
            list(
                iter_score_rows(
                    self.path,
                    descriptor,
                    expected_catalog_ids=self.ids,
                    expected_binding_sha256=self.binding,
                    expected_popularity=(4, 0, 11),
                    popularity_arm=False,
                )
            )

    def test_batch_edges_and_no_query_payload(self) -> None:
        writer = BlockScoreWriter(
            self.path,
            catalog_ids=self.ids,
            dimensions=ScoreDimensions(3, 129, 129),
            popularity=self.popularity,
            popularity_arm=False,
            binding_sha256=self.binding,
        )
        for n in range(129):
            writer.write_query("q1_" + f"{n:064x}", 0, None)
        descriptor = writer.close()
        self.assertEqual(descriptor["query_block_count"], 2)
        self.assertEqual(len(self._read(descriptor)), 129)
        self.path.unlink()
        writer = BlockScoreWriter(
            self.path,
            catalog_ids=self.ids,
            dimensions=ScoreDimensions(3, 0, 0),
            popularity=self.popularity,
            popularity_arm=True,
            binding_sha256=self.binding,
        )
        empty = writer.close()
        self.assertEqual(empty["query_block_count"], 0)
        self.assertEqual(self.path.stat().st_size, FILE.size + BLOCK.size + 12)

    def test_quantization_half_even_and_application_bound(self) -> None:
        vector = ScoreVector(
            values=np.array([0.5e-9, 1.5e-9, -2.5e-9], dtype=np.float64),
            available=np.array([True, True, True], dtype=np.bool_),
            supported_seeds=1,
        )
        checked = _checked_vector(vector, 1, self.ids)
        self.assertEqual(checked.values.tolist(), [0, 2, -2])
        with self.assertRaises(ResearchError):
            _checked_vector(
                ScoreVector(
                    values=np.array([1e8, 0.0, 0.0]),
                    available=np.array([True, True, True]),
                    supported_seeds=1,
                ),
                1,
                self.ids,
            )

    def test_inner_headers_padding_null_payload_and_zlib_tail(self) -> None:
        descriptor = self._write()
        original = self.path.read_bytes()
        block_offsets = []
        offset = FILE.size
        while offset < len(original):
            fields = BLOCK.unpack(original[offset : offset + BLOCK.size])
            block_offsets.append((offset, fields))
            offset += BLOCK.size + fields[5]
        self.assertEqual([fields[0] for _, fields in block_offsets], [1, 2, 3])
        for name, index, replacement in (
            ("catalog-kind", 0, {"kind": 2}),
            ("reserved", 0, {"reserved": 1}),
            ("bad-codec", 0, {"codec": 2}),
            ("wrong-first", 2, {"first": 1}),
            ("wrong-count", 2, {"count": 2}),
            ("excessive-expansion", 2, {"raw_length": 4_194_305}),
        ):
            with self.subTest(name=name):
                position, fields = block_offsets[index]
                changed = list(fields)
                for key, value in replacement.items():
                    changed[
                        {
                            "kind": 0,
                            "codec": 1,
                            "reserved": 2,
                            "count": 3,
                            "raw_length": 4,
                            "first": 6,
                        }[key]
                    ] = value
                forged = (
                    original[:position] + BLOCK.pack(*changed) + original[position + BLOCK.size :]
                )
                self.path.write_bytes(forged)
                with self.assertRaises(ResearchError):
                    self._read({**descriptor, "sha256": hashlib.sha256(forged).hexdigest()})

        # Rebuild the query block and both descriptors so the semantic mask check,
        # rather than an outer checksum, rejects each invalid vector.
        query_offset, query_fields = block_offsets[2]
        raw_query = zlib.decompress(original[query_offset + BLOCK.size :])
        for name, position in (("padding", 66), ("null-payload", 125)):
            with self.subTest(name=name):
                mutated_raw = bytearray(raw_query)
                mutated_raw[position] = 0x80 if name == "padding" else 1
                forged, forged_desc = self._replace_query_block(
                    original, descriptor, bytes(mutated_raw)
                )
                self.path.write_bytes(forged)
                with self.assertRaises(ResearchError):
                    self._read(forged_desc)

        query_payload = original[query_offset + BLOCK.size :]
        second_member = zlib.compress(b"second-member")
        new_fields = list(query_fields)
        new_fields[5] += len(second_member)
        forged = original[:query_offset] + BLOCK.pack(*new_fields) + query_payload + second_member
        self.path.write_bytes(forged)
        with self.assertRaises(ResearchError):
            self._read(
                {
                    **descriptor,
                    "bytes": len(forged),
                    "sha256": hashlib.sha256(forged).hexdigest(),
                }
            )

    def _replace_query_block(
        self, original: bytes, descriptor: dict[str, object], raw: bytes
    ) -> tuple[bytes, dict[str, object]]:
        offset = FILE.size
        decoded = bytearray(original[: FILE.size])
        stored = bytearray(original[: FILE.size])
        while offset < len(original):
            fields = list(BLOCK.unpack(original[offset : offset + BLOCK.size]))
            payload = original[offset + BLOCK.size : offset + BLOCK.size + fields[5]]
            if fields[0] == 3:
                fields[8] = hashlib.sha256(raw).digest()
                compressed = zlib.compress(raw, level=1)
                fields[1] = int(len(compressed) < len(raw))
                payload = compressed if fields[1] else raw
                fields[5] = len(payload)
            stored += BLOCK.pack(*fields) + payload
            block_raw = zlib.decompress(payload) if fields[1] else payload
            fields[1] = 0
            fields[5] = fields[4]
            decoded += BLOCK.pack(*fields) + block_raw
            offset += BLOCK.size + BLOCK.unpack(original[offset : offset + BLOCK.size])[5]
        return bytes(stored), {
            **descriptor,
            "bytes": len(stored),
            "sha256": hashlib.sha256(stored).hexdigest(),
            "uncompressed_bytes": len(decoded),
            "decoded_sha256": hashlib.sha256(decoded).hexdigest(),
        }

    def test_bound_oracle_and_bool_metadata(self) -> None:
        self.assertEqual(
            score_upper_bound(ScoreDimensions(28_955, 45_938, 0), popularity_arm=True) + 16384,
            1_906_586,
        )
        self.assertEqual(
            score_upper_bound(ScoreDimensions(28_955, 45_938, 0), popularity_arm=False) + 16384,
            10_809_430_482,
        )
        for dimensions in (
            ScoreDimensions(True, 1, 0),
            ScoreDimensions(3, 1, 2),
            ScoreDimensions(600_000, 1, 0),
        ):
            with self.assertRaises(ResearchError):
                score_upper_bound(dimensions, popularity_arm=False)
        descriptor = self._write()
        self.assertEqual(check_block_descriptor(descriptor), descriptor)
        with self.assertRaises(ResearchError):
            check_block_descriptor({**descriptor, "rows": True})
        self.assertEqual(hashlib.sha256(self.path.read_bytes()).hexdigest(), descriptor["sha256"])

    def test_valid_alternative_zlib_stream_preserves_decoded_identity(self) -> None:
        descriptor = self._write()
        original = self.path.read_bytes()
        offset = FILE.size
        while offset < len(original):
            fields = list(BLOCK.unpack(original[offset : offset + BLOCK.size]))
            if fields[0] == 3:
                break
            offset += BLOCK.size + fields[5]
        self.assertEqual(fields[0], 3)
        stored = original[offset + BLOCK.size :]
        raw = zlib.decompress(stored) if fields[1] else stored
        compressor = zlib.compressobj(level=1, strategy=zlib.Z_RLE)
        alternative = compressor.compress(raw) + compressor.flush()
        self.assertNotEqual(alternative, stored)
        self.assertLess(len(alternative), len(raw))
        fields[1] = 1
        fields[5] = len(alternative)
        changed = original[:offset] + BLOCK.pack(*fields) + alternative
        self.path.write_bytes(changed)
        changed_descriptor = {
            **descriptor,
            "bytes": len(changed),
            "sha256": hashlib.sha256(changed).hexdigest(),
        }
        changed_rows = self._read(changed_descriptor)
        self.path.write_bytes(original)
        original_rows = self._read(descriptor)
        self.assertEqual(
            [
                (row.query_id, row.mode, row.values.tolist(), row.available.tolist())
                for row in changed_rows
            ],
            [
                (row.query_id, row.mode, row.values.tolist(), row.available.tolist())
                for row in original_rows
            ],
        )
        self.assertEqual(changed_descriptor["decoded_sha256"], descriptor["decoded_sha256"])
