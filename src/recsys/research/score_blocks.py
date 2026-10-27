"""Bounded, lossless validation score container (arb-score-blocks-v1)."""

from __future__ import annotations

import hashlib
import os
import re
import struct
import zlib
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypedDict, cast

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

from recsys.research.artifacts import MAX_STREAM, safe_path
from recsys.research.protocol import fail

FORMAT = "arb-score-blocks-v1"
MAGIC = b"ARBSCR1\x00"
FILE = struct.Struct("<8sHHIQQ32s")
BLOCK = struct.Struct("<BBHIIIQQ32s")
MAX_BLOCK = 4 * 1024 * 1024
MAX_ROWS = 128
MAX_MANIFEST = 16384
_QUERY = re.compile(r"q1_([0-9a-f]{64})\Z")


class BlockScoreDescriptor(TypedDict):
    bytes: int
    sha256: str
    rows: int
    uncompressed_bytes: int
    decoded_sha256: str
    catalog_count: int
    dense_rows: int
    query_block_count: int


@dataclass(frozen=True, slots=True)
class EncodedVector:
    values: NDArray[np.int64]
    available: NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class DecodedScoreRow:
    query_id: str
    mode: Literal["popularity", "dense"]
    values: NDArray[np.int64]
    available: NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class ScoreDimensions:
    catalog_count: int
    query_count: int
    zero_seed_queries: int


def _dimensions(value: ScoreDimensions) -> tuple[int, int, int, int, int]:
    if type(value) is not ScoreDimensions:
        fail("research_prediction_invalid")
    c, q, q0 = value.catalog_count, value.query_count, value.zero_seed_queries
    if (
        type(c) is not int
        or type(q) is not int
        or type(q0) is not int
        or not 1 <= c <= 0xFFFFFFFF
        or not 0 <= q <= 0xFFFFFFFFFFFFFFFF
        or not 0 <= q0 <= q
    ):
        fail("research_prediction_invalid")
    v = 8 * c + (c + 7) // 8
    if 33 + v > MAX_BLOCK or 4 * c > MAX_BLOCK:
        fail("research_resource_limit")
    return c, q, q0, v, min(MAX_ROWS, MAX_BLOCK // (33 + v))


def score_upper_bound(dimensions: ScoreDimensions, *, popularity_arm: bool) -> int:
    c, q, q0, v, batch = _dimensions(dimensions)
    dense = 0 if popularity_arm else q - q0
    blocks = (q + batch - 1) // batch if not popularity_arm else (q + MAX_ROWS - 1) // MAX_ROWS
    total = FILE.size + BLOCK.size + 4 * c + int(q > 0) * (BLOCK.size + v)
    total += blocks * BLOCK.size + 33 * q + dense * v
    if total > MAX_STREAM:
        fail("research_resource_limit")
    return total


def check_block_descriptor(value: object) -> BlockScoreDescriptor:
    keys = {
        "bytes",
        "sha256",
        "rows",
        "uncompressed_bytes",
        "decoded_sha256",
        "catalog_count",
        "dense_rows",
        "query_block_count",
    }
    if type(value) is not dict or set(value) != keys:
        fail("research_prediction_invalid")
    result = cast("BlockScoreDescriptor", value)
    for key in (
        "bytes",
        "rows",
        "uncompressed_bytes",
        "catalog_count",
        "dense_rows",
        "query_block_count",
    ):
        item = cast("dict[str, object]", result)[key]
        if type(item) is not int or item < 0 or item > MAX_STREAM:
            fail("research_prediction_invalid")
    for key in ("sha256", "decoded_sha256"):
        item = cast("dict[str, object]", result)[key]
        if type(item) is not str or re.fullmatch(r"[0-9a-f]{64}", item) is None:
            fail("research_prediction_invalid")
    if result["catalog_count"] < 1 or result["dense_rows"] > result["rows"]:
        fail("research_prediction_invalid")
    return result


def _catalog(ids: Sequence[int]) -> tuple[int, ...]:
    if not isinstance(ids, (tuple, list)) or not ids:
        fail("research_prediction_invalid")
    values = tuple(ids)
    if any(type(x) is not int or not 1 <= x <= 0x7FFFFFFF for x in values):
        fail("research_prediction_invalid")
    if tuple(sorted(set(values))) != values:
        fail("research_prediction_invalid")
    return values


def _popularity(values: Sequence[int], c: int) -> EncodedVector:
    import numpy as np

    if len(values) != c or any(
        type(x) is not int or not 0 <= x <= 0x7FFFFFFFFFFFFFFF for x in values
    ):
        fail("research_prediction_invalid")
    answer = EncodedVector(np.asarray(values, dtype=np.int64), np.ones(c, dtype=np.bool_))
    answer.values.flags.writeable = False
    answer.available.flags.writeable = False
    return answer


def _vector_raw(vector: EncodedVector, c: int) -> bytes:
    import numpy as np

    if (
        type(vector) is not EncodedVector
        or not isinstance(vector.values, np.ndarray)
        or vector.values.dtype != np.dtype("int64")
        or vector.values.shape != (c,)
        or not isinstance(vector.available, np.ndarray)
        or vector.available.dtype != np.dtype("bool")
        or vector.available.shape != (c,)
        or np.any(vector.values[~vector.available] != 0)
    ):
        fail("research_prediction_invalid")
    bitmap = np.packbits(vector.available, bitorder="little").tobytes()
    return bitmap + vector.values.astype("<i8", copy=False).tobytes()


def _read_vector(raw: bytes, c: int) -> EncodedVector:
    import numpy as np

    bitmap_size = (c + 7) // 8
    if len(raw) != bitmap_size + 8 * c:
        fail("research_prediction_invalid")
    if c % 8 and raw[bitmap_size - 1] >> (c % 8):
        fail("research_prediction_invalid")
    available = np.unpackbits(np.frombuffer(raw[:bitmap_size], dtype=np.uint8), bitorder="little")[
        :c
    ].astype(np.bool_)
    values = np.frombuffer(raw[bitmap_size:], dtype="<i8").astype(np.int64, copy=True)
    if np.any(values[~available] != 0):
        fail("research_prediction_invalid")
    values.flags.writeable = False
    available.flags.writeable = False
    return EncodedVector(values, available)


class BlockScoreWriter:
    def __init__(
        self,
        path: Path,
        *,
        catalog_ids: Sequence[int],
        dimensions: ScoreDimensions,
        popularity: Sequence[int],
        popularity_arm: bool,
        binding_sha256: str,
        storage_guard: Callable[[int], None] | None = None,
    ) -> None:
        self.catalog_ids = _catalog(catalog_ids)
        self.c, self.q, self.q0, self.v, base_batch = _dimensions(dimensions)
        if self.c != len(self.catalog_ids) or type(popularity_arm) is not bool:
            fail("research_prediction_invalid")
        if type(binding_sha256) is not str or re.fullmatch(r"[0-9a-f]{64}", binding_sha256) is None:
            fail("research_prediction_invalid")
        self.popularity_arm = popularity_arm
        self.popularity = _popularity(popularity, self.c)
        self.batch = MAX_ROWS if popularity_arm else base_batch
        self.guard = storage_guard
        self.stored_hash = hashlib.sha256()
        self.decoded_hash = hashlib.sha256()
        self.stored_bytes = self.decoded_bytes = self.rows = self.dense_rows = self.query_blocks = 0
        self.pending = bytearray()
        self.pending_rows = self.pending_dense = 0
        self.closed = False
        self.file = open(path, "xb", opener=lambda name, flags: os.open(name, flags, 0o600))
        try:
            self._emit(
                FILE.pack(
                    MAGIC,
                    1,
                    0,
                    self.c,
                    self.q,
                    0 if popularity_arm else self.q - self.q0,
                    bytes.fromhex(binding_sha256),
                )
            )
            self._block(1, self.c, 0, 0, struct.pack(f"<{self.c}I", *self.catalog_ids))
            if self.q:
                self._block(2, 1, 0, 0, _vector_raw(self.popularity, self.c))
        except BaseException:
            self.abort()
            raise

    def _emit(self, raw: bytes, *, decoded: bytes | None = None) -> None:
        if (
            self.stored_bytes + len(raw) > MAX_STREAM
            or self.decoded_bytes + len(decoded or raw) > MAX_STREAM
        ):
            fail("research_resource_limit")
        if self.guard is not None:
            self.guard(len(raw))
        self.file.write(raw)
        self.stored_hash.update(raw)
        self.stored_bytes += len(raw)
        self.decoded_hash.update(decoded if decoded is not None else raw)
        self.decoded_bytes += len(decoded) if decoded is not None else len(raw)

    def _block(self, kind: int, count: int, first: int, dense: int, raw: bytes) -> None:
        if len(raw) > MAX_BLOCK:
            fail("research_resource_limit")
        compressed = zlib.compress(raw, level=1)
        codec, stored = (1, compressed) if len(compressed) < len(raw) else (0, raw)
        normalized = BLOCK.pack(
            kind, 0, 0, count, len(raw), len(raw), first, dense, hashlib.sha256(raw).digest()
        )
        actual = BLOCK.pack(
            kind, codec, 0, count, len(raw), len(stored), first, dense, hashlib.sha256(raw).digest()
        )
        self._emit(actual, decoded=normalized)
        self._emit(stored, decoded=raw)

    def write_query(self, query_id: str, n: int, vector: EncodedVector | None) -> None:
        match = _QUERY.fullmatch(query_id) if type(query_id) is str else None
        if self.closed or self.rows >= self.q or match is None or type(n) is not int or n < 0:
            fail("research_prediction_invalid")
        broadcast = self.popularity_arm or n == 0
        if (vector is None) != broadcast:
            fail("research_prediction_invalid")
        self.pending += bytes.fromhex(match.group(1)) + bytes([0 if broadcast else 1])
        if not broadcast:
            self.pending += _vector_raw(cast("EncodedVector", vector), self.c)
            self.pending_dense += 1
            self.dense_rows += 1
        self.rows += 1
        self.pending_rows += 1
        if self.pending_rows == self.batch:
            self._flush_queries()

    def _flush_queries(self) -> None:
        if self.pending_rows:
            self._block(
                3,
                self.pending_rows,
                self.rows - self.pending_rows,
                self.pending_dense,
                bytes(self.pending),
            )
            self.query_blocks += 1
            self.pending.clear()
            self.pending_rows = self.pending_dense = 0

    def close(self) -> BlockScoreDescriptor:
        if (
            self.closed
            or self.rows != self.q
            or self.dense_rows != (0 if self.popularity_arm else self.q - self.q0)
        ):
            fail("research_prediction_invalid")
        self._flush_queries()
        self.file.flush()
        os.fsync(self.file.fileno())
        self.file.close()
        self.closed = True
        return {
            "bytes": self.stored_bytes,
            "sha256": self.stored_hash.hexdigest(),
            "rows": self.rows,
            "uncompressed_bytes": self.decoded_bytes,
            "decoded_sha256": self.decoded_hash.hexdigest(),
            "catalog_count": self.c,
            "dense_rows": self.dense_rows,
            "query_block_count": self.query_blocks,
        }

    def abort(self) -> None:
        if not self.file.closed:
            self.file.close()
        self.closed = True


def require_query_binding(
    row: DecodedScoreRow, *, query_id: str, n: int, popularity_arm: bool
) -> None:
    if row.query_id != query_id or (row.mode == "popularity") != (popularity_arm or n == 0):
        fail("research_prediction_invalid")


def iter_score_rows(
    path: Path,
    descriptor: object,
    *,
    expected_catalog_ids: Sequence[int],
    expected_binding_sha256: str,
    expected_popularity: Sequence[int],
    popularity_arm: bool,
    progress_guard: Callable[[int], None] | None = None,
) -> Iterator[DecodedScoreRow]:
    import numpy as np

    desc = check_block_descriptor(descriptor)
    ids = _catalog(expected_catalog_ids)
    dims = ScoreDimensions(len(ids), desc["rows"], desc["rows"] - desc["dense_rows"])
    c, q, _, v, base_batch = _dimensions(dims)
    batch = MAX_ROWS if popularity_arm else base_batch
    bound = score_upper_bound(dims, popularity_arm=popularity_arm)
    if (
        desc["catalog_count"] != c
        or desc["query_block_count"] != (q + batch - 1) // batch
        or (popularity_arm and desc["dense_rows"])
    ):
        fail("research_prediction_invalid")
    if desc["uncompressed_bytes"] != bound or desc["bytes"] > bound:
        fail("research_prediction_invalid")
    expected_pop = _popularity(expected_popularity, c)
    if (
        type(expected_binding_sha256) is not str
        or re.fullmatch(r"[0-9a-f]{64}", expected_binding_sha256) is None
    ):
        fail("research_prediction_invalid")
    stored_hash, decoded_hash = hashlib.sha256(), hashlib.sha256()
    stored_bytes = decoded_bytes = 0

    def account(stored: bytes, decoded: bytes | None = None) -> None:
        nonlocal stored_bytes, decoded_bytes
        stored_hash.update(stored)
        stored_bytes += len(stored)
        decoded_hash.update(decoded if decoded is not None else stored)
        decoded_bytes += len(decoded) if decoded is not None else len(stored)
        if stored_bytes > MAX_STREAM or decoded_bytes > MAX_STREAM:
            fail("research_resource_limit")

    path = safe_path(path)
    if path.stat().st_size != desc["bytes"]:
        fail("research_prediction_invalid")
    with path.open("rb") as source:

        def exact_read(n: int) -> bytes:
            if progress_guard is not None:
                progress_guard(n)
            raw = source.read(n)
            if len(raw) != n:
                fail("research_prediction_invalid")
            return raw

        header = exact_read(FILE.size)
        account(header)
        magic, version, flags, file_c, file_q, file_d, binding = FILE.unpack(header)
        if (magic, version, flags, file_c, file_q, file_d, binding) != (
            MAGIC,
            1,
            0,
            c,
            q,
            desc["dense_rows"],
            bytes.fromhex(expected_binding_sha256),
        ):
            fail("research_prediction_invalid")

        def read_block(kind: int, count: int, first: int, dense: int, raw_len: int) -> bytes:
            if not 0 < raw_len <= MAX_BLOCK or not 0 < count <= 0xFFFFFFFF:
                fail("research_prediction_invalid")
            block_header = exact_read(BLOCK.size)
            (
                k,
                codec,
                reserved,
                actual_count,
                actual_raw,
                stored_len,
                actual_first,
                actual_dense,
                digest,
            ) = BLOCK.unpack(block_header)
            if (
                (k, reserved, actual_count, actual_raw, actual_first, actual_dense)
                != (kind, 0, count, raw_len, first, dense)
                or codec not in (0, 1)
                or (codec == 0 and stored_len != raw_len)
                or (codec == 1 and not 0 < stored_len < raw_len)
            ):
                fail("research_prediction_invalid")
            payload = exact_read(stored_len)
            if codec == 1:
                try:
                    decoder = zlib.decompressobj()
                    raw = decoder.decompress(payload, raw_len + 1)
                except zlib.error:
                    fail("research_prediction_invalid")
                if not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
                    fail("research_prediction_invalid")
            else:
                raw = payload
            if len(raw) != raw_len or hashlib.sha256(raw).digest() != digest:
                fail("research_prediction_invalid")
            normalized = BLOCK.pack(kind, 0, 0, count, raw_len, raw_len, first, dense, digest)
            account(block_header, normalized)
            account(payload, raw)
            return raw

        catalog_raw = read_block(1, c, 0, 0, 4 * c)
        if struct.unpack(f"<{c}I", catalog_raw) != ids:
            fail("research_prediction_invalid")
        pop = expected_pop
        if q:
            observed_pop = _read_vector(read_block(2, 1, 0, 0, v), c)
            if not observed_pop.available.all() or not np.array_equal(
                observed_pop.values, expected_pop.values
            ):
                fail("research_prediction_invalid")
            pop = observed_pop
        rows = dense_rows = 0
        while rows < q:
            count = min(batch, q - rows)
            # The count of dense rows is encoded in the header; validate its feasible range before payload read.
            position = source.tell()
            raw_header = exact_read(BLOCK.size)
            _, _, _, _, _, _, _, dense_count, _ = BLOCK.unpack(raw_header)
            source.seek(position)
            if dense_count > count or (popularity_arm and dense_count):
                fail("research_prediction_invalid")
            raw = read_block(3, count, rows, dense_count, 33 * count + v * dense_count)
            offset = 0
            decoded: list[DecodedScoreRow] = []
            for _ in range(count):
                suffix = raw[offset : offset + 32].hex()
                mode = raw[offset + 32]
                offset += 33
                if mode == 0:
                    vector = pop
                elif mode == 1 and not popularity_arm:
                    vector = _read_vector(raw[offset : offset + v], c)
                    offset += v
                else:
                    fail("research_prediction_invalid")
                decoded.append(
                    DecodedScoreRow(
                        "q1_" + suffix,
                        "dense" if mode else "popularity",
                        vector.values,
                        vector.available,
                    )
                )
            if offset != len(raw) or sum(row.mode == "dense" for row in decoded) != dense_count:
                fail("research_prediction_invalid")
            rows += count
            dense_rows += dense_count
            yield from decoded
        if dense_rows != desc["dense_rows"] or source.read(1):
            fail("research_prediction_invalid")
        if (
            stored_bytes != desc["bytes"]
            or decoded_bytes != desc["uncompressed_bytes"]
            or stored_hash.hexdigest() != desc["sha256"]
            or decoded_hash.hexdigest() != desc["decoded_sha256"]
        ):
            fail("research_prediction_invalid")
