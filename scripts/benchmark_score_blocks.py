"""Synthetic-only size and timing evidence for the ARB-065 score transport."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import statistics
import subprocess
import sys
import tempfile
import time
import zlib
from collections.abc import Iterator
from heapq import nsmallest
from pathlib import Path
from typing import TypedDict, cast

import numpy as np

from recsys.research.artifacts import StreamDescriptor, stream_jsonl
from recsys.research.predict import GzipOneWriter
from recsys.research.score_blocks import (
    BlockScoreDescriptor,
    BlockScoreWriter,
    EncodedVector,
    ScoreDimensions,
    iter_score_rows,
)

C = 28_955
COUNTS = (41, 1024)
PROFILES = (
    "repeated-popularity",
    "low-cardinality-ties",
    "signed-high-entropy",
    "sparse-null-zero",
)
REPEATS = 5
SEED = 65_2026


class FormatResult(TypedDict):
    bytes: int
    write_seconds: float
    verify_decode_seconds: float
    evaluation_seconds: float
    decoded_sha256: str
    ranking_sha256: str


class CaseResult(TypedDict):
    profile: str
    queries: int
    repeat: int
    order: list[str]
    zero_seed_queries: int
    process_peak_rss_bytes: int
    results: dict[str, FormatResult]


def _vector(profile: str, ordinal: int, ids: tuple[int, ...]) -> tuple[int, EncodedVector | None]:
    if profile == "repeated-popularity" or profile == "sparse-null-zero" and ordinal % 7 == 0:
        return 0, None
    rng = np.random.default_rng(SEED + ordinal * 1_000_003 + PROFILES.index(profile) * 17)
    if profile == "low-cardinality-ties":
        values = rng.integers(-3, 4, size=len(ids), dtype=np.int64)
        available = np.ones(len(ids), dtype=np.bool_)
    elif profile == "signed-high-entropy":
        values = rng.integers(-(2**53), 2**53 + 1, size=len(ids), dtype=np.int64)
        available = np.ones(len(ids), dtype=np.bool_)
    else:
        values = rng.integers(-100, 101, size=len(ids), dtype=np.int64)
        available = rng.random(len(ids)) < 0.1
        values[~available] = 0
        values[ordinal % len(ids)] = 0
        available[ordinal % len(ids)] = True
    return 1, EncodedVector(values, available)


def _row(
    profile: str, ordinal: int, ids: tuple[int, ...], popularity: tuple[int, ...], *, legacy: bool
) -> tuple[str, int, EncodedVector | None, list[list[int | None]] | None]:
    query_id = "q1_" + hashlib.sha256(f"synthetic:{profile}:{ordinal}".encode()).hexdigest()
    n, vector = _vector(profile, ordinal, ids)
    pairs: list[list[int | None]] | None
    if not legacy:
        pairs = None
    elif vector is None:
        pairs = [[mal_id, count] for mal_id, count in zip(ids, popularity, strict=True)]
    else:
        pairs = [
            [mal_id, int(value) if available else None]
            for mal_id, value, available in zip(ids, vector.values, vector.available, strict=True)
        ]
    return query_id, n, vector, pairs


def _rank(pairs: list[list[int | None]]) -> tuple[int, ...]:
    def candidates() -> Iterator[tuple[int, int]]:
        for mal_id, score in pairs:
            if score is not None:
                yield -score, cast("int", mal_id)

    return tuple(mal_id for _, mal_id in nsmallest(10, candidates()))


def _case(profile: str, queries: int, repeat: int, root: Path) -> CaseResult:
    ids = tuple(range(1, C + 1))
    popularity = tuple((item * 17) % 509 for item in ids)
    zero = (
        queries
        if profile == "repeated-popularity"
        else (queries + 6) // 7
        if profile == "sparse-null-zero"
        else 0
    )
    order = ("jsonl-v1", "binary-v1") if repeat % 2 == 0 else ("binary-v1", "jsonl-v1")
    results: dict[str, FormatResult] = {}
    for fmt in order:
        path = root / ("scores.jsonl.gz" if fmt == "jsonl-v1" else "scores.bin")
        start = time.perf_counter()
        writer: GzipOneWriter | BlockScoreWriter
        if fmt == "jsonl-v1":
            writer = GzipOneWriter(path)
        else:
            writer = BlockScoreWriter(
                path,
                catalog_ids=ids,
                dimensions=ScoreDimensions(C, queries, zero),
                popularity=popularity,
                popularity_arm=profile == "repeated-popularity",
                binding_sha256="a" * 64,
            )
        for index in range(queries):
            query_id, n, vector, pairs = _row(
                profile, index, ids, popularity, legacy=fmt == "jsonl-v1"
            )
            if isinstance(writer, GzipOneWriter):
                assert pairs is not None
                writer.write_json({"query_id": query_id, "scores": pairs})
            else:
                writer.write_query(query_id, n, vector)
        descriptor = writer.close()
        write_seconds = time.perf_counter() - start

        def rows() -> Iterator[tuple[str, list[list[int | None]]]]:
            if fmt == "jsonl-v1":
                for item in stream_jsonl(path, cast("StreamDescriptor", descriptor)):
                    record = cast("dict[str, object]", item)
                    yield (
                        cast("str", record["query_id"]),
                        cast("list[list[int | None]]", record["scores"]),
                    )
            else:
                for item in iter_score_rows(
                    path,
                    cast("BlockScoreDescriptor", descriptor),
                    expected_catalog_ids=ids,
                    expected_binding_sha256="a" * 64,
                    expected_popularity=popularity,
                    popularity_arm=profile == "repeated-popularity",
                ):
                    yield (
                        item.query_id,
                        [
                            [mal_id, int(value) if available else None]
                            for mal_id, value, available in zip(
                                ids, item.values, item.available, strict=True
                            )
                        ],
                    )

        start = time.perf_counter()
        digest = hashlib.sha256()
        count = 0
        for query_id, pairs in rows():
            digest.update(json.dumps([query_id, pairs], separators=(",", ":")).encode())
            count += 1
        verify_decode_seconds = time.perf_counter() - start
        assert count == queries
        start = time.perf_counter()
        rank_digest = hashlib.sha256()
        for _query_id, pairs in rows():
            rank_digest.update(json.dumps(_rank(pairs), separators=(",", ":")).encode())
        evaluation_seconds = time.perf_counter() - start
        results[fmt] = {
            "bytes": path.stat().st_size,
            "write_seconds": write_seconds,
            "verify_decode_seconds": verify_decode_seconds,
            "evaluation_seconds": evaluation_seconds,
            "decoded_sha256": digest.hexdigest(),
            "ranking_sha256": rank_digest.hexdigest(),
        }
        path.unlink()
    assert results["jsonl-v1"]["decoded_sha256"] == results["binary-v1"]["decoded_sha256"]
    assert results["jsonl-v1"]["ranking_sha256"] == results["binary-v1"]["ranking_sha256"]
    return {
        "profile": profile,
        "queries": queries,
        "repeat": repeat,
        "order": list(order),
        "zero_seed_queries": zero,
        "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--profile", choices=PROFILES)
    parser.add_argument("--queries", type=int, choices=COUNTS)
    parser.add_argument("--repeat", type=int)
    args = parser.parse_args()
    if args.worker:
        assert args.profile and args.queries and args.repeat is not None
        with tempfile.TemporaryDirectory() as temporary:
            print(json.dumps(_case(args.profile, args.queries, args.repeat, Path(temporary))))
        return
    if args.output is None or not args.output.is_absolute():
        parser.error("--output must be an absolute private file path")
    cases: list[CaseResult] = []
    for queries in COUNTS:
        for profile in PROFILES:
            for repeat in range(REPEATS):
                completed = subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--worker",
                        "--profile",
                        profile,
                        "--queries",
                        str(queries),
                        "--repeat",
                        str(repeat),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                cases.append(cast("CaseResult", json.loads(completed.stdout)))
    aggregates = []
    for queries in COUNTS:
        for profile in PROFILES:
            selected = [
                case for case in cases if case["queries"] == queries and case["profile"] == profile
            ]
            aggregates.append(
                {
                    "queries": queries,
                    "profile": profile,
                    "zero_seed_queries": selected[0]["zero_seed_queries"],
                    "median_process_peak_rss_bytes": statistics.median(
                        case["process_peak_rss_bytes"] for case in selected
                    ),
                    "formats": {
                        fmt: {
                            key: statistics.median(case["results"][fmt][key] for case in selected)
                            for key in (
                                "bytes",
                                "write_seconds",
                                "verify_decode_seconds",
                                "evaluation_seconds",
                            )
                        }
                        for fmt in ("jsonl-v1", "binary-v1")
                    },
                }
            )
    report = {
        "kind": "arb065-synthetic-score-block-benchmark",
        "catalog_count": C,
        "query_counts": COUNTS,
        "profiles": PROFILES,
        "seed": SEED,
        "alternating_repeats": REPEATS,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "zlib": zlib.ZLIB_VERSION,
        "runtime_zlib": zlib.ZLIB_RUNTIME_VERSION,
        "cases": cases,
        "aggregates": aggregates,
        "limits": "Synthetic transport and ranking only; full ML-02 completion and scientific quality are not measured.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    args.output.write_text(json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
