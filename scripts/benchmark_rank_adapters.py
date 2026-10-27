"""Synthetic, paired production ranking-adapter timings for ARB-065."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import TypedDict, cast

import numpy as np
from benchmark_score_blocks import COUNTS, PROFILES, SEED, C, _row

from recsys.research.evaluation.evaluate import EvaluationTrainStats, _rank, _rank_binary
from recsys.research.score_blocks import DecodedScoreRow

REPEATS = 5
FORMATS = ("jsonl-v1", "binary-v1")


class AdapterResult(TypedDict):
    rank_seconds: float
    ranking_sha256: str


class CaseResult(TypedDict):
    profile: str
    queries: int
    repeat: int
    order: list[str]
    excluded_count: int
    process_peak_rss_bytes: int
    results: dict[str, AdapterResult]


def _case(profile: str, queries: int, repeat: int) -> CaseResult:
    ids = list(range(1, C + 1))
    catalog_ids = tuple(ids)
    popularity = tuple((item * 17) % 509 for item in ids)
    popularity_values = np.asarray(popularity, dtype=np.int64)
    popularity_available = np.ones(C, dtype=np.bool_)
    stats: EvaluationTrainStats = {
        "schema_version": 1,
        "train_users": 0,
        "train_completed_rows": 0,
        "items": {
            str(mal_id): {
                "completed_users": 0,
                "positive_users": count,
                "rating_sum": 0,
                "rating_count": 0,
                "band": "all",
            }
            for mal_id, count in zip(ids, popularity, strict=True)
        },
    }
    excluded = set(ids[::2711]) | {C}
    order = FORMATS if repeat % 2 == 0 else FORMATS[::-1]
    results: dict[str, AdapterResult] = {}
    for fmt in order:
        elapsed = 0.0
        digest = hashlib.sha256()
        for ordinal in range(queries):
            query_id, n, vector, pairs = _row(
                profile, ordinal, catalog_ids, popularity, legacy=True
            )
            assert pairs is not None
            if vector is None:
                row = DecodedScoreRow(
                    query_id, "popularity", popularity_values, popularity_available
                )
            else:
                row = DecodedScoreRow(query_id, "dense", vector.values, vector.available)
            start = time.perf_counter()
            if fmt == "jsonl-v1":
                ranking = _rank(pairs, ids, excluded, n0=n == 0, stats=stats)
            else:
                ranking = _rank_binary(row, ids, excluded)
            elapsed += time.perf_counter() - start
            digest.update(json.dumps([query_id, ranking], separators=(",", ":")).encode())
        results[fmt] = {"rank_seconds": elapsed, "ranking_sha256": digest.hexdigest()}
    assert results["jsonl-v1"]["ranking_sha256"] == results["binary-v1"]["ranking_sha256"]
    return {
        "profile": profile,
        "queries": queries,
        "repeat": repeat,
        "order": list(order),
        "excluded_count": len(excluded),
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
        assert args.profile is not None and args.queries is not None and args.repeat is not None
        print(json.dumps(_case(args.profile, args.queries, args.repeat), sort_keys=True))
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
    aggregates = [
        {
            "queries": queries,
            "profile": profile,
            "excluded_count": next(
                c["excluded_count"]
                for c in cases
                if c["queries"] == queries and c["profile"] == profile
            ),
            "median_process_peak_rss_bytes": statistics.median(
                c["process_peak_rss_bytes"]
                for c in cases
                if c["queries"] == queries and c["profile"] == profile
            ),
            "formats": {
                fmt: {
                    "median_rank_seconds": statistics.median(
                        c["results"][fmt]["rank_seconds"]
                        for c in cases
                        if c["queries"] == queries and c["profile"] == profile
                    )
                }
                for fmt in FORMATS
            },
        }
        for queries in COUNTS
        for profile in PROFILES
    ]
    report = {
        "kind": "arb065-synthetic-production-rank-adapters",
        "catalog_count": C,
        "query_counts": COUNTS,
        "profiles": PROFILES,
        "seed": SEED,
        "alternating_repeats": REPEATS,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "cases": cases,
        "aggregates": aggregates,
        "limits": (
            "Only production _rank and _rank_binary calls are timed on preconstructed synthetic "
            "rows; parsing, binary block verification, query binding, full evaluation, ML-02 "
            "completion, and scientific quality are not measured. RSS spans both formats per case."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    args.output.write_text(json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
