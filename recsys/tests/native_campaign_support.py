"""Synthetic full-catalog validation queries for the fixed native arm."""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict, cast

from recsys.contracts.bundle import canonical_bytes
from recsys.research.artifacts import StreamDescriptor, StreamWriter, read_json, stream_jsonl
from recsys.research.protocol import sha


class NativeQuery(TypedDict):
    query_id: str
    cohort: str
    seed_policy: str
    order: str
    n: int
    seed_mal_ids: list[int]


def native_query_rows() -> list[NativeQuery]:
    cases = (
        (),
        (1,),
        (1, 9),
        (9,),
        (7,),
        (2, 3),
    )
    return [
        {
            "query_id": f"q1_{index:064x}",
            "cohort": "main",
            "seed_policy": "popularity" if not seeds else "liked",
            "order": "popularity" if not seeds else f"random-liked-{min(len(seeds), 3)}",
            "n": len(seeds),
            "seed_mal_ids": list(seeds),
        }
        for index, seeds in enumerate(cases, start=1)
    ]


def replace_validation_queries(prepared: Path) -> tuple[str, list[NativeQuery]]:
    """Change only a disposable fixture preparation, preserving canonical descriptors."""
    manifest_path = prepared / "prepare-manifest.json"
    value, _ = read_json(manifest_path)
    if type(value) is not dict:
        raise AssertionError("fixture preparation invalid")
    manifest = cast("dict[str, object]", value)
    files = cast("dict[str, object]", manifest["files"])
    query_path = prepared / "development" / "queries.jsonl.gz"
    old = list(
        stream_jsonl(query_path, cast("StreamDescriptor", files["development/queries.jsonl.gz"]))
    )
    if not old:
        raise AssertionError("fixture query stream empty")
    query_path.unlink()
    rows = native_query_rows()
    writer = StreamWriter(query_path)
    for row in rows:
        writer.write_json(row)
    files["development/queries.jsonl.gz"] = writer.close()
    raw = canonical_bytes(manifest)
    manifest_path.write_bytes(raw)
    return "sha256:" + sha(raw), rows
