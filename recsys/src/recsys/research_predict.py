"""Validation-only full-catalog ML-02 prediction publication."""

from __future__ import annotations

import hashlib
import os
import re
import struct
import zlib
from pathlib import Path

from .bundle import canonical_bytes
from .research_artifacts import (MAX_LINE, MAX_STREAM, cleanup, open_stage, publish,
                                 read_json, separate, stream_jsonl, write_json)
from .research_models import ScoreVector, load_model, read_scoring_inputs
from .research_protocol import exact, fail, load_protocol, sha


class GzipOneWriter:
    """Level-1 single gzip member with the frozen ML-01 header and bounded rows."""

    def __init__(self, path):
        self.file = open(path, "xb", opener=lambda name, flags: os.open(name, flags, 0o600))
        self.compressor = zlib.compressobj(level=1, wbits=-15)
        self.compressed_hash = hashlib.sha256()
        self.plain_hash = hashlib.sha256()
        self.compressed_bytes = 0
        self.plain_bytes = 0
        self.rows = 0
        self.crc = 0
        self._emit(bytes.fromhex("1f8b08000000000000ff"))

    def _emit(self, raw):
        self.file.write(raw)
        self.compressed_hash.update(raw)
        self.compressed_bytes += len(raw)
        if self.compressed_bytes > MAX_STREAM:
            fail("research_resource_limit")

    def write_json(self, row):
        raw = canonical_bytes(row)
        if len(raw) > MAX_LINE or self.plain_bytes + len(raw) > MAX_STREAM:
            fail("research_resource_limit")
        self._emit(self.compressor.compress(raw))
        self.plain_hash.update(raw)
        self.plain_bytes += len(raw)
        self.crc = zlib.crc32(raw, self.crc)
        self.rows += 1

    def close(self):
        self._emit(self.compressor.flush())
        self._emit(struct.pack("<II", self.crc & 0xffffffff, self.plain_bytes & 0xffffffff))
        self.file.flush()
        os.fsync(self.file.fileno())
        self.file.close()
        return {"bytes": self.compressed_bytes, "sha256": self.compressed_hash.hexdigest(),
                "rows": self.rows, "uncompressed_bytes": self.plain_bytes,
                "canonical_sha256": self.plain_hash.hexdigest()}

    def abort(self):
        if not self.file.closed:
            self.file.close()


def _query(query, ids, seen, protocol):
    exact(query, {"query_id", "cohort", "seed_policy", "order", "n", "seed_mal_ids"})
    qid, n, seeds = query["query_id"], query["n"], query["seed_mal_ids"]
    if (type(qid) is not str or re.fullmatch(r"q1_[0-9a-f]{64}", qid) is None or
            qid in seen or query["cohort"] not in ("main", "short") or
            type(n) is not int or not 0 <= n <= protocol["query"]["max_n"] or
            type(seeds) is not list or len(seeds) != n or
            any(type(item) is not int or item not in ids for item in seeds) or
            len(set(seeds)) != n):
        fail("research_prediction_invalid")
    if n == 0:
        if query["order"] != "popularity" or query["seed_policy"] != "popularity":
            fail("research_prediction_invalid")
    elif (query["order"] not in protocol["query"]["orders"] or
          query["seed_policy"] != "liked"):
        fail("research_prediction_invalid")
    seen.add(qid)
    return tuple(seeds)


def _encode_vector(vector, n, ids):
    import numpy as np
    if (type(vector) is not ScoreVector or not isinstance(vector.values, np.ndarray) or
            vector.values.dtype != np.dtype("float64") or vector.values.shape != (len(ids),) or
            not isinstance(vector.available, np.ndarray) or
            vector.available.dtype != np.dtype("bool") or
            vector.available.shape != (len(ids),) or
            type(vector.supported_seeds) is not int or not 0 <= vector.supported_seeds <= n or
            not np.isfinite(vector.values).all()):
        fail("research_prediction_invalid")
    scaled = np.rint(vector.values * 1_000_000_000)
    if not np.isfinite(scaled).all() or np.any(np.abs(scaled[vector.available]) > 2**53):
        fail("research_prediction_invalid")
    return [[item, int(value) if available else None]
            for item, value, available in zip(ids, scaled, vector.available, strict=True)]


def predict_validation(prepared_dir, model_dir, output_dir, *, expected_prepared_identity):
    prepared_dir, model_dir, output_dir = map(Path, (prepared_dir, model_dir, output_dir))
    separate(prepared_dir, model_dir, output_dir)
    inputs = read_scoring_inputs(prepared_dir, expected_prepared_identity=expected_prepared_identity)
    model, scorer = load_model(model_dir, inputs=inputs)
    prep, _ = read_json(prepared_dir / "prepare-manifest.json")
    query_desc = prep["files"]["development/queries.jsonl.gz"]
    ids = inputs.mal_ids
    id_set = set(ids)
    protocol = load_protocol()
    popularity = [[item, inputs.train_stats["items"][str(item)]["positive_users"]] for item in ids]
    stage = open_stage(output_dir)
    writer = None
    try:
        writer = GzipOneWriter(stage / "scores.jsonl.gz")
        seen = set()
        for query in stream_jsonl(prepared_dir / "development/queries.jsonl.gz", query_desc):
            seeds = _query(query, id_set, seen, protocol)
            scores = (popularity if not seeds or model["variant_id"] == "popularity"
                      else _encode_vector(scorer.score(seeds), len(seeds), ids))
            writer.write_json({"query_id": query["query_id"], "scores": scores})
        scores_desc = writer.close()
        writer = None
        manifest = {"schema_version": 1, "kind": "arb-ml-predictions",
                    "prepared_identity": inputs.prepared_identity,
                    "query_descriptor": query_desc,
                    "catalog_descriptor": inputs.catalog_descriptor,
                    "split": "validation", "model_id": model["variant_id"],
                    "config_sha256": sha(canonical_bytes(model)),
                    "fit_identity": inputs.fit_identity,
                    "train_interactions": inputs.train_interactions_descriptor,
                    "train_stats": inputs.train_stats_descriptor,
                    "code_revision": model["producer"]["code_revision"],
                    "zero_seed_policy": "popularity-train-positive-user-counts",
                    "scores": scores_desc}
        write_json(stage / "prediction-manifest.json", manifest)
        publish(stage, output_dir)
        return output_dir / "prediction-manifest.json"
    except Exception:
        if writer is not None:
            writer.abort()
        cleanup(stage)
        raise
