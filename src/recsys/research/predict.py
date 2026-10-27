"""Validation-only full-catalog ML-02 prediction publication."""

from __future__ import annotations

import hashlib
import os
import re
import stat
import struct
import zlib
from collections.abc import Callable
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from time import perf_counter_ns
from typing import Literal, TypedDict, cast

from recsys.contracts.bundle import canonical_bytes
from recsys.research.artifacts import (
    MAX_LINE,
    MAX_STREAM,
    StreamDescriptor,
    cleanup,
    open_stage,
    publish,
    read_json,
    safe_path,
    separate,
    stream_jsonl,
    write_json,
)
from recsys.research.models.inputs import read_scoring_inputs
from recsys.research.models.package import load_model
from recsys.research.models.spec import require_cpu_backend
from recsys.research.models.types import (
    BatchScorer,
    CloseableScorer,
    Scorer,
    ScoreVector,
    ScoringInputs,
)
from recsys.research.protocol import ProtocolSpec, exact, fail, load_protocol, sha
from recsys.research.score_blocks import (
    FORMAT,
    MAX_MANIFEST,
    BlockScoreDescriptor,
    BlockScoreWriter,
    EncodedVector,
    ScoreDimensions,
)


class PredictionQuery(TypedDict):
    query_id: str
    cohort: str
    seed_policy: str
    order: str
    n: int
    seed_mal_ids: list[int]


type EncodedScores = list[list[int | None]]
type ValidatedQuery = tuple[PredictionQuery, tuple[int, ...]]


@dataclass(frozen=True)
class PredictionModelMetadata:
    variant_id: str
    config_sha256: str
    code_revision: str


class GzipOneWriter:
    """Level-1 single gzip member with the frozen ML-01 header and bounded rows."""

    def __init__(self, path: Path, storage_guard: Callable[[int], None] | None = None) -> None:
        self.file = open(path, "xb", opener=lambda name, flags: os.open(name, flags, 0o600))
        self.compressor = zlib.compressobj(level=1, wbits=-15)
        self.compressed_hash = hashlib.sha256()
        self.plain_hash = hashlib.sha256()
        self.compressed_bytes = 0
        self.plain_bytes = 0
        self.rows = 0
        self.crc = 0
        self.storage_guard = storage_guard
        try:
            self._emit(bytes.fromhex("1f8b08000000000000ff"))
        except BaseException:
            self.file.close()
            raise

    def _emit(self, raw: bytes) -> None:
        if self.compressed_bytes + len(raw) > MAX_STREAM:
            fail("research_resource_limit")
        if self.storage_guard is not None:
            self.storage_guard(len(raw))
        self.file.write(raw)
        self.compressed_hash.update(raw)
        self.compressed_bytes += len(raw)

    def write_json(self, row: object) -> None:
        raw = canonical_bytes(row)
        if len(raw) > MAX_LINE or self.plain_bytes + len(raw) > MAX_STREAM:
            fail("research_resource_limit")
        self._emit(self.compressor.compress(raw))
        self.plain_hash.update(raw)
        self.plain_bytes += len(raw)
        self.crc = zlib.crc32(raw, self.crc)
        self.rows += 1

    def close(self) -> StreamDescriptor:
        self._emit(self.compressor.flush())
        self._emit(struct.pack("<II", self.crc & 0xFFFFFFFF, self.plain_bytes & 0xFFFFFFFF))
        self.file.flush()
        os.fsync(self.file.fileno())
        self.file.close()
        return {
            "bytes": self.compressed_bytes,
            "sha256": self.compressed_hash.hexdigest(),
            "rows": self.rows,
            "uncompressed_bytes": self.plain_bytes,
            "canonical_sha256": self.plain_hash.hexdigest(),
        }

    def abort(self) -> None:
        if not self.file.closed:
            self.file.close()


def _query(query: object, ids: set[int], seen: set[str], protocol: ProtocolSpec) -> tuple[int, ...]:
    exact(query, {"query_id", "cohort", "seed_policy", "order", "n", "seed_mal_ids"})
    query = cast("PredictionQuery", query)
    qid, n, seeds = query["query_id"], query["n"], query["seed_mal_ids"]
    if (
        type(qid) is not str
        or re.fullmatch(r"q1_[0-9a-f]{64}", qid) is None
        or qid in seen
        or query["cohort"] not in ("main", "short")
        or type(n) is not int
        or not 0 <= n <= protocol["query"]["max_n"]
        or type(seeds) is not list
        or len(seeds) != n
        or any(type(item) is not int or item not in ids for item in seeds)
        or len(set(seeds)) != n
    ):
        fail("research_prediction_invalid")
    if n == 0:
        if query["order"] != "popularity" or query["seed_policy"] != "popularity":
            fail("research_prediction_invalid")
    elif query["order"] not in protocol["query"]["orders"] or query["seed_policy"] != "liked":
        fail("research_prediction_invalid")
    seen.add(qid)
    return tuple(seeds)


def _checked_vector(vector: object, n: int, ids: tuple[int, ...]) -> EncodedVector:
    import numpy as np

    if (
        type(vector) is not ScoreVector
        or not isinstance(vector.values, np.ndarray)
        or vector.values.dtype != np.dtype("float64")
        or vector.values.shape != (len(ids),)
        or not isinstance(vector.available, np.ndarray)
        or vector.available.dtype != np.dtype("bool")
        or vector.available.shape != (len(ids),)
        or type(vector.supported_seeds) is not int
        or not 0 <= vector.supported_seeds <= n
        or not np.isfinite(vector.values).all()
    ):
        fail("research_prediction_invalid")
    scaled = np.rint(vector.values * 1_000_000_000)
    if not np.isfinite(scaled).all() or np.any(np.abs(scaled[vector.available]) > 2**53):
        fail("research_prediction_invalid")
    scaled[~vector.available] = 0
    values = scaled.astype(np.int64)
    return EncodedVector(values, vector.available)


def _encode_vector(vector: object, n: int, ids: tuple[int, ...]) -> EncodedScores:
    checked = _checked_vector(vector, n, ids)
    return [
        [item, int(value) if available else None]
        for item, value, available in zip(ids, checked.values, checked.available, strict=True)
    ]


def _vectors(scorer: Scorer, seed_sets: tuple[tuple[int, ...], ...]) -> tuple[ScoreVector, ...]:
    if not seed_sets:
        return ()
    rows = (
        scorer.score_many(seed_sets)
        if isinstance(scorer, BatchScorer)
        else tuple(scorer.score(seeds) for seeds in seed_sets)
    )
    if type(rows) is not tuple or len(rows) != len(seed_sets):
        fail("research_prediction_invalid")
    return rows


def _score_stream(
    prepared_dir: Path,
    *,
    inputs: ScoringInputs,
    model: PredictionModelMetadata,
    scorer: Scorer,
    writer: GzipOneWriter | BlockScoreWriter,
) -> tuple[StreamDescriptor | BlockScoreDescriptor, int, int, int, int]:
    prep, _ = read_json(prepared_dir / "prepare-manifest.json")
    if not isinstance(prep, dict) or not isinstance(prep.get("files"), dict):
        fail("research_prediction_invalid")
    query_desc = cast("StreamDescriptor", prep["files"]["development/queries.jsonl.gz"])
    ids = inputs.mal_ids
    id_set = set(ids)
    protocol = load_protocol()
    popularity = [[item, inputs.train_stats["items"][str(item)]["positive_users"]] for item in ids]
    seen: set[str] = set()
    source = stream_jsonl(prepared_dir / "development/queries.jsonl.gz", query_desc)
    validated = (
        (cast("PredictionQuery", query), _query(query, id_set, seen, protocol)) for query in source
    )
    scoring_ns = writing_ns = query_count = score_count = 0
    while block := tuple(islice(validated, 128)):
        wanted = tuple(seeds for _, seeds in block if seeds and model.variant_id != "popularity")
        tick = perf_counter_ns()
        scored = iter(_vectors(scorer, wanted))
        scoring_ns += perf_counter_ns() - tick
        for query, seeds in block:
            tick = perf_counter_ns()
            if isinstance(writer, BlockScoreWriter):
                vector = (
                    None
                    if not seeds or model.variant_id == "popularity"
                    else _checked_vector(next(scored), len(seeds), ids)
                )
                writer.write_query(query["query_id"], len(seeds), vector)
            else:
                scores = (
                    popularity
                    if not seeds or model.variant_id == "popularity"
                    else _encode_vector(next(scored), len(seeds), ids)
                )
                writer.write_json({"query_id": query["query_id"], "scores": scores})
            writing_ns += perf_counter_ns() - tick
            query_count += 1
            score_count += len(ids)
    scores_desc = writer.close()
    return scores_desc, query_count, score_count, scoring_ns, writing_ns


def prediction_dimensions(
    prepared_dir: Path, query_desc: StreamDescriptor, ids: tuple[int, ...]
) -> ScoreDimensions:
    protocol = load_protocol()
    seen: set[str] = set()
    id_set = set(ids)
    queries = zero = 0
    for value in stream_jsonl(prepared_dir / "development/queries.jsonl.gz", query_desc):
        seeds = _query(value, id_set, seen, protocol)
        queries += 1
        zero += not bool(seeds)
    return ScoreDimensions(len(ids), queries, zero)


def _measurement_target(
    prepared_dir: Path, output_dir: Path, measurement_output: Path | None
) -> None:
    if measurement_output is None:
        return
    safe_path(measurement_output)
    separate(prepared_dir, output_dir, measurement_output)
    parent = measurement_output.parent
    if (
        measurement_output.exists()
        or not parent.is_dir()
        or parent.stat().st_uid != os.getuid()
        or stat.S_IMODE(parent.stat().st_mode) != 0o700
    ):
        fail("research_unsafe_path")


def _write_validation_scores(
    prepared_dir: Path,
    output_dir: Path,
    *,
    inputs: ScoringInputs,
    model: PredictionModelMetadata,
    scorer: Scorer,
    measurements: dict[str, object] | None,
    measurement_output: Path | None,
    storage_guard: Callable[[int], None] | None,
    start: int,
    info: dict[str, object],
    score_format: Literal["binary-v1", "jsonl-v1"],
) -> Path:
    stage: Path | None = None
    writer: GzipOneWriter | BlockScoreWriter | None = None
    published = False
    query_count = score_count = 0
    try:
        prep, _ = read_json(prepared_dir / "prepare-manifest.json")
        if not isinstance(prep, dict) or not isinstance(prep.get("files"), dict):
            fail("research_prediction_invalid")
        query_desc = cast("StreamDescriptor", prep["files"]["development/queries.jsonl.gz"])
        manifest: dict[str, object] = {
            "schema_version": 2 if score_format == "binary-v1" else 1,
            "kind": "arb-ml-predictions",
            "prepared_identity": inputs.prepared_identity,
            "query_descriptor": query_desc,
            "catalog_descriptor": inputs.catalog_descriptor,
            "split": "validation",
            "model_id": model.variant_id,
            "config_sha256": model.config_sha256,
            "fit_identity": inputs.fit_identity,
            "train_interactions": inputs.train_interactions_descriptor,
            "train_stats": inputs.train_stats_descriptor,
            "code_revision": model.code_revision,
            "zero_seed_policy": "popularity-train-positive-user-counts",
        }
        if score_format == "binary-v1":
            manifest["score_format"] = FORMAT
            dimensions = prediction_dimensions(prepared_dir, query_desc, inputs.mal_ids)
            popularity_values = tuple(
                inputs.train_stats["items"][str(item)]["positive_users"] for item in inputs.mal_ids
            )
        stage = open_stage(output_dir)
        if score_format == "binary-v1":
            writer = BlockScoreWriter(
                stage / "scores.bin",
                catalog_ids=inputs.mal_ids,
                dimensions=dimensions,
                popularity=popularity_values,
                popularity_arm=model.variant_id == "popularity",
                binding_sha256=sha(canonical_bytes(manifest)),
                storage_guard=storage_guard,
            )
        else:
            writer = GzipOneWriter(stage / "scores.jsonl.gz", storage_guard)
        scores_desc, query_count, score_count, scoring_ns, writing_ns = _score_stream(
            prepared_dir, inputs=inputs, model=model, scorer=scorer, writer=writer
        )
        writer = None
        manifest["scores"] = scores_desc
        manifest_raw = canonical_bytes(manifest)
        if score_format == "binary-v1" and len(manifest_raw) > MAX_MANIFEST:
            fail("research_resource_limit")
        if storage_guard is not None:
            storage_guard(len(manifest_raw))
        write_json(stage / "prediction-manifest.json", manifest)
        info.update(
            {
                "queries": query_count,
                "score_pairs": score_count,
                "synchronized_scoring_seconds": scoring_ns / 1e9,
                "encoding_write_seconds"
                if score_format == "binary-v1"
                else "encoding_gzip_write_seconds": writing_ns / 1e9,
                "staging_wall_seconds": (perf_counter_ns() - start) / 1e9,
            }
        )
        publish(stage, output_dir)
        published = True
        info.update(
            {"status": "published", "publication_wall_seconds": (perf_counter_ns() - start) / 1e9}
        )
        if measurements is not None:
            measurements.update(info)
        if measurement_output is not None:
            write_json(measurement_output, info)
        return output_dir / "prediction-manifest.json"
    except BaseException:
        if writer is not None:
            writer.abort()
        if stage is not None:
            cleanup(stage)
        failure = {
            **info,
            "status": "published_measurement_failed" if published else "failed",
            "queries": query_count,
            "score_pairs": score_count,
            "elapsed_seconds": (perf_counter_ns() - start) / 1e9,
        }
        if measurements is not None:
            measurements.clear()
            measurements.update(failure)
        if measurement_output is not None and not measurement_output.exists():
            write_json(measurement_output, failure)
        raise


def write_validation_scores(
    prepared_dir: Path,
    output_dir: Path,
    *,
    inputs: ScoringInputs,
    model: PredictionModelMetadata,
    scorer: Scorer,
    measurements: dict[str, object] | None = None,
    measurement_output: Path | None = None,
    storage_guard: Callable[[int], None] | None = None,
    score_format: Literal["binary-v1", "jsonl-v1"] = "binary-v1",
) -> Path:
    """Publish one verified scorer's full validation stream with the frozen encoder."""
    separate(prepared_dir, output_dir)
    _measurement_target(prepared_dir, output_dir, measurement_output)
    if measurements is not None and (type(measurements) is not dict or measurements):
        fail("research_prediction_invalid")
    if model.variant_id == "" or len(model.config_sha256) != 64:
        fail("research_prediction_invalid")
    if score_format not in ("binary-v1", "jsonl-v1"):
        fail("research_prediction_invalid")
    start = perf_counter_ns()
    return _write_validation_scores(
        prepared_dir,
        output_dir,
        inputs=inputs,
        model=model,
        scorer=scorer,
        measurements=measurements,
        measurement_output=measurement_output,
        storage_guard=storage_guard,
        start=start,
        info={"backend": "cpu"},
        score_format=score_format,
    )


def predict_validation(
    prepared_dir: str | Path,
    model_dir: str | Path,
    output_dir: str | Path,
    *,
    expected_prepared_identity: str,
    backend: str = "cpu",
    measurements: dict[str, object] | None = None,
    measurement_output: str | Path | None = None,
    storage_guard: Callable[[int], None] | None = None,
    score_format: Literal["binary-v1", "jsonl-v1"] = "binary-v1",
) -> Path:
    require_cpu_backend(backend)
    if score_format not in ("binary-v1", "jsonl-v1"):
        fail("research_prediction_invalid")
    prepared_dir, model_dir, output_dir = map(Path, (prepared_dir, model_dir, output_dir))
    separate(prepared_dir, model_dir, output_dir)
    measurement_path = Path(measurement_output) if measurement_output is not None else None
    _measurement_target(prepared_dir, output_dir, measurement_path)
    if measurement_path is not None:
        separate(model_dir, measurement_path)
    if measurements is not None and (type(measurements) is not dict or measurements):
        fail("research_prediction_invalid")
    start = perf_counter_ns()
    scorer: Scorer | None = None
    entered_writer = False
    info: dict[str, object] = {"backend": backend}
    try:
        info["session_acquire_seconds"] = (perf_counter_ns() - start) / 1e9
        tick = perf_counter_ns()
        inputs = read_scoring_inputs(
            prepared_dir, expected_prepared_identity=expected_prepared_identity
        )
        info["input_validation_seconds"] = (perf_counter_ns() - tick) / 1e9
        tick = perf_counter_ns()
        model, scorer = load_model(model_dir, inputs=inputs, backend=backend)
        info["model_validation_upload_seconds"] = (perf_counter_ns() - tick) / 1e9
        info["input_model_init_upload_seconds"] = (perf_counter_ns() - start) / 1e9
        metadata = PredictionModelMetadata(
            model["variant_id"], sha(canonical_bytes(model)), model["producer"]["code_revision"]
        )
        entered_writer = True
        return _write_validation_scores(
            prepared_dir,
            output_dir,
            inputs=inputs,
            model=metadata,
            scorer=scorer,
            measurements=measurements,
            measurement_output=measurement_path,
            storage_guard=storage_guard,
            start=start,
            info=info,
            score_format=score_format,
        )
    except BaseException:
        if not entered_writer:
            failure = {
                **info,
                "status": "failed",
                "queries": 0,
                "score_pairs": 0,
                "elapsed_seconds": (perf_counter_ns() - start) / 1e9,
            }
            if measurements is not None:
                measurements.clear()
                measurements.update(failure)
            if measurement_path is not None and not measurement_path.exists():
                write_json(measurement_path, failure)
        raise
    finally:
        if scorer is not None and isinstance(scorer, CloseableScorer):
            scorer.close()
