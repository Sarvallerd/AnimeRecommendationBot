"""Validation-only full-catalog ML-02 prediction publication."""

from __future__ import annotations

import hashlib
from contextlib import nullcontext
from itertools import chain, islice
import os
import re
import statistics
import stat
import struct
import tempfile
import zlib
from pathlib import Path
from time import perf_counter_ns

from .bundle import canonical_bytes
from .research_artifacts import (MAX_LINE, MAX_STREAM, cleanup, open_stage, publish,
                                 read_json, safe_path, separate, stream_jsonl, write_json)
from .research_models import ScoreVector, _backend, load_model, read_scoring_inputs
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


def _vectors(scorer, seed_sets):
    if not seed_sets:
        return ()
    many = getattr(scorer, "score_many", None)
    rows = many(tuple(seed_sets)) if callable(many) else tuple(scorer.score(seeds)
                                                            for seeds in seed_sets)
    if type(rows) is not tuple or len(rows) != len(seed_sets):
        fail("research_prediction_invalid")
    return rows


def _rank(vector, encoded, seeds, ids):
    import numpy as np
    excluded = set(seeds)
    candidates = np.flatnonzero(vector.available &
                                np.asarray([item not in excluded for item in ids],
                                           dtype=np.bool_))
    values = np.asarray([encoded[index][1] for index in candidates], dtype=np.int64)
    order = np.lexsort((np.asarray(ids, dtype=np.int64)[candidates], -values))
    return candidates[order]


def _admit_gpu(first, model_dir, model, gpu_scorer, inputs, query_desc):
    """Compare complete candidate masks and rankings using the same validated payload."""
    import numpy as np
    variant = model["variant_id"]
    reference = _backend(variant).load(model_dir, model, inputs, backend="cpu")
    compared = differences = 0
    max_abs = 0.0
    try:
        seeds = tuple(group for _, group in first if group)
        cpu_rows = _vectors(reference, seeds)
        gpu_rows = _vectors(gpu_scorer, seeds)
        for group, cpu, gpu in zip(seeds, cpu_rows, gpu_rows, strict=True):
            cpu_encoded = _encode_vector(cpu, len(group), inputs.mal_ids)
            gpu_encoded = _encode_vector(gpu, len(group), inputs.mal_ids)
            if (cpu.supported_seeds != gpu.supported_seeds or
                    not np.array_equal(cpu.available, gpu.available) or
                    not np.isclose(cpu.values, gpu.values, atol=1e-12,
                                   rtol=1e-12).all() or
                    not np.array_equal(_rank(cpu, cpu_encoded, group, inputs.mal_ids),
                                       _rank(gpu, gpu_encoded, group, inputs.mal_ids))):
                fail("research_gpu_parity_failed")
            differences += sum(a[1] != b[1] for a, b in zip(cpu_encoded, gpu_encoded,
                                                             strict=True))
            max_abs = max(max_abs, float(np.max(np.abs(cpu.values - gpu.values))))
            compared += 1
    finally:
        close = getattr(reference, "close", None)
        if callable(close):
            close()
    if compared == 0:
        fail("research_gpu_parity_failed")
    return {"purpose": "parity-only-gpu-admission", "canonical_queries": len(first),
            "nonempty_queries": compared, "raw_atol": 1e-12, "raw_rtol": 1e-12,
            "raw_max_abs_difference": max_abs,
            "encoded_score_differences": differences,
            "complete_available_rankings_equal": True,
            "model_sha256": sha(canonical_bytes(model)),
            "query_sha256": query_desc["sha256"],
            "scoring_provenance": model["producer"]["execution"]["scoring"],
            "code_files_sha256": model["producer"]["code_files_sha256"]}


def predict_validation(prepared_dir, model_dir, output_dir, *, expected_prepared_identity,
                       backend="cpu", measurements=None, measurement_output=None):
    prepared_dir, model_dir, output_dir = map(Path, (prepared_dir, model_dir, output_dir))
    separate(prepared_dir, model_dir, output_dir)
    if measurement_output is not None:
        measurement_output = safe_path(Path(measurement_output))
        separate(prepared_dir, model_dir, output_dir, measurement_output)
        parent = measurement_output.parent
        if (measurement_output.exists() or not parent.is_dir() or
                parent.stat().st_uid != os.getuid() or
                stat.S_IMODE(parent.stat().st_mode) != 0o700):
            fail("research_unsafe_path")
    if backend not in ("cpu", "gpu") or type(backend) is not str:
        fail("research_backend_unsupported")
    if measurements is not None and (type(measurements) is not dict or measurements):
        fail("research_prediction_invalid")
    start = perf_counter_ns()
    stage = None
    writer = None
    scorer = None
    scoring_ns = writing_ns = query_count = score_count = 0
    info = {"backend": backend}
    try:
        if backend == "gpu":
            from .research_gpu import gpu_execution, gpu_memory_usage
            session = gpu_execution()
        else:
            session = nullcontext()
        with session:
            info["session_acquire_seconds"] = (perf_counter_ns() - start) / 1e9
            tick = perf_counter_ns()
            inputs = read_scoring_inputs(prepared_dir,
                                         expected_prepared_identity=expected_prepared_identity)
            info["input_validation_seconds"] = (perf_counter_ns() - tick) / 1e9
            tick = perf_counter_ns()
            model, scorer = load_model(model_dir, inputs=inputs, backend=backend)
            info["model_validation_upload_seconds"] = (perf_counter_ns() - tick) / 1e9
            info["input_model_init_upload_seconds"] = (perf_counter_ns() - start) / 1e9
            try:
                prep, _ = read_json(prepared_dir / "prepare-manifest.json")
                query_desc = prep["files"]["development/queries.jsonl.gz"]
                ids = inputs.mal_ids
                id_set = set(ids)
                protocol = load_protocol()
                popularity = [[item, inputs.train_stats["items"][str(item)]["positive_users"]]
                              for item in ids]
                seen = set()
                source = stream_jsonl(prepared_dir / "development/queries.jsonl.gz", query_desc)
                validated = ((query, _query(query, id_set, seen, protocol)) for query in source)
                first = tuple(islice(validated, 128)) if backend == "gpu" else ()
                if backend == "gpu":
                    admission_start = perf_counter_ns()
                    info["gpu_admission"] = _admit_gpu(first, model_dir, model, scorer,
                                                        inputs, query_desc)
                    info["gpu_admission_seconds"] = (perf_counter_ns() - admission_start) / 1e9
                stage = open_stage(output_dir)
                writer = GzipOneWriter(stage / "scores.jsonl.gz")
                rows = chain(first, validated)
                while block := tuple(islice(rows, 128)):
                    wanted = tuple(seeds for _, seeds in block
                                   if seeds and model["variant_id"] != "popularity")
                    tick = perf_counter_ns()
                    scored = iter(_vectors(scorer, wanted))
                    scoring_ns += perf_counter_ns() - tick
                    for query, seeds in block:
                        tick = perf_counter_ns()
                        scores = (popularity if not seeds or model["variant_id"] == "popularity"
                                  else _encode_vector(next(scored), len(seeds), ids))
                        writer.write_json({"query_id": query["query_id"], "scores": scores})
                        writing_ns += perf_counter_ns() - tick
                        query_count += 1
                        score_count += len(scores)
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
                if backend == "gpu":
                    info["dedicated_pool_session_including_admission"] = gpu_memory_usage()
                    info["memory_scope"] = "dedicated-pool-only-excludes-context-and-cublas"
            finally:
                if writer is not None:
                    writer.abort()
                    writer = None
                if scorer is not None:
                    close = getattr(scorer, "close", None)
                    if callable(close):
                        close()
        info.update({"status": "staged", "queries": query_count, "score_pairs": score_count,
                     "synchronized_scoring_seconds": scoring_ns / 1e9,
                     "encoding_gzip_write_seconds": writing_ns / 1e9,
                     "staging_wall_seconds": (perf_counter_ns() - start) / 1e9})
        if measurements is not None:
            measurements.update(info)
        if measurement_output is not None:
            write_json(measurement_output, info)
        publish(stage, output_dir)
        return output_dir / "prediction-manifest.json"
    except BaseException:
        if writer is not None:
            writer.abort()
        if stage is not None:
            cleanup(stage)
        if measurements is not None:
            measurements.clear()
            measurements.update({**info, "status": "failed", "queries": query_count,
                                 "score_pairs": score_count,
                                 "elapsed_seconds": (perf_counter_ns() - start) / 1e9})
        if measurement_output is not None and not measurement_output.exists():
            write_json(measurement_output, {**info, "status": "failed", "queries": query_count,
                                            "score_pairs": score_count,
                                            "elapsed_seconds": (perf_counter_ns() - start) / 1e9})
        raise


def gpu_preflight(prepared_dir, model_dir, output_dir, *, expected_prepared_identity):
    """Private, label-free GPU parity and comparable single/batch timing evidence."""
    from contextlib import ExitStack
    from .research_gpu import gpu_execution, gpu_memory_usage

    prepared_dir, model_dir, output_dir = map(Path, (prepared_dir, model_dir, output_dir))
    separate(prepared_dir, model_dir, output_dir)
    stage = None
    try:
        started = perf_counter_ns()
        with ExitStack() as stack:
            stack.enter_context(gpu_execution())
            session_acquire = (perf_counter_ns() - started) / 1e9
            inputs = read_scoring_inputs(prepared_dir,
                                         expected_prepared_identity=expected_prepared_identity)
            tick = perf_counter_ns()
            model, scorer = load_model(model_dir, inputs=inputs, backend="gpu")
            stack.callback(scorer.close)
            initial_upload = (perf_counter_ns() - tick) / 1e9
            prep, _ = read_json(prepared_dir / "prepare-manifest.json")
            descriptor = prep["files"]["development/queries.jsonl.gz"]
            seen = set()
            protocol = load_protocol()
            ids = inputs.mal_ids
            id_set = set(ids)
            source = stream_jsonl(prepared_dir / "development/queries.jsonl.gz", descriptor)
            validated = ((query, _query(query, id_set, seen, protocol)) for query in source)
            first = tuple(islice(validated, 128))
            admission = _admit_gpu(first, model_dir, model, scorer, inputs, descriptor)
            total = len(first)
            for _ in validated:
                total += 1
            sample = tuple(group for _, group in first if group)
            single = cpu_single = batch = None
            if sample:
                tick = perf_counter_ns()
                cpu_scorer = _backend(model["variant_id"]).load(model_dir, model, inputs,
                                                                 backend="cpu")
                cpu_close = getattr(cpu_scorer, "close", None)
                if callable(cpu_close):
                    stack.callback(cpu_close)
                cpu_init = (perf_counter_ns() - tick) / 1e9
                cpu_traces = []
                for _ in range(2):
                    durations = []
                    for group in sample:
                        tick = perf_counter_ns()
                        cpu_scorer.score(group)
                        durations.append((perf_counter_ns() - tick) / 1e6)
                    cpu_traces.append({"median_ms": statistics.median(durations),
                                       "p95_ms": sorted(durations)[min(len(durations) - 1,
                                                                          int(.95 * len(durations)))],
                                       "queries": len(durations)})
                cpu_single = {"fresh_scorer_init_seconds": cpu_init,
                              "first_cache_empty_trace": cpu_traces[0],
                              "warmed_repeat_trace": cpu_traces[1]}
                tick = perf_counter_ns()
                single_model, single_scorer = load_model(model_dir, inputs=inputs, backend="gpu")
                stack.callback(single_scorer.close)
                single_upload = (perf_counter_ns() - tick) / 1e9
                if sha(canonical_bytes(single_model)) != admission["model_sha256"]:
                    fail("research_backend_mismatch")
                traces = []
                for _ in range(2):
                    durations = []
                    for group in sample:
                        tick = perf_counter_ns()
                        single_scorer.score(group)
                        durations.append((perf_counter_ns() - tick) / 1e6)
                    traces.append({"median_ms": statistics.median(durations),
                                   "p95_ms": sorted(durations)[min(len(durations) - 1,
                                                                      int(.95 * len(durations)))],
                                   "queries": len(durations)})
                single = {"fresh_scorer_init_upload_seconds": single_upload,
                          "first_cache_empty_trace": traces[0],
                          "warmed_repeat_trace": traces[1]}
                tick = perf_counter_ns()
                batch_model, batch_scorer = load_model(model_dir, inputs=inputs, backend="gpu")
                stack.callback(batch_scorer.close)
                batch_upload = (perf_counter_ns() - tick) / 1e9
                if sha(canonical_bytes(batch_model)) != admission["model_sha256"]:
                    fail("research_backend_mismatch")
                traces = []
                for _ in range(2):
                    tick = perf_counter_ns()
                    vectors = _vectors(batch_scorer, sample)
                    seconds = (perf_counter_ns() - tick) / 1e9
                    traces.append({"wall_seconds": seconds,
                                   "queries_per_second": len(sample) / seconds,
                                   "queries": len(sample)})
                batch = {"fresh_scorer_init_upload_seconds": batch_upload,
                         "first_cache_empty_trace": traces[0],
                         "warmed_repeat_trace": traces[1]}
                with tempfile.TemporaryDirectory(prefix="arb-ml02-gpu-prefix-") as temporary:
                    tick = perf_counter_ns()
                    writer = GzipOneWriter(Path(temporary) / "scores.jsonl.gz")
                    try:
                        for (query, group), vector in zip((item for item in first if item[1]),
                                                         vectors, strict=True):
                            writer.write_json({"query_id": query["query_id"],
                                               "scores": _encode_vector(vector, len(group), ids)})
                        writer.close()
                    except BaseException:
                        writer.abort()
                        raise
                    prefix_seconds = (perf_counter_ns() - tick) / 1e9
            else:
                prefix_seconds = 0.0
            memory = gpu_memory_usage()
        report = {"schema_version": 1, "kind": "arb-ml02-gpu-preflight",
                  "status": "passed", "backend": "gpu", "admission": admission,
                  "validated_development_queries": total,
                  "session_acquire_seconds": session_acquire,
                  "initial_model_init_upload_seconds": initial_upload,
                  "cpu_single_query_latency": cpu_single,
                  "single_query_latency": single,
                  "score_many_batch_throughput": batch,
                  "sample_prefix_encode_gzip_seconds": prefix_seconds,
                  "dedicated_pool_session_including_admission_and_benchmark": memory,
                  "memory_scope": "dedicated-pool-only-excludes-context-and-cublas",
                  "timing_conditions": {"shared_device_session": True,
                                        "first_trace_scorer_cache_empty": True,
                                        "second_trace_warmed_repeat": True,
                                        "single_query_latency_is_not_batch_latency": True},
                  "diagnostic_wall_seconds": (perf_counter_ns() - started) / 1e9}
        stage = open_stage(output_dir)
        write_json(stage / "preflight.json", report)
        publish(stage, output_dir)
        return output_dir / "preflight.json"
    except BaseException:
        if stage is not None:
            cleanup(stage)
        raise
