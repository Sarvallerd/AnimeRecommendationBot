"""Pinned three-method comparison and fresh-process measurements for ARB-018."""

import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import subprocess
import sys
import tempfile
import time
from importlib import resources
from pathlib import Path

from .assessment import canonical, make_packet, make_template, sha
from .legacy import (ANIME_COLUMNS, SYNOPSIS_COLUMNS, FEATURES, KMEANS, RETAINED_SHA256,
                     build_legacy_vectors, cluster_legacy, legacy_neighbors, legacy_tokens,
                     prepare_legacy_rows, retained_ids_sha)


class EvaluationError(ValueError):
    """An input, worker, or deterministic comparison check failed."""


METHODS = ("glove-selected", "genre-jaccard", "legacy-repaired")
VERSIONS = {"glove-selected": "arb009-v1", "genre-jaccard": "1",
            "legacy-repaired": "arb018-legacy-v1"}
PARAMETERS = {
    "glove-selected": {"k": 5, "positive_only": True, "reduction": "numpy.einsum_ik_jk_ij_optimize_false",
                       "tie_break": "numeric_MAL_ID_ascending", "universe": "normalized_catalog"},
    "genre-jaccard": {"k": 5, "positive_only": True,
                      "tie_break": "numeric_MAL_ID_ascending", "universe": "normalized_catalog"},
    "legacy-repaired": {"k": 5, "top_cosine": 99, "same_cluster": True,
                        "score_year_sort": True, "feature_order": list(FEATURES),
                        "kmeans": KMEANS, "universe": "notebook_eligible"},
}
ORIGINALS = ("dev/notebooks/EDA + data preprocessing.ipynb",
             "dev/notebooks/create embeddings.ipynb", "src/app/recsys/utils.py")
CODE_FILES = ("legacy.py", "evaluation.py", "assessment.py", "assessment_prompt_v1.md",
              "cli.py", "quality.py", "build.py", "normalize.py", "sources.py", "bundle.py", "export.py")


def _id_sha(ids):
    return sha("".join(f"{i}\n" for i in ids).encode())


def _load_pins(bundle_dir, normalization_report, build_report, anime_csv, synopsis_csv,
               glove, lockfile):
    """Validate every pinned public input before launching measured workers."""
    from .build import _read_canonical
    from .bundle import check_bundle, read_canonical
    from .export import _validate_report
    from .normalize import _read_csv
    from .quality import (_load_json, _validate_catalog, _validate_query_set,
                          _validate_spec_metadata)
    from .sources import load_registry

    registry, registry_sha = load_registry()
    descriptors = {d["id"]: d for d in registry["sources"]}
    identity, count = check_bundle(bundle_dir)
    catalog, catalog_raw = read_canonical(bundle_dir / "catalog.json")
    neighbors, neighbors_raw = read_canonical(bundle_dir / "neighbors.json")
    manifest, manifest_raw = read_canonical(bundle_dir / "manifest.json")
    norm, norm_raw = _read_canonical(normalization_report)
    build, build_raw = _read_canonical(build_report)
    _validate_report(build, catalog, neighbors,
                     {"catalog.json": catalog_raw, "neighbors.json": neighbors_raw},
                     registry, registry_sha, norm, norm_raw)
    if manifest.get("algorithm") != build["algorithm"]:
        raise EvaluationError("manifest algorithm and build report disagree")
    spec_raw = resources.files("recsys").joinpath("quality_queries_v1.json").read_bytes()
    spec = _load_json(spec_raw, "quality query specification")
    _validate_spec_metadata(spec)
    if sha(catalog_raw) != spec.get("catalog_sha256") or len(catalog["anime"]) != spec["catalog_count"]:
        raise EvaluationError("catalog does not match pinned quality query set")
    anime = _validate_catalog(catalog, spec["catalog_count"])
    queries = _validate_query_set(spec, anime)
    anime_rows = _read_csv(anime_csv, descriptors["mal_anime"], ANIME_COLUMNS)
    synopsis_rows = _read_csv(synopsis_csv, descriptors["mal_synopsis"], SYNOPSIS_COLUMNS)
    for name, rows in (("anime", anime_rows), ("synopsis", synopsis_rows)):
        observed = set()
        for row_number, row in rows:
            raw_id = row["MAL_ID"]
            if not raw_id.isascii() or not raw_id.isdecimal() or int(raw_id) < 1 or int(raw_id) > 2_147_483_647:
                raise EvaluationError(f"{name} CSV row {row_number}: malformed MAL_ID")
            mal_id = int(raw_id)
            if mal_id in observed:
                raise EvaluationError(f"{name} CSV row {row_number}: duplicate MAL_ID {mal_id}")
            observed.add(mal_id)
    if len(anime_rows) != 17562:
        raise EvaluationError("unexpected MAL anime source row count")
    # Verify GloVe bytes in each relevant worker as part of its measured preparation.
    if not glove.is_file() or glove.stat().st_size != descriptors["glove_300d"]["size"]:
        raise EvaluationError("GloVe missing or wrong size")
    lock_raw = lockfile.read_bytes()
    versions = {name: importlib.metadata.version(name) for name in
                ("anime-recommendation-recsys", "numpy", "pandas", "scikit-learn", "scipy", "joblib", "threadpoolctl")}
    if sys.version_info[:2] != (3, 12) or versions["numpy"] != "2.5.3" or versions["pandas"] != "3.0.6" or versions["scikit-learn"] != "1.9.1":
        raise EvaluationError("Python or pinned dependency version mismatch")
    repository_root = lockfile.resolve().parent.parent
    originals = {p: sha((repository_root / p).read_bytes()) for p in ORIGINALS}
    expected_originals = {ORIGINALS[0]: "a08391a7161e23e65b9da87dbdefb25e2ccbdabb836b86e1f44ab864617ae9d0",
                          ORIGINALS[1]: "c032878aa33fd9a9db0fae69a01f25a919d0c001e6221cfb15c2e196c77a7a96",
                          ORIGINALS[2]: "2faf2555c00dfc67626ea00a2d926324686b501707a05177047cc3881f6dddc6"}
    if originals != expected_originals:
        raise EvaluationError("original notebook or ranker SHA256 mismatch")
    code_dir = Path(__file__).parent
    return {"catalog": catalog, "neighbors": neighbors, "queries": queries,
            "spec": spec, "spec_sha": sha(spec_raw), "registry": registry,
            "provenance": {"registry_sha256": registry_sha,
                           "raw_sources": [{"id": d["id"], "size": d["size"], "sha256": d["sha256"]}
                                           for d in (descriptors["mal_anime"], descriptors["mal_synopsis"],
                                                     descriptors["glove_300d"])],
                           "bundle_identity": identity, "manifest_sha256": sha(manifest_raw),
                           "neighbors_sha256": sha(neighbors_raw),
                           "normalization_report_sha256": sha(norm_raw),
                           "build_report_sha256": sha(build_raw),
                           "code_files": {name: sha((code_dir / name).read_bytes()) for name in CODE_FILES},
                           "lockfile_sha256": sha(lock_raw), "versions": versions,
                           "legacy_originals": originals},
            "catalog_sha": sha(catalog_raw), "count": count}


def _selected_queries(anime, queries, glove, descriptor):
    import numpy as np
    from .build import _tokens, build_vectors, load_embeddings
    vocabulary = {token for entry in anime.values() for token in _tokens(entry)}
    embeddings, glove_info = load_embeddings(glove, vocabulary, descriptor)
    ids, vectors, diagnostics = build_vectors(anime, embeddings)
    del embeddings
    ready = np.any(vectors != 0, axis=1)
    id_array = np.asarray(ids, dtype=np.int64)
    position = {mal_id: index for index, mal_id in enumerate(ids)}
    rank_started = time.perf_counter_ns()
    results = []
    for query in queries:
        mal_id = query["mal_id"]
        index = position[mal_id]
        if not ready[index]:
            recommendations = []
            status = diagnostics["anime"][str(mal_id)]["vector_status"]
        else:
            scores = np.einsum("ik,jk->ij", vectors[index:index + 1], vectors,
                               optimize=False)[0]
            if not np.isfinite(scores).all() or np.any(scores > 1 + 1e-12):
                raise EvaluationError("selected cosine score invalid")
            np.minimum(scores, 1.0, out=scores)
            scores[index] = 0.0
            scores[~ready] = 0.0
            candidates = np.flatnonzero(scores > 0)
            if len(candidates) > 5:
                threshold = np.partition(scores[candidates], -5)[-5]
                candidates = candidates[scores[candidates] >= threshold]
            order = np.lexsort((id_array[candidates], -scores[candidates]))
            recommendations = [{"rank": rank, "mal_id": int(id_array[j]),
                                "score": float(scores[j])} for rank, j in enumerate(candidates[order[:5]], 1)]
            status = "ok" if recommendations else "no_positive_candidates"
        results.append({"query_mal_id": mal_id, "status": status, "reasons": [],
                        "recommendations": recommendations})
    return ids, results, glove_info, time.perf_counter_ns() - rank_started


def _baseline_queries(anime, queries):
    from .quality import genre_neighbors
    result = []
    for query in queries:
        mal_id = query["mal_id"]
        items = genre_neighbors(anime, mal_id)
        status = "empty_genres" if not anime[str(mal_id)]["genres"] else "ok" if items else "no_positive_candidates"
        result.append({"query_mal_id": mal_id, "status": status, "reasons": [],
                       "recommendations": [{"rank": rank, **item} for rank, item in enumerate(items, 1)]})
    return sorted(int(i) for i in anime), result


def _legacy_queries(anime_csv, synopsis_csv, registry, queries, glove):
    from .build import load_embeddings
    from .normalize import _read_csv
    descriptors = {d["id"]: d for d in registry["sources"]}
    anime_rows = _read_csv(anime_csv, descriptors["mal_anime"], ANIME_COLUMNS)
    synopsis_rows = _read_csv(synopsis_csv, descriptors["mal_synopsis"], SYNOPSIS_COLUMNS)
    dataset = prepare_legacy_rows(anime_rows, synopsis_rows)
    if dataset.counts["anime_input"] != 17562 or dataset.counts["metadata_retained"] != 12173 or dataset.counts["synopsis_join_retained"] != 10882 or retained_ids_sha(dataset) != RETAINED_SHA256:
        raise EvaluationError(f"legacy eligibility mismatch: {dataset.counts}, {retained_ids_sha(dataset)}")
    if 35102 in dataset.records or 39619 not in dataset.records:
        raise EvaluationError("legacy query sentinel mismatch")
    labels, clustering = cluster_legacy(dataset)
    vocabulary = {token for i, record in dataset.records.items()
                  for token in legacy_tokens(record, labels[i])}
    embeddings, glove_info = load_embeddings(glove, vocabulary, descriptors["glove_300d"])
    ids, vectors, vector_diagnostics = build_legacy_vectors(dataset, labels, embeddings)
    del embeddings
    rank_started = time.perf_counter_ns()
    results = [legacy_neighbors(dataset, ids, vectors, labels, q["mal_id"]) for q in queries]
    rank_ns = time.perf_counter_ns() - rank_started
    for row in results:
        if row["status"] == "zero_vector":
            row["status"] = vector_diagnostics["statuses"][row["query_mal_id"]]
    diagnostics = {"counts": dataset.counts, "retained_ids_sha256": retained_ids_sha(dataset),
                   "query_exclusions": {str(q["mal_id"]): dataset.exclusions.get(q["mal_id"], [])
                                        for q in queries if q["mal_id"] not in dataset.records},
                   "clustering": clustering,
                   "vectors": {k: v for k, v in vector_diagnostics.items() if k != "statuses"},
                   "glove": glove_info,
                   "repairs": ["stable_MAL_ID", "explicit_KMeans_profile", "fixed_feature_order",
                               "explicit_scalar_types", "self_excluded_by_ID", "numeric_ties",
                               "nonready_vectors_excluded", "cosine_clamp", "top99_before_cluster_no_refill"]}
    return ids, results, diagnostics, rank_ns


def _worker(input_path, output_path):
    import threadpoolctl
    started = time.perf_counter_ns()
    task = json.loads(Path(input_path).read_text())
    method = task["method"]
    paths = {k: Path(v) for k, v in task["paths"].items()}
    with threadpoolctl.threadpool_limits(limits=1):
        pins = _load_pins(paths["bundle_dir"], paths["normalization_report"],
                          paths["build_report"], paths["anime_csv"], paths["synopsis_csv"],
                          paths["glove"], paths["lockfile"])
        anime, queries = pins["catalog"]["anime"], pins["queries"]
        descriptor = {d["id"]: d for d in pins["registry"]["sources"]}["glove_300d"]
        # Loading vectors and features is preparation; query ranking starts after it.
        if method == "glove-selected":
            # The selected adapter is one pass to preserve the exact published reduction.
            ids, rows, glove_info, rank_ns = _selected_queries(anime, queries, paths["glove"],
                                                                descriptor)
            prepared = time.perf_counter_ns()
            extra = {"glove": glove_info}
        elif method == "genre-jaccard":
            ids = sorted(int(i) for i in anime)
            prepared = time.perf_counter_ns()
            rank_start = time.perf_counter_ns()
            _, rows = _baseline_queries(anime, queries)
            rank_ns = time.perf_counter_ns() - rank_start
            extra = {}
        elif method == "legacy-repaired":
            ids, rows, diagnostics, rank_ns = _legacy_queries(paths["anime_csv"], paths["synopsis_csv"],
                                                     pins["registry"], queries, paths["glove"])
            prepared = time.perf_counter_ns()
            extra = {"legacy_diagnostics": diagnostics}
        else:
            raise EvaluationError("unknown worker method")
        prepare_ns = prepared - started - rank_ns
        result = {"method": method, "universe_count": len(ids),
                  "universe_ids_sha256": _id_sha(ids), "queries": rows, "extra": extra}
        completed = time.perf_counter_ns()
        pools = threadpoolctl.threadpool_info()
        if any(pool.get("num_threads") != 1 for pool in pools):
            raise EvaluationError("native threadpool limit was not effective")
        output = {"result": result, "prepare_wall_ns": prepare_ns,
                  "rank20_wall_ns": rank_ns,
                  "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                  "effective_threadpools": pools,
                  "worker_wall_ns": completed - started}
    Path(output_path).write_bytes(canonical(output))


def compare(bundle_dir, normalization_report, build_report, anime_csv, synopsis_csv,
            glove, lockfile, code_revision, output_dir, *, repetitions=3):
    if type(repetitions) is not int or not 1 <= repetitions <= 10:
        raise EvaluationError("repetitions must be 1..10")
    if not isinstance(code_revision, str) or len(code_revision) != 40 or any(c not in "0123456789abcdef" for c in code_revision):
        raise EvaluationError("code revision must be a lowercase commit SHA")
    paths = {k: Path(v).resolve() for k, v in locals().items()
             if k in ("bundle_dir", "normalization_report", "build_report", "anime_csv", "synopsis_csv", "glove", "lockfile")}
    output_dir = Path(output_dir)
    if output_dir.is_symlink() or (output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir()))):
        raise EvaluationError("output directory must be new or empty")
    pins = _load_pins(**paths)
    pins["provenance"]["code_revision"] = code_revision
    output_dir.mkdir(parents=True, exist_ok=True)
    schedule, workers, first = [], [], {}
    try:
        for repetition in range(1, repetitions + 1):
            order = [METHODS[(j + repetition - 1) % len(METHODS)] for j in range(len(METHODS))]
            schedule.append({"repetition": repetition, "order": order})
            for method in order:
                with tempfile.NamedTemporaryFile(mode="wb", dir=output_dir, prefix=".worker-input-", delete=False) as stream:
                    input_path = Path(stream.name)
                    stream.write(canonical({"method": method, "paths": {k: str(v) for k, v in paths.items()}}))
                with tempfile.NamedTemporaryFile(mode="wb", dir=output_dir, prefix=".worker-output-", delete=False) as stream:
                    output_path = Path(stream.name)
                env = os.environ.copy()
                for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
                    env[key] = "1"
                start = time.perf_counter_ns()
                process = subprocess.run([sys.executable, "-m", "recsys.evaluation", "worker",
                                          str(input_path), str(output_path)], env=env,
                                         capture_output=True, text=True, check=False)
                elapsed = time.perf_counter_ns() - start
                input_path.unlink(missing_ok=True)
                if process.returncode:
                    output_path.unlink(missing_ok=True)
                    raise EvaluationError(f"{method} repetition {repetition} worker failed: {process.stderr[-3000:]}")
                output = json.loads(output_path.read_text())
                output_path.unlink(missing_ok=True)
                result = output["result"]
                if method == "glove-selected":
                    for row in result["queries"]:
                        expected = pins["neighbors"]["neighbors"][str(row["query_mal_id"])]
                        actual = [{"mal_id": rec["mal_id"], "similarity": rec["score"]}
                                  for rec in row["recommendations"]]
                        if actual != expected:
                            raise EvaluationError(f"selected query {row['query_mal_id']} differs from published bundle")
                digest = sha(canonical(result))
                if method in first and result != first[method]:
                    raise EvaluationError(f"{method} repetition output mismatch")
                first.setdefault(method, result)
                workers.append({"method": method, "repetition": repetition,
                                "exit_code": process.returncode, "output_sha256": digest,
                                "end_to_end_wall_ns": elapsed,
                                "prepare_wall_ns": output["prepare_wall_ns"],
                                "rank20_wall_ns": output["rank20_wall_ns"],
                                "peak_rss_bytes": output["peak_rss_bytes"],
                                "effective_threadpools": output["effective_threadpools"]})
        legacy_diagnostics = first["legacy-repaired"]["extra"]["legacy_diagnostics"]
        methods = [{"id": method, "version": VERSIONS[method],
                    "parameters": PARAMETERS[method], "parameters_sha256": sha(canonical(PARAMETERS[method])),
                    "universe_count": first[method]["universe_count"],
                    "universe_ids_sha256": first[method]["universe_ids_sha256"],
                    "queries": first[method]["queries"]} for method in METHODS]
        comparison = {"schema_version": 1, "protocol_version": "arb018-v1",
                      "query_set_version": pins["spec"]["query_set_version"],
                      "query_set_sha256": pins["spec_sha"], "catalog_sha256": pins["catalog_sha"],
                      "provenance": pins["provenance"], "methods": methods,
                      "legacy_diagnostics_sha256": sha(canonical(legacy_diagnostics))}
        comparison_sha = sha(canonical(comparison))
        measurements = {"schema_version": 1, "protocol_version": "arb018-v1",
                        "comparison_sha256": comparison_sha,
                        "scope": "fresh-process-prepare-and-rank20-v1",
                        "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                                    "executable": sys.executable},
                        "thread_environment": {k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")},
                        "schedule": schedule, "workers": workers}
        packet = make_packet(comparison, pins["catalog"])
        template = make_template(packet)
        from .quality import _write_atomic
        for name, obj in (("legacy-diagnostics.json", legacy_diagnostics),
                          ("measurements.json", measurements),
                          ("assessment-packet.json", packet),
                          ("assessment-template.json", template)):
            _write_atomic(output_dir / name, canonical(obj))
        _write_atomic(output_dir / "comparison.json", canonical(comparison))
        return output_dir / "comparison.json"
    except Exception:
        for path in output_dir.glob(".worker-*"):
            path.unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "worker":
        _worker(sys.argv[2], sys.argv[3])
    else:
        raise SystemExit("internal worker invocation only")


def _validate_comparison(comparison, catalog, comparison_raw):
    import math
    from .quality import _load_json, _validate_catalog, _validate_query_set, _validate_spec_metadata
    from .bundle import check_catalog
    if not isinstance(comparison, dict) or set(comparison) != {"schema_version", "protocol_version", "query_set_version", "query_set_sha256", "catalog_sha256", "provenance", "methods", "legacy_diagnostics_sha256"}:
        raise EvaluationError("invalid comparison fields")
    if type(comparison["schema_version"]) is not int or comparison["schema_version"] != 1 or comparison["protocol_version"] != "arb018-v1":
        raise EvaluationError("unsupported comparison protocol")
    spec_raw = resources.files("recsys").joinpath("quality_queries_v1.json").read_bytes()
    spec = _load_json(spec_raw, "packaged query specification")
    _validate_spec_metadata(spec)
    if comparison["query_set_sha256"] != sha(spec_raw) or comparison["query_set_version"] != spec["query_set_version"]:
        raise EvaluationError("query set mismatch")
    check_catalog(catalog)
    anime = _validate_catalog(catalog, spec["catalog_count"])
    queries = _validate_query_set(spec, anime)
    if comparison["catalog_sha256"] != spec["catalog_sha256"]:
        raise EvaluationError("comparison catalog pin mismatch")
    methods = comparison["methods"]
    if not isinstance(methods, list) or len(methods) != 3 or any(not isinstance(m, dict) for m in methods) or [m.get("id") for m in methods] != list(METHODS):
        raise EvaluationError("comparison method order mismatch")
    for method in methods:
        ident = method["id"]
        if set(method) != {"id", "version", "parameters", "parameters_sha256", "universe_count", "universe_ids_sha256", "queries"}:
            raise EvaluationError("invalid comparison method fields")
        if method["version"] != VERSIONS[ident] or method["parameters"] != PARAMETERS[ident] or method["parameters_sha256"] != sha(canonical(PARAMETERS[ident])):
            raise EvaluationError("unsupported method parameters")
        if type(method["universe_count"]) is not int or method["universe_count"] < 1 or not isinstance(method["universe_ids_sha256"], str) or len(method["universe_ids_sha256"]) != 64:
            raise EvaluationError("invalid method universe")
        if ident != "legacy-repaired" and (method["universe_count"] != len(anime) or method["universe_ids_sha256"] != _id_sha(sorted(int(i) for i in anime))):
            raise EvaluationError("normalized method universe mismatch")
        if ident == "legacy-repaired" and (method["universe_count"] != 10882 or method["universe_ids_sha256"] != RETAINED_SHA256):
            raise EvaluationError("legacy method universe mismatch")
        rows = method["queries"]
        if not isinstance(rows, list) or len(rows) != 20:
            raise EvaluationError("comparison needs 20 queries per method")
        for row, query in zip(rows, queries):
            if not isinstance(row, dict) or set(row) != {"query_mal_id", "status", "reasons", "recommendations"} or type(row["query_mal_id"]) is not int or row["query_mal_id"] != query["mal_id"]:
                raise EvaluationError("invalid comparison query")
            allowed = ({"ok", "no_tokens", "oov_only", "zero_vector", "no_positive_candidates"}
                       if ident == "glove-selected" else
                       {"ok", "empty_genres", "no_positive_candidates"} if ident == "genre-jaccard" else
                       {"ok", "legacy_exclusion", "no_tokens", "oov_only", "zero_vector", "no_candidates_after_cluster"})
            if row["status"] not in allowed or not isinstance(row["reasons"], list) or any(not isinstance(r, str) for r in row["reasons"]):
                raise EvaluationError("invalid comparison status or reasons")
            recommendations = row["recommendations"]
            if not isinstance(recommendations, list) or len(recommendations) > 5 or (row["status"] == "ok") != bool(recommendations):
                raise EvaluationError("invalid comparison shortlist length or status")
            seen = set()
            for rank, rec in enumerate(recommendations, 1):
                if (not isinstance(rec, dict) or set(rec) != {"rank", "mal_id", "score"}
                        or type(rec["rank"]) is not int or rec["rank"] != rank
                        or type(rec["mal_id"]) is not int or rec["mal_id"] == query["mal_id"]
                        or str(rec["mal_id"]) not in anime or rec["mal_id"] in seen
                        or type(rec["score"]) not in (int, float) or not math.isfinite(rec["score"])
                        or rec["score"] < -1 - 1e-12 or rec["score"] > 1 + 1e-12
                        or (ident != "legacy-repaired" and rec["score"] <= 0)):
                    raise EvaluationError("invalid comparison recommendation")
                seen.add(rec["mal_id"])
    return queries


def _performance(measurements, comparison_raw):
    import statistics
    if (not isinstance(measurements, dict) or measurements.get("schema_version") != 1
            or measurements.get("protocol_version") != "arb018-v1"
            or measurements.get("comparison_sha256") != sha(comparison_raw)
            or measurements.get("scope") != "fresh-process-prepare-and-rank20-v1"):
        raise EvaluationError("measurements do not match comparison or scope")
    workers = measurements.get("workers")
    if not isinstance(workers, list) or len(workers) % 3 or not workers:
        raise EvaluationError("invalid measurement worker count")
    schedule = measurements.get("schedule")
    repetitions = len(workers) // 3
    if (not isinstance(schedule, list) or len(schedule) != repetitions
            or schedule != [{"repetition": i,
                             "order": [METHODS[(j + i - 1) % 3] for j in range(3)]}
                            for i in range(1, repetitions + 1)]
            or [(w.get("repetition"), w.get("method")) for w in workers]
            != [(entry["repetition"], method) for entry in schedule for method in entry["order"]]):
        raise EvaluationError("measurement schedule mismatch")
    grouped = {}
    for method in METHODS:
        samples = [w for w in workers if w.get("method") == method]
        if len(samples) * 3 != len(workers) or [w.get("repetition") for w in samples] != list(range(1, len(samples) + 1)):
            raise EvaluationError("measurement repetitions incomplete")
        digests = [row.get("output_sha256") for row in samples]
        if any(not isinstance(d, str) or len(d) != 64 or any(c not in "0123456789abcdef" for c in d) for d in digests) or len(set(digests)) != 1:
            raise EvaluationError("worker outputs differ across repetitions")
        for row in samples:
            if not isinstance(row.get("effective_threadpools"), list):
                raise EvaluationError("worker threadpool metadata missing")
            if row.get("exit_code") != 0 or any(type(row.get(key)) is not int or row[key] < 0 for key in
                                                 ("end_to_end_wall_ns", "prepare_wall_ns", "rank20_wall_ns", "peak_rss_bytes")):
                raise EvaluationError("invalid measurement units or worker result")
        grouped[method] = {key: {"samples": [row[key] for row in samples],
                                 "median": statistics.median(row[key] for row in samples),
                                 "min": min(row[key] for row in samples),
                                 "max": max(row[key] for row in samples)}
                           for key in ("end_to_end_wall_ns", "prepare_wall_ns", "rank20_wall_ns", "peak_rss_bytes")}
    return grouped


def summarize(comparison_path, catalog_path, measurements_path, assessment_paths, output_dir):
    from .assessment import validate_assessment, score_method, make_packet
    from .bundle import read_canonical
    if not assessment_paths:
        raise EvaluationError("at least one assessment required")
    comparison, comparison_raw = read_canonical(comparison_path)
    catalog, catalog_raw = read_canonical(catalog_path)
    measurements, measurements_raw = read_canonical(measurements_path)
    if sha(catalog_raw) != comparison.get("catalog_sha256"):
        raise EvaluationError("catalog SHA256 mismatch")
    _validate_comparison(comparison, catalog, comparison_raw)
    performance = _performance(measurements, comparison_raw)
    packet = make_packet(comparison, catalog)
    assessments, seen = [], set()
    for path in assessment_paths:
        assessment, raw = read_canonical(path)
        assessor_id = validate_assessment(assessment, packet)
        if assessor_id in seen:
            raise EvaluationError(f"duplicate assessor ID {assessor_id}")
        seen.add(assessor_id)
        assessments.append({"assessor": assessment["assessor"], "assessment_sha256": sha(raw),
                            "methods": [score_method(method, assessment["judgments"])
                                        for method in comparison["methods"]]})
    summary = {"schema_version": 1, "protocol_version": "arb018-v1",
               "comparison_sha256": sha(comparison_raw), "catalog_sha256": sha(catalog_raw),
               "measurements_sha256": sha(measurements_raw),
               "packet_sha256": sha(canonical(packet)), "performance": performance,
               "assessments": assessments}
    out = Path(output_dir)
    if out.is_symlink() or (out.exists() and (not out.is_dir() or any(out.iterdir()))):
        raise EvaluationError("summary output directory must be new or empty")
    out.mkdir(parents=True, exist_ok=True)
    from .quality import _write_atomic
    _write_atomic(out / "summary.json", canonical(summary))
    lines = ["# ARB-018 comparison", "", "The fixed, purposive set has 20 queries. Each method is ranked against its own stated universe.",
             "Returned results fill up to five slots per query; missing slots contribute zero. Unknown returned judgments stay unknown.",
             "Bounds are arithmetic uncertainty bounds, not confidence intervals. No general quality threshold was set.", "",
             f"Comparison SHA-256: `{summary['comparison_sha256']}`. Packet SHA-256: `{summary['packet_sha256']}`.", "",
             "## Performance", "", "Each sample starts a fresh process, prepares its method, and ranks the same 20 queries. The OS page cache was uncontrolled.", ""]
    for method, metrics in performance.items():
        lines.append(f"- {method}:")
        for key, unit in (("end_to_end_wall_ns", "ns"), ("prepare_wall_ns", "ns"),
                          ("rank20_wall_ns", "ns"), ("peak_rss_bytes", "bytes")):
            item = metrics[key]
            lines.append(f"  - {key} ({unit}): samples {item['samples']}; median {item['median']}; min {item['min']}; max {item['max']}.")
    lines += ["", "## Provenance and limits", "",
              f"Catalog SHA-256: `{summary['catalog_sha256']}`; measurements SHA-256: `{summary['measurements_sha256']}`.",
              "The raw MAL 2020 CSV and pinned GloVe bytes were verified. The original fitted clusters and vectors were unavailable, so the legacy method is a repaired comparator.",
              "MAL ID 35102 is excluded from the legacy universe because Aired is missing. MAL ID 39619 retains raw boilerplate synopsis text there, while its normalized catalog synopsis is null.",
              "Assessors use only supplied catalog evidence; missing synopsis text and LLM uncertainty can widen reported bounds. These data do not establish general recommendation quality.",
              "", "## Reproduction", "",
              "Run `recsys quality compare` with the pinned bundle, normalization and build reports, raw CSVs, GloVe file, lockfile and code revision, then `recsys quality summarize` with complete declared assessment files. Exact CLI forms are in `recsys/EVALUATION.md`.", ""]
    for assessment in assessments:
        assessor = assessment["assessor"]
        lines += ["", f"## Assessor {assessor['id']} ({assessor['kind']})", "",
                  f"Method version: `{assessor['method_version']}`. Model: `{assessor['model']}`. Prompt: `{assessor['prompt_version']}`.", ""]
        for method in assessment["methods"]:
            macro = method["macro"]
            lines += [f"### {method['method_id']}", "",
                      f"Returned {macro['returned']}/100; assessed {macro['assessed']}; coverage {macro['coverage']}; full-five queries {macro['full_five_queries']}.",
                      f"Mean grade@5 {macro['mean_grade_at_5']} bounds {macro['mean_grade_bounds']}; precision@5 {macro['precision_at_5']} bounds {macro['precision_bounds']}.",
                      f"Assessor-judged same-franchise@5 {macro['same_franchise_at_5']} bounds {macro['same_franchise_bounds']}; franchise coverage {macro['franchise_coverage']}.", "",
                      "| Query MAL_ID | Status | Returned | Assessed | Unknown | Mean grade@5 | Grade bounds | Precision@5 | Precision bounds | Franchise@5 | Franchise bounds |", "| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | --- | ---: | --- |"]
            for row in method["queries"]:
                lines.append(f"| {row['query_mal_id']} | {row['status']} | {row['returned']} | {row['assessed']} | {row['unknown_returned']} | {row['mean_grade_at_5']} | {row['mean_grade_bounds']} | {row['precision_at_5']} | {row['precision_bounds']} | {row['same_franchise_at_5']} | {row['same_franchise_bounds']} |")
            lines.append("")
    _write_atomic(out / "summary.md", ("\n".join(lines) + "\n").encode("utf-8"))
    return out / "summary.json"
