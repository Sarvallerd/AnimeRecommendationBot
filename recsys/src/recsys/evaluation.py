"""Pinned three-method comparison and fresh-process measurements for ARB-018."""

import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import re
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
DISTRIBUTIONS = ("anime-recommendation-recsys", "numpy", "pandas", "scikit-learn",
                 "scipy", "joblib", "threadpoolctl")
THREAD_ENVIRONMENT = {name: "1" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                                 "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                                                 "NUMEXPR_NUM_THREADS")}
ORIGINAL_HASHES = {
    ORIGINALS[0]: "a08391a7161e23e65b9da87dbdefb25e2ccbdabb836b86e1f44ab864617ae9d0",
    ORIGINALS[1]: "c032878aa33fd9a9db0fae69a01f25a919d0c001e6221cfb15c2e196c77a7a96",
    ORIGINALS[2]: "2faf2555c00dfc67626ea00a2d926324686b501707a05177047cc3881f6dddc6",
}


def _hex(value, length=64):
    return isinstance(value, str) and re.fullmatch(rf"[0-9a-f]{{{length}}}", value) is not None


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
    versions = {name: importlib.metadata.version(name) for name in DISTRIBUTIONS}
    if sys.version_info[:2] != (3, 12) or versions["numpy"] != "2.5.3" or versions["pandas"] != "3.0.6" or versions["scikit-learn"] != "1.9.1":
        raise EvaluationError("Python or pinned dependency version mismatch")
    repository_root = lockfile.resolve().parent.parent
    originals = {p: sha((repository_root / p).read_bytes()) for p in ORIGINALS}
    if originals != ORIGINAL_HASHES:
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
                env.update(THREAD_ENVIRONMENT)
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
                        "runtime": {"python": platform.python_version(), "implementation": platform.python_implementation(),
                                    "executable": sys.executable},
                        "machine": {"platform": platform.platform(), "machine": platform.machine(),
                                    "processor": platform.processor(), "cpu_count": os.cpu_count()},
                        "thread_environment": THREAD_ENVIRONMENT,
                        "workload": {"query_count": 20, "universe_counts":
                                     {method: first[method]["universe_count"] for method in METHODS}},
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


def _validate_provenance(value, catalog_sha):
    from .sources import load_registry
    fields = {"registry_sha256", "raw_sources", "bundle_identity", "manifest_sha256",
              "neighbors_sha256", "normalization_report_sha256", "build_report_sha256",
              "code_revision", "code_files", "lockfile_sha256", "versions", "legacy_originals"}
    if not isinstance(value, dict) or set(value) != fields:
        raise EvaluationError("invalid comparison provenance fields")
    registry, registry_sha = load_registry()
    sources = {item["id"]: item for item in registry["sources"]}
    expected_sources = [{"id": d["id"], "size": d["size"], "sha256": d["sha256"]}
                        for d in (sources["mal_anime"], sources["mal_synopsis"],
                                  sources["glove_300d"])]
    if value["registry_sha256"] != registry_sha or value["raw_sources"] != expected_sources:
        raise EvaluationError("comparison source provenance mismatch")
    for name in ("manifest_sha256", "neighbors_sha256", "normalization_report_sha256",
                 "build_report_sha256", "lockfile_sha256"):
        if not _hex(value[name]):
            raise EvaluationError(f"invalid comparison {name}")
    if value["bundle_identity"] != "sha256:" + value["manifest_sha256"]:
        raise EvaluationError("bundle identity does not match manifest SHA256")
    if not _hex(value["code_revision"], 40):
        raise EvaluationError("invalid comparison code revision")
    if value["legacy_originals"] != ORIGINAL_HASHES:
        raise EvaluationError("legacy source hashes mismatch")
    code_dir = Path(__file__).parent
    expected_code = {name: sha((code_dir / name).read_bytes()) for name in CODE_FILES}
    if value["code_files"] != expected_code:
        raise EvaluationError("comparison code file hashes mismatch")
    expected_versions = {name: importlib.metadata.version(name) for name in DISTRIBUTIONS}
    if value["versions"] != expected_versions:
        raise EvaluationError("comparison dependency versions mismatch")
    if not _hex(catalog_sha):
        raise EvaluationError("invalid catalog SHA256")


def _validate_bundle_links(provenance, catalog_path, catalog_raw):
    from .bundle import check_bundle_bytes, read_canonical
    bundle_dir = Path(catalog_path).parent
    manifest, manifest_raw = read_canonical(bundle_dir / "manifest.json")
    _, neighbors_raw = read_canonical(bundle_dir / "neighbors.json")
    identity, _ = check_bundle_bytes(catalog_raw, neighbors_raw, manifest_raw)
    if (identity != provenance["bundle_identity"]
            or sha(manifest_raw) != provenance["manifest_sha256"]
            or sha(neighbors_raw) != provenance["neighbors_sha256"]):
        raise EvaluationError("comparison bundle file hashes mismatch")
    sources = {item["name"]: item["sha256"] for item in manifest["sources"]}
    for name, key in (("build_report", "build_report_sha256"),
                      ("normalization_report", "normalization_report_sha256"),
                      ("source_registry", "registry_sha256")):
        if sources.get(name) != provenance[key]:
            raise EvaluationError(f"comparison {name} manifest link mismatch")
    raw_by_id = {item["id"]: item["sha256"] for item in provenance["raw_sources"]}
    for name in ("mal_anime", "mal_synopsis", "glove_300d"):
        if sources.get(name) != raw_by_id[name]:
            raise EvaluationError(f"comparison {name} source link mismatch")


def _validate_legacy_diagnostics(value, raw, expected_sha):
    import math
    from .sources import load_registry
    if not _hex(expected_sha) or sha(raw) != expected_sha:
        raise EvaluationError("legacy diagnostics SHA256 mismatch")
    if not isinstance(value, dict) or set(value) != {"counts", "retained_ids_sha256",
                                                      "query_exclusions", "clustering", "vectors",
                                                      "glove", "repairs"}:
        raise EvaluationError("invalid legacy diagnostics fields")
    counts = value["counts"]
    expected_missing = {"Aired": 309, "Duration": 555, "Episodes": 516,
                        "Genres": 63, "Rating": 688, "Score": 5141, "Type": 37}
    if (not isinstance(counts, dict) or set(counts) != {"anime_input", "metadata_retained",
                                                      "synopsis_join_retained", "metadata_excluded",
                                                      "synopsis_join_excluded", "metadata_missing_by_field"}
            or counts["anime_input"] != 17562 or counts["metadata_retained"] != 12173
            or counts["synopsis_join_retained"] != 10882
            or counts["metadata_excluded"] != 5389
            or counts["synopsis_join_excluded"] != 1291
            or counts["metadata_missing_by_field"] != expected_missing
            or value["retained_ids_sha256"] != RETAINED_SHA256
            or value["query_exclusions"] != {"35102": ["Aired"]}):
        raise EvaluationError("legacy diagnostics universe mismatch")
    clustering = value["clustering"]
    if (not isinstance(clustering, dict) or set(clustering) != {"feature_order", "kmeans_parameters",
                                                             "cluster_histogram", "inertia", "n_iter",
                                                             "assignment_sha256"}
            or clustering["feature_order"] != list(FEATURES)
            or clustering["kmeans_parameters"] != KMEANS
            or not _hex(clustering["assignment_sha256"])
            or type(clustering["n_iter"]) is not int or not 1 <= clustering["n_iter"] <= 300
            or type(clustering["inertia"]) not in (int, float)
            or not math.isfinite(clustering["inertia"]) or clustering["inertia"] < 0):
        raise EvaluationError("legacy diagnostics clustering mismatch")
    histogram = clustering["cluster_histogram"]
    if (not isinstance(histogram, dict) or set(histogram) != {str(i) for i in range(20)}
            or any(type(count) is not int or count < 1 for count in histogram.values())
            or sum(histogram.values()) != 10882):
        raise EvaluationError("legacy diagnostics cluster histogram mismatch")
    vectors = value["vectors"]
    if (not isinstance(vectors, dict) or set(vectors) != {"serialized_text_sha256", "vector_sha256",
                                                    "status_counts", "nonready_ids"}
            or not _hex(vectors["serialized_text_sha256"])
            or not _hex(vectors["vector_sha256"])):
        raise EvaluationError("legacy diagnostics vector hashes invalid")
    statuses = vectors["status_counts"]
    nonready = vectors["nonready_ids"]
    if (not isinstance(statuses, dict) or set(statuses) != {"no_tokens", "oov_only", "zero_vector", "ready"}
            or any(type(count) is not int or count < 0 for count in statuses.values())
            or sum(statuses.values()) != 10882
            or not isinstance(nonready, list) or len(nonready) != 10882 - statuses["ready"]
            or any(type(i) is not int or i < 1 for i in nonready)
            or nonready != sorted(set(nonready))):
        raise EvaluationError("legacy diagnostics vector status mismatch")
    registry, _ = load_registry()
    glove = next(source for source in registry["sources"] if source["id"] == "glove_300d")
    if value["glove"] != {"sha256": glove["sha256"], "size": glove["size"]}:
        raise EvaluationError("legacy diagnostics GloVe hash mismatch")
    repairs = ["stable_MAL_ID", "explicit_KMeans_profile", "fixed_feature_order",
               "explicit_scalar_types", "self_excluded_by_ID", "numeric_ties",
               "nonready_vectors_excluded", "cosine_clamp", "top99_before_cluster_no_refill"]
    if value["repairs"] != repairs:
        raise EvaluationError("legacy diagnostics repair profile mismatch")


def _validate_comparison(comparison, catalog, comparison_raw, diagnostics, diagnostics_raw):
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
    _validate_provenance(comparison["provenance"], comparison["catalog_sha256"])
    _validate_legacy_diagnostics(diagnostics, diagnostics_raw, comparison["legacy_diagnostics_sha256"])
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


def _performance(measurements, comparison_raw, comparison, diagnostics):
    import statistics
    fields = {"schema_version", "protocol_version", "comparison_sha256", "scope",
              "runtime", "machine", "thread_environment", "workload", "schedule", "workers"}
    if not isinstance(measurements, dict) or set(measurements) != fields:
        raise EvaluationError("invalid measurements fields")
    if (type(measurements["schema_version"]) is not int or measurements["schema_version"] != 1
            or measurements["protocol_version"] != "arb018-v1"
            or measurements["comparison_sha256"] != sha(comparison_raw)
            or measurements["scope"] != "fresh-process-prepare-and-rank20-v1"):
        raise EvaluationError("measurements do not match comparison or scope")
    runtime = measurements["runtime"]
    if (not isinstance(runtime, dict) or set(runtime) != {"python", "implementation", "executable"}
            or not isinstance(runtime["python"], str)
            or re.fullmatch(r"3\.12\.[0-9]+", runtime["python"]) is None
            or runtime["implementation"] != "CPython"
            or not isinstance(runtime["executable"], str)
            or not Path(runtime["executable"]).is_absolute()):
        raise EvaluationError("invalid measurement runtime metadata")
    machine = measurements["machine"]
    if (not isinstance(machine, dict) or set(machine) != {"platform", "machine", "processor", "cpu_count"}
            or any(not isinstance(machine[key], str) for key in ("platform", "machine", "processor"))
            or not machine["platform"] or not machine["machine"]
            or type(machine["cpu_count"]) is not int or machine["cpu_count"] < 1):
        raise EvaluationError("invalid measurement machine metadata")
    if measurements["thread_environment"] != THREAD_ENVIRONMENT:
        raise EvaluationError("measurement native thread environment mismatch")
    expected_workload = {"query_count": 20,
                         "universe_counts": {method["id"]: method["universe_count"]
                                             for method in comparison["methods"]}}
    if measurements["workload"] != expected_workload:
        raise EvaluationError("measurement workload mismatch")
    workers = measurements["workers"]
    if not isinstance(workers, list) or not workers or len(workers) % 3:
        raise EvaluationError("invalid measurement worker count")
    repetitions = len(workers) // 3
    if not 1 <= repetitions <= 10:
        raise EvaluationError("invalid measurement repetition count")
    schedule = measurements["schedule"]
    expected_schedule = [{"repetition": i,
                          "order": [METHODS[(j + i - 1) % 3] for j in range(3)]}
                         for i in range(1, repetitions + 1)]
    if (schedule != expected_schedule
            or any(not isinstance(worker, dict) for worker in workers)
            or [(worker.get("repetition"), worker.get("method")) for worker in workers]
            != [(entry["repetition"], method) for entry in expected_schedule
                for method in entry["order"]]):
        raise EvaluationError("measurement schedule mismatch")
    sources = {source["id"]: source for source in comparison["provenance"]["raw_sources"]}
    selected_glove = {"sha256": sources["glove_300d"]["sha256"],
                      "size": sources["glove_300d"]["size"]}
    expected_digests = {}
    for method in comparison["methods"]:
        name = method["id"]
        extra = ({"glove": selected_glove} if name == "glove-selected" else
                 {"legacy_diagnostics": diagnostics} if name == "legacy-repaired" else {})
        result = {"method": name, "universe_count": method["universe_count"],
                  "universe_ids_sha256": method["universe_ids_sha256"],
                  "queries": method["queries"], "extra": extra}
        expected_digests[name] = sha(canonical(result))
    grouped = {}
    worker_fields = {"method", "repetition", "exit_code", "output_sha256",
                     "end_to_end_wall_ns", "prepare_wall_ns", "rank20_wall_ns",
                     "peak_rss_bytes", "effective_threadpools"}
    for method in METHODS:
        samples = [worker for worker in workers if worker["method"] == method]
        if [worker["repetition"] for worker in samples] != list(range(1, repetitions + 1)):
            raise EvaluationError("measurement repetitions incomplete")
        for row in samples:
            if set(row) != worker_fields or type(row["exit_code"]) is not int or row["exit_code"] != 0:
                raise EvaluationError("invalid measurement worker fields or exit code")
            if row["output_sha256"] != expected_digests[method]:
                raise EvaluationError("worker output digest does not match comparison")
            for key in ("end_to_end_wall_ns", "prepare_wall_ns", "rank20_wall_ns", "peak_rss_bytes"):
                if type(row[key]) is not int or row[key] <= 0:
                    raise EvaluationError("invalid measurement units")
            if row["end_to_end_wall_ns"] < row["prepare_wall_ns"] + row["rank20_wall_ns"]:
                raise EvaluationError("worker timing exceeds parent elapsed time")
            pools = row["effective_threadpools"]
            if not isinstance(pools, list) or (method != "genre-jaccard" and not pools):
                raise EvaluationError("worker threadpool metadata missing")
            for pool in pools:
                if (not isinstance(pool, dict) or type(pool.get("num_threads")) is not int
                        or pool["num_threads"] != 1
                        or not isinstance(pool.get("user_api"), str) or not pool["user_api"]
                        or not isinstance(pool.get("internal_api"), str) or not pool["internal_api"]):
                    raise EvaluationError("worker native threadpool limit mismatch")
        grouped[method] = {key: {"samples": [row[key] for row in samples],
                                 "median": statistics.median(row[key] for row in samples),
                                 "min": min(row[key] for row in samples),
                                 "max": max(row[key] for row in samples)}
                           for key in ("end_to_end_wall_ns", "prepare_wall_ns", "rank20_wall_ns", "peak_rss_bytes")}
    return grouped


def _comparison_context(comparison):
    methods = []
    for method in comparison["methods"]:
        statuses = {}
        for row in method["queries"]:
            statuses[row["status"]] = statuses.get(row["status"], 0) + 1
        methods.append({"id": method["id"], "version": method["version"],
                        "parameters": method["parameters"],
                        "parameters_sha256": method["parameters_sha256"],
                        "universe_count": method["universe_count"],
                        "universe_ids_sha256": method["universe_ids_sha256"],
                        "returned_slots": sum(len(row["recommendations"]) for row in method["queries"]),
                        "unavailable_queries": sum(not row["recommendations"] for row in method["queries"]),
                        "short_queries": sum(0 < len(row["recommendations"]) < 5 for row in method["queries"]),
                        "status_counts": dict(sorted(statuses.items()))})
    return {"query_set_version": comparison["query_set_version"],
            "query_set_sha256": comparison["query_set_sha256"],
            "catalog_sha256": comparison["catalog_sha256"],
            "legacy_diagnostics_sha256": comparison["legacy_diagnostics_sha256"],
            "provenance": comparison["provenance"], "methods": methods}


def _measurement_context(measurements):
    return {"scope": measurements["scope"], "runtime": measurements["runtime"],
            "machine": measurements["machine"],
            "thread_environment": measurements["thread_environment"],
            "workload": measurements["workload"], "schedule": measurements["schedule"],
            "verification_scope": {
                "every_worker": "bundle, manifest, catalog, build and normalization reports, both MAL CSV byte hashes and parsed IDs, lock/dependency versions, original notebook hashes",
                "glove_selected_and_legacy": "full GloVe stream size and SHA256",
                "genre_jaccard_glove": "file size check only; GloVe contents are not read",
                "interpretation": "shared input validation is included in preparation time; genre Jaccard repeats CSV and report checks beyond its ranking needs",
            }}


def summarize(comparison_path, catalog_path, measurements_path, assessment_paths, output_dir):
    from .assessment import validate_assessment, score_method, make_packet
    from .bundle import read_canonical
    if not assessment_paths:
        raise EvaluationError("at least one assessment required")
    comparison, comparison_raw = read_canonical(comparison_path)
    catalog, catalog_raw = read_canonical(catalog_path)
    measurements, measurements_raw = read_canonical(measurements_path)
    diagnostics, diagnostics_raw = read_canonical(Path(comparison_path).with_name("legacy-diagnostics.json"))
    if sha(catalog_raw) != comparison.get("catalog_sha256"):
        raise EvaluationError("catalog SHA256 mismatch")
    _validate_comparison(comparison, catalog, comparison_raw, diagnostics, diagnostics_raw)
    _validate_bundle_links(comparison["provenance"], catalog_path, catalog_raw)
    performance = _performance(measurements, comparison_raw, comparison, diagnostics)
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
               "packet_sha256": sha(canonical(packet)),
               "comparison_context": _comparison_context(comparison),
               "measurement_context": _measurement_context(measurements),
               "performance": performance, "assessments": assessments}
    out = Path(output_dir)
    if out.is_symlink() or (out.exists() and (not out.is_dir() or any(out.iterdir()))):
        raise EvaluationError("summary output directory must be new or empty")
    out.mkdir(parents=True, exist_ok=True)
    from .quality import _write_atomic
    _write_atomic(out / "summary.json", canonical(summary))
    context = summary["comparison_context"]
    measure_context = summary["measurement_context"]
    provenance = context["provenance"]
    lines = ["# ARB-018 comparison", "",
             "The fixed, purposive set has 20 queries. Each method ranks against its stated universe.",
             "Returned results fill up to five slots per query; missing slots contribute zero. Unknown returned judgments stay unknown.",
             "Bounds are arithmetic uncertainty bounds, not confidence intervals. No general quality threshold was set.", "",
             "## Methods and source provenance", "",
             "| Method | Version | Universe | Universe ID SHA-256 | Parameter SHA-256 | Returned slots | Unavailable | Short | Status counts |",
             "| --- | --- | ---: | --- | --- | ---: | ---: | ---: | --- |"]
    for method in context["methods"]:
        lines.append(f"| {method['id']} | {method['version']} | {method['universe_count']} | `{method['universe_ids_sha256']}` | `{method['parameters_sha256']}` | {method['returned_slots']}/100 | {method['unavailable_queries']} | {method['short_queries']} | `{json.dumps(method['status_counts'], sort_keys=True)}` |")
    lines.append("")
    for method in context["methods"]:
        lines.append(f"- {method['id']} parameters: `{json.dumps(method['parameters'], sort_keys=True, separators=(',', ':'))}`.")
    lines += ["", f"Query set `{context['query_set_version']}` SHA-256: `{context['query_set_sha256']}`.",
              f"Catalog SHA-256: `{context['catalog_sha256']}`; legacy diagnostics SHA-256: `{context['legacy_diagnostics_sha256']}`.",
              f"Code revision: `{provenance['code_revision']}`; lockfile SHA-256: `{provenance['lockfile_sha256']}`.",
              f"Registry SHA-256: `{provenance['registry_sha256']}`; bundle identity: `{provenance['bundle_identity']}`.",
              f"Manifest SHA-256: `{provenance['manifest_sha256']}`; neighbors SHA-256: `{provenance['neighbors_sha256']}`.",
              f"Normalization report SHA-256: `{provenance['normalization_report_sha256']}`; build report SHA-256: `{provenance['build_report_sha256']}`.",
              f"Comparison SHA-256: `{summary['comparison_sha256']}`; measurements SHA-256: `{summary['measurements_sha256']}`; packet SHA-256: `{summary['packet_sha256']}`.", "",
              "Raw sources (size in bytes, SHA-256):", ""]
    for source in provenance["raw_sources"]:
        lines.append(f"- {source['id']}: {source['size']}; `{source['sha256']}`")
    lines += ["", "Code file SHA-256 hashes:", ""]
    for name, digest in sorted(provenance["code_files"].items()):
        lines.append(f"- {name}: `{digest}`")
    lines += ["", "## Measurement context", "",
              f"Scope: `{measure_context['scope']}`; workload: {measure_context['workload']['query_count']} queries, universe counts `{json.dumps(measure_context['workload']['universe_counts'], sort_keys=True)}`.",
              f"Machine: `{json.dumps(measure_context['machine'], sort_keys=True)}`.",
              f"Runtime: `{json.dumps(measure_context['runtime'], sort_keys=True)}`.",
              f"Native thread environment: `{json.dumps(measure_context['thread_environment'], sort_keys=True)}`.",
              "Each worker rechecks the bundle, catalog, manifest, build and normalization reports, and both MAL CSV hashes, headers and IDs during measured preparation. This shared validation cost is included even for genre Jaccard.",
              "Selected GloVe and repaired legacy workers stream and hash the complete GloVe file. Genre Jaccard checks only its file size; it does not read GloVe contents.",
              "Each sample starts a fresh process, prepares its method, and ranks the same 20 queries. The OS page cache, background host activity, and CPU scheduling were uncontrolled on this shared development machine; full production neighbor generation was outside scope.", "",
              "## Performance", ""]
    for method, metrics in performance.items():
        lines.append(f"- {method}:")
        for key, unit in (("end_to_end_wall_ns", "ns"), ("prepare_wall_ns", "ns"),
                          ("rank20_wall_ns", "ns"), ("peak_rss_bytes", "bytes")):
            item = metrics[key]
            lines.append(f"  - {key} ({unit}): samples {item['samples']}; median {item['median']}; min {item['min']}; max {item['max']}.")
        pool_samples = [worker["effective_threadpools"] for worker in measurements["workers"]
                        if worker["method"] == method]
        pool_counts = [[pool["num_threads"] for pool in pools] for pools in pool_samples]
        lines.append(f"  - Effective native threads per sample (each detected pool): {pool_counts}.")
    lines += ["", "## Provenance and limits", "",
              f"Catalog SHA-256: `{summary['catalog_sha256']}`; measurements SHA-256: `{summary['measurements_sha256']}`.",
              "The raw MAL 2020 CSVs were verified by every worker; GloVe bytes were fully verified by selected and legacy workers. The original fitted clusters and vectors were unavailable, so the legacy method is a repaired comparator.",
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
