"""Synthetic comparison behavior and measurement validation."""

import copy
import hashlib
import importlib.metadata
import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import cast
from unittest.mock import patch

from recsys.contracts.bundle import AnimeRecord, Catalog
from recsys.contracts.provenance import SOURCE_FILES, SOURCE_LAYOUT
from recsys.data.sources import load_registry
from recsys.historical.assessment import HistoricalComparison, HistoricalQuery, canonical, sha
from recsys.historical.comparison import (
    DISTRIBUTIONS,
    ORIGINAL_HASHES,
    ORIGINALS,
    THREAD_ENVIRONMENT,
    ComparisonArtifact,
    ComparisonMethod,
    ComparisonProvenance,
    EvaluationError,
    MeasurementArtifact,
    MeasurementWorker,
    RawSource,
    WorkerResult,
    _baseline_queries,
    _performance,
    _validate_bundle_links,
    _validate_legacy_diagnostics,
    _validate_provenance,
    _verify_originals,
    _worker,
    compare,
)
from recsys.historical.genre_baseline import GenreAnime, GenreQuery, genre_neighbors


class RootLockOriginalsTests(unittest.TestCase):
    def test_root_lock_checks_pinned_original_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock = root / "uv.lock"
            lock.write_bytes(b"synthetic lock\n")
            original_bytes = {
                name: f"synthetic original {index}\n".encode()
                for index, name in enumerate(ORIGINALS)
            }
            expected = {
                name: hashlib.sha256(raw).hexdigest() for name, raw in original_bytes.items()
            }
            for name, raw in original_bytes.items():
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(raw)
            with patch("recsys.historical.comparison.ORIGINAL_HASHES", expected):
                self.assertEqual(_verify_originals(lock, originals_root=root), expected)
                changed = root / ORIGINALS[0]
                changed.write_bytes(changed.read_bytes() + b"changed")
                with self.assertRaisesRegex(
                    EvaluationError, "original notebook or ranker SHA256 mismatch"
                ):
                    _verify_originals(lock, originals_root=root)
                changed.write_bytes(original_bytes[ORIGINALS[0]])
                changed.unlink()
                with self.assertRaisesRegex(EvaluationError, "historical originals missing"):
                    _verify_originals(lock, originals_root=root)


class EvaluationTests(unittest.TestCase):
    def test_jaccard_adapter_preserves_baseline_scores_and_order(self) -> None:
        anime: dict[str, AnimeRecord] = {
            key: {
                "title": key,
                "aliases": [],
                "genres": genres,
                "score": None,
                "year": None,
                "type": None,
                "episodes": None,
                "synopsis": None,
            }
            for key, genres in {
                "1": ["Action", "Drama"],
                "2": ["Drama"],
                "3": ["Action", "Drama"],
                "4": [],
            }.items()
        }
        queries: list[GenreQuery] = [
            {
                "mal_id": item,
                "title": str(item),
                "popularity_rank": item,
                "members": 1,
                "popularity_band": "test",
                "rationale": "synthetic baseline query",
            }
            for item in (1, 4)
        ]
        ids, rows = _baseline_queries(anime, queries)
        self.assertEqual(ids, [1, 2, 3, 4])
        self.assertEqual(
            [{"mal_id": x["mal_id"], "score": x["score"]} for x in rows[0]["recommendations"]],
            genre_neighbors({key: cast("GenreAnime", value) for key, value in anime.items()}, 1),
        )
        self.assertEqual(rows[1]["status"], "empty_genres")

    def test_selected_worker_uses_real_vectors_and_records_timing(self) -> None:
        def record(title: str) -> AnimeRecord:
            return {
                "title": title,
                "aliases": [],
                "genres": [],
                "score": None,
                "year": None,
                "type": None,
                "episodes": None,
                "synopsis": None,
            }

        anime = {"1": record("Blue"), "2": record("Blue"), "3": record("Absent")}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            glove = root / "glove.6B.300d.txt"
            glove_raw = ("blue " + " ".join(["1"] + ["0"] * 299) + "\n").encode()
            glove.write_bytes(glove_raw)
            descriptor = {
                "id": "glove_300d",
                "size": len(glove_raw),
                "sha256": hashlib.sha256(glove_raw).hexdigest(),
            }
            pins = {
                "catalog": {"anime": anime},
                "queries": [{"mal_id": 1}, {"mal_id": 3}],
                "registry": {"sources": [descriptor]},
            }
            request = root / "request.json"
            output = root / "output.json"
            request.write_text(
                json.dumps(
                    {
                        "method": "glove-selected",
                        "paths": {
                            name: str(glove if name == "glove" else root / name)
                            for name in (
                                "bundle_dir",
                                "normalization_report",
                                "build_report",
                                "anime_csv",
                                "synopsis_csv",
                                "glove",
                                "lockfile",
                            )
                        },
                    }
                )
            )
            thread_env = {
                name: "1"
                for name in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            }
            with (
                patch.dict(os.environ, thread_env),
                patch("recsys.historical.comparison._load_pins", return_value=pins),
            ):
                _worker(request, output)
            result = json.loads(output.read_text())
        rows = result["result"]["queries"]
        self.assertEqual(result["result"]["universe_count"], 3)
        self.assertEqual(
            rows[0],
            {
                "query_mal_id": 1,
                "status": "ok",
                "reasons": [],
                "recommendations": [{"rank": 1, "mal_id": 2, "score": 1.0}],
            },
        )
        self.assertEqual(
            rows[1], {"query_mal_id": 3, "status": "oov_only", "reasons": [], "recommendations": []}
        )
        self.assertEqual(result["result"]["extra"]["glove"]["sha256"], descriptor["sha256"])
        self.assertGreater(result["rank20_wall_ns"], 0)
        self.assertGreater(result["prepare_wall_ns"], 0)
        self.assertTrue(all(pool["num_threads"] == 1 for pool in result["effective_threadpools"]))

    def test_provenance_rejects_null_wrong_source_and_garbage_digest(self) -> None:
        import recsys.historical.comparison as evaluation

        registry, registry_sha = load_registry()
        descriptors = {entry["id"]: entry for entry in registry["sources"]}
        raw_sources: list[RawSource] = [
            {"id": d["id"], "size": d["size"], "sha256": d["sha256"]}
            for d in (
                descriptors["mal_anime"],
                descriptors["mal_synopsis"],
                descriptors["glove_300d"],
            )
        ]
        code_dir = Path(evaluation.__file__).parents[1]
        provenance: ComparisonProvenance = {
            "registry_sha256": registry_sha,
            "raw_sources": raw_sources,
            "bundle_identity": "sha256:" + "a" * 64,
            "manifest_sha256": "a" * 64,
            "neighbors_sha256": "b" * 64,
            "normalization_report_sha256": "c" * 64,
            "build_report_sha256": "d" * 64,
            "code_revision": "e" * 40,
            "source_layout": SOURCE_LAYOUT,
            "code_files": {name: sha((code_dir / name).read_bytes()) for name in SOURCE_FILES},
            "lockfile_sha256": "f" * 64,
            "versions": {name: importlib.metadata.version(name) for name in DISTRIBUTIONS},
            "legacy_originals": ORIGINAL_HASHES,
        }
        _validate_provenance(provenance, "1" * 64)
        for field, bad in (
            ("raw_sources", None),
            ("manifest_sha256", "garbage"),
            ("bundle_identity", "sha256:" + "b" * 64),
            ("code_files", {}),
            ("versions", None),
            ("legacy_originals", {}),
        ):
            with self.subTest(field=field):
                mutated: dict[str, object] = dict(copy.deepcopy(provenance))
                mutated[field] = bad
                with self.assertRaises(EvaluationError):
                    _validate_provenance(mutated, "1" * 64)

    def test_bundle_provenance_links_to_actual_files(self) -> None:
        from recsys.contracts.bundle import Manifest, read_canonical

        fixture = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "bundle"
        _, catalog_raw = read_canonical(fixture / "catalog.json")
        _, neighbors_raw = read_canonical(fixture / "neighbors.json")
        manifest_value, _ = read_canonical(fixture / "manifest.json")
        assert isinstance(manifest_value, dict)
        assert isinstance(manifest_value.get("sources"), list)
        manifest = cast("Manifest", manifest_value)
        names = (
            "build_report",
            "glove_300d",
            "mal_anime",
            "mal_synopsis",
            "normalization_report",
            "source_registry",
        )
        manifest["sources"] = [
            {"name": name, "version": "1", "sha256": str(i + 1) * 64, "metadata": {}}
            for i, name in enumerate(names)
        ]
        manifest_raw = canonical(manifest)
        sources = {item["name"]: item["sha256"] for item in manifest["sources"]}
        provenance: ComparisonProvenance = {
            "code_revision": "a" * 40,
            "source_layout": SOURCE_LAYOUT,
            "code_files": {},
            "lockfile_sha256": "b" * 64,
            "versions": {},
            "legacy_originals": {},
            "bundle_identity": "sha256:" + sha(manifest_raw),
            "manifest_sha256": sha(manifest_raw),
            "neighbors_sha256": sha(neighbors_raw),
            "build_report_sha256": sources["build_report"],
            "normalization_report_sha256": sources["normalization_report"],
            "registry_sha256": sources["source_registry"],
            "raw_sources": [
                {"id": name, "sha256": sources[name], "size": 1}
                for name in ("mal_anime", "mal_synopsis", "glove_300d")
            ],
        }
        with tempfile.TemporaryDirectory() as temp:
            bundle = Path(temp)
            for name, raw in (
                ("catalog.json", catalog_raw),
                ("neighbors.json", neighbors_raw),
                ("manifest.json", manifest_raw),
            ):
                (bundle / name).write_bytes(raw)
            _validate_bundle_links(provenance, bundle / "catalog.json", catalog_raw)
            wrong = copy.deepcopy(provenance)
            wrong["neighbors_sha256"] = "0" * 64
            with self.assertRaisesRegex(EvaluationError, "bundle file hashes"):
                _validate_bundle_links(wrong, bundle / "catalog.json", catalog_raw)

    def test_diagnostics_digest_must_match_actual_bytes(self) -> None:
        from recsys.historical.legacy_vectors import FEATURES, KMEANS, RETAINED_SHA256

        registry, _ = load_registry()
        glove = next(item for item in registry["sources"] if item["id"] == "glove_300d")
        histogram = {str(i): 1 for i in range(20)}
        histogram["19"] += 10862
        diagnostics = {
            "counts": {
                "anime_input": 17562,
                "metadata_retained": 12173,
                "synopsis_join_retained": 10882,
                "metadata_excluded": 5389,
                "synopsis_join_excluded": 1291,
                "metadata_missing_by_field": {
                    "Aired": 309,
                    "Duration": 555,
                    "Episodes": 516,
                    "Genres": 63,
                    "Rating": 688,
                    "Score": 5141,
                    "Type": 37,
                },
            },
            "retained_ids_sha256": RETAINED_SHA256,
            "query_exclusions": {"35102": ["Aired"]},
            "clustering": {
                "feature_order": list(FEATURES),
                "kmeans_parameters": KMEANS,
                "cluster_histogram": histogram,
                "inertia": 12.0,
                "n_iter": 5,
                "assignment_sha256": "a" * 64,
            },
            "vectors": {
                "serialized_text_sha256": "b" * 64,
                "vector_sha256": "c" * 64,
                "status_counts": {"no_tokens": 0, "oov_only": 0, "zero_vector": 0, "ready": 10882},
                "nonready_ids": [],
            },
            "glove": {"sha256": glove["sha256"], "size": glove["size"]},
            "repairs": [
                "stable_MAL_ID",
                "explicit_KMeans_profile",
                "fixed_feature_order",
                "explicit_scalar_types",
                "self_excluded_by_ID",
                "numeric_ties",
                "nonready_vectors_excluded",
                "cosine_clamp",
                "top99_before_cluster_no_refill",
            ],
        }
        raw = canonical(diagnostics)
        _validate_legacy_diagnostics(diagnostics, raw, sha(raw))
        with self.assertRaisesRegex(EvaluationError, "diagnostics SHA256"):
            _validate_legacy_diagnostics(diagnostics, raw, "garbage")
        wrong = copy.deepcopy(diagnostics)
        clustering = wrong["clustering"]
        assert isinstance(clustering, dict)
        histogram = clustering["cluster_histogram"]
        assert isinstance(histogram, dict)
        histogram["0"] = 99
        with self.assertRaises(EvaluationError):
            _validate_legacy_diagnostics(wrong, canonical(wrong), sha(canonical(wrong)))

    def test_measurements_reject_missing_metadata_bad_threads_and_output_digest(self) -> None:
        comparison_raw = b"{}\n"
        diagnostics = {"fixture": True}
        source: RawSource = {"id": "glove_300d", "sha256": "a" * 64, "size": 123}
        methods: list[ComparisonMethod] = [
            {
                "id": name,
                "version": "fixture-v1",
                "parameters": {},
                "parameters_sha256": "a" * 64,
                "universe_count": 3,
                "universe_ids_sha256": "b" * 64,
                "queries": [],
            }
            for name in ("glove-selected", "genre-jaccard", "legacy-repaired")
        ]
        comparison: ComparisonArtifact = {
            "schema_version": 1,
            "protocol_version": "arb018-v1",
            "query_set_version": "fixture-v1",
            "query_set_sha256": "a" * 64,
            "catalog_sha256": "b" * 64,
            "legacy_diagnostics_sha256": "c" * 64,
            "methods": methods,
            "provenance": {
                "registry_sha256": "a" * 64,
                "raw_sources": [source],
                "bundle_identity": "sha256:" + "a" * 64,
                "manifest_sha256": "a" * 64,
                "neighbors_sha256": "b" * 64,
                "normalization_report_sha256": "c" * 64,
                "build_report_sha256": "d" * 64,
                "code_revision": "e" * 40,
                "source_layout": SOURCE_LAYOUT,
                "code_files": {},
                "lockfile_sha256": "f" * 64,
                "versions": {},
                "legacy_originals": {},
            },
        }
        extras: dict[str, dict[str, object]] = {
            "glove-selected": {"glove": {"sha256": source["sha256"], "size": 123}},
            "genre-jaccard": {},
            "legacy-repaired": {"legacy_diagnostics": diagnostics},
        }
        workers: list[MeasurementWorker] = []
        for method in methods:
            name = method["id"]
            worker_output: WorkerResult = {
                "method": name,
                "universe_count": 3,
                "universe_ids_sha256": "b" * 64,
                "queries": [],
                "extra": extras[name],
            }
            workers.append(
                {
                    "method": name,
                    "repetition": 1,
                    "exit_code": 0,
                    "output_sha256": sha(canonical(worker_output)),
                    "end_to_end_wall_ns": 20,
                    "prepare_wall_ns": 10,
                    "rank20_wall_ns": 5,
                    "peak_rss_bytes": 100,
                    "effective_threadpools": []
                    if name == "genre-jaccard"
                    else [{"num_threads": 1, "user_api": "blas", "internal_api": "openblas"}],
                }
            )
        measurements: MeasurementArtifact = {
            "schema_version": 1,
            "protocol_version": "arb018-v1",
            "comparison_sha256": hashlib.sha256(comparison_raw).hexdigest(),
            "scope": "fresh-process-prepare-and-rank20-v1",
            "runtime": {
                "python": "3.12.3",
                "implementation": "CPython",
                "executable": "/usr/bin/python3.12",
            },
            "machine": {
                "platform": "Linux-test",
                "machine": "x86_64",
                "processor": "",
                "cpu_count": 2,
            },
            "thread_environment": THREAD_ENVIRONMENT,
            "workload": {
                "query_count": 20,
                "universe_counts": {method["id"]: 3 for method in methods},
            },
            "schedule": [
                {"repetition": 1, "order": ["glove-selected", "genre-jaccard", "legacy-repaired"]}
            ],
            "workers": workers,
        }
        performance = _performance(measurements, comparison_raw, comparison, diagnostics)
        self.assertEqual(performance["glove-selected"]["rank20_wall_ns"]["median"], 5)
        mutations: list[object] = []
        wrong = copy.deepcopy(measurements)
        cast("dict[str, object]", wrong).pop("runtime")
        mutations.append(wrong)
        wrong = copy.deepcopy(measurements)
        cast("dict[str, object]", wrong).pop("thread_environment")
        mutations.append(wrong)
        wrong = copy.deepcopy(measurements)
        cast("dict[str, object]", wrong)["machine"] = None
        mutations.append(wrong)
        wrong = copy.deepcopy(measurements)
        wrong["workers"][0]["effective_threadpools"][0]["num_threads"] = 99
        mutations.append(wrong)
        wrong = copy.deepcopy(measurements)
        wrong["workers"][0]["output_sha256"] = "0" * 64
        mutations.append(wrong)
        wrong = copy.deepcopy(measurements)
        wrong["workers"][0]["peak_rss_bytes"] = -1
        mutations.append(wrong)
        for mutated in mutations:
            with self.subTest(mutated=mutated):
                with self.assertRaises(EvaluationError):
                    _performance(mutated, comparison_raw, comparison, diagnostics)

    def test_bad_repetition_never_creates_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "out"
            with self.assertRaisesRegex(EvaluationError, "repetitions"):
                compare(
                    Path("missing"),
                    Path("missing"),
                    Path("missing"),
                    Path("missing"),
                    Path("missing"),
                    Path("missing"),
                    Path("missing"),
                    "a" * 40,
                    out,
                    repetitions=0,
                )
            self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()


class SummaryReportTests(unittest.TestCase):
    def test_summary_includes_methods_machine_scope_and_short_counts(self) -> None:
        from recsys.historical.assessment import make_packet, make_template
        from recsys.historical.comparison import (
            METHODS,
            PARAMETERS,
            THREAD_ENVIRONMENT,
            VERSIONS,
            summarize,
        )

        def record(title: str) -> AnimeRecord:
            return {
                "title": title,
                "aliases": [],
                "genres": ["Drama"],
                "type": "TV",
                "episodes": 12,
                "year": 2020,
                "synopsis": f"Story of {title}",
                "score": 8.0,
            }

        catalog: Catalog = {
            "schema_version": 1,
            "anime": {"1": record("First"), "2": record("Second")},
        }
        queries: list[HistoricalQuery] = [
            {
                "query_mal_id": 1,
                "status": "ok",
                "reasons": [],
                "recommendations": [{"rank": 1, "mal_id": 2, "score": 0.5}],
            }
            for _ in range(20)
        ]
        methods: list[ComparisonMethod] = [
            {
                "id": name,
                "version": VERSIONS[name],
                "parameters": PARAMETERS[name],
                "parameters_sha256": sha(canonical(PARAMETERS[name])),
                "universe_count": 2,
                "universe_ids_sha256": "b" * 64,
                "queries": queries,
            }
            for name in METHODS
        ]
        provenance = {
            "code_revision": "a" * 40,
            "lockfile_sha256": "b" * 64,
            "registry_sha256": "c" * 64,
            "bundle_identity": "sha256:" + "d" * 64,
            "manifest_sha256": "d" * 64,
            "neighbors_sha256": "e" * 64,
            "normalization_report_sha256": "f" * 64,
            "build_report_sha256": "1" * 64,
            "raw_sources": [{"id": "mal_anime", "size": 100, "sha256": "2" * 64}],
            "code_files": {"evaluation.py": "3" * 64},
        }
        catalog_sha = sha(canonical(catalog))
        query_set_sha = "4" * 64
        comparison = {
            "query_set_version": "synthetic-v1",
            "query_set_sha256": query_set_sha,
            "catalog_sha256": catalog_sha,
            "legacy_diagnostics_sha256": "5" * 64,
            "provenance": provenance,
            "methods": methods,
        }
        measurements = {
            "scope": "fresh-process-prepare-and-rank20-v1",
            "runtime": {
                "python": "3.12.3",
                "implementation": "CPython",
                "executable": "/usr/bin/python3.12",
            },
            "machine": {
                "platform": "Linux-test",
                "machine": "x86_64",
                "processor": "",
                "cpu_count": 2,
            },
            "thread_environment": THREAD_ENVIRONMENT,
            "workload": {"query_count": 20, "universe_counts": {name: 2 for name in METHODS}},
            "schedule": [],
            "workers": [
                {"method": name, "effective_threadpools": [{"num_threads": 1}]} for name in METHODS
            ],
        }
        # The packet hashes the complete comparison bytes, including context fields.
        packet = make_packet(cast("HistoricalComparison", comparison), catalog)
        assessment = make_template(packet)
        assessment["assessor"] = {
            "id": "synthetic-fixture",
            "kind": "human",
            "method_version": "fixture-v1",
            "model": None,
            "prompt_version": None,
            "prompt_sha256": None,
        }
        judgments = assessment["judgments"]
        assert isinstance(judgments, list)
        for value in judgments:
            assert isinstance(value, dict)
            row = cast("dict[str, object]", value)
            row["rationale"] = "Catalog evidence is insufficient."
            row["franchise_rationale"] = "Catalog evidence is insufficient."
        perf = {
            name: {
                key: {"samples": [10], "median": 10, "min": 10, "max": 10}
                for key in (
                    "end_to_end_wall_ns",
                    "prepare_wall_ns",
                    "rank20_wall_ns",
                    "peak_rss_bytes",
                )
            }
            for name in METHODS
        }
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name, value in (
                ("comparison.json", comparison),
                ("catalog.json", catalog),
                ("measurements.json", measurements),
                ("legacy-diagnostics.json", {}),
                ("assessment.json", assessment),
            ):
                (root / name).write_bytes(canonical(value))
            with (
                patch("recsys.historical.comparison._validate_comparison"),
                patch("recsys.historical.comparison._validate_bundle_links"),
                patch("recsys.historical.comparison._performance", return_value=perf),
            ):
                summarize(
                    root / "comparison.json",
                    root / "catalog.json",
                    root / "measurements.json",
                    [root / "assessment.json"],
                    root / "out",
                )
            report = json.loads((root / "out" / "summary.json").read_text())
            markdown = (root / "out" / "summary.md").read_text()
        self.assertEqual(
            report["comparison_context"]["methods"][0]["version"], VERSIONS[METHODS[0]]
        )
        self.assertEqual(report["comparison_context"]["methods"][0]["short_queries"], 20)
        self.assertEqual(report["measurement_context"]["machine"]["cpu_count"], 2)
        self.assertEqual(report["measurement_context"]["thread_environment"], THREAD_ENVIRONMENT)
        self.assertIsNone(report["assessments"][0]["methods"][0]["macro"]["mean_grade_at_5"])
        for phrase in (
            "Method | Version",
            "Universe ID SHA-256",
            "Unavailable",
            "Short",
            "Native thread environment",
            "Genre Jaccard checks only its file size",
            "shared validation cost",
            "Code file SHA-256 hashes",
            "background host activity",
            "CPU scheduling",
        ):
            self.assertIn(phrase, markdown)
