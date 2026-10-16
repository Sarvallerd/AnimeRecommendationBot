"""Paired user bootstrap and prespecified minimum-N rule."""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

from recsys.research.artifacts import (
    StreamDescriptor,
    check_descriptor,
    cleanup,
    open_stage,
    publish,
    read_json,
    safe_path,
    separate,
    stream_jsonl,
    write_json,
)
from recsys.research.evaluation.evaluate import METRICS, EvaluationManifest, PrivateMetric
from recsys.research.protocol import ProtocolSpec, exact, fail, sha

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

type EvaluationRows = dict[tuple[str, str], PrivateMetric]
type EvaluationRecord = tuple[EvaluationManifest, str, EvaluationRows]
type PrimaryIndex = dict[str, dict[tuple[str, int], tuple[str, float]]]


def _private_metric(value: object) -> PrivateMetric:
    exact(
        value,
        {
            "query_id",
            "user_key",
            "split",
            "cohort",
            "mask",
            "order",
            "n",
            "history_bin",
            "seed_band",
            "ranking",
            "metrics",
            "known_history_repeats",
            "completed_repeats",
            "recent_2024_seed_count",
            "genre_diversity",
        },
    )
    row = cast("dict[str, object]", value)
    if any(
        type(row[field]) is not str
        for field in (
            "query_id",
            "user_key",
            "split",
            "cohort",
            "mask",
            "order",
            "history_bin",
            "seed_band",
        )
    ):
        fail()
    if any(
        type(row[field]) is not int
        for field in ("n", "known_history_repeats", "completed_repeats", "recent_2024_seed_count")
    ):
        fail()
    ranking = row["ranking"]
    if type(ranking) is not list or any(type(item) is not int for item in ranking):
        fail()
    metric_value = row["metrics"]
    exact(metric_value, METRICS)
    metric_map = cast("dict[str, object]", metric_value)
    for key in METRICS:
        metric = metric_map[key]
        if type(metric) is not float or not math.isfinite(metric):
            fail()
    diversity_value = row["genre_diversity"]
    exact(diversity_value, {"mean", "valid_pair_coverage"})
    diversity = cast("dict[str, object]", diversity_value)
    mean = diversity["mean"]
    coverage = diversity["valid_pair_coverage"]
    if (
        (mean is not None and (type(mean) is not float or not math.isfinite(mean)))
        or type(coverage) is not float
        or not math.isfinite(coverage)
    ):
        fail()
    return cast("PrivateMetric", row)


def _evaluation(path: str | Path, expected_prepared_identity: str) -> EvaluationRecord:
    path = safe_path(path, directory=True)
    if {item.name for item in path.iterdir()} != {
        "evaluation-manifest.json",
        "report.json",
        "user-metrics.jsonl.gz",
    }:
        fail()
    value, raw = read_json(path / "evaluation-manifest.json")
    exact(
        value,
        {
            "schema_version",
            "kind",
            "prepared_identity",
            "prediction_identity",
            "split",
            "model_id",
            "config_sha256",
            "fit_identity",
            "prediction_code_revision",
            "selection_identity",
            "files",
        },
    )
    raw_manifest = cast("dict[str, object]", value)
    if (
        type(raw_manifest["schema_version"]) is not int
        or raw_manifest["schema_version"] != 1
        or raw_manifest["kind"] != "arb-ml-evaluation"
        or raw_manifest["split"] != "validation"
        or raw_manifest["prepared_identity"] != expected_prepared_identity
        or raw_manifest["selection_identity"] is not None
        or any(
            type(raw_manifest[field]) is not str
            for field in (
                "prediction_identity",
                "model_id",
                "config_sha256",
                "fit_identity",
                "prediction_code_revision",
            )
        )
    ):
        fail()
    files_value = raw_manifest["files"]
    exact(files_value, {"report.json", "user-metrics.jsonl.gz"})
    files = cast("dict[str, object]", files_value)
    check_descriptor(files["report.json"])
    check_descriptor(files["user-metrics.jsonl.gz"], stream=True)
    manifest = cast("EvaluationManifest", raw_manifest)
    report_value, _ = read_json(path / "report.json", manifest["files"]["report.json"])
    exact(
        report_value,
        {
            "schema_version",
            "kind",
            "split",
            "prediction_identity",
            "prepared_identity",
            "queries",
            "groups",
        },
    )
    report = cast("dict[str, object]", report_value)
    if (
        type(report["schema_version"]) is not int
        or report["schema_version"] != 1
        or report["kind"] != "arb-ml-evaluation-report"
        or report["split"] != "validation"
        or report["prepared_identity"] != expected_prepared_identity
        or report["prediction_identity"] != manifest["prediction_identity"]
        or type(report["queries"]) is not int
        or type(report["groups"]) is not list
    ):
        fail()
    rows: EvaluationRows = {}
    per_query: defaultdict[str, set[str]] = defaultdict(set)
    for row_value in stream_jsonl(
        path / "user-metrics.jsonl.gz",
        cast("StreamDescriptor", manifest["files"]["user-metrics.jsonl.gz"]),
    ):
        row = _private_metric(row_value)
        key = (row["query_id"], row["mask"])
        if (
            key in rows
            or row["split"] != "validation"
            or type(row["metrics"]["ndcg5"]) not in (int, float)
            or not 0 <= row["metrics"]["ndcg5"] <= 1
        ):
            fail()
        rows[key] = row
        per_query[row["query_id"]].add(row["mask"])
    if (
        any(masks != {"controlled", "product"} for masks in per_query.values())
        or type(report["queries"]) is not int
        or report["queries"] != len(per_query)
    ):
        fail()
    return manifest, "sha256:" + sha(raw), rows


def _primary(rows: EvaluationRows) -> tuple[list[str], PrimaryIndex]:
    by_user: defaultdict[str, dict[tuple[str, int], tuple[str, float]]] = defaultdict(dict)
    for row in rows.values():
        if (
            row["cohort"] == "main"
            and row["mask"] == "controlled"
            and row["order"].startswith("random-liked-")
            and 1 <= row["n"] <= 10
        ):
            key = (row["order"], row["n"])
            if key in by_user[row["user_key"]]:
                fail()
            by_user[row["user_key"]][key] = (row["query_id"], row["metrics"]["ndcg5"])
    users = sorted(by_user)
    wanted = {(f"random-liked-{order}", n) for order in (1, 2, 3) for n in range(1, 11)}
    if any(set(by_user[user]) != wanted for user in users):
        fail("research_unpaired_evaluation")
    return users, by_user


def _quantile(
    values: NDArray[np.float64], probability: float, method: Literal["higher", "lower"]
) -> float:
    import numpy as np

    return float(np.quantile(values, probability, method=method))


def _bootstrap(
    matrix: NDArray[np.float64], seed: int, replicates: int
) -> NDArray[np.float64] | None:
    import numpy as np

    rng = np.random.Generator(np.random.PCG64(seed))
    nusers = matrix.shape[0]
    if nusers < 2:
        return None
    return (
        np.empty((replicates, matrix.shape[1]), dtype=np.float64)
        if not nusers
        else np.stack(
            [matrix[rng.integers(0, nusers, nusers)].mean(axis=0) for _ in range(replicates)]
        )
    )


def _policy(matrix: NDArray[np.float64], protocol: ProtocolSpec) -> dict[str, object]:
    import numpy as np

    epsilon = protocol["comparison"]["epsilon_ndcg5"]
    nusers = matrix.shape[0]
    if nusers < 2:
        return {
            "status": "insufficient-users",
            "chosen_n": None,
            "epsilon": epsilon,
            "users": nusers,
            "means": None,
            "upper_regret": None,
        }
    mean = matrix.mean(axis=0)
    bootstrap = _bootstrap(
        matrix, protocol["comparison"]["seed"], protocol["comparison"]["bootstrap_replicates"]
    )
    if bootstrap is None:
        fail("research_unpaired_evaluation")
    errors: list[NDArray[np.float64]] = []
    for n in range(10):
        for j in range(10):
            if j != n:
                errors.append((bootstrap[:, j] - bootstrap[:, n]) - (mean[j] - mean[n]))
    c = _quantile(np.max(np.stack(errors, axis=1), axis=1), 0.95, "higher")
    upper = [
        max(0.0, max(float(mean[j] - mean[n]) for j in range(10)) + max(0.0, c)) for n in range(10)
    ]
    chosen = next((n + 1 for n, value in enumerate(upper) if value <= epsilon), None)
    ci = [
        [_quantile(bootstrap[:, n], 0.025, "lower"), _quantile(bootstrap[:, n], 0.975, "higher")]
        for n in range(10)
    ]
    return {
        "status": "selected" if chosen is not None else "inconclusive",
        "chosen_n": chosen,
        "epsilon": epsilon,
        "users": nusers,
        "means": [float(x) for x in mean],
        "ci95_by_n": ci,
        "upper_regret": upper,
        "simultaneous_error_95": c,
        "bootstrap_replicates": protocol["comparison"]["bootstrap_replicates"],
    }


def _comparison_report(
    evaluations: list[EvaluationRecord], prepared_identity: str, protocol: ProtocolSpec
) -> dict[str, object]:
    """Derive every public comparison field from verified private validation rows."""
    import numpy as np

    if not evaluations:
        fail()
    ids = [item[1] for item in evaluations]
    if len(ids) != len(set(ids)):
        fail()
    reference_keys = set(evaluations[0][2])
    reference_users, reference_primary = _primary(evaluations[0][2])
    models: list[dict[str, object]] = []
    matrices: list[NDArray[np.float64]] = []
    for manifest, eval_id, rows in evaluations:
        if set(rows) != reference_keys:
            fail("research_unpaired_evaluation")
        for key, row in rows.items():
            baseline = evaluations[0][2][key]
            if any(
                row[field] != baseline[field]
                for field in (
                    "user_key",
                    "split",
                    "cohort",
                    "mask",
                    "order",
                    "n",
                    "history_bin",
                    "seed_band",
                    "recent_2024_seed_count",
                )
            ):
                fail("research_unpaired_evaluation")
        users, primary = _primary(rows)
        if users != reference_users or any(
            primary[user][key][0] != reference_primary[user][key][0]
            for user in users
            for key in reference_primary[user]
        ):
            fail("research_unpaired_evaluation")
        matrix = np.asarray(
            [
                [
                    sum(primary[user][(f"random-liked-{order}", n)][1] for order in (1, 2, 3)) / 3
                    for n in range(1, 11)
                ]
                for user in users
            ],
            dtype=np.float64,
        )
        matrices.append(matrix)
        models.append(
            {
                "evaluation_identity": eval_id,
                "prediction_identity": manifest["prediction_identity"],
                "model_id": manifest["model_id"],
                "config_sha256": manifest["config_sha256"],
                "fit_identity": manifest["fit_identity"],
                "code_revision": manifest["prediction_code_revision"],
                "n_policy": _policy(matrix, protocol),
            }
        )
    pairs: list[dict[str, object]] = []
    if len(reference_users) >= 2:
        base = matrices[0]
        for index in range(1, len(matrices)):
            difference = matrices[index] - base
            bootstrap = _bootstrap(
                difference,
                protocol["comparison"]["seed"],
                protocol["comparison"]["bootstrap_replicates"],
            )
            if bootstrap is None:
                fail("research_unpaired_evaluation")
            pairs.append(
                {
                    "left": ids[0],
                    "right": ids[index],
                    "users": len(reference_users),
                    "mean_difference_by_n": [float(x) for x in difference.mean(axis=0)],
                    "ci95_by_n": [
                        [
                            _quantile(bootstrap[:, n], 0.025, "lower"),
                            _quantile(bootstrap[:, n], 0.975, "higher"),
                        ]
                        for n in range(10)
                    ],
                }
            )
    report = {
        "schema_version": 1,
        "kind": "arb-ml-comparison",
        "prepared_identity": prepared_identity,
        "metric": "main-controlled-random-liked-user-mean-NDCG@5",
        "models": models,
        "paired_differences": pairs,
        "users": len(reference_users),
        "selection_limitations": "Validation only; uncertainty and finite sample do not guarantee product benefit.",
    }
    return report


def compare_evaluations(
    evaluation_dirs: list[str | Path] | tuple[str | Path, ...],
    output_dir: str | Path,
    *,
    expected_prepared_identity: str,
) -> Path:
    if type(evaluation_dirs) not in (list, tuple) or not evaluation_dirs:
        fail()
    output_dir = Path(output_dir)
    separate(*(list(evaluation_dirs) + [output_dir]))
    from recsys.research.protocol import load_protocol

    prepared_identity = expected_prepared_identity
    evaluations = [_evaluation(path, prepared_identity) for path in evaluation_dirs]
    report = _comparison_report(evaluations, prepared_identity, load_protocol())
    ids = [item[1] for item in evaluations]
    stage = open_stage(output_dir)
    try:
        desc = write_json(stage / "comparison.json", report)
        write_json(
            stage / "comparison-manifest.json",
            {
                "schema_version": 1,
                "kind": "arb-ml-comparison-artifact",
                "prepared_identity": prepared_identity,
                "evaluation_identities": ids,
                "report": desc,
            },
        )
        publish(stage, output_dir)
        return output_dir / "comparison-manifest.json"
    except Exception:
        cleanup(stage)
        raise
