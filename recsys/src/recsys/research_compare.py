"""Paired user bootstrap and prespecified minimum-N rule."""

from collections import defaultdict
from pathlib import Path

from .research_artifacts import (cleanup, identity, open_stage, publish, read_json,
                                 safe_path, separate, stream_jsonl, write_json)
from .research_protocol import exact, fail, sha


def _evaluation(path, expected_prepared_identity):
    path = safe_path(path, directory=True)
    if {item.name for item in path.iterdir()} != {"evaluation-manifest.json", "report.json", "user-metrics.jsonl.gz"}:
        fail()
    manifest, raw = read_json(path / "evaluation-manifest.json")
    exact(manifest, {"schema_version", "kind", "prepared_identity", "prediction_identity", "split",
                     "model_id", "config_sha256", "fit_identity", "prediction_code_revision",
                     "selection_identity", "files"})
    if (type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1 or
            manifest["kind"] != "arb-ml-evaluation" or manifest["split"] != "validation" or
            manifest["prepared_identity"] != expected_prepared_identity or
            manifest["selection_identity"] is not None or
            set(manifest["files"]) != {"report.json", "user-metrics.jsonl.gz"}):
        fail()
    report, _ = read_json(path / "report.json", manifest["files"]["report.json"])
    if (report["split"] != "validation" or report["prepared_identity"] != expected_prepared_identity or
            report["prediction_identity"] != manifest["prediction_identity"]):
        fail()
    rows = {}
    per_query = defaultdict(set)
    for row in stream_jsonl(path / "user-metrics.jsonl.gz", manifest["files"]["user-metrics.jsonl.gz"]):
        exact(row, {"query_id", "user_key", "split", "cohort", "mask", "order", "n",
                    "history_bin", "seed_band", "ranking", "metrics",
                    "known_history_repeats", "completed_repeats", "recent_2024_seed_count",
                    "genre_diversity"})
        key = (row["query_id"], row["mask"])
        if (key in rows or row["split"] != "validation" or
                type(row["metrics"]["ndcg5"]) not in (int, float) or
                not 0 <= row["metrics"]["ndcg5"] <= 1):
            fail()
        rows[key] = row
        per_query[row["query_id"]].add(row["mask"])
    if (any(masks != {"controlled", "product"} for masks in per_query.values()) or
            type(report["queries"]) is not int or report["queries"] != len(per_query)):
        fail()
    return manifest, "sha256:" + sha(raw), rows


def _primary(rows):
    by_user = defaultdict(dict)
    for row in rows.values():
        if row["cohort"] == "main" and row["mask"] == "controlled" and row["order"].startswith("random-liked-") and 1 <= row["n"] <= 10:
            key = (row["order"], row["n"])
            if key in by_user[row["user_key"]]:
                fail()
            by_user[row["user_key"]][key] = (row["query_id"], row["metrics"]["ndcg5"])
    users = sorted(by_user)
    wanted = {(f"random-liked-{order}", n) for order in (1, 2, 3) for n in range(1, 11)}
    if any(set(by_user[user]) != wanted for user in users):
        fail("research_unpaired_evaluation")
    return users, by_user


def _quantile(values, probability, method):
    import numpy as np
    return float(np.quantile(values, probability, method=method))


def _bootstrap(matrix, seed, replicates):
    import numpy as np
    rng = np.random.Generator(np.random.PCG64(seed))
    nusers = matrix.shape[0]
    if nusers < 2:
        return None
    return np.empty((replicates, matrix.shape[1]), dtype=np.float64) if not nusers else np.stack(
        [matrix[rng.integers(0, nusers, nusers)].mean(axis=0) for _ in range(replicates)])


def _policy(matrix, protocol):
    import numpy as np
    epsilon = protocol["comparison"]["epsilon_ndcg5"]
    nusers = matrix.shape[0]
    if nusers < 2:
        return {"status": "insufficient-users", "chosen_n": None, "epsilon": epsilon,
                "users": nusers, "means": None, "upper_regret": None}
    mean = matrix.mean(axis=0)
    bootstrap = _bootstrap(matrix, protocol["comparison"]["seed"],
                           protocol["comparison"]["bootstrap_replicates"])
    errors = []
    for n in range(10):
        for j in range(10):
            if j != n:
                errors.append((bootstrap[:, j] - bootstrap[:, n]) - (mean[j] - mean[n]))
    c = _quantile(np.max(np.stack(errors, axis=1), axis=1), 0.95, "higher")
    upper = [max(0.0, max(float(mean[j] - mean[n]) for j in range(10)) + max(0.0, c))
             for n in range(10)]
    chosen = next((n + 1 for n, value in enumerate(upper) if value <= epsilon), None)
    ci = [[_quantile(bootstrap[:, n], .025, "lower"),
           _quantile(bootstrap[:, n], .975, "higher")] for n in range(10)]
    return {"status": "selected" if chosen is not None else "inconclusive",
            "chosen_n": chosen, "epsilon": epsilon, "users": nusers,
            "means": [float(x) for x in mean], "ci95_by_n": ci, "upper_regret": upper,
            "simultaneous_error_95": c,
            "bootstrap_replicates": protocol["comparison"]["bootstrap_replicates"]}


def compare_evaluations(evaluation_dirs, output_dir, *, expected_prepared_identity):
    import numpy as np
    if type(evaluation_dirs) not in (list, tuple) or not evaluation_dirs:
        fail()
    output_dir = Path(output_dir)
    separate(*(list(evaluation_dirs) + [output_dir]))
    # Read only protocol/catalog/fit metadata, never hidden test labels.
    prepared_identity = expected_prepared_identity
    # Protocol is embedded in each evaluation through prepared identity. Caller supplies
    # prepared path via the evaluation manifest's reproducible companion? Explicitly use
    # the packaged frozen policy; production cannot override it.
    from .research_protocol import load_protocol
    protocol = load_protocol()
    evaluations = [_evaluation(path, prepared_identity) for path in evaluation_dirs]
    ids = [item[1] for item in evaluations]
    if len(ids) != len(set(ids)):
        fail()
    reference_keys = set(evaluations[0][2])
    reference_users, reference_primary = _primary(evaluations[0][2])
    models = []
    matrices = []
    for manifest, eval_id, rows in evaluations:
        if set(rows) != reference_keys:
            fail("research_unpaired_evaluation")
        for key, row in rows.items():
            baseline = evaluations[0][2][key]
            if any(row[field] != baseline[field] for field in
                   ("user_key", "split", "cohort", "mask", "order", "n",
                    "history_bin", "seed_band", "recent_2024_seed_count")):
                fail("research_unpaired_evaluation")
        users, primary = _primary(rows)
        if users != reference_users or any(
                primary[user][key][0] != reference_primary[user][key][0]
                for user in users for key in reference_primary[user]):
            fail("research_unpaired_evaluation")
        matrix = np.asarray([[sum(primary[user][(f"random-liked-{order}", n)][1]
                                  for order in (1, 2, 3)) / 3 for n in range(1, 11)]
                             for user in users], dtype=np.float64)
        matrices.append(matrix)
        models.append({"evaluation_identity": eval_id, "prediction_identity": manifest["prediction_identity"],
                       "model_id": manifest["model_id"], "config_sha256": manifest["config_sha256"],
                       "fit_identity": manifest["fit_identity"],
                       "code_revision": manifest["prediction_code_revision"],
                       "n_policy": _policy(matrix, protocol)})
    pairs = []
    if len(reference_users) >= 2:
        base = matrices[0]
        for index in range(1, len(matrices)):
            difference = matrices[index] - base
            bootstrap = _bootstrap(difference, protocol["comparison"]["seed"],
                                   protocol["comparison"]["bootstrap_replicates"])
            pairs.append({"left": ids[0], "right": ids[index], "users": len(reference_users),
                          "mean_difference_by_n": [float(x) for x in difference.mean(axis=0)],
                          "ci95_by_n": [[_quantile(bootstrap[:, n], .025, "lower"),
                                         _quantile(bootstrap[:, n], .975, "higher")]
                                        for n in range(10)]})
    report = {"schema_version": 1, "kind": "arb-ml-comparison", "prepared_identity": prepared_identity,
              "metric": "main-controlled-random-liked-user-mean-NDCG@5", "models": models,
              "paired_differences": pairs, "users": len(reference_users),
              "selection_limitations": "Validation only; uncertainty and finite sample do not guarantee product benefit."}
    stage = open_stage(output_dir)
    try:
        desc = write_json(stage / "comparison.json", report)
        write_json(stage / "comparison-manifest.json", {"schema_version": 1,
                   "kind": "arb-ml-comparison-artifact", "prepared_identity": prepared_identity,
                   "evaluation_identities": ids, "report": desc})
        publish(stage, output_dir)
        return output_dir / "comparison-manifest.json"
    except Exception:
        cleanup(stage)
        raise
