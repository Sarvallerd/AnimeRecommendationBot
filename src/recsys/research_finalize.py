"""Irreversible validation-selection receipt and controlled test query release."""

from pathlib import Path

from .research_artifacts import (StreamWriter, check_descriptor, check_identity, cleanup, identity,
                                 open_stage, publish, read_json, safe_path, separate, stream_jsonl, write_json)
from .bundle import canonical_bytes
from .research_protocol import exact, fail, load_protocol, sha


def read_selection(selection_path, expected_selection_identity, expected_prepared_identity):
    selection_path = safe_path(selection_path, directory=False)
    check_identity(expected_selection_identity)
    selection, raw = read_json(selection_path)
    exact(selection, {"schema_version", "kind", "prepared_identity", "comparison_identity",
                      "evaluation_identity", "model_id", "config_sha256", "fit_identity",
                      "n", "code_revision", "allowed_arms", "declared_comparison_arms"})
    if ("sha256:" + sha(raw) != expected_selection_identity or
            type(selection["schema_version"]) is not int or selection["schema_version"] != 1 or
            selection["kind"] != "arb-ml-selection" or
            selection["prepared_identity"] != expected_prepared_identity or
            type(selection["n"]) is not int or not 1 <= selection["n"] <= 10 or
            selection["allowed_arms"] != [{"cohort": "main", "order": f"random-liked-{i}",
                                            "n": selection["n"]} for i in (1, 2, 3)]):
        fail("research_selection_invalid")
    return selection, expected_selection_identity


def _comparison(path, prepared_identity):
    path = safe_path(path, directory=True)
    if {item.name for item in path.iterdir()} != {"comparison.json", "comparison-manifest.json"}:
        fail()
    manifest, raw = read_json(path / "comparison-manifest.json")
    exact(manifest, {"schema_version", "kind", "prepared_identity", "evaluation_identities", "report"})
    if (type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1 or
            manifest["kind"] != "arb-ml-comparison-artifact" or
            manifest["prepared_identity"] != prepared_identity or
            type(manifest["evaluation_identities"]) is not list or
            not manifest["evaluation_identities"]):
        fail()
    check_identity(prepared_identity)
    check_descriptor(manifest["report"])
    evaluation_ids = manifest["evaluation_identities"]
    for value in evaluation_ids:
        check_identity(value)
        if value != value.lower():
            fail()
    if len(set(evaluation_ids)) != len(evaluation_ids):
        fail()
    report, _ = read_json(path / "comparison.json", manifest["report"])
    exact(report, {"schema_version", "kind", "prepared_identity", "metric", "models",
                   "paired_differences", "users", "selection_limitations"})
    if (type(report["schema_version"]) is not int or report["schema_version"] != 1 or
            report["kind"] != "arb-ml-comparison" or
            report["prepared_identity"] != prepared_identity or
            report["metric"] != "main-controlled-random-liked-user-mean-NDCG@5" or
            type(report["models"]) is not list or
            type(report["paired_differences"]) is not list or
            type(report["users"]) is not int or report["users"] < 0 or
            type(report["selection_limitations"]) is not str or
            len(report["models"]) != len(evaluation_ids)):
        fail()
    model_ids = []
    for model in report["models"]:
        exact(model, {"evaluation_identity", "prediction_identity", "model_id", "config_sha256",
                      "fit_identity", "code_revision", "n_policy"})
        model_ids.append(model["evaluation_identity"])
    if model_ids != evaluation_ids:
        fail()
    for paired in report["paired_differences"]:
        exact(paired, {"left", "right", "users", "mean_difference_by_n", "ci95_by_n"})
    return report, "sha256:" + sha(raw), evaluation_ids


def finalize_selection(prepared_dir, decision_path, validation_reports, output_dir):
    from .research_evaluate import _prepared
    from .research_compare import _comparison_report, _evaluation
    prepared_dir, decision_path, output_dir = map(Path, (prepared_dir, decision_path, output_dir))
    if type(validation_reports) not in (list, tuple) or len(validation_reports) < 2:
        fail()
    separate(prepared_dir, decision_path, output_dir, *validation_reports)
    prepared_identity = identity(prepared_dir / "prepare-manifest.json")
    _prepared(prepared_dir, prepared_identity)
    decision, _ = read_json(decision_path)
    exact(decision, {"schema_version", "prepared_identity", "comparison_identity",
                     "evaluation_identity", "model_id", "config_sha256", "fit_identity",
                     "n", "code_revision", "declared_comparison_arms"})
    if (type(decision["schema_version"]) is not int or decision["schema_version"] != 1 or
            decision["prepared_identity"] != prepared_identity or
            type(decision["n"]) is not int or not 1 <= decision["n"] <= 10 or
            decision["declared_comparison_arms"] != [f"random-liked-{i}" for i in (1, 2, 3)]):
        fail("research_selection_invalid")
    evaluations = []
    comparisons = []
    for path in validation_reports:
        names = {item.name for item in safe_path(path, directory=True).iterdir()}
        if "evaluation-manifest.json" in names:
            evaluations.append(_evaluation(path, prepared_identity))
        elif "comparison-manifest.json" in names:
            comparisons.append(_comparison(path, prepared_identity))
        else:
            fail()
    if len(comparisons) != 1 or not evaluations:
        fail()
    comparison, comparison_identity, comparison_ids = comparisons[0]
    evaluation_ids = [item[1] for item in evaluations]
    if (comparison_identity != decision["comparison_identity"] or
            len(set(evaluation_ids)) != len(evaluation_ids) or
            set(evaluation_ids) != set(comparison_ids)):
        fail("research_selection_invalid")
    by_identity = {item[1]: item for item in evaluations}
    recomputed = _comparison_report([by_identity[value] for value in comparison_ids],
                                    prepared_identity, load_protocol())
    if canonical_bytes(comparison) != canonical_bytes(recomputed):
        fail("research_selection_invalid")
    selected = next((item for item in evaluations if item[1] == decision["evaluation_identity"]), None)
    if selected is None:
        fail()
    evaluation = selected[0]
    model = next((item for item in comparison["models"]
                  if item["evaluation_identity"] == decision["evaluation_identity"]), None)
    if (model is None or model["n_policy"]["status"] != "selected" or
            model["n_policy"]["chosen_n"] != decision["n"] or
            any(model[key] != decision[key] or evaluation[key] != decision[key]
                for key in ("model_id", "config_sha256", "fit_identity")) or
            model["code_revision"] != decision["code_revision"] or
            evaluation["prediction_code_revision"] != decision["code_revision"] or
            set(comparison["models"][i]["evaluation_identity"] for i in range(len(comparison["models"]))) !=
            {item[1] for item in evaluations}):
        fail("research_selection_invalid")
    selection = {**decision, "kind": "arb-ml-selection",
                 "allowed_arms": [{"cohort": "main", "order": arm, "n": decision["n"]}
                                  for arm in decision["declared_comparison_arms"]]}
    stage = open_stage(output_dir)
    try:
        write_json(stage / "selection.json", selection)
        publish(stage, output_dir)
        return output_dir / "selection.json"
    except Exception:
        cleanup(stage)
        raise


def export_test_queries(prepared_dir, selection_path, output_dir, *, expected_selection_identity):
    from .research_evaluate import _paired_rows, _prepared
    prepared_dir, selection_path, output_dir = map(Path, (prepared_dir, selection_path, output_dir))
    separate(prepared_dir, selection_path, output_dir)
    prepared_identity = identity(prepared_dir / "prepare-manifest.json")
    prep, _, _, _, _, _ = _prepared(prepared_dir, prepared_identity)
    selection, _ = read_selection(selection_path, expected_selection_identity, prepared_identity)
    allowed = {(arm["cohort"], arm["order"], arm["n"]) for arm in selection["allowed_arms"]}
    stage = open_stage(output_dir)
    try:
        stream = StreamWriter(stage / "test-queries.jsonl.gz")
        for query, _ in _paired_rows(prepared_dir, prep, "test", allowed):
            stream.write_json(query)
        desc = stream.close()
        write_json(stage / "test-query-manifest.json", {"schema_version": 1,
                   "kind": "arb-ml-test-queries", "prepared_identity": prepared_identity,
                   "selection_identity": expected_selection_identity, "queries": desc})
        publish(stage, output_dir)
        return output_dir / "test-query-manifest.json"
    except Exception:
        cleanup(stage)
        raise
