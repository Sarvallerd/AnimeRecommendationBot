"""Irreversible validation-selection receipt and controlled test query release."""

from pathlib import Path
from typing import TypedDict, cast

from recsys.contracts.bundle import canonical_bytes
from recsys.research.artifacts import (
    FileDescriptor,
    StreamWriter,
    check_descriptor,
    check_identity,
    cleanup,
    identity,
    open_stage,
    publish,
    read_json,
    safe_path,
    separate,
    write_json,
)
from recsys.research.evaluation.selection import read_selection
from recsys.research.protocol import exact, fail, load_protocol, sha


class ComparisonManifest(TypedDict):
    schema_version: int
    kind: str
    prepared_identity: str
    evaluation_identities: list[str]
    report: dict[str, object]


class PolicyChoice(TypedDict):
    status: str
    chosen_n: int | None


class ComparisonModel(TypedDict):
    evaluation_identity: str
    prediction_identity: str
    model_id: str
    config_sha256: str
    fit_identity: str
    code_revision: str
    n_policy: PolicyChoice


class ComparisonReport(TypedDict):
    schema_version: int
    kind: str
    prepared_identity: str
    metric: str
    models: list[ComparisonModel]
    paired_differences: list[dict[str, object]]
    users: int
    selection_limitations: str


class SelectionDecision(TypedDict):
    schema_version: int
    prepared_identity: str
    comparison_identity: str
    evaluation_identity: str
    model_id: str
    config_sha256: str
    fit_identity: str
    n: int
    code_revision: str
    declared_comparison_arms: list[str]


def _comparison(
    path: str | Path, prepared_identity: str
) -> tuple[ComparisonReport, str, list[str]]:
    path = safe_path(path, directory=True)
    if {item.name for item in path.iterdir()} != {"comparison.json", "comparison-manifest.json"}:
        fail()
    manifest_value, raw = read_json(path / "comparison-manifest.json")
    exact(
        manifest_value,
        {"schema_version", "kind", "prepared_identity", "evaluation_identities", "report"},
    )
    manifest_record = cast("dict[str, object]", manifest_value)
    if (
        type(manifest_record["schema_version"]) is not int
        or manifest_record["schema_version"] != 1
        or manifest_record["kind"] != "arb-ml-comparison-artifact"
        or manifest_record["prepared_identity"] != prepared_identity
        or type(manifest_record["evaluation_identities"]) is not list
        or not manifest_record["evaluation_identities"]
    ):
        fail()
    check_identity(prepared_identity)
    check_descriptor(manifest_record["report"])
    evaluation_ids: list[str] = []
    for value in cast("list[object]", manifest_record["evaluation_identities"]):
        check_identity(value)
        if type(value) is not str:
            fail()
        if value != value.lower():
            fail()
        evaluation_ids.append(value)
    if len(set(evaluation_ids)) != len(evaluation_ids):
        fail()
    manifest = cast("ComparisonManifest", manifest_record)
    report_value, _ = read_json(
        path / "comparison.json", cast("FileDescriptor", manifest["report"])
    )
    exact(
        report_value,
        {
            "schema_version",
            "kind",
            "prepared_identity",
            "metric",
            "models",
            "paired_differences",
            "users",
            "selection_limitations",
        },
    )
    report_record = cast("dict[str, object]", report_value)
    if (
        type(report_record["schema_version"]) is not int
        or report_record["schema_version"] != 1
        or report_record["kind"] != "arb-ml-comparison"
        or report_record["prepared_identity"] != prepared_identity
        or report_record["metric"] != "main-controlled-random-liked-user-mean-NDCG@5"
        or type(report_record["models"]) is not list
        or type(report_record["paired_differences"]) is not list
        or type(report_record["users"]) is not int
        or report_record["users"] < 0
        or type(report_record["selection_limitations"]) is not str
        or len(report_record["models"]) != len(evaluation_ids)
    ):
        fail()
    model_ids: list[str] = []
    for model_value in cast("list[object]", report_record["models"]):
        exact(
            model_value,
            {
                "evaluation_identity",
                "prediction_identity",
                "model_id",
                "config_sha256",
                "fit_identity",
                "code_revision",
                "n_policy",
            },
        )
        model = cast("dict[str, object]", model_value)
        if any(
            type(model[field]) is not str
            for field in (
                "evaluation_identity",
                "prediction_identity",
                "model_id",
                "config_sha256",
                "fit_identity",
                "code_revision",
            )
        ):
            fail()
        policy_value = model["n_policy"]
        if type(policy_value) is not dict:
            fail()
        policy = cast("dict[str, object]", policy_value)
        chosen_n = policy.get("chosen_n")
        if (
            type(policy.get("status")) is not str
            or "chosen_n" not in policy
            or (chosen_n is not None and type(chosen_n) is not int)
        ):
            fail()
        evaluation_identity = model["evaluation_identity"]
        if type(evaluation_identity) is not str:
            fail()
        model_ids.append(evaluation_identity)
    if model_ids != evaluation_ids:
        fail()
    for paired in cast("list[object]", report_record["paired_differences"]):
        exact(paired, {"left", "right", "users", "mean_difference_by_n", "ci95_by_n"})
    report = cast("ComparisonReport", report_record)
    return report, "sha256:" + sha(raw), evaluation_ids


def finalize_selection(
    prepared_dir: str | Path,
    decision_path: str | Path,
    validation_reports: list[str | Path] | tuple[str | Path, ...],
    output_dir: str | Path,
) -> Path:
    from recsys.research.evaluation.compare import _comparison_report, _evaluation
    from recsys.research.evaluation.evaluate import _prepared

    prepared_dir, decision_path, output_dir = map(Path, (prepared_dir, decision_path, output_dir))
    if type(validation_reports) not in (list, tuple) or len(validation_reports) < 2:
        fail()
    separate(prepared_dir, decision_path, output_dir, *validation_reports)
    prepared_identity = identity(prepared_dir / "prepare-manifest.json")
    _prepared(prepared_dir, prepared_identity)
    decision_value, _ = read_json(decision_path)
    exact(
        decision_value,
        {
            "schema_version",
            "prepared_identity",
            "comparison_identity",
            "evaluation_identity",
            "model_id",
            "config_sha256",
            "fit_identity",
            "n",
            "code_revision",
            "declared_comparison_arms",
        },
    )
    decision_record = cast("dict[str, object]", decision_value)
    if (
        type(decision_record["schema_version"]) is not int
        or decision_record["schema_version"] != 1
        or decision_record["prepared_identity"] != prepared_identity
        or type(decision_record["n"]) is not int
        or not 1 <= decision_record["n"] <= 10
        or any(
            type(decision_record[field]) is not str
            for field in (
                "comparison_identity",
                "evaluation_identity",
                "model_id",
                "config_sha256",
                "fit_identity",
                "code_revision",
            )
        )
        or decision_record["declared_comparison_arms"] != [f"random-liked-{i}" for i in (1, 2, 3)]
    ):
        fail("research_selection_invalid")
    decision = cast("SelectionDecision", decision_record)
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
    if (
        comparison_identity != decision["comparison_identity"]
        or len(set(evaluation_ids)) != len(evaluation_ids)
        or set(evaluation_ids) != set(comparison_ids)
    ):
        fail("research_selection_invalid")
    by_identity = {item[1]: item for item in evaluations}
    recomputed = _comparison_report(
        [by_identity[value] for value in comparison_ids], prepared_identity, load_protocol()
    )
    if canonical_bytes(comparison) != canonical_bytes(recomputed):
        fail("research_selection_invalid")
    selected = next(
        (item for item in evaluations if item[1] == decision["evaluation_identity"]), None
    )
    if selected is None:
        fail()
    evaluation = selected[0]
    model = next(
        (
            item
            for item in comparison["models"]
            if item["evaluation_identity"] == decision["evaluation_identity"]
        ),
        None,
    )
    if (
        model is None
        or model["n_policy"]["status"] != "selected"
        or model["n_policy"]["chosen_n"] != decision["n"]
        or any(
            model[key] != decision[key] or evaluation[key] != decision[key]
            for key in ("model_id", "config_sha256", "fit_identity")
        )
        or model["code_revision"] != decision["code_revision"]
        or evaluation["prediction_code_revision"] != decision["code_revision"]
        or set(
            comparison["models"][i]["evaluation_identity"] for i in range(len(comparison["models"]))
        )
        != {item[1] for item in evaluations}
    ):
        fail("research_selection_invalid")
    selection = {
        **decision,
        "kind": "arb-ml-selection",
        "allowed_arms": [
            {"cohort": "main", "order": arm, "n": decision["n"]}
            for arm in decision["declared_comparison_arms"]
        ],
    }
    stage = open_stage(output_dir)
    try:
        write_json(stage / "selection.json", selection)
        publish(stage, output_dir)
        return output_dir / "selection.json"
    except Exception:
        cleanup(stage)
        raise


def export_test_queries(
    prepared_dir: str | Path,
    selection_path: str | Path,
    output_dir: str | Path,
    *,
    expected_selection_identity: str,
) -> Path:
    from recsys.research.evaluation.evaluate import _paired_rows, _prepared

    prepared_dir, selection_path, output_dir = map(Path, (prepared_dir, selection_path, output_dir))
    separate(prepared_dir, selection_path, output_dir)
    prepared_identity = identity(prepared_dir / "prepare-manifest.json")
    prep, protocol, catalog, _, _, _ = _prepared(prepared_dir, prepared_identity)
    selection, _ = read_selection(selection_path, expected_selection_identity, prepared_identity)
    allowed = {(arm["cohort"], arm["order"], arm["n"]) for arm in selection["allowed_arms"]}
    stage = open_stage(output_dir)
    try:
        stream = StreamWriter(stage / "test-queries.jsonl.gz")
        for query, _ in _paired_rows(
            prepared_dir,
            prep,
            "test",
            allowed,
            protocol=protocol,
            ids=set(catalog["anime"]),
        ):
            stream.write_json(query)
        desc = stream.close()
        write_json(
            stage / "test-query-manifest.json",
            {
                "schema_version": 1,
                "kind": "arb-ml-test-queries",
                "prepared_identity": prepared_identity,
                "selection_identity": expected_selection_identity,
                "queries": desc,
            },
        )
        publish(stage, output_dir)
        return output_dir / "test-query-manifest.json"
    except Exception:
        cleanup(stage)
        raise
