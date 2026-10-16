"""Independent validation-selection receipt reader."""

from pathlib import Path
from typing import TypedDict, cast

from recsys.research.artifacts import check_identity, read_json, safe_path
from recsys.research.protocol import exact, fail, sha


class SelectionArm(TypedDict):
    cohort: str
    order: str
    n: int


class Selection(TypedDict):
    schema_version: int
    kind: str
    prepared_identity: str
    comparison_identity: str
    evaluation_identity: str
    model_id: str
    config_sha256: str
    fit_identity: str
    n: int
    code_revision: str
    allowed_arms: list[SelectionArm]
    declared_comparison_arms: list[str]


def read_selection(
    selection_path: str | Path,
    expected_selection_identity: str,
    expected_prepared_identity: str,
) -> tuple[Selection, str]:
    selection_path = safe_path(selection_path, directory=False)
    check_identity(expected_selection_identity)
    value, raw = read_json(selection_path)
    exact(
        value,
        {
            "schema_version",
            "kind",
            "prepared_identity",
            "comparison_identity",
            "evaluation_identity",
            "model_id",
            "config_sha256",
            "fit_identity",
            "n",
            "code_revision",
            "allowed_arms",
            "declared_comparison_arms",
        },
    )
    record = cast("dict[str, object]", value)
    if (
        "sha256:" + sha(raw) != expected_selection_identity
        or type(record["schema_version"]) is not int
        or record["schema_version"] != 1
        or record["kind"] != "arb-ml-selection"
        or record["prepared_identity"] != expected_prepared_identity
        or type(record["n"]) is not int
        or not 1 <= record["n"] <= 10
        or any(
            type(record[field]) is not str
            for field in (
                "comparison_identity",
                "evaluation_identity",
                "model_id",
                "config_sha256",
                "fit_identity",
                "code_revision",
            )
        )
    ):
        fail("research_selection_invalid")
    arms = record["allowed_arms"]
    expected_arms = [
        {"cohort": "main", "order": f"random-liked-{i}", "n": record["n"]} for i in (1, 2, 3)
    ]
    if type(arms) is not list or len(arms) != 3:
        fail("research_selection_invalid")
    for arm in arms:
        if (
            type(arm) is not dict
            or set(arm) != {"cohort", "order", "n"}
            or type(arm["cohort"]) is not str
            or type(arm["order"]) is not str
            or type(arm["n"]) is not int
        ):
            fail("research_selection_invalid")
    if arms != expected_arms or record["declared_comparison_arms"] != [
        f"random-liked-{i}" for i in (1, 2, 3)
    ]:
        fail("research_selection_invalid")
    selection = cast("Selection", record)
    return selection, expected_selection_identity
