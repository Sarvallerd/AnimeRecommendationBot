"""Independent validation-selection receipt reader."""

from recsys.research.artifacts import check_identity, read_json, safe_path
from recsys.research.protocol import exact, fail, sha


def read_selection(selection_path, expected_selection_identity, expected_prepared_identity):
    selection_path = safe_path(selection_path, directory=False)
    check_identity(expected_selection_identity)
    selection, raw = read_json(selection_path)
    exact(
        selection,
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
    if (
        "sha256:" + sha(raw) != expected_selection_identity
        or type(selection["schema_version"]) is not int
        or selection["schema_version"] != 1
        or selection["kind"] != "arb-ml-selection"
        or selection["prepared_identity"] != expected_prepared_identity
        or type(selection["n"]) is not int
        or not 1 <= selection["n"] <= 10
        or selection["allowed_arms"]
        != [
            {"cohort": "main", "order": f"random-liked-{i}", "n": selection["n"]} for i in (1, 2, 3)
        ]
    ):
        fail("research_selection_invalid")
    return selection, expected_selection_identity
