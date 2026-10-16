"""Private ML-02 model inputs, fixed variants, and immutable model packages."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from recsys.contracts.bundle import canonical_bytes, check_catalog
from recsys.data.dataset import HEADER
from recsys.release.lock import release_info
from recsys.research.artifacts import (
    check_descriptor,
    check_identity,
    read_json,
    safe_path,
    stream_tsv,
    validate_preparation_producer,
)
from recsys.research.models.types import ScoringInputs, TrainingInputs
from recsys.research.protocol import exact, fail, load_protocol, sha, split_user

PREPARED_FILES = {
    "protocol.json",
    "model-catalog.json",
    "fit/manifest.json",
    "fit/train-interactions.tsv.gz",
    "fit/train-stats.json",
    "development/queries.jsonl.gz",
    "evaluator/splits.tsv.gz",
    "evaluator/validation-labels.jsonl.gz",
    "evaluator/test-labels.jsonl.gz",
}


def _stats(stats, ids):
    exact(stats, {"schema_version", "train_users", "train_completed_rows", "items"})
    if (
        type(stats["schema_version"]) is not int
        or stats["schema_version"] != 1
        or type(stats["train_users"]) is not int
        or stats["train_users"] < 1
        or type(stats["train_completed_rows"]) is not int
        or stats["train_completed_rows"] < 0
        or type(stats["items"]) is not dict
        or set(stats["items"]) != set(ids)
    ):
        fail("research_model_invalid")
    total = 0
    positive = []
    for key, value in stats["items"].items():
        exact(value, {"completed_users", "positive_users", "rating_sum", "rating_count", "band"})
        c, p, s, r = (
            value[name]
            for name in ("completed_users", "positive_users", "rating_sum", "rating_count")
        )
        if (
            any(type(n) is not int for n in (c, p, s, r))
            or not 0 <= p <= c <= stats["train_users"]
            or r != c
            or not c <= s <= 10 * c
        ):
            fail("research_model_invalid")
        total += c
        if p:
            positive.append(int(key))
    if total != stats["train_completed_rows"]:
        fail("research_model_invalid")
    ranked = sorted(positive, key=lambda item: (-stats["items"][str(item)]["positive_users"], item))
    head = set(ranked[: (len(ranked) + 4) // 5])
    for key, value in stats["items"].items():
        expected = "cold" if not value["positive_users"] else "head" if int(key) in head else "tail"
        if value["band"] != expected:
            fail("research_model_invalid")


def read_scoring_inputs(prepared_dir, *, expected_prepared_identity):
    """Read only development-visible preparation metadata, catalog, and train stats."""
    prepared_dir = safe_path(prepared_dir, directory=True)
    check_identity(expected_prepared_identity)
    prep, raw = read_json(prepared_dir / "prepare-manifest.json")
    exact(
        prep,
        {
            "schema_version",
            "kind",
            "protocol_identity",
            "release_lock_sha256",
            "release_id",
            "dataset_identity",
            "catalog_release_identity",
            "catalog_sha256",
            "producer",
            "files",
            "counts",
        },
    )
    validate_preparation_producer(prep)
    lock = release_info()
    if (
        "sha256:" + sha(raw) != expected_prepared_identity
        or type(prep["schema_version"]) is not int
        or prep["schema_version"] != 2
        or prep["kind"] != "arb-ml-preparation"
        or prep["release_id"] != lock["release_id"]
        or prep["release_lock_sha256"] != sha(canonical_bytes(lock))
        or prep["dataset_identity"] != lock["inputs"]["base"]["identity"]
        or prep["catalog_release_identity"] != lock["inputs"]["catalog_release"]["identity"]
        or prep["catalog_sha256"] != lock["inputs"]["catalog_release"]["catalog_sha256"]
    ):
        fail("research_model_invalid")
    files = prep["files"]
    if type(files) is not dict or set(files) != PREPARED_FILES:
        fail("research_model_invalid")
    for name, entry in files.items():
        check_descriptor(entry, stream=name.endswith(".gz"))
    protocol, _ = read_json(prepared_dir / "protocol.json", files["protocol.json"])
    if protocol != load_protocol() or prep["protocol_identity"] != "sha256:" + sha(
        canonical_bytes(protocol)
    ):
        fail("research_model_invalid")
    catalog, _ = read_json(prepared_dir / "model-catalog.json", files["model-catalog.json"])
    ids = check_catalog(catalog)
    if len(ids) != lock["inputs"]["catalog_release"]["catalog_count"] or any(
        item["score"] is not None for item in catalog["anime"].values()
    ):
        fail("research_model_invalid")
    fit, fit_raw = read_json(prepared_dir / "fit/manifest.json", files["fit/manifest.json"])
    if fit != {
        "schema_version": 1,
        "kind": "arb-ml-fit",
        "dataset_identity": prep["dataset_identity"],
        "protocol_identity": prep["protocol_identity"],
        "train_interactions": files["fit/train-interactions.tsv.gz"],
        "train_stats": files["fit/train-stats.json"],
    }:
        fail("research_model_invalid")
    stats, _ = read_json(prepared_dir / "fit/train-stats.json", files["fit/train-stats.json"])
    _stats(stats, ids)
    return ScoringInputs(
        expected_prepared_identity,
        prep["protocol_identity"],
        files["model-catalog.json"],
        "sha256:" + sha(fit_raw),
        files["fit/train-interactions.tsv.gz"],
        files["fit/train-stats.json"],
        tuple(sorted(map(int, ids))),
        catalog,
        stats,
    )


def read_training_inputs(prepared_dir, *, expected_prepared_identity):
    scores = read_scoring_inputs(
        prepared_dir, expected_prepared_identity=expected_prepared_identity
    )
    path = safe_path(Path(prepared_dir) / "fit/train-interactions.tsv.gz", directory=False)
    return TrainingInputs(**vars(scores), train_interactions_path=path)


def iter_training_rows(inputs: TrainingInputs):
    """Validate every train row and compressed/plain stream descriptor through EOF."""
    lines = iter(stream_tsv(inputs.train_interactions_path, inputs.train_interactions_descriptor))
    if next(lines, None) != HEADER:
        fail("research_model_invalid")
    ids = set(inputs.mal_ids)
    previous = None
    count = 0
    protocol = load_protocol()
    for raw in lines:
        fields = raw.removesuffix(b"\n").split(b"\t")
        if len(fields) != 4:
            fail("research_model_invalid")
        try:
            user = fields[0].decode("ascii")
            item = int(fields[1])
            rating = int(fields[2])
            if (
                str(item).encode() != fields[1]
                or str(rating).encode() != fields[2]
                or split_user(user, protocol) != "train"
                or fields[3] != b"completed"
                or item not in ids
                or not 1 <= rating <= 10
                or (previous is not None and (user, item) <= previous)
            ):
                fail("research_model_invalid")
        except (ValueError, UnicodeError):
            fail("research_model_invalid")
        previous = user, item
        count += 1
        yield user, item, rating
    if count != inputs.train_stats["train_completed_rows"]:
        fail("research_model_invalid")
