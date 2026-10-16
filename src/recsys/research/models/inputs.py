"""Private ML-02 model inputs, fixed variants, and immutable model packages."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import cast

from recsys.contracts.bundle import Catalog, canonical_bytes, check_catalog
from recsys.data.dataset import HEADER
from recsys.release.lock import release_info
from recsys.research.artifacts import (
    FileDescriptor,
    StreamDescriptor,
    check_descriptor,
    check_identity,
    read_json,
    safe_path,
    stream_tsv,
    validate_preparation_producer,
)
from recsys.research.models.types import ScoringInputs, TrainingInputs, TrainStats
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


def _stats(stats: object, ids: set[str]) -> None:
    exact(stats, {"schema_version", "train_users", "train_completed_rows", "items"})
    stats = cast("dict[str, object]", stats)  # exact checked the outer map.
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
    items = cast("dict[str, object]", stats["items"])  # Validated exact item keys above.
    train_users = stats["train_users"]
    train_completed_rows = stats["train_completed_rows"]
    total = 0
    positive: list[int] = []
    for key, value in items.items():
        exact(value, {"completed_users", "positive_users", "rating_sum", "rating_count", "band"})
        value = cast("dict[str, object]", value)  # exact checked the record fields.
        c, p, s, r = (
            value[name]
            for name in ("completed_users", "positive_users", "rating_sum", "rating_count")
        )
        if (
            any(type(n) is not int for n in (c, p, s, r))
            or not 0 <= cast("int", p) <= cast("int", c) <= train_users
            or r != c
            or not cast("int", c) <= cast("int", s) <= 10 * cast("int", c)
        ):
            fail("research_model_invalid")
        total += cast("int", c)
        if p:
            positive.append(int(key))
    if total != train_completed_rows:
        fail("research_model_invalid")
    ranked = sorted(
        positive,
        key=lambda item: (
            -cast("int", cast("dict[str, object]", items[str(item)])["positive_users"]),
            item,
        ),
    )
    head = set(ranked[: (len(ranked) + 4) // 5])
    for key, raw_value in items.items():
        value = cast("dict[str, object]", raw_value)  # Records checked in the first pass.
        expected = "cold" if not value["positive_users"] else "head" if int(key) in head else "tail"
        if value["band"] != expected:
            fail("research_model_invalid")


def read_scoring_inputs(
    prepared_dir: str | Path, *, expected_prepared_identity: str
) -> ScoringInputs:
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
    prep = cast("dict[str, object]", prep)  # Exact fields and producer validated above.
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
    file_map = cast("dict[str, object]", files)  # Exact prepared-file set checked above.
    protocol, _ = read_json(
        prepared_dir / "protocol.json", cast("FileDescriptor", file_map["protocol.json"])
    )
    if protocol != load_protocol() or prep["protocol_identity"] != "sha256:" + sha(
        canonical_bytes(protocol)
    ):
        fail("research_model_invalid")
    catalog, _ = read_json(
        prepared_dir / "model-catalog.json", cast("FileDescriptor", file_map["model-catalog.json"])
    )
    ids = check_catalog(catalog)
    catalog = cast("Catalog", catalog)  # Full catalog wire validation above.
    if len(ids) != lock["inputs"]["catalog_release"]["catalog_count"] or any(
        item["score"] is not None for item in catalog["anime"].values()
    ):
        fail("research_model_invalid")
    fit, fit_raw = read_json(
        prepared_dir / "fit/manifest.json", cast("FileDescriptor", file_map["fit/manifest.json"])
    )
    if fit != {
        "schema_version": 1,
        "kind": "arb-ml-fit",
        "dataset_identity": prep["dataset_identity"],
        "protocol_identity": prep["protocol_identity"],
        "train_interactions": file_map["fit/train-interactions.tsv.gz"],
        "train_stats": file_map["fit/train-stats.json"],
    }:
        fail("research_model_invalid")
    stats, _ = read_json(
        prepared_dir / "fit/train-stats.json",
        cast("FileDescriptor", file_map["fit/train-stats.json"]),
    )
    _stats(stats, ids)
    stats = cast("TrainStats", stats)  # Complete stat keys and numeric ranges checked above.
    return ScoringInputs(
        expected_prepared_identity,
        prep["protocol_identity"],
        cast("FileDescriptor", file_map["model-catalog.json"]),
        "sha256:" + sha(fit_raw),
        cast("StreamDescriptor", file_map["fit/train-interactions.tsv.gz"]),
        cast("FileDescriptor", file_map["fit/train-stats.json"]),
        tuple(sorted(map(int, ids))),
        catalog,
        stats,
    )


def read_training_inputs(
    prepared_dir: str | Path, *, expected_prepared_identity: str
) -> TrainingInputs:
    scores = read_scoring_inputs(
        prepared_dir, expected_prepared_identity=expected_prepared_identity
    )
    path = safe_path(Path(prepared_dir) / "fit/train-interactions.tsv.gz", directory=False)
    return TrainingInputs(
        scores.prepared_identity,
        scores.protocol_identity,
        scores.catalog_descriptor,
        scores.fit_identity,
        scores.train_interactions_descriptor,
        scores.train_stats_descriptor,
        scores.mal_ids,
        scores.catalog,
        scores.train_stats,
        path,
    )


def iter_training_rows(inputs: TrainingInputs) -> Iterator[tuple[str, int, int]]:
    """Validate every train row and compressed/plain stream descriptor through EOF."""
    lines = iter(stream_tsv(inputs.train_interactions_path, inputs.train_interactions_descriptor))
    if next(lines, None) != HEADER:
        fail("research_model_invalid")
    ids = set(inputs.mal_ids)
    previous: tuple[str, int] | None = None
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
