"""Derive an ItemKNN bundle from a verified GloVe bundle and the pinned rating interactions."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
from array import array
from pathlib import Path
from typing import TYPE_CHECKING

from recsys.contracts.bundle import ContractError, canonical_bytes, check_bundle_bytes, require
from recsys.release.export import _publish

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

NAME = "itemknn-shrunk-cosine"
MAX_NEIGHBORS = 5
POSITIVE_MIN_RATING = 8
SHRINKAGE = 10
USER_CHUNK = 16384
HEADER = b"dataset_user_key\tmal_id\trating\twatch_status\n"
ALGORITHM = {
    "name": NAME,
    "version": "arb069-v1",
    "max_neighbors": MAX_NEIGHBORS,
    "similarity": "cosine",
    "parameters": {
        "fallback": "glove-mean-cosine-when-no-itemknn-neighbor",
        "positive_min_rating": POSITIVE_MIN_RATING,
        "positive_status": "completed",
        "shrinkage": SHRINKAGE,
        "tie_break": "numeric_MAL_ID_ascending",
        "users": "all-dataset-users",
        "weight": "c/sqrt(p_i*p_j)*c/(c+shrinkage)",
    },
}
# Identity of the bundle derived from the pinned current release; a rebuild must reproduce it.
CURRENT_IDENTITY = "sha256:a9f4cf27520460a10d9dc2ec18d3d0e556730fbfb61f6e60f666733ce166e2a2"


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _object(value: object, location: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ContractError(f"{location}: expected object")
    return {str(key): item for key, item in value.items()}


def _field(value: dict[str, object], key: str, location: str) -> object:
    if key not in value:
        raise ContractError(f"{location}: missing {key}")
    return value[key]


def positive_matrix(
    raw_gz: bytes, columns: dict[int, int], expected_rows: int
) -> tuple[NDArray[np.int64], NDArray[np.int32]]:
    """Return per-user CSR pointers and item columns of completed ratings >= 8."""
    import numpy as np

    indptr = array("q", [0])
    indices = array("i")
    rows = 0
    previous = b""
    with gzip.GzipFile(fileobj=io.BytesIO(raw_gz)) as stream:
        require(stream.readline() == HEADER, "interactions.tsv.gz", "unexpected header")
        for line in stream:
            rows += 1
            user, mal_id, rating, status = line.rstrip(b"\n").split(b"\t")
            if user != previous:
                require(user > previous, "interactions.tsv.gz", "rows must be sorted by user")
                if len(indices) > indptr[-1]:
                    indptr.append(len(indices))
                previous = user
            if status != b"completed" or not rating or int(rating) < POSITIVE_MIN_RATING:
                continue
            column = columns.get(int(mal_id))
            if column is None:
                raise ContractError("interactions.tsv.gz: MAL ID missing from catalog")
            indices.append(column)
    require(rows == expected_rows, "interactions.tsv.gz", "row count mismatch")
    if len(indices) > indptr[-1]:
        indptr.append(len(indices))
    return np.frombuffer(indptr, dtype=np.int64), np.frombuffer(indices, dtype=np.int32)


def top_neighbors(
    indptr: NDArray[np.int64], indices: NDArray[np.int32], items: int, limit: int = MAX_NEIGHBORS
) -> list[list[tuple[int, float]]]:
    """Return each item's strongest (column, similarity) pairs, ordered like the research model."""
    import numpy as np

    users = len(indptr) - 1
    lengths = np.diff(indptr)
    positive = np.bincount(indices, minlength=items)
    entry_users = np.repeat(np.arange(users, dtype=np.int64), lengths)
    by_item = entry_users[np.argsort(indices, kind="stable")]
    item_ptr = np.concatenate(([0], np.cumsum(positive)))
    result: list[list[tuple[int, float]]] = []
    for column in range(items):
        liked = by_item[item_ptr[column] : item_ptr[column + 1]]
        cooccurrence = np.zeros(items, dtype=np.int64)
        for start in range(0, len(liked), USER_CHUNK):
            chunk = liked[start : start + USER_CHUNK]
            starts, sizes = indptr[chunk], lengths[chunk]
            offsets = np.repeat(starts - (np.cumsum(sizes) - sizes), sizes)
            gathered = indices[offsets + np.arange(int(sizes.sum()))]
            cooccurrence += np.bincount(gathered, minlength=items)
        cooccurrence[column] = 0
        candidate = np.flatnonzero(cooccurrence)
        if not candidate.size:
            result.append([])
            continue
        shared = cooccurrence[candidate].astype(np.float64)
        similarity = (
            shared
            / np.sqrt(float(positive[column]) * positive[candidate])
            * shared
            / (shared + SHRINKAGE)
        )
        chosen = np.lexsort((candidate, -similarity))[:limit]
        result.append([(int(candidate[i]), float(similarity[i])) for i in chosen])
    return result


def derive_itemknn_bundle(
    base_bundle: Path,
    dataset_dir: Path,
    store: Path,
    *,
    expected_base_identity: str,
    expected_dataset_identity: str,
    expected_identity: str = "",
) -> tuple[Path, str, int]:
    """Publish the derived bundle and return (path, identity, catalog record count).

    A nonempty ``expected_identity`` must match before anything is published.
    """
    base = {name: (base_bundle / name).read_bytes() for name in ("catalog.json", "neighbors.json")}
    base_manifest = (base_bundle / "manifest.json").read_bytes()
    base_identity, count = check_bundle_bytes(
        base["catalog.json"], base["neighbors.json"], base_manifest
    )
    require(base_identity == expected_base_identity, str(base_bundle), "unexpected base bundle")
    manifest_raw = (dataset_dir / "dataset-manifest.json").read_bytes()
    require(
        "sha256:" + _sha(manifest_raw) == expected_dataset_identity,
        str(dataset_dir),
        "unexpected dataset",
    )
    dataset = _object(json.loads(manifest_raw), "dataset-manifest.json")
    files = _object(_field(dataset, "files", "dataset-manifest.json"), "files")
    location = "files.interactions.tsv.gz"
    descriptor = _object(_field(files, "interactions.tsv.gz", "files"), location)
    raw_gz = (dataset_dir / "interactions.tsv.gz").read_bytes()
    require(
        len(raw_gz) == _field(descriptor, "bytes", location)
        and _sha(raw_gz) == _field(descriptor, "sha256", location),
        "interactions.tsv.gz",
        "bytes differ from the dataset manifest",
    )

    catalog = _object(json.loads(base["catalog.json"]), "catalog.json")
    glove = _object(_object(json.loads(base["neighbors.json"]), "neighbors.json")["neighbors"], "n")
    ids = sorted(int(key) for key in _object(catalog["anime"], "catalog.json.anime"))
    rows = _field(descriptor, "rows", location)
    if not isinstance(rows, int):
        raise ContractError("files.interactions.tsv.gz.rows: expected integer")
    columns = {mal_id: column for column, mal_id in enumerate(ids)}
    indptr, indices = positive_matrix(raw_gz, columns, rows)
    fitted = top_neighbors(indptr, indices, len(ids))

    neighbors: dict[str, object] = {}
    itemknn_items = fallback_items = 0
    for mal_id, pairs in zip(ids, fitted, strict=True):
        if pairs:
            itemknn_items += 1
            neighbors[str(mal_id)] = [
                {"mal_id": ids[column], "similarity": weight} for column, weight in pairs
            ]
        else:
            fallback_items += bool(glove[str(mal_id)])
            neighbors[str(mal_id)] = glove[str(mal_id)]
    neighbors_raw = canonical_bytes({"schema_version": 1, "neighbors": neighbors})
    sources = [
        {
            "name": "glove_bundle",
            "version": "1",
            "sha256": base_identity.removeprefix("sha256:"),
            "metadata": {"identity": base_identity, "role": "catalog-and-fallback-neighbors"},
        },
        {
            "name": "rating_interactions",
            "version": "1",
            "sha256": expected_dataset_identity.removeprefix("sha256:"),
            "metadata": {
                "identity": expected_dataset_identity,
                "interactions_sha256": descriptor["sha256"],
                "positive_ratings": len(indices),
                "users_with_positive": len(indptr) - 1,
                "itemknn_items": itemknn_items,
                "fallback_items": fallback_items,
                "empty_items": len(ids) - itemknn_items - fallback_items,
            },
        },
    ]
    manifest = canonical_bytes(
        {
            "schema_version": 1,
            "algorithm": ALGORITHM,
            "sources": sources,
            "files": {
                "catalog.json": {"sha256": _sha(base["catalog.json"])},
                "neighbors.json": {"sha256": _sha(neighbors_raw)},
            },
        }
    )
    output = {
        "catalog.json": base["catalog.json"],
        "neighbors.json": neighbors_raw,
        "manifest.json": manifest,
    }
    identity, published_count = check_bundle_bytes(output["catalog.json"], neighbors_raw, manifest)
    require(published_count == count, "catalog.json", "record count changed")
    require(
        not expected_identity or identity == expected_identity,
        identity,
        "derived bundle differs from the pinned identity",
    )
    from recsys.release.prepare import _published_permissions

    published = _publish(store, output, identity, count)
    # The bot and prepare containers read the bundle as another UID.
    _published_permissions(published)
    return published, identity, count


def derive_current(data_dir: Path) -> tuple[Path, str, int]:
    """Derive from the pinned current release inside a prepared data directory."""
    from recsys.release.lock import release_info

    lock = release_info("current")
    base_identity = lock["reference"]["bundle_identity"]
    dataset_identity = lock["inputs"]["base"]["identity"]
    base_bundle = data_dir / "bundles" / base_identity.replace(":", "-")
    if not base_bundle.is_dir():
        raise ContractError(f"{base_bundle}: run `recsys release prepare` first")
    return derive_itemknn_bundle(
        base_bundle,
        data_dir / "datasets" / dataset_identity.replace(":", "-"),
        data_dir / "bundles",
        expected_base_identity=base_identity,
        expected_dataset_identity=dataset_identity,
        expected_identity=CURRENT_IDENTITY,
    )
