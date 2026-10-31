"""Derive an ItemKNN bundle from a verified GloVe bundle and the pinned rating interactions."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
from array import array
from collections.abc import Sequence
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
# AniList relation types that keep two anime in one franchise; spin-offs and crossovers do not.
LINKING = frozenset(
    {
        "ALTERNATIVE",
        "COMPILATION",
        "CONTAINS",
        "PARENT",
        "PREQUEL",
        "SEQUEL",
        "SIDE_STORY",
        "SUMMARY",
    }
)
SERIES = frozenset({"TV", "TV_SHORT"})
MAIN_ENTRIES = frozenset({"MOVIE", "ONA", "OVA"})
ALGORITHM = {
    "name": NAME,
    "version": "arb069-v2",
    "max_neighbors": MAX_NEIGHBORS,
    "similarity": "cosine",
    "parameters": {
        "fallback": "glove-mean-cosine-when-no-itemknn-neighbor",
        "franchise": "one-card-per-franchise-shown-as-first-entry-seed-franchise-excluded",
        "franchise_relations": sorted(LINKING),
        "franchise_crossovers": "parents-in-several-franchises-stay-alone",
        "franchise_main_formats": sorted(MAIN_ENTRIES),
        "franchise_popularity": "list-entries-except-plan_to_watch",
        "franchise_representative": (
            "most-watched-series-without-earlier-series-prequel"
            "-else-earliest-watched-main-format-else-earliest-watched"
        ),
        "franchise_series_formats": sorted(SERIES),
        "positive_min_rating": POSITIVE_MIN_RATING,
        "positive_status": "completed",
        "shrinkage": SHRINKAGE,
        "tie_break": "numeric_MAL_ID_ascending",
        "users": "all-dataset-users",
        "weight": "c/sqrt(p_i*p_j)*c/(c+shrinkage)",
    },
}
RELATIONS_SNAPSHOT = "anilist-relations-2026-v1"
RELATIONS_SHA256 = "e4a42d0618446cc02ef95076ea1bb1309e39f778070d9eb93205260184389348"
# Identity of the bundle derived from the pinned current release; a rebuild must reproduce it.
CURRENT_IDENTITY = "sha256:6eec1413efcfdb129cbb51618917972efd01392ba815534e81bbe5fb5336cbf2"


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
) -> tuple[NDArray[np.int64], NDArray[np.int32], list[int]]:
    """Return per-user CSR pointers and item columns of completed ratings >= 8, and watchers.

    Watchers count every list entry except plan_to_watch. MAL does not allow «completed» for
    an airing series, so positives alone miss titles such as One Piece.
    """
    import numpy as np

    indptr = array("q", [0])
    indices = array("i")
    watchers = [0] * len(columns)
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
            if status == b"plan_to_watch":
                continue
            column = columns.get(int(mal_id))
            if column is None:
                raise ContractError("interactions.tsv.gz: MAL ID missing from catalog")
            watchers[column] += 1
            if status != b"completed" or not rating or int(rating) < POSITIVE_MIN_RATING:
                continue
            indices.append(column)
    require(rows == expected_rows, "interactions.tsv.gz", "row count mismatch")
    if len(indices) > indptr[-1]:
        indptr.append(len(indices))
    return (
        np.frombuffer(indptr, dtype=np.int64),
        np.frombuffer(indices, dtype=np.int32),
        watchers,
    )


def _start_key(record: dict[str, object]) -> tuple[int, int, int]:
    parts = record.get("start_date")
    values = parts if isinstance(parts, list) and len(parts) == 3 else [None, None, None]
    year, month, day = (
        value if isinstance(value, int) and not isinstance(value, bool) else unknown
        for value, unknown in zip(values, (9999, 99, 99), strict=True)
    )
    return year, month, day


def franchise_groups(
    raw_gz: bytes, ids: list[int], watchers: Sequence[int] | None = None
) -> tuple[list[int], list[int], dict[str, int]]:
    """Group catalog columns into franchises and pick the card that stands for each one.

    Returns the group of every column, the representative column of every group and counts.
    Titles without a usable relation stay in a group of their own. The representative is the
    most watched first season: a TV series with no earlier TV prequel in the franchise, so a
    remake such as Fullmetal Alchemist: Brotherhood wins over the older series. Franchises
    without a watched TV series show their earliest watched movie, ONA or OVA, then their
    earliest watched entry, so a prologue special does not stand for the film.
    """
    columns = {mal_id: column for column, mal_id in enumerate(ids)}
    parent = list(range(len(ids)))

    def root(column: int) -> int:
        while parent[column] != column:
            parent[column] = parent[parent[column]]
            column = parent[column]
        return column

    def union(a: int, b: int) -> None:
        a, b = root(a), root(b)
        if a != b:
            parent[max(a, b)] = min(a, b)

    records: dict[int, dict[str, object]] = {}
    edges: list[tuple[int, int, str]] = []
    for line in gzip.decompress(raw_gz).splitlines():
        record = _object(json.loads(line), "relations.jsonl")
        mal_id = record.get("mal_id")
        if not isinstance(mal_id, int) or mal_id not in columns:
            raise ContractError("relations.jsonl: MAL ID missing from catalog")
        records[columns[mal_id]] = record
        relations = record.get("relations")
        if not isinstance(relations, list):
            raise ContractError("relations.jsonl: expected relations array")
        for relation in relations:
            if not isinstance(relation, list) or len(relation) != 2:
                raise ContractError("relations.jsonl: expected [type, MAL ID] pairs")
            kind, target = relation
            if isinstance(kind, str) and isinstance(target, int) and target in columns:
                edges.append((columns[mal_id], columns[target], kind))

    parents: dict[int, set[int]] = {}
    for source, target, kind in edges:
        if kind == "PARENT":
            parents.setdefault(source, set()).add(target)
    # A title with several parents joins only if they turn out to share one franchise; a
    # crossover such as Lupin III vs. Detective Conan must not merge two franchises.
    pending = {column for column, found in parents.items() if len(found) > 1}
    for source, target, kind in edges:
        if kind in LINKING and source not in pending and target not in pending:
            union(source, target)
    crossovers = 0
    for column in sorted(pending):
        roots = {root(target) for target in parents[column]}
        if len(roots) == 1:
            union(column, roots.pop())
        else:
            crossovers += 1

    formats = [records.get(column, {}).get("format") for column in range(len(ids))]
    starts = [_start_key(records.get(column, {})) for column in range(len(ids))]
    # AniList orders PREQUEL by story, not release: Nekomonogatari: Kuro (2012) is a prequel of
    # Bakemonogatari (2009). Only a TV prequel released earlier makes a season a continuation.
    continued: set[int] = set()
    for source, target, kind in edges:
        later, earlier = (source, target) if kind == "PREQUEL" else (target, source)
        if (
            kind in ("PREQUEL", "SEQUEL")
            and formats[later] in SERIES
            and formats[earlier] in SERIES
            and starts[earlier] < starts[later]
            and root(later) == root(earlier)
        ):
            continued.add(later)
    # Without watcher counts every title counts as watched.
    watched = watchers if watchers is not None else [1] * len(ids)

    def preference(column: int) -> tuple[int, int, tuple[int, int, int], int]:
        if not watched[column]:
            # An announced or never listed title stands for its franchise only as a last resort.
            rank = 4
        elif formats[column] in SERIES:
            rank = 1 if column in continued else 0
        else:
            rank = 2 if formats[column] in MAIN_ENTRIES else 3
        # Watchers decide among TV seasons only; other entries keep release order.
        weight = -watched[column] if rank < 2 else 0
        return (rank, weight, starts[column], ids[column])

    members: dict[int, list[int]] = {}
    for column in range(len(ids)):
        members.setdefault(root(column), []).append(column)
    groups = [0] * len(ids)
    representatives: list[int] = []
    for index, cluster in enumerate(sorted(members.values())):
        for column in cluster:
            groups[column] = index
        representatives.append(min(cluster, key=preference))
    sizes = [len(cluster) for cluster in members.values()]
    stats = {
        "titles_with_relations": len(records),
        "franchises": sum(size > 1 for size in sizes),
        "grouped_titles": sum(size for size in sizes if size > 1),
        "largest_franchise": max(sizes),
        "crossovers": crossovers,
    }
    return groups, representatives, stats


def top_neighbors(
    indptr: NDArray[np.int64],
    indices: NDArray[np.int32],
    items: int,
    limit: int = MAX_NEIGHBORS,
    groups: Sequence[int] | None = None,
) -> list[list[tuple[int, float]]]:
    """Return each item's strongest (column, similarity) pairs, ordered like the research model.

    With ``groups``, the item's own group is skipped and only the strongest member of each
    other group is kept.
    """
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
        order = np.lexsort((candidate, -similarity))
        if groups is None:
            chosen = order[:limit].tolist()
        else:
            seen = {groups[column]}
            chosen = []
            # Iterate lazily: the loop usually stops after a few groups.
            for i in order:
                group = groups[int(candidate[i])]
                if group not in seen:
                    seen.add(group)
                    chosen.append(int(i))
                    if len(chosen) == limit:
                        break
        result.append([(int(candidate[i]), float(similarity[i])) for i in chosen])
    return result


def _as_first_entries(
    pairs: list[tuple[int, float]], ids: list[int], groups: list[int], representatives: list[int]
) -> list[dict[str, object]]:
    shown = [(ids[representatives[groups[column]]], weight) for column, weight in pairs]
    return [
        {"mal_id": mal_id, "similarity": weight}
        for mal_id, weight in sorted(shown, key=lambda pair: (-pair[1], pair[0]))
    ]


def derive_itemknn_bundle(
    base_bundle: Path,
    dataset_dir: Path,
    store: Path,
    *,
    expected_base_identity: str,
    expected_dataset_identity: str,
    expected_identity: str = "",
    relations: Path | None = None,
    expected_relations_sha256: str = "",
) -> tuple[Path, str, int]:
    """Publish the derived bundle and return (path, identity, catalog record count).

    A nonempty ``expected_identity`` must match before anything is published. ``relations``
    is an AniList relations snapshot; without it every title is its own franchise.
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
    relations_raw = b""
    if relations is not None:
        relations_raw = (relations / "relations.jsonl.gz").read_bytes()
        require(
            _sha(relations_raw) == expected_relations_sha256,
            str(relations),
            "unexpected relations snapshot",
        )
    indptr, indices, watchers = positive_matrix(raw_gz, columns, rows)
    if relations is None:
        groups, representatives = list(range(len(ids))), list(range(len(ids)))
        franchise_stats: dict[str, int] = {}
    else:
        groups, representatives, franchise_stats = franchise_groups(relations_raw, ids, watchers)
    fitted = top_neighbors(indptr, indices, len(ids), groups=groups)

    neighbors: dict[str, object] = {}
    itemknn_items = fallback_items = 0
    for column, (mal_id, pairs) in enumerate(zip(ids, fitted, strict=True)):
        if not pairs:
            seen = {groups[column]}
            for entry in _glove_list(glove[str(mal_id)]):
                target = columns[entry[0]]
                if groups[target] not in seen and len(pairs) < MAX_NEIGHBORS:
                    seen.add(groups[target])
                    pairs.append((target, entry[1]))
            fallback_items += bool(pairs)
        else:
            itemknn_items += 1
        neighbors[str(mal_id)] = _as_first_entries(pairs, ids, groups, representatives)
    neighbors_raw = canonical_bytes({"schema_version": 1, "neighbors": neighbors})
    sources: list[dict[str, object]] = []
    if relations is not None:
        sources.append(
            {
                "name": "anilist_relations",
                "version": "1",
                "sha256": _sha(relations_raw),
                "metadata": {"snapshot_id": RELATIONS_SNAPSHOT, **franchise_stats},
            }
        )
    sources += [
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


def _glove_list(value: object) -> list[tuple[int, int | float]]:
    if not isinstance(value, list):
        raise ContractError("neighbors.json: expected neighbor list")
    pairs = []
    for entry in value:
        entry = _object(entry, "neighbors.json")
        mal_id, weight = entry.get("mal_id"), entry.get("similarity")
        if not isinstance(mal_id, int) or not isinstance(weight, int | float):
            raise ContractError("neighbors.json: invalid neighbor")
        pairs.append((mal_id, weight))
    return pairs


def derive_current(data_dir: Path) -> tuple[Path, str, int]:
    """Derive from the pinned current release inside a prepared data directory."""
    from recsys.release.lock import release_info

    lock = release_info("current")
    base_identity = lock["reference"]["bundle_identity"]
    dataset_identity = lock["inputs"]["base"]["identity"]
    base_bundle = data_dir / "bundles" / base_identity.replace(":", "-")
    if not base_bundle.is_dir():
        raise ContractError(f"{base_bundle}: run `recsys release prepare` first")
    relations = data_dir / "raw" / RELATIONS_SNAPSHOT
    if not (relations / "relations.jsonl.gz").is_file():
        raise ContractError(f"{relations}: restore the AniList relations snapshot from DVC")
    return derive_itemknn_bundle(
        base_bundle,
        data_dir / "datasets" / dataset_identity.replace(":", "-"),
        data_dir / "bundles",
        expected_base_identity=base_identity,
        expected_dataset_identity=dataset_identity,
        expected_identity=CURRENT_IDENTITY,
        relations=relations,
        expected_relations_sha256=RELATIONS_SHA256,
    )
