"""Stream a pinned dataset into private ML-01 fit, queries, and hidden labels."""

import heapq
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TypedDict, cast

from recsys.contracts.bundle import Catalog, canonical_bytes, check_catalog
from recsys.data.dataset import HEADER, DatasetSpec, validate_dataset
from recsys.data.metadata import MetadataFact, MetadataFacts
from recsys.release.lock import ReleaseLock, release_info
from recsys.research.artifacts import (
    FileDescriptor,
    StreamDescriptor,
    StreamWriter,
    _stream_lines,
    check_descriptor,
    check_identity,
    cleanup,
    descriptor,
    open_stage,
    producer,
    publish,
    read_json,
    safe_path,
    separate,
    stream_jsonl,
    stream_tsv,
    validate_preparation_producer,
    write_json,
)
from recsys.research.protocol import (
    ProtocolSpec,
    exact,
    fail,
    load_protocol,
    order_hash,
    protocol_identity,
    sample_hash,
    sha,
    split_user,
)

type InteractionRow = tuple[str, int, int, str, bytes]
type Profile = list[InteractionRow]
type HeapEntry = tuple[int, int, str]
type Heaps = dict[tuple[str, str], list[HeapEntry]]
type UserPayload = tuple[
    str, tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...]
]


class PreparationManifest(TypedDict):
    schema_version: int
    kind: str
    protocol_identity: str
    release_lock_sha256: str
    release_id: str
    dataset_identity: str
    catalog_release_identity: str
    catalog_sha256: str
    producer: dict[str, object]
    files: dict[str, FileDescriptor]
    counts: dict[str, int]


class PreparationLabel(TypedDict):
    cohort: str
    user_key: str
    split: str
    positive_ids: list[int]
    target_ids: list[int]
    history_ids: list[int]
    completed_ids: list[int]
    history_count: int
    query_id: str
    order: str
    n: int
    seed_mal_ids: list[int]
    reserved_seed_ids: list[int]
    recent_2024_seed_count: int
    positive_ratings: list[list[int]]


class PreparationTrainItem(TypedDict):
    completed_users: int
    positive_users: int
    rating_sum: int
    rating_count: int
    band: str


class PreparationTrainStats(TypedDict):
    schema_version: int
    train_users: int
    train_completed_rows: int
    items: dict[str, PreparationTrainItem]


FILES = (
    "protocol.json",
    "model-catalog.json",
    "fit/manifest.json",
    "fit/train-interactions.tsv.gz",
    "fit/train-stats.json",
    "development/queries.jsonl.gz",
    "evaluator/splits.tsv.gz",
    "evaluator/validation-labels.jsonl.gz",
    "evaluator/test-labels.jsonl.gz",
)


def _checked_preparation_manifest(value: object) -> PreparationManifest:
    """Establish the wire shape after the caller has checked producer provenance."""
    exact(
        value,
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
    raw = cast("dict[str, object]", value)
    if (
        type(raw["schema_version"]) is not int
        or any(
            type(raw[key]) is not str
            for key in (
                "kind",
                "protocol_identity",
                "release_lock_sha256",
                "release_id",
                "dataset_identity",
                "catalog_release_identity",
                "catalog_sha256",
            )
        )
        or type(raw["producer"]) is not dict
        or type(raw["files"]) is not dict
        or type(raw["counts"]) is not dict
    ):
        fail()
    files = cast("dict[str, object]", raw["files"])
    counts = cast("dict[str, object]", raw["counts"])
    if set(files) != set(FILES) or any(
        type(key) is not str or type(item) is not int for key, item in counts.items()
    ):
        fail()
    for name, entry in files.items():
        check_descriptor(entry, stream=name.endswith(".gz"))
    return cast("PreparationManifest", raw)


def _release(
    release_dir: str | Path,
    dataset_dir: str | Path,
    *,
    _lock_bytes: bytes | None = None,
) -> tuple[ReleaseLock, Catalog, dict[str, object], bytes]:
    lock = release_info(_lock_bytes=_lock_bytes)
    release_dir = safe_path(release_dir, directory=True)
    if {path.name for path in release_dir.iterdir()} != {"catalog.json", "release-manifest.json"}:
        fail()
    manifest_value, raw = read_json(release_dir / "release-manifest.json")
    catalog_value, catalog_raw = read_json(release_dir / "catalog.json")
    _, base_raw = read_json(Path(dataset_dir) / "dataset-manifest.json")
    if not isinstance(manifest_value, dict):
        fail("research_release_mismatch")
    manifest_inputs = manifest_value.get("inputs")
    if not isinstance(manifest_inputs, dict):
        fail("research_release_mismatch")
    base_input = manifest_inputs.get("base")
    alias_input = manifest_inputs.get("aliases")
    if not isinstance(base_input, dict) or not isinstance(alias_input, dict):
        fail("research_release_mismatch")
    counts = manifest_value.get("counts")
    if not isinstance(counts, dict):
        fail("research_release_mismatch")
    manifest = cast("dict[str, object]", manifest_value)
    locked = lock["inputs"]
    if (
        "sha256:" + sha(raw) != locked["catalog_release"]["identity"]
        or "sha256:" + sha(base_raw) != locked["base"]["identity"]
        or manifest.get("kind") != "arb-catalog-release"
        or manifest.get("policy") != "ordered-alias-overlay-v1"
        or base_input.get("identity") != locked["base"]["identity"]
        or alias_input.get("identity") != locked["aliases"]["identity"]
        or manifest.get("files") != {"catalog.json": descriptor(catalog_raw)}
        or sha(catalog_raw) != locked["catalog_release"]["catalog_sha256"]
        or counts.get("catalog_count") != locked["catalog_release"]["catalog_count"]
    ):
        fail("research_release_mismatch")
    keys = check_catalog(catalog_value)
    if len(keys) != locked["catalog_release"]["catalog_count"]:
        fail("research_release_mismatch")
    return lock, cast("Catalog", catalog_value), manifest, raw


def _rows(dataset_dir: str | Path) -> Iterator[InteractionRow]:
    path = safe_path(Path(dataset_dir) / "interactions.tsv.gz", directory=False)
    lines = iter(_stream_lines(path))
    if next(lines, None) != HEADER:
        fail()
    last_key: str | None = None
    last_id = 0
    for raw in lines:
        fields = raw.rstrip(b"\n").split(b"\t")
        if len(fields) != 4:
            fail()
        try:
            user_raw, item_raw, rating_raw, status_raw = fields
            user = user_raw.decode("ascii")
            item = int(item_raw)
            rating = int(rating_raw)
            status = status_raw.decode("ascii")
        except (ValueError, UnicodeError):
            fail()
        if user != last_key:
            if last_key is not None and user <= last_key:
                fail()
            last_key, last_id = user, 0
        if item <= last_id or not 1 <= rating <= 10:
            fail()
        last_id = item
        yield user, item, rating, status, raw


def _profiles(rows: Iterable[InteractionRow]) -> Iterator[tuple[str, Profile]]:
    user: str | None = None
    current: Profile = []
    for row in rows:
        if row[0] != user:
            if user is not None:
                yield user, current
            user, current = row[0], []
        current.append(row)
    if user is not None:
        yield user, current


def _cohort(positives: set[int] | dict[int, int] | list[int], protocol: ProtocolSpec) -> str | None:
    count = len(positives)
    if count >= protocol["cohorts"]["main_min_positives"]:
        return "main"
    if (
        protocol["cohorts"]["short_min_positives"]
        <= count
        <= protocol["cohorts"]["short_max_positives"]
    ):
        return "short"
    return None


def _select(heaps: Heaps, split: str, cohort: str, user: str, protocol: ProtocolSpec) -> None:
    limit = protocol["query"][
        "sample_main_per_split" if cohort == "main" else "sample_short_per_split"
    ]
    key = sample_hash(protocol["query"]["sample_seed"], split, cohort, user)
    # The integer user key resolves theoretical hash collisions independently of input order.
    rank = (int.from_bytes(key, "big"), int(user[3:], 16))
    entry = (-rank[0], -rank[1], user)
    heap = heaps[split, cohort]
    if len(heap) < limit:
        heapq.heappush(heap, entry)
    elif entry > heap[0]:
        heapq.heapreplace(heap, entry)


def _query_id(user: str, cohort: str, order: str, n: int) -> str:
    raw = (
        b"arb-ml-query-v1\0"
        + user.encode()
        + b"\0"
        + cohort.encode()
        + b"\0"
        + order.encode()
        + b"\0"
        + str(n).encode()
    )
    return "q1_" + sha(raw)


def _emit_user(
    user: str,
    profile: Profile,
    split: str,
    cohort: str,
    queries: StreamWriter | None,
    labels: StreamWriter,
    protocol: ProtocolSpec,
    catalog_ids: set[int],
    facts: dict[str, MetadataFact],
) -> None:
    positives = {
        item: rating
        for _, item, rating, status, _ in profile
        if status == "completed" and rating >= 8
    }
    history = sorted({row[1] for row in profile})
    completed = sorted({row[1] for row in profile if row[3] == "completed"})
    if _cohort(positives, protocol) != cohort or not set(history) <= catalog_ids:
        fail()
    count = len(positives)
    target_count = (
        protocol["query"]["main_targets"] if cohort == "main" else min(5, max(1, count // 3))
    )
    targets = sorted(
        positives, key=lambda item: (order_hash(protocol["query"]["target_seed"], user, item), item)
    )[:target_count]
    target_set = set(targets)
    pool = set(positives) - target_set
    nmax = min(protocol["query"]["max_n"], len(pool))
    if cohort == "main" and nmax != 10:
        fail()
    common = {
        "cohort": cohort,
        "user_key": user,
        "split": split,
        "positive_ids": sorted(positives),
        "positive_ratings": [[item, positives[item]] for item in sorted(positives)],
        "target_ids": sorted(targets),
        "history_ids": history,
        "completed_ids": completed,
        "history_count": len(history),
    }
    plans: list[tuple[str, list[int]]] = [("popularity", [])]
    for name, seed in zip(protocol["query"]["orders"][:3], protocol["query"]["order_seeds"]):
        plans.append(
            (name, sorted(pool, key=lambda item: (order_hash(seed, user, item), item))[:nmax])
        )
    plans.append(
        (
            "top-rated",
            sorted(
                pool,
                key=lambda item: (
                    -positives[item],
                    order_hash(protocol["query"]["toprated_tie_seed"], user, item),
                    item,
                ),
            )[:nmax],
        )
    )
    for order, reserved in plans:
        ns = (0,) if order == "popularity" else range(1, nmax + 1)
        for n in ns:
            seeds = reserved[:n]
            query_id = _query_id(user, cohort, order, n)
            query = {
                "query_id": query_id,
                "cohort": cohort,
                "seed_policy": "popularity" if n == 0 else "liked",
                "order": order,
                "n": n,
                "seed_mal_ids": seeds,
            }
            recent = sum(
                facts[str(item)]["airing_status"] in ("finished_airing", "currently_airing")
                and (first_date := facts[str(item)]["first_release_date"]) is not None
                and "2024-01-01" <= first_date <= "2025-10-31"
                for item in seeds
            )
            label = {
                **common,
                "query_id": query_id,
                "order": order,
                "n": n,
                "seed_mal_ids": seeds,
                "reserved_seed_ids": reserved,
                "recent_2024_seed_count": recent,
            }
            if queries is not None:
                queries.write_json(query)
            labels.write_json(label)


def prepare_research(
    dataset_dir: str | Path,
    release_dir: str | Path,
    output_dir: str | Path,
    *,
    code_revision: str,
    _dataset_spec: DatasetSpec | None = None,
    _lock_bytes: bytes | None = None,
    _protocol_bytes: bytes | None = None,
) -> Path:
    """Validate source, then two streaming passes; publish only complete immutable output."""
    dataset_dir, release_dir, output_dir = map(Path, (dataset_dir, release_dir, output_dir))
    separate(dataset_dir, release_dir, output_dir)
    protocol = load_protocol(_protocol_bytes=_protocol_bytes)
    lock, catalog, _, lock_release_raw = _release(release_dir, dataset_dir, _lock_bytes=_lock_bytes)
    summary = validate_dataset(
        dataset_dir, expected_identity=lock["inputs"]["base"]["identity"], _spec=_dataset_spec
    )
    catalog_ids = {int(item) for item in catalog["anime"]}
    facts_doc, _ = read_json(dataset_dir / "item-facts.json")
    if (
        not isinstance(facts_doc, dict)
        or set(facts_doc) != {"schema_version", "anime"}
        or type(facts_doc["schema_version"]) is not int
        or facts_doc["schema_version"] != 1
        or type(facts_doc["anime"]) is not dict
        or set(facts_doc["anime"]) != set(catalog["anime"])
    ):
        fail()
    for fact in facts_doc["anime"].values():
        exact(fact, {"first_release_date", "airing_status", "seasonal_year"})
        if (
            fact["first_release_date"] is not None
            and type(fact["first_release_date"]) is not str
            or type(fact["airing_status"]) is not str
            or fact["seasonal_year"] is not None
            and type(fact["seasonal_year"]) is not int
        ):
            fail()
    facts = cast("MetadataFacts", facts_doc)["anime"]
    model_catalog = {
        **catalog,
        "anime": {key: {**item, "score": None} for key, item in catalog["anime"].items()},
    }
    stage = open_stage(output_dir)
    try:
        for sub in ("fit", "development", "evaluator"):
            (stage / sub).mkdir(mode=0o700)
        files: dict[str, FileDescriptor] = {}
        files["protocol.json"] = write_json(stage / "protocol.json", protocol)
        files["model-catalog.json"] = write_json(stage / "model-catalog.json", model_catalog)
        train = StreamWriter(stage / "fit/train-interactions.tsv.gz")
        splits = StreamWriter(stage / "evaluator/splits.tsv.gz")
        train.write(HEADER)
        splits.write(b"dataset_user_key\tsplit\tcohort\n")
        stats: dict[str, PreparationTrainItem] = {
            str(item): {
                "completed_users": 0,
                "positive_users": 0,
                "rating_sum": 0,
                "rating_count": 0,
                "band": "cold",
            }
            for item in sorted(catalog_ids)
        }
        counters: Counter[str] = Counter()
        heaps: Heaps = {
            (split, cohort): [] for split in ("validation", "test") for cohort in ("main", "short")
        }
        for user, profile in _profiles(_rows(dataset_dir)):
            split = split_user(user, protocol)
            positives = {
                item
                for _, item, rating, status, _ in profile
                if status == "completed" and rating >= 8
            }
            cohort = _cohort(positives, protocol)
            counters[f"users_{split}"] += 1
            if cohort:
                counters[f"eligible_{split}_{cohort}"] += 1
            splits.write(f"{user}\t{split}\t{cohort or 'ineligible'}\n".encode())
            if split == "train":
                counters["train_users"] += 1
                for _, item, rating, status, raw in profile:
                    if item not in catalog_ids:
                        fail()
                    if status != "completed":
                        continue
                    train.write(raw)
                    counters["train_completed_rows"] += 1
                    stat = stats[str(item)]
                    stat["completed_users"] += 1
                    stat["rating_sum"] += rating
                    stat["rating_count"] += 1
                    if rating >= 8:
                        stat["positive_users"] += 1
            elif cohort:
                _select(heaps, split, cohort, user, protocol)
        files["fit/train-interactions.tsv.gz"] = train.close()
        files["evaluator/splits.tsv.gz"] = splits.close()
        nonzero = sorted(
            (item for item in catalog_ids if stats[str(item)]["positive_users"]),
            key=lambda item: (-stats[str(item)]["positive_users"], item),
        )
        head_count = (len(nonzero) + 4) // 5
        head = set(nonzero[:head_count])
        for item in catalog_ids:
            stat = stats[str(item)]
            stat["band"] = (
                "cold" if not stat["positive_users"] else "head" if item in head else "tail"
            )
        train_stats = {
            "schema_version": 1,
            "train_users": counters["train_users"],
            "train_completed_rows": counters["train_completed_rows"],
            "items": stats,
        }
        files["fit/train-stats.json"] = write_json(stage / "fit/train-stats.json", train_stats)
        selected = {item[2] for heap in heaps.values() for item in heap}
        qwriter = StreamWriter(stage / "development/queries.jsonl.gz")
        val = StreamWriter(stage / "evaluator/validation-labels.jsonl.gz")
        test = StreamWriter(stage / "evaluator/test-labels.jsonl.gz")
        seen: set[str] = set()
        for user, profile in _profiles(_rows(dataset_dir)):
            if user not in selected:
                continue
            split = split_user(user, protocol)
            positives = {
                item
                for _, item, rating, status, _ in profile
                if status == "completed" and rating >= 8
            }
            cohort = _cohort(positives, protocol)
            if split not in ("validation", "test") or cohort is None:
                fail()
            _emit_user(
                user,
                profile,
                split,
                cohort,
                qwriter if split == "validation" else None,
                val if split == "validation" else test,
                protocol,
                catalog_ids,
                facts,
            )
            seen.add(user)
        if seen != selected:
            fail()
        files["development/queries.jsonl.gz"] = qwriter.close()
        files["evaluator/validation-labels.jsonl.gz"] = val.close()
        files["evaluator/test-labels.jsonl.gz"] = test.close()
        fit_manifest = {
            "schema_version": 1,
            "kind": "arb-ml-fit",
            "dataset_identity": summary.identity,
            "protocol_identity": protocol_identity(_protocol_bytes=_protocol_bytes),
            "train_interactions": files["fit/train-interactions.tsv.gz"],
            "train_stats": files["fit/train-stats.json"],
        }
        files["fit/manifest.json"] = write_json(stage / "fit/manifest.json", fit_manifest)
        manifest = {
            "schema_version": 2,
            "kind": "arb-ml-preparation",
            "protocol_identity": protocol_identity(_protocol_bytes=_protocol_bytes),
            "release_lock_sha256": sha(canonical_bytes(lock)),
            "release_id": lock["release_id"],
            "dataset_identity": summary.identity,
            "catalog_release_identity": lock["inputs"]["catalog_release"]["identity"],
            "catalog_sha256": lock["inputs"]["catalog_release"]["catalog_sha256"],
            "producer": producer(code_revision),
            "files": files,
            "counts": {**dict(sorted(counters.items())), "sampled_users": len(selected)},
        }
        write_json(stage / "prepare-manifest.json", manifest)
        publish(stage, output_dir)
        return output_dir / "prepare-manifest.json"
    except Exception:
        cleanup(stage)
        raise


def _id_list(values: object, ids: set[str], *, ordered: bool = False) -> None:
    if type(values) is not list or any(
        type(item) is not int or str(item) not in ids for item in values
    ):
        fail()
    values = cast("list[int]", values)
    if len(values) != len(set(values)) or (ordered and values != sorted(values)):
        fail()


def _check_label(label: object, protocol: ProtocolSpec, ids: set[str]) -> PreparationLabel:
    exact(
        label,
        {
            "cohort",
            "user_key",
            "split",
            "positive_ids",
            "target_ids",
            "history_ids",
            "completed_ids",
            "history_count",
            "query_id",
            "order",
            "n",
            "seed_mal_ids",
            "reserved_seed_ids",
            "recent_2024_seed_count",
            "positive_ratings",
        },
    )
    raw = cast("dict[str, object]", label)
    if any(
        type(raw[key]) is not str for key in ("cohort", "user_key", "split", "query_id", "order")
    ) or any(type(raw[key]) is not int for key in ("history_count", "n", "recent_2024_seed_count")):
        fail()
    if (
        raw["split"] != split_user(cast("str", raw["user_key"]), protocol)
        or raw["split"] == "train"
    ):
        fail()
    for key in (
        "positive_ids",
        "target_ids",
        "history_ids",
        "completed_ids",
        "seed_mal_ids",
        "reserved_seed_ids",
    ):
        _id_list(raw[key], ids, ordered=key in ("positive_ids", "history_ids", "completed_ids"))
    if type(raw["positive_ratings"]) is not list or any(
        type(pair) is not list or len(pair) != 2 or any(type(value) is not int for value in pair)
        for pair in cast("list[object]", raw["positive_ratings"])
    ):
        fail()
    label = cast("PreparationLabel", raw)
    positives = label["positive_ids"]
    rated = label["positive_ratings"]
    if (
        type(rated) is not list
        or len(rated) != len(positives)
        or any(
            type(pair) is not list
            or len(pair) != 2
            or type(pair[0]) is not int
            or pair[0] != item
            or type(pair[1]) is not int
            or not 8 <= pair[1] <= 10
            for pair, item in zip(rated, positives)
        )
    ):
        fail()
    ratings = {pair[0]: pair[1] for pair in rated}
    if (
        label["cohort"] != _cohort(positives, protocol)
        or not set(positives) <= set(label["completed_ids"]) <= set(label["history_ids"])
        or type(label["history_count"]) is not int
        or label["history_count"] != len(label["history_ids"])
        or type(label["n"]) is not int
        or not 0 <= label["n"] <= 10
        or type(label["recent_2024_seed_count"]) is not int
        or not 0 <= label["recent_2024_seed_count"] <= label["n"]
        or label["query_id"]
        != _query_id(label["user_key"], label["cohort"], label["order"], label["n"])
    ):
        fail()
    expected_targets = 5 if label["cohort"] == "main" else min(5, max(1, len(positives) // 3))
    targets = sorted(
        positives,
        key=lambda item: (
            order_hash(protocol["query"]["target_seed"], label["user_key"], item),
            item,
        ),
    )[:expected_targets]
    pool = set(positives) - set(targets)
    nmax = min(10, len(pool))
    if label["target_ids"] != sorted(targets) or len(label["reserved_seed_ids"]) != (
        0 if label["n"] == 0 else nmax
    ):
        fail()
    if label["n"] == 0:
        if label["order"] != "popularity" or label["seed_mal_ids"] or label["reserved_seed_ids"]:
            fail()
    else:
        if (
            label["order"] not in protocol["query"]["orders"]
            or label["n"] > nmax
            or label["seed_mal_ids"] != label["reserved_seed_ids"][: label["n"]]
            or (
                set(label["reserved_seed_ids"]) != pool
                if len(pool) <= 10
                else not set(label["reserved_seed_ids"]) <= pool
            )
        ):
            fail()
        if label["order"] == "top-rated":
            expected = sorted(
                pool,
                key=lambda item: (
                    -ratings[item],
                    order_hash(protocol["query"]["toprated_tie_seed"], label["user_key"], item),
                    item,
                ),
            )[:nmax]
            if label["reserved_seed_ids"] != expected:
                fail()
        if label["order"].startswith("random-liked-"):
            index = protocol["query"]["orders"].index(label["order"])
            expected = sorted(
                pool,
                key=lambda item: (
                    order_hash(protocol["query"]["order_seeds"][index], label["user_key"], item),
                    item,
                ),
            )[:nmax]
            if label["reserved_seed_ids"] != expected:
                fail()
    return label


def _verify_semantics(
    prepared_dir: Path,
    manifest: PreparationManifest,
    protocol: ProtocolSpec,
    ids: set[str],
) -> None:
    files = manifest["files"]
    expected_paths = {"prepare-manifest.json", *FILES}
    observed = list(prepared_dir.rglob("*"))
    if any(path.is_symlink() for path in observed):
        fail()
    actual_paths = {str(path.relative_to(prepared_dir)) for path in observed if path.is_file()}
    actual_dirs = {str(path.relative_to(prepared_dir)) for path in observed if path.is_dir()}
    if actual_paths != expected_paths or actual_dirs != {"fit", "development", "evaluator"}:
        fail()
    heaps: Heaps = {
        (split, cohort): [] for split in ("validation", "test") for cohort in ("main", "short")
    }
    split_count: Counter[str] = Counter()
    last_user = None
    splits = stream_tsv(
        prepared_dir / "evaluator/splits.tsv.gz",
        cast("StreamDescriptor", files["evaluator/splits.tsv.gz"]),
    )
    if next(splits, None) != b"dataset_user_key\tsplit\tcohort\n":
        fail()
    for line in splits:
        parts = line.rstrip(b"\n").split(b"\t")
        if len(parts) != 3:
            fail()
        user, split, cohort = (value.decode("ascii") for value in parts)
        if (
            last_user is not None
            and user <= last_user
            or split != split_user(user, protocol)
            or cohort not in ("main", "short", "ineligible")
        ):
            fail()
        last_user = user
        split_count[f"users_{split}"] += 1
        if cohort != "ineligible":
            split_count[f"eligible_{split}_{cohort}"] += 1
            if split != "train":
                _select(heaps, split, cohort, user, protocol)
    selected = {item[2] for heap in heaps.values() for item in heap}
    stat_value, _ = read_json(prepared_dir / "fit/train-stats.json", files["fit/train-stats.json"])
    exact(stat_value, {"schema_version", "train_users", "train_completed_rows", "items"})
    stat_doc = cast("dict[str, object]", stat_value)
    if type(stat_doc["items"]) is not dict or set(stat_doc["items"]) != ids:
        fail()
    calculated: dict[str, PreparationTrainItem] = {
        key: {
            "completed_users": 0,
            "positive_users": 0,
            "rating_sum": 0,
            "rating_count": 0,
            "band": "cold",
        }
        for key in ids
    }
    train = stream_tsv(
        prepared_dir / "fit/train-interactions.tsv.gz",
        cast("StreamDescriptor", files["fit/train-interactions.tsv.gz"]),
    )
    if next(train, None) != HEADER:
        fail()
    last_pair = None
    train_count = 0
    for line in train:
        parts = line.rstrip(b"\n").split(b"\t")
        if len(parts) != 4:
            fail()
        user = parts[0].decode("ascii")
        item_text, rating_text, status = parts[1:]
        if (
            split_user(user, protocol) != "train"
            or status != b"completed"
            or not item_text.isdigit()
            or not rating_text.isdigit()
        ):
            fail()
        item, rating = int(item_text), int(rating_text)
        if (
            str(item) not in ids
            or str(item).encode() != item_text
            or not 1 <= rating <= 10
            or (last_pair is not None and (user, item) <= last_pair)
        ):
            fail()
        last_pair = (user, item)
        stat = calculated[str(item)]
        stat["completed_users"] += 1
        stat["positive_users"] += rating >= 8
        stat["rating_sum"] += rating
        stat["rating_count"] += 1
        train_count += 1
    nonzero = sorted(
        (int(key) for key, stat in calculated.items() if stat["positive_users"]),
        key=lambda item: (-calculated[str(item)]["positive_users"], item),
    )
    head = set(nonzero[: (len(nonzero) + 4) // 5])
    for key, stat in calculated.items():
        item = int(key)
        stat["band"] = "cold" if not stat["positive_users"] else "head" if item in head else "tail"
    if (
        stat_doc["items"] != calculated
        or stat_doc["train_users"] != split_count["users_train"]
        or stat_doc["train_completed_rows"] != train_count
    ):
        fail()
    seen: set[str] = set()
    user_queries: dict[tuple[str, str], list[tuple[str, int]]] = {}
    user_payload: dict[tuple[str, str], UserPayload] = {}
    val_queries = iter(
        stream_jsonl(
            prepared_dir / "development/queries.jsonl.gz",
            cast("StreamDescriptor", files["development/queries.jsonl.gz"]),
        )
    )
    sampled: set[str] = set()
    for split in ("validation", "test"):
        for label_value in stream_jsonl(
            prepared_dir / f"evaluator/{split}-labels.jsonl.gz",
            cast("StreamDescriptor", files[f"evaluator/{split}-labels.jsonl.gz"]),
        ):
            label = _check_label(label_value, protocol, ids)
            if (
                label["split"] != split
                or label["query_id"] in seen
                or label["user_key"] not in selected
            ):
                fail()
            seen.add(label["query_id"])
            sampled.add(label["user_key"])
            user_key = (split, label["user_key"])
            payload: UserPayload = (
                label["cohort"],
                tuple(map(str, label["positive_ids"])),
                tuple(map(str, label["positive_ratings"])),
                tuple(map(str, label["target_ids"])),
                tuple(map(str, label["history_ids"])),
                tuple(map(str, label["completed_ids"])),
            )
            if user_key in user_payload and user_payload[user_key] != payload:
                fail()
            user_payload[user_key] = payload
            user_queries.setdefault(user_key, []).append((label["order"], label["n"]))
            if split == "validation":
                query = next(val_queries, None)
                if query is None or query != {
                    "query_id": label["query_id"],
                    "cohort": label["cohort"],
                    "seed_policy": "popularity" if label["n"] == 0 else "liked",
                    "order": label["order"],
                    "n": label["n"],
                    "seed_mal_ids": label["seed_mal_ids"],
                }:
                    fail()
    for user_key, arms in user_queries.items():
        count = len(user_payload[user_key][1])
        targets = 5 if user_payload[user_key][0] == "main" else min(5, max(1, count // 3))
        nmax = min(10, count - targets)
        expected_arms = [("popularity", 0)] + [
            (order, n) for order in protocol["query"]["orders"] for n in range(1, nmax + 1)
        ]
        if arms != expected_arms:
            fail()
    if (
        next(val_queries, None) is not None
        or sampled != selected
        or manifest["counts"]["sampled_users"] != len(sampled)
    ):
        fail()
    for key, value in split_count.items():
        if manifest["counts"].get(key) != value:
            fail()
    if (
        manifest["counts"].get("train_users") != split_count["users_train"]
        or manifest["counts"].get("train_completed_rows") != train_count
    ):
        fail()


def validate_preparation(
    prepared_dir: str | Path, *, expected_identity: str
) -> PreparationManifest:
    """Full self-contained validation, including both private label files."""
    prepared_dir = safe_path(prepared_dir, directory=True)
    check_identity(expected_identity)
    manifest_value, raw = read_json(prepared_dir / "prepare-manifest.json")
    if "sha256:" + sha(raw) != expected_identity:
        fail()
    exact(
        manifest_value,
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
    validate_preparation_producer(manifest_value)
    manifest = _checked_preparation_manifest(manifest_value)
    if manifest["kind"] != "arb-ml-preparation":
        fail()
    files = manifest["files"]
    for name, desc in files.items():
        path = prepared_dir / name
        if name.endswith(".jsonl.gz"):
            for _ in stream_jsonl(path, cast("StreamDescriptor", desc)):
                pass
        elif name.endswith(".tsv.gz"):
            for _ in stream_tsv(path, cast("StreamDescriptor", desc)):
                pass
        else:
            read_json(path, desc)
    protocol_value, protocol_raw = read_json(prepared_dir / "protocol.json", files["protocol.json"])
    if "sha256:" + sha(canonical_bytes(protocol_value)) != manifest["protocol_identity"]:
        fail()
    protocol = load_protocol(_protocol_bytes=protocol_raw)
    catalog_value, _ = read_json(prepared_dir / "model-catalog.json", files["model-catalog.json"])
    ids = check_catalog(catalog_value)
    catalog = cast("Catalog", catalog_value)
    if any(entry["score"] is not None for entry in catalog["anime"].values()):
        fail()
    fit, _ = read_json(prepared_dir / "fit/manifest.json", files["fit/manifest.json"])
    if fit != {
        "schema_version": 1,
        "kind": "arb-ml-fit",
        "dataset_identity": manifest["dataset_identity"],
        "protocol_identity": manifest["protocol_identity"],
        "train_interactions": files["fit/train-interactions.tsv.gz"],
        "train_stats": files["fit/train-stats.json"],
    }:
        fail()
    _verify_semantics(prepared_dir, manifest, protocol, ids)
    return manifest
