"""Stream a pinned dataset into private ML-01 fit, queries, and hidden labels."""

import hashlib
import heapq
import os
from collections import Counter
from importlib import resources
from pathlib import Path

from .bundle import canonical_bytes, check_catalog
from .dataset import DatasetSpec, HEADER, _stream_lines, validate_dataset
from .release_prepare import release_info
from .research_artifacts import (StreamWriter, check_identity, cleanup, descriptor, identity,
                                 module_fingerprints, open_stage, producer, publish, read_json,
                                 safe_path, separate, stream_jsonl, stream_tsv, write_json)
from .research_protocol import (ResearchError, exact, fail, load_protocol, order_hash,
                                protocol_identity, sample_hash, sha, split_user)

FILES = ("protocol.json", "model-catalog.json", "fit/manifest.json",
         "fit/train-interactions.tsv.gz", "fit/train-stats.json",
         "development/queries.jsonl.gz", "evaluator/splits.tsv.gz",
         "evaluator/validation-labels.jsonl.gz", "evaluator/test-labels.jsonl.gz")


def _release(release_dir, dataset_dir, *, _lock_bytes=None):
    lock = release_info(_lock_bytes=_lock_bytes)
    release_dir = safe_path(release_dir, directory=True)
    if {path.name for path in release_dir.iterdir()} != {"catalog.json", "release-manifest.json"}:
        fail()
    manifest, raw = read_json(release_dir / "release-manifest.json")
    catalog, catalog_raw = read_json(release_dir / "catalog.json")
    base_manifest, base_raw = read_json(Path(dataset_dir) / "dataset-manifest.json")
    locked = lock["inputs"]
    if ("sha256:" + sha(raw) != locked["catalog_release"]["identity"] or
            "sha256:" + sha(base_raw) != locked["base"]["identity"] or
            manifest.get("kind") != "arb-catalog-release" or
            manifest.get("policy") != "ordered-alias-overlay-v1" or
            manifest.get("inputs", {}).get("base", {}).get("identity") != locked["base"]["identity"] or
            manifest.get("inputs", {}).get("aliases", {}).get("identity") != locked["aliases"]["identity"] or
            manifest.get("files") != {"catalog.json": descriptor(catalog_raw)} or
            sha(catalog_raw) != locked["catalog_release"]["catalog_sha256"] or
            manifest.get("counts", {}).get("catalog_count") != locked["catalog_release"]["catalog_count"]):
        fail("research_release_mismatch")
    keys = check_catalog(catalog)
    if len(keys) != locked["catalog_release"]["catalog_count"]:
        fail("research_release_mismatch")
    return lock, catalog, manifest, raw


def _rows(dataset_dir):
    path = safe_path(Path(dataset_dir) / "interactions.tsv.gz", directory=False)
    lines = iter(_stream_lines(path))
    if next(lines, None) != HEADER:
        fail()
    last_key, last_id = None, 0
    for raw in lines:
        fields = raw.rstrip(b"\n").split(b"\t")
        if len(fields) != 4:
            fail()
        try:
            user, item, rating, status = fields
            user = user.decode("ascii")
            item = int(item)
            rating = int(rating)
            status = status.decode("ascii")
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


def _profiles(rows):
    user = None
    current = []
    for row in rows:
        if row[0] != user:
            if user is not None:
                yield user, current
            user, current = row[0], []
        current.append(row)
    if user is not None:
        yield user, current


def _cohort(positives, protocol):
    count = len(positives)
    if count >= protocol["cohorts"]["main_min_positives"]:
        return "main"
    if protocol["cohorts"]["short_min_positives"] <= count <= protocol["cohorts"]["short_max_positives"]:
        return "short"
    return None


def _select(heaps, split, cohort, user, protocol):
    limit = protocol["query"]["sample_main_per_split" if cohort == "main" else "sample_short_per_split"]
    key = sample_hash(protocol["query"]["sample_seed"], split, cohort, user)
    # The integer user key resolves theoretical hash collisions independently of input order.
    rank = (int.from_bytes(key, "big"), int(user[3:], 16))
    entry = (-rank[0], -rank[1], user)
    heap = heaps[split, cohort]
    if len(heap) < limit:
        heapq.heappush(heap, entry)
    elif entry > heap[0]:
        heapq.heapreplace(heap, entry)


def _query_id(user, cohort, order, n):
    raw = b"arb-ml-query-v1\0" + user.encode() + b"\0" + cohort.encode() + b"\0" + order.encode() + b"\0" + str(n).encode()
    return "q1_" + sha(raw)


def _emit_user(user, profile, split, cohort, queries, labels, protocol, catalog_ids, facts):
    positives = {item: rating for _, item, rating, status, _ in profile
                 if status == "completed" and rating >= 8}
    history = sorted({row[1] for row in profile})
    completed = sorted({row[1] for row in profile if row[3] == "completed"})
    if _cohort(positives, protocol) != cohort or not set(history) <= catalog_ids:
        fail()
    count = len(positives)
    target_count = protocol["query"]["main_targets"] if cohort == "main" else min(5, max(1, count // 3))
    targets = sorted(positives, key=lambda item: (order_hash(protocol["query"]["target_seed"], user, item), item))[:target_count]
    target_set = set(targets)
    pool = set(positives) - target_set
    nmax = min(protocol["query"]["max_n"], len(pool))
    if cohort == "main" and nmax != 10:
        fail()
    common = {"cohort": cohort, "user_key": user, "split": split,
              "positive_ids": sorted(positives),
              "positive_ratings": [[item, positives[item]] for item in sorted(positives)],
              "target_ids": sorted(targets),
              "history_ids": history, "completed_ids": completed,
              "history_count": len(history)}
    plans = [("popularity", [])]
    for name, seed in zip(protocol["query"]["orders"][:3], protocol["query"]["order_seeds"]):
        plans.append((name, sorted(pool, key=lambda item: (order_hash(seed, user, item), item))[:nmax]))
    plans.append(("top-rated", sorted(pool, key=lambda item: (-positives[item],
                  order_hash(protocol["query"]["toprated_tie_seed"], user, item), item))[:nmax]))
    for order, reserved in plans:
        ns = (0,) if order == "popularity" else range(1, nmax + 1)
        for n in ns:
            seeds = reserved[:n]
            query_id = _query_id(user, cohort, order, n)
            query = {"query_id": query_id, "cohort": cohort, "seed_policy": "popularity" if n == 0 else "liked",
                     "order": order, "n": n, "seed_mal_ids": seeds}
            recent = sum(facts[str(item)]["airing_status"] in ("finished_airing", "currently_airing")
                         and facts[str(item)]["first_release_date"] is not None
                         and "2024-01-01" <= facts[str(item)]["first_release_date"] <= "2025-10-31"
                         for item in seeds)
            label = {**common, "query_id": query_id, "order": order, "n": n,
                     "seed_mal_ids": seeds, "reserved_seed_ids": reserved,
                     "recent_2024_seed_count": recent}
            if queries is not None:
                queries.write_json(query)
            labels.write_json(label)


def prepare_research(dataset_dir, release_dir, output_dir, *, code_revision,
                     _dataset_spec: DatasetSpec | None = None, _lock_bytes=None,
                     _protocol_bytes=None):
    """Validate source, then two streaming passes; publish only complete immutable output."""
    dataset_dir, release_dir, output_dir = map(Path, (dataset_dir, release_dir, output_dir))
    separate(dataset_dir, release_dir, output_dir)
    protocol = load_protocol(_protocol_bytes=_protocol_bytes)
    lock, catalog, _, lock_release_raw = _release(release_dir, dataset_dir, _lock_bytes=_lock_bytes)
    summary = validate_dataset(dataset_dir, expected_identity=lock["inputs"]["base"]["identity"],
                               _spec=_dataset_spec)
    catalog_ids = {int(item) for item in catalog["anime"]}
    facts_doc, _ = read_json(dataset_dir / "item-facts.json")
    if (set(facts_doc) != {"schema_version", "anime"} or
            type(facts_doc["schema_version"]) is not int or facts_doc["schema_version"] != 1 or
            set(facts_doc["anime"]) != set(catalog["anime"])):
        fail()
    facts = facts_doc["anime"]
    model_catalog = {**catalog, "anime": {key: {**item, "score": None}
                                           for key, item in catalog["anime"].items()}}
    stage = open_stage(output_dir)
    try:
        for sub in ("fit", "development", "evaluator"):
            (stage / sub).mkdir(mode=0o700)
        files = {}
        files["protocol.json"] = write_json(stage / "protocol.json", protocol)
        files["model-catalog.json"] = write_json(stage / "model-catalog.json", model_catalog)
        train = StreamWriter(stage / "fit/train-interactions.tsv.gz")
        splits = StreamWriter(stage / "evaluator/splits.tsv.gz")
        train.write(HEADER)
        splits.write(b"dataset_user_key\tsplit\tcohort\n")
        stats = {str(item): {"completed_users": 0, "positive_users": 0,
                             "rating_sum": 0, "rating_count": 0} for item in sorted(catalog_ids)}
        counters = Counter()
        heaps = {(split, cohort): [] for split in ("validation", "test") for cohort in ("main", "short")}
        for user, profile in _profiles(_rows(dataset_dir)):
            split = split_user(user, protocol)
            positives = {item for _, item, rating, status, _ in profile
                         if status == "completed" and rating >= 8}
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
        nonzero = sorted((item for item in catalog_ids if stats[str(item)]["positive_users"]),
                         key=lambda item: (-stats[str(item)]["positive_users"], item))
        head_count = (len(nonzero) + 4) // 5
        head = set(nonzero[:head_count])
        for item in catalog_ids:
            stat = stats[str(item)]
            stat["band"] = "cold" if not stat["positive_users"] else "head" if item in head else "tail"
        train_stats = {"schema_version": 1, "train_users": counters["train_users"],
                       "train_completed_rows": counters["train_completed_rows"], "items": stats}
        files["fit/train-stats.json"] = write_json(stage / "fit/train-stats.json", train_stats)
        selected = {item[2] for heap in heaps.values() for item in heap}
        qwriter = StreamWriter(stage / "development/queries.jsonl.gz")
        val = StreamWriter(stage / "evaluator/validation-labels.jsonl.gz")
        test = StreamWriter(stage / "evaluator/test-labels.jsonl.gz")
        seen = set()
        for user, profile in _profiles(_rows(dataset_dir)):
            if user not in selected:
                continue
            split = split_user(user, protocol)
            positives = {item for _, item, rating, status, _ in profile
                         if status == "completed" and rating >= 8}
            cohort = _cohort(positives, protocol)
            if split not in ("validation", "test") or cohort is None:
                fail()
            _emit_user(user, profile, split, cohort, qwriter if split == "validation" else None,
                       val if split == "validation" else test,
                       protocol, catalog_ids, facts)
            seen.add(user)
        if seen != selected:
            fail()
        files["development/queries.jsonl.gz"] = qwriter.close()
        files["evaluator/validation-labels.jsonl.gz"] = val.close()
        files["evaluator/test-labels.jsonl.gz"] = test.close()
        fit_manifest = {"schema_version": 1, "kind": "arb-ml-fit", "dataset_identity": summary.identity,
                        "protocol_identity": protocol_identity(_protocol_bytes=_protocol_bytes),
                        "train_interactions": files["fit/train-interactions.tsv.gz"],
                        "train_stats": files["fit/train-stats.json"]}
        files["fit/manifest.json"] = write_json(stage / "fit/manifest.json", fit_manifest)
        manifest = {"schema_version": 1, "kind": "arb-ml-preparation", "protocol_identity": protocol_identity(_protocol_bytes=_protocol_bytes),
                    "release_lock_sha256": sha(canonical_bytes(lock)), "release_id": lock["release_id"],
                    "dataset_identity": summary.identity, "catalog_release_identity": lock["inputs"]["catalog_release"]["identity"],
                    "catalog_sha256": lock["inputs"]["catalog_release"]["catalog_sha256"],
                    "producer": producer(code_revision), "files": files,
                    "counts": {**dict(sorted(counters.items())), "sampled_users": len(selected)}}
        write_json(stage / "prepare-manifest.json", manifest)
        publish(stage, output_dir)
        return output_dir / "prepare-manifest.json"
    except Exception:
        cleanup(stage)
        raise


def _id_list(values, ids, *, ordered=False):
    if type(values) is not list or any(type(item) is not int or str(item) not in ids for item in values):
        fail()
    if len(values) != len(set(values)) or (ordered and values != sorted(values)):
        fail()


def _check_label(label, protocol, ids):
    exact(label, {"cohort", "user_key", "split", "positive_ids", "target_ids",
                  "history_ids", "completed_ids", "history_count", "query_id", "order",
                  "n", "seed_mal_ids", "reserved_seed_ids", "recent_2024_seed_count",
                  "positive_ratings"})
    if label["split"] != split_user(label["user_key"], protocol) or label["split"] == "train":
        fail()
    for key in ("positive_ids", "target_ids", "history_ids", "completed_ids", "seed_mal_ids", "reserved_seed_ids"):
        _id_list(label[key], ids, ordered=key in ("positive_ids", "history_ids", "completed_ids"))
    positives = label["positive_ids"]
    rated = label["positive_ratings"]
    if (type(rated) is not list or len(rated) != len(positives) or
            any(type(pair) is not list or len(pair) != 2 or
                type(pair[0]) is not int or pair[0] != item or
                type(pair[1]) is not int or not 8 <= pair[1] <= 10
                for pair, item in zip(rated, positives))):
        fail()
    ratings = dict(rated)
    if (label["cohort"] != _cohort(positives, protocol) or
            not set(positives) <= set(label["completed_ids"]) <= set(label["history_ids"]) or
            type(label["history_count"]) is not int or label["history_count"] != len(label["history_ids"]) or
            type(label["n"]) is not int or not 0 <= label["n"] <= 10 or
            type(label["recent_2024_seed_count"]) is not int or
            not 0 <= label["recent_2024_seed_count"] <= label["n"] or
            label["query_id"] != _query_id(label["user_key"], label["cohort"], label["order"], label["n"])):
        fail()
    expected_targets = 5 if label["cohort"] == "main" else min(5, max(1, len(positives) // 3))
    targets = sorted(positives, key=lambda item: (order_hash(protocol["query"]["target_seed"],
                                                  label["user_key"], item), item))[:expected_targets]
    pool = set(positives) - set(targets)
    nmax = min(10, len(pool))
    if label["target_ids"] != sorted(targets) or len(label["reserved_seed_ids"]) != (0 if label["n"] == 0 else nmax):
        fail()
    if label["n"] == 0:
        if label["order"] != "popularity" or label["seed_mal_ids"] or label["reserved_seed_ids"]:
            fail()
    else:
        if (label["order"] not in protocol["query"]["orders"] or
                label["n"] > nmax or label["seed_mal_ids"] != label["reserved_seed_ids"][:label["n"]] or
                (set(label["reserved_seed_ids"]) != pool if len(pool) <= 10 else
                 not set(label["reserved_seed_ids"]) <= pool)) :
            fail()
        if label["order"] == "top-rated":
            expected = sorted(pool, key=lambda item: (-ratings[item],
                order_hash(protocol["query"]["toprated_tie_seed"], label["user_key"], item), item))[:nmax]
            if label["reserved_seed_ids"] != expected:
                fail()
        if label["order"].startswith("random-liked-"):
            index = protocol["query"]["orders"].index(label["order"])
            expected = sorted(pool, key=lambda item: (order_hash(protocol["query"]["order_seeds"][index],
                                                                label["user_key"], item), item))[:nmax]
            if label["reserved_seed_ids"] != expected:
                fail()


def _verify_semantics(prepared_dir, manifest, protocol, ids):
    files = manifest["files"]
    expected_paths = {"prepare-manifest.json", *FILES}
    observed = list(prepared_dir.rglob("*"))
    if any(path.is_symlink() for path in observed):
        fail()
    actual_paths = {str(path.relative_to(prepared_dir)) for path in observed if path.is_file()}
    actual_dirs = {str(path.relative_to(prepared_dir)) for path in observed if path.is_dir()}
    if actual_paths != expected_paths or actual_dirs != {"fit", "development", "evaluator"}:
        fail()
    heaps = {(split, cohort): [] for split in ("validation", "test") for cohort in ("main", "short")}
    split_count = Counter()
    last_user = None
    splits = stream_tsv(prepared_dir / "evaluator/splits.tsv.gz", files["evaluator/splits.tsv.gz"])
    if next(splits, None) != b"dataset_user_key\tsplit\tcohort\n":
        fail()
    for line in splits:
        parts = line.rstrip(b"\n").split(b"\t")
        if len(parts) != 3:
            fail()
        user, split, cohort = (value.decode("ascii") for value in parts)
        if (last_user is not None and user <= last_user or split != split_user(user, protocol) or
                cohort not in ("main", "short", "ineligible")):
            fail()
        last_user = user
        split_count[f"users_{split}"] += 1
        if cohort != "ineligible":
            split_count[f"eligible_{split}_{cohort}"] += 1
            if split != "train":
                _select(heaps, split, cohort, user, protocol)
    selected = {item[2] for heap in heaps.values() for item in heap}
    stat_doc, _ = read_json(prepared_dir / "fit/train-stats.json", files["fit/train-stats.json"])
    exact(stat_doc, {"schema_version", "train_users", "train_completed_rows", "items"})
    if set(stat_doc["items"]) != ids:
        fail()
    calculated = {key: {"completed_users": 0, "positive_users": 0,
                        "rating_sum": 0, "rating_count": 0} for key in ids}
    train = stream_tsv(prepared_dir / "fit/train-interactions.tsv.gz", files["fit/train-interactions.tsv.gz"])
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
        if (split_user(user, protocol) != "train" or status != b"completed" or
                not item_text.isdigit() or not rating_text.isdigit()):
            fail()
        item, rating = int(item_text), int(rating_text)
        if (str(item) not in ids or str(item).encode() != item_text or
                not 1 <= rating <= 10 or
                (last_pair is not None and (user, item) <= last_pair)):
            fail()
        last_pair = (user, item)
        stat = calculated[str(item)]
        stat["completed_users"] += 1
        stat["positive_users"] += rating >= 8
        stat["rating_sum"] += rating
        stat["rating_count"] += 1
        train_count += 1
    nonzero = sorted((int(key) for key, stat in calculated.items() if stat["positive_users"]),
                     key=lambda item: (-calculated[str(item)]["positive_users"], item))
    head = set(nonzero[:(len(nonzero) + 4) // 5])
    for key, stat in calculated.items():
        item = int(key)
        stat["band"] = "cold" if not stat["positive_users"] else "head" if item in head else "tail"
    if (stat_doc["items"] != calculated or stat_doc["train_users"] != split_count["users_train"] or
            stat_doc["train_completed_rows"] != train_count):
        fail()
    seen = set()
    user_queries = {}
    user_payload = {}
    val_queries = iter(stream_jsonl(prepared_dir / "development/queries.jsonl.gz",
                                    files["development/queries.jsonl.gz"]))
    sampled = set()
    for split in ("validation", "test"):
        for label in stream_jsonl(prepared_dir / f"evaluator/{split}-labels.jsonl.gz",
                                  files[f"evaluator/{split}-labels.jsonl.gz"]):
            _check_label(label, protocol, ids)
            if label["split"] != split or label["query_id"] in seen or label["user_key"] not in selected:
                fail()
            seen.add(label["query_id"])
            sampled.add(label["user_key"])
            key = (split, label["user_key"])
            payload = tuple(label[name] if name == "cohort" else tuple(map(str, label[name]))
                            for name in ("cohort", "positive_ids", "positive_ratings",
                                         "target_ids", "history_ids", "completed_ids"))
            if key in user_payload and user_payload[key] != payload:
                fail()
            user_payload[key] = payload
            user_queries.setdefault(key, []).append((label["order"], label["n"]))
            if split == "validation":
                query = next(val_queries, None)
                if (query is None or query != {"query_id": label["query_id"],
                        "cohort": label["cohort"], "seed_policy": "popularity" if label["n"] == 0 else "liked",
                        "order": label["order"], "n": label["n"], "seed_mal_ids": label["seed_mal_ids"]}):
                    fail()
    for key, arms in user_queries.items():
        count = len(user_payload[key][1])
        targets = 5 if user_payload[key][0] == "main" else min(5, max(1, count // 3))
        nmax = min(10, count - targets)
        expected_arms = [("popularity", 0)] + [(order, n)
                         for order in protocol["query"]["orders"] for n in range(1, nmax + 1)]
        if arms != expected_arms:
            fail()
    if next(val_queries, None) is not None or sampled != selected or manifest["counts"]["sampled_users"] != len(sampled):
        fail()
    for key, value in split_count.items():
        if manifest["counts"].get(key) != value:
            fail()
    if (manifest["counts"].get("train_users") != split_count["users_train"] or
            manifest["counts"].get("train_completed_rows") != train_count):
        fail()


def validate_preparation(prepared_dir, *, expected_identity):
    """Full self-contained validation, including both private label files."""
    prepared_dir = safe_path(prepared_dir, directory=True)
    check_identity(expected_identity)
    manifest, raw = read_json(prepared_dir / "prepare-manifest.json")
    if "sha256:" + sha(raw) != expected_identity:
        fail()
    exact(manifest, {"schema_version", "kind", "protocol_identity", "release_lock_sha256",
                     "release_id", "dataset_identity", "catalog_release_identity", "catalog_sha256",
                     "producer", "files", "counts"})
    if (type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1 or
            manifest["kind"] != "arb-ml-preparation" or set(manifest["files"]) != set(FILES) or
            manifest["producer"]["code_files_sha256"] != module_fingerprints()):
        fail()
    files = manifest["files"]
    for name, desc in files.items():
        path = prepared_dir / name
        if name.endswith(".jsonl.gz"):
            for _ in stream_jsonl(path, desc):
                pass
        elif name.endswith(".tsv.gz"):
            for _ in stream_tsv(path, desc):
                pass
        else:
            read_json(path, desc)
    protocol, _ = read_json(prepared_dir / "protocol.json", files["protocol.json"])
    if "sha256:" + sha(canonical_bytes(protocol)) != manifest["protocol_identity"]:
        fail()
    catalog, _ = read_json(prepared_dir / "model-catalog.json", files["model-catalog.json"])
    ids = check_catalog(catalog)
    if any(entry["score"] is not None for entry in catalog["anime"].values()):
        fail()
    fit, _ = read_json(prepared_dir / "fit/manifest.json", files["fit/manifest.json"])
    if (fit != {"schema_version": 1, "kind": "arb-ml-fit",
                "dataset_identity": manifest["dataset_identity"],
                "protocol_identity": manifest["protocol_identity"],
                "train_interactions": files["fit/train-interactions.tsv.gz"],
                "train_stats": files["fit/train-stats.json"]}):
        fail()
    _verify_semantics(prepared_dir, manifest, protocol, ids)
    return manifest
