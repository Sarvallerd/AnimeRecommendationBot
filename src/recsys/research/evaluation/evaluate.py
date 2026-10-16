"""Strict full-catalog ML-01 scoring and held-out metrics."""

import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from recsys.contracts.bundle import canonical_bytes, check_catalog
from recsys.research.artifacts import (
    StreamWriter,
    check_descriptor,
    check_identity,
    cleanup,
    open_stage,
    publish,
    read_json,
    safe_path,
    separate,
    stream_jsonl,
    validate_preparation_producer,
    write_json,
)
from recsys.research.protocol import exact, fail, finite, sha

METRICS = ("ndcg5", "ndcg10", "recall5", "recall10", "hit5", "hit10")


def metrics(ranking, targets):
    if type(ranking) is not list or not targets or len(set(ranking)) != len(ranking):
        fail()
    targets = set(targets)
    answer = {}
    for k in (5, 10):
        hits = [int(item in targets) for item in ranking[:k]]
        dcg = sum(hit / math.log2(index + 2) for index, hit in enumerate(hits))
        ideal = sum(1 / math.log2(index + 2) for index in range(min(k, len(targets))))
        answer[f"ndcg{k}"] = dcg / ideal
        answer[f"recall{k}"] = sum(hits) / len(targets)
        answer[f"hit{k}"] = float(bool(sum(hits)))
    return answer


def _prepared(prepared_dir, expected_identity):
    check_identity(expected_identity)
    prepared_dir = safe_path(prepared_dir, directory=True)
    manifest, raw = read_json(prepared_dir / "prepare-manifest.json")
    validate_preparation_producer(manifest)
    if "sha256:" + sha(raw) != expected_identity or manifest.get("kind") != "arb-ml-preparation":
        fail()
    files = manifest["files"]
    protocol, _ = read_json(prepared_dir / "protocol.json", files["protocol.json"])
    catalog, _ = read_json(prepared_dir / "model-catalog.json", files["model-catalog.json"])
    ids = check_catalog(catalog)
    if any(item["score"] is not None for item in catalog["anime"].values()):
        fail()
    fit, fit_raw = read_json(prepared_dir / "fit/manifest.json", files["fit/manifest.json"])
    stats, _ = read_json(prepared_dir / "fit/train-stats.json", files["fit/train-stats.json"])
    if (
        fit["train_stats"] != files["fit/train-stats.json"]
        or fit["train_interactions"] != files["fit/train-interactions.tsv.gz"]
        or set(stats["items"]) != ids
        or "sha256:" + sha(canonical_bytes(protocol)) != manifest["protocol_identity"]
    ):
        fail()
    return manifest, protocol, catalog, fit, "sha256:" + sha(fit_raw), stats


def _predictions(path, prepared_manifest, prepared_identity, split, fit_identity):
    path = safe_path(path, directory=True)
    if {item.name for item in path.iterdir()} != {"prediction-manifest.json", "scores.jsonl.gz"}:
        fail()
    manifest, raw = read_json(path / "prediction-manifest.json")
    exact(
        manifest,
        {
            "schema_version",
            "kind",
            "prepared_identity",
            "query_descriptor",
            "catalog_descriptor",
            "split",
            "model_id",
            "config_sha256",
            "fit_identity",
            "train_interactions",
            "train_stats",
            "code_revision",
            "zero_seed_policy",
            "scores",
        },
    )
    if (
        type(manifest["schema_version"]) is not int
        or manifest["schema_version"] != 1
        or manifest["kind"] != "arb-ml-predictions"
        or manifest["prepared_identity"] != prepared_identity
        or manifest["split"] != split
        or (
            split == "validation"
            and manifest["query_descriptor"]
            != prepared_manifest["files"]["development/queries.jsonl.gz"]
        )
        or manifest["catalog_descriptor"] != prepared_manifest["files"]["model-catalog.json"]
        or manifest["fit_identity"] != fit_identity
        or manifest["train_interactions"]
        != prepared_manifest["files"]["fit/train-interactions.tsv.gz"]
        or manifest["train_stats"] != prepared_manifest["files"]["fit/train-stats.json"]
        or type(manifest["model_id"]) is not str
        or re.fullmatch(r"[a-z][a-z0-9-]{0,79}", manifest["model_id"]) is None
        or type(manifest["config_sha256"]) is not str
        or re.fullmatch(r"[0-9a-f]{64}", manifest["config_sha256"]) is None
        or type(manifest["code_revision"]) is not str
        or re.fullmatch(r"[0-9a-f]{40}", manifest["code_revision"]) is None
        or manifest["zero_seed_policy"] != "popularity-train-positive-user-counts"
        or type(manifest["scores"]) is not dict
        or set(manifest["scores"])
        != {"bytes", "sha256", "rows", "uncompressed_bytes", "canonical_sha256"}
    ):
        fail("research_prediction_invalid")
    check_descriptor(manifest["scores"], stream=True)
    check_descriptor(manifest["query_descriptor"], stream=True)
    check_descriptor(manifest["catalog_descriptor"])
    check_descriptor(manifest["train_interactions"], stream=True)
    check_descriptor(manifest["train_stats"])
    return manifest, "sha256:" + sha(raw)


def _paired_rows(prepared_dir, prep, split, allowed=None):
    """Validation reads only development queries and validation labels."""
    files = prep["files"]
    labels = stream_jsonl(
        prepared_dir / f"evaluator/{split}-labels.jsonl.gz",
        files[f"evaluator/{split}-labels.jsonl.gz"],
    )
    queries = (
        stream_jsonl(
            prepared_dir / "development/queries.jsonl.gz", files["development/queries.jsonl.gz"]
        )
        if split == "validation"
        else None
    )
    seen = set()
    for label in labels:
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
        query = (
            next(queries, None)
            if queries is not None
            else {
                "query_id": label["query_id"],
                "cohort": label["cohort"],
                "seed_policy": "popularity" if label["n"] == 0 else "liked",
                "order": label["order"],
                "n": label["n"],
                "seed_mal_ids": label["seed_mal_ids"],
            }
        )
        if query is None:
            fail()
        exact(query, {"query_id", "cohort", "seed_policy", "order", "n", "seed_mal_ids"})
        qid = query["query_id"]
        if qid in seen:
            fail()
        seen.add(qid)
        if (
            label["split"] != split
            or qid != label["query_id"]
            or query["cohort"] != label["cohort"]
            or query["order"] != label["order"]
            or query["n"] != label["n"]
            or query["seed_mal_ids"] != label["seed_mal_ids"]
            or query["seed_policy"] != ("popularity" if query["n"] == 0 else "liked")
            or type(query["n"]) is not int
            or not 0 <= query["n"] <= 10
            or not label["target_ids"]
            or set(label["target_ids"]) & set(label["reserved_seed_ids"])
            or label["seed_mal_ids"] != label["reserved_seed_ids"][: label["n"]]
            or not set(label["positive_ids"]) <= set(label["history_ids"])
        ):
            fail()
        if allowed is None or (query["cohort"], query["order"], query["n"]) in allowed:
            yield query, label
    if queries is not None and next(queries, None) is not None:
        fail()


def _rank(scores, ids, excluded, *, n0=False, stats=None):
    if type(scores) is not list or len(scores) != len(ids):
        fail("research_prediction_invalid")
    values = []
    for entry, expected in zip(scores, ids):
        if (
            type(entry) is not list
            or len(entry) != 2
            or type(entry[0]) is not int
            or entry[0] != expected
        ):
            fail("research_prediction_invalid")
        score = entry[1]
        if score is not None:
            finite(score)
        if n0 and score != stats["items"][str(expected)]["positive_users"]:
            fail("research_prediction_invalid")
        if expected not in excluded and score is not None:
            values.append((-score, expected))
    values.sort()
    return [item for _, item in values[:10]]


def _history_bin(count):
    if count < 20:
        return "lt20"
    if count < 50:
        return "20-49"
    if count < 100:
        return "50-99"
    if count < 200:
        return "100-199"
    return "200plus"


def _seed_band(seeds, stats):
    if not seeds:
        return "none"
    bands = {stats["items"][str(item)]["band"] for item in seeds}
    return next(iter(bands)) if len(bands) == 1 else "mixed"


def _genre_diversity(ranking, catalog):
    pairs = len(ranking) * (len(ranking) - 1) // 2
    valid = []
    for index, left in enumerate(ranking):
        genres_left = set(catalog["anime"][str(left)]["genres"])
        for right in ranking[index + 1 :]:
            genres_right = set(catalog["anime"][str(right)]["genres"])
            if genres_left and genres_right:
                valid.append(1 - len(genres_left & genres_right) / len(genres_left | genres_right))
    return {
        "mean": sum(valid) / len(valid) if valid else None,
        "valid_pair_coverage": len(valid) / pairs if pairs else 0.0,
    }


def _slice(rows):
    by_user = defaultdict(list)
    for row in rows:
        by_user[row["user_key"]].append(row)
    if len(by_user) < 20:
        return None
    means = {
        metric: sum(
            sum(item["metrics"][metric] for item in user_rows) / len(user_rows)
            for user_rows in by_user.values()
        )
        / len(by_user)
        for metric in METRICS
    }
    diversity = [
        row["genre_diversity"]["mean"] for row in rows if row["genre_diversity"]["mean"] is not None
    ]
    return {
        "users": len(by_user),
        "queries": len(rows),
        "metrics": means,
        "genre_diversity": sum(diversity) / len(diversity) if diversity else None,
        "valid_pair_coverage": sum(row["genre_diversity"]["valid_pair_coverage"] for row in rows)
        / len(rows),
        "recent_2024_seed_share": sum(row["recent_2024_seed_count"] for row in rows)
        / max(1, sum(row["n"] for row in rows)),
    }


def _aggregate(private_rows, catalog, stats):
    groups = defaultdict(list)
    users = defaultdict(set)
    coverage = defaultdict(set)
    short = Counter()
    repeat = Counter()
    for row in private_rows:
        key = (row["split"], row["cohort"], row["mask"], row["order"], row["n"])
        groups[key].append(row)
        users[key].add(row["user_key"])
        coverage[key].update(row["ranking"])
        short[key] += len(row["ranking"]) < 5
        repeat[key] += row["known_history_repeats"]
    report = []
    for key in sorted(groups):
        rows = groups[key]
        aggregate = _slice(rows)
        if aggregate is None:
            continue
        slices = {}
        for field in ("history_bin", "seed_band"):
            values = sorted({row[field] for row in rows})
            slices[field] = {
                value: segment
                for value in values
                if (segment := _slice([row for row in rows if row[field] == value])) is not None
            }
        report.append(
            {
                "split": key[0],
                "cohort": key[1],
                "mask": key[2],
                "order": key[3],
                "n": key[4],
                **aggregate,
                "coverage": len(coverage[key]) / len(catalog["anime"]),
                "empty_rate": sum(not row["ranking"] for row in rows) / len(rows),
                "short_rate": short[key] / len(rows),
                "known_history_repeat_rate": repeat[key]
                / max(1, sum(len(row["ranking"]) for row in rows)),
                "completed_repeat_rate": sum(row["completed_repeats"] for row in rows)
                / max(1, sum(len(row["ranking"]) for row in rows)),
                "slices": slices,
            }
        )
    return report


def evaluate_predictions(
    prepared_dir,
    prediction_dir,
    output_dir,
    *,
    split,
    expected_prepared_identity,
    selection_path=None,
    expected_selection_identity=None,
):
    if split not in ("validation", "test"):
        fail()
    prepared_dir, prediction_dir, output_dir = map(Path, (prepared_dir, prediction_dir, output_dir))
    separate(prepared_dir, prediction_dir, output_dir)
    prep, protocol, catalog, fit, fit_identity, stats = _prepared(
        prepared_dir, expected_prepared_identity
    )
    prediction, prediction_identity = _predictions(
        prediction_dir, prep, expected_prepared_identity, split, fit_identity
    )
    allowed = None
    selection_identity = None
    if split == "test":
        if selection_path is None or expected_selection_identity is None:
            fail("research_selection_required")
        from recsys.research.evaluation.selection import read_selection

        selection, selection_identity = read_selection(
            selection_path, expected_selection_identity, expected_prepared_identity
        )
        if (
            prediction["model_id"] != selection["model_id"]
            or prediction["config_sha256"] != selection["config_sha256"]
            or prediction["fit_identity"] != selection["fit_identity"]
            or prediction["code_revision"] != selection["code_revision"]
        ):
            fail("research_selection_mismatch")
        allowed = {(arm["cohort"], arm["order"], arm["n"]) for arm in selection["allowed_arms"]}
    elif selection_path is not None or expected_selection_identity is not None:
        fail()
    ids = sorted(int(item) for item in catalog["anime"])
    label_query = _paired_rows(prepared_dir, prep, split, allowed)
    score_rows = stream_jsonl(prediction_dir / "scores.jsonl.gz", prediction["scores"])
    stage = open_stage(output_dir)
    try:
        private = StreamWriter(stage / "user-metrics.jsonl.gz")
        test_queries = (
            StreamWriter(stage / "recomputed-test-queries.jsonl.gz") if split == "test" else None
        )
        rows_for_report = []
        nrows = 0
        from recsys.research.prepare import _check_label

        for (query, label), row in zip(label_query, score_rows, strict=True):
            _check_label(label, protocol, set(catalog["anime"]))
            if test_queries is not None:
                test_queries.write_json(query)
            exact(row, {"query_id", "scores"})
            if type(row["query_id"]) is not str or row["query_id"] != query["query_id"]:
                fail("research_prediction_invalid")
            history, targets, seeds = (
                set(label[key]) for key in ("history_ids", "target_ids", "seed_mal_ids")
            )
            positive = set(label["positive_ids"])
            if not (
                targets <= positive <= history and seeds <= positive and targets.isdisjoint(seeds)
            ):
                fail()
            controlled = _rank(
                row["scores"], ids, history - targets, n0=query["n"] == 0, stats=stats
            )
            product = _rank(row["scores"], ids, seeds, n0=query["n"] == 0, stats=stats)
            for mask, ranking, relevance in (
                ("controlled", controlled, targets),
                ("product", product, positive - seeds),
            ):
                if (
                    not relevance
                    or len(ranking) != len(set(ranking))
                    or any(
                        item not in ids
                        or item in (history - targets if mask == "controlled" else seeds)
                        for item in ranking
                    )
                ):
                    fail()
                result = {
                    "query_id": query["query_id"],
                    "user_key": label["user_key"],
                    "split": split,
                    "cohort": query["cohort"],
                    "mask": mask,
                    "order": query["order"],
                    "n": query["n"],
                    "history_bin": _history_bin(label["history_count"]),
                    "seed_band": _seed_band(seeds, stats),
                    "recent_2024_seed_count": label["recent_2024_seed_count"],
                    "ranking": ranking,
                    "metrics": metrics(ranking, relevance),
                    "genre_diversity": _genre_diversity(ranking, catalog),
                    "known_history_repeats": len(set(ranking) & history),
                    "completed_repeats": len(set(ranking) & set(label["completed_ids"])),
                }
                private.write_json(result)
                rows_for_report.append(result)
            nrows += 1
        private_desc = private.close()
        if test_queries is not None:
            test_desc = test_queries.close()
            if test_desc != prediction["query_descriptor"]:
                fail("research_prediction_invalid")
            (stage / "recomputed-test-queries.jsonl.gz").unlink()
        report = {
            "schema_version": 1,
            "kind": "arb-ml-evaluation-report",
            "split": split,
            "prediction_identity": prediction_identity,
            "prepared_identity": expected_prepared_identity,
            "queries": nrows,
            "groups": _aggregate(rows_for_report, catalog, stats),
        }
        report_desc = write_json(stage / "report.json", report)
        output_manifest = {
            "schema_version": 1,
            "kind": "arb-ml-evaluation",
            "prepared_identity": expected_prepared_identity,
            "prediction_identity": prediction_identity,
            "split": split,
            "model_id": prediction["model_id"],
            "config_sha256": prediction["config_sha256"],
            "fit_identity": fit_identity,
            "prediction_code_revision": prediction["code_revision"],
            "selection_identity": selection_identity,
            "files": {"report.json": report_desc, "user-metrics.jsonl.gz": private_desc},
        }
        write_json(stage / "evaluation-manifest.json", output_manifest)
        publish(stage, output_dir)
        return output_dir / "evaluation-manifest.json"
    except Exception:
        cleanup(stage)
        raise
