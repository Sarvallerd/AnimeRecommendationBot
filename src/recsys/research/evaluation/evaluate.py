"""Strict full-catalog ML-01 scoring and held-out metrics."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from heapq import nsmallest
from pathlib import Path
from typing import TypedDict, cast

from recsys.contracts.bundle import Catalog, canonical_bytes, check_catalog
from recsys.research.artifacts import (
    FileDescriptor,
    StreamDescriptor,
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
from recsys.research.prepare import (
    PreparationLabel,
    PreparationManifest,
    _check_label,
    _checked_preparation_manifest,
)
from recsys.research.protocol import ProtocolSpec, exact, fail, finite, load_protocol, sha
from recsys.research.score_blocks import (
    FORMAT,
    MAX_MANIFEST,
    BlockScoreDescriptor,
    DecodedScoreRow,
    check_block_descriptor,
    iter_score_rows,
    require_query_binding,
)


class EvaluationQuery(TypedDict):
    query_id: str
    cohort: str
    seed_policy: str
    order: str
    n: int
    seed_mal_ids: list[int]


class EvaluationTrainItem(TypedDict):
    completed_users: int
    positive_users: int
    rating_sum: int
    rating_count: int
    band: str


class EvaluationTrainStats(TypedDict):
    schema_version: int
    train_users: int
    train_completed_rows: int
    items: dict[str, EvaluationTrainItem]


def _checked_train_stats(value: object, ids: set[str]) -> EvaluationTrainStats:
    exact(value, {"schema_version", "train_users", "train_completed_rows", "items"})
    raw = cast("dict[str, object]", value)
    if (
        type(raw["schema_version"]) is not int
        or type(raw["train_users"]) is not int
        or type(raw["train_completed_rows"]) is not int
        or type(raw["items"]) is not dict
    ):
        fail()
    items = cast("dict[str, object]", raw["items"])
    if set(items) != ids:
        fail()
    for item in items.values():
        exact(item, {"completed_users", "positive_users", "rating_sum", "rating_count", "band"})
        record = cast("dict[str, object]", item)
        if (
            any(
                type(record[key]) is not int
                for key in ("completed_users", "positive_users", "rating_sum", "rating_count")
            )
            or type(record["band"]) is not str
        ):
            fail()
    return cast("EvaluationTrainStats", raw)


class PredictionManifest(TypedDict):
    schema_version: int
    kind: str
    prepared_identity: str
    query_descriptor: StreamDescriptor
    catalog_descriptor: FileDescriptor
    split: str
    model_id: str
    config_sha256: str
    fit_identity: str
    train_interactions: StreamDescriptor
    train_stats: FileDescriptor
    code_revision: str
    zero_seed_policy: str
    scores: StreamDescriptor | BlockScoreDescriptor
    score_format: str


class EvaluationManifest(TypedDict):
    schema_version: int
    kind: str
    prepared_identity: str
    prediction_identity: str
    split: str
    model_id: str
    config_sha256: str
    fit_identity: str
    prediction_code_revision: str
    selection_identity: str | None
    files: dict[str, FileDescriptor]


class GenreDiversity(TypedDict):
    mean: float | None
    valid_pair_coverage: float


class PrivateMetric(TypedDict):
    query_id: str
    user_key: str
    split: str
    cohort: str
    mask: str
    order: str
    n: int
    history_bin: str
    seed_band: str
    ranking: list[int]
    metrics: dict[str, float]
    known_history_repeats: int
    completed_repeats: int
    recent_2024_seed_count: int
    genre_diversity: GenreDiversity


METRICS = ("ndcg5", "ndcg10", "recall5", "recall10", "hit5", "hit10")


def metrics(ranking: list[int], targets: set[int] | list[int]) -> dict[str, float]:
    if type(ranking) is not list or not targets or len(set(ranking)) != len(ranking):
        fail()
    targets = set(targets)
    answer: dict[str, float] = {}
    for k in (5, 10):
        hits = [int(item in targets) for item in ranking[:k]]
        dcg = sum(hit / math.log2(index + 2) for index, hit in enumerate(hits))
        ideal = sum(1 / math.log2(index + 2) for index in range(min(k, len(targets))))
        answer[f"ndcg{k}"] = dcg / ideal
        answer[f"recall{k}"] = sum(hits) / len(targets)
        answer[f"hit{k}"] = float(bool(sum(hits)))
    return answer


def _prepared(
    prepared_dir: str | Path, expected_identity: str
) -> tuple[
    PreparationManifest, ProtocolSpec, Catalog, dict[str, object], str, EvaluationTrainStats
]:
    check_identity(expected_identity)
    prepared_dir = safe_path(prepared_dir, directory=True)
    manifest_value, raw = read_json(prepared_dir / "prepare-manifest.json")
    validate_preparation_producer(manifest_value)
    raw_manifest = cast("dict[str, object]", manifest_value)
    if (
        "sha256:" + sha(raw) != expected_identity
        or raw_manifest.get("kind") != "arb-ml-preparation"
    ):
        fail()
    manifest = _checked_preparation_manifest(raw_manifest)
    files = manifest["files"]
    protocol_value, protocol_raw = read_json(prepared_dir / "protocol.json", files["protocol.json"])
    catalog_value, _ = read_json(prepared_dir / "model-catalog.json", files["model-catalog.json"])
    ids = check_catalog(catalog_value)
    catalog = cast("Catalog", catalog_value)
    if any(item["score"] is not None for item in catalog["anime"].values()):
        fail()
    fit_value, fit_raw = read_json(prepared_dir / "fit/manifest.json", files["fit/manifest.json"])
    stats_value, _ = read_json(prepared_dir / "fit/train-stats.json", files["fit/train-stats.json"])
    if not isinstance(fit_value, dict):
        fail()
    fit = cast("dict[str, object]", fit_value)
    stats = _checked_train_stats(stats_value, ids)
    if (
        fit
        != {
            "schema_version": 1,
            "kind": "arb-ml-fit",
            "dataset_identity": manifest["dataset_identity"],
            "protocol_identity": manifest["protocol_identity"],
            "train_interactions": files["fit/train-interactions.tsv.gz"],
            "train_stats": files["fit/train-stats.json"],
        }
        or "sha256:" + sha(canonical_bytes(protocol_value)) != manifest["protocol_identity"]
    ):
        fail()
    protocol = load_protocol(_protocol_bytes=protocol_raw)
    return (
        manifest,
        protocol,
        catalog,
        fit,
        "sha256:" + sha(fit_raw),
        stats,
    )


def _predictions(
    path: str | Path,
    prepared_manifest: PreparationManifest,
    prepared_identity: str,
    split: str,
    fit_identity: str,
) -> tuple[PredictionManifest, str]:
    path = safe_path(path, directory=True)
    value, raw = read_json(path / "prediction-manifest.json")
    if type(value) is not dict or type(value.get("schema_version")) is not int:
        fail("research_prediction_invalid")
    version = value["schema_version"]
    if version not in (1, 2):
        fail("research_prediction_invalid")
    if {item.name for item in path.iterdir()} != {
        "prediction-manifest.json",
        "scores.bin" if version == 2 else "scores.jsonl.gz",
    }:
        fail("research_prediction_invalid")
    if version == 2 and len(raw) > MAX_MANIFEST:
        fail("research_prediction_invalid")
    exact(
        value,
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
        }
        | ({"score_format"} if version == 2 else set()),
    )
    manifest = cast("PredictionManifest", value)
    if (
        type(manifest["schema_version"]) is not int
        or manifest["schema_version"] != version
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
        or (version == 2 and manifest["score_format"] != FORMAT)
    ):
        fail("research_prediction_invalid")
    if version == 2:
        check_block_descriptor(manifest["scores"])
    else:
        check_descriptor(manifest["scores"], stream=True)
    check_descriptor(manifest["query_descriptor"], stream=True)
    check_descriptor(manifest["catalog_descriptor"])
    check_descriptor(manifest["train_interactions"], stream=True)
    check_descriptor(manifest["train_stats"])
    return manifest, "sha256:" + sha(raw)


def _paired_rows(
    prepared_dir: Path,
    prep: PreparationManifest,
    split: str,
    allowed: set[tuple[str, str, int]] | None = None,
    *,
    protocol: ProtocolSpec,
    ids: set[str],
) -> Iterator[tuple[EvaluationQuery, PreparationLabel]]:
    """Validation reads only development queries and validation labels."""
    files = prep["files"]
    labels = stream_jsonl(
        prepared_dir / f"evaluator/{split}-labels.jsonl.gz",
        cast("StreamDescriptor", files[f"evaluator/{split}-labels.jsonl.gz"]),
    )
    queries = (
        stream_jsonl(
            prepared_dir / "development/queries.jsonl.gz",
            cast("StreamDescriptor", files["development/queries.jsonl.gz"]),
        )
        if split == "validation"
        else None
    )
    seen: set[str] = set()
    for label_value in labels:
        exact(
            label_value,
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
        label = _check_label(label_value, protocol, ids)
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
        raw_query = cast("dict[str, object]", query)
        if (
            any(
                type(raw_query[key]) is not str
                for key in ("query_id", "cohort", "seed_policy", "order")
            )
            or type(raw_query["n"]) is not int
            or type(raw_query["seed_mal_ids"]) is not list
            or any(type(item) is not int for item in raw_query["seed_mal_ids"])
        ):
            fail()
        query = cast("EvaluationQuery", raw_query)
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


def _rank(
    scores: object,
    ids: list[int],
    excluded: set[int],
    *,
    n0: bool = False,
    stats: EvaluationTrainStats | None = None,
) -> list[int]:
    if type(scores) is not list or len(scores) != len(ids):
        fail("research_prediction_invalid")

    def candidates() -> Iterator[tuple[float, int]]:
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
            if n0 and (stats is None or score != stats["items"][str(expected)]["positive_users"]):
                fail("research_prediction_invalid")
            if expected not in excluded and score is not None:
                yield (-cast("float", score), expected)

    return [item for _, item in nsmallest(10, candidates())]


def _rank_binary(row: DecodedScoreRow, ids: list[int], excluded: set[int]) -> list[int]:
    def candidates() -> Iterator[tuple[int, int]]:
        for index, mal_id in enumerate(ids):
            if mal_id not in excluded and row.available[index]:
                yield (-int(row.values[index]), mal_id)

    return [item for _, item in nsmallest(10, candidates())]


def _history_bin(count: int) -> str:
    if count < 20:
        return "lt20"
    if count < 50:
        return "20-49"
    if count < 100:
        return "50-99"
    if count < 200:
        return "100-199"
    return "200plus"


def _seed_band(seeds: set[int], stats: EvaluationTrainStats) -> str:
    if not seeds:
        return "none"
    bands = {stats["items"][str(item)]["band"] for item in seeds}
    return next(iter(bands)) if len(bands) == 1 else "mixed"


def _genre_diversity(ranking: list[int], catalog: Catalog) -> GenreDiversity:
    pairs = len(ranking) * (len(ranking) - 1) // 2
    valid: list[float] = []
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


def _slice(rows: list[PrivateMetric]) -> dict[str, object] | None:
    by_user: defaultdict[str, list[PrivateMetric]] = defaultdict(list)
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


def _aggregate(
    private_rows: Iterable[PrivateMetric], catalog: Catalog, stats: EvaluationTrainStats
) -> list[dict[str, object]]:
    groups: defaultdict[tuple[str, str, str, str, int], list[PrivateMetric]] = defaultdict(list)
    users: defaultdict[tuple[str, str, str, str, int], set[str]] = defaultdict(set)
    coverage: defaultdict[tuple[str, str, str, str, int], set[int]] = defaultdict(set)
    short: Counter[tuple[str, str, str, str, int]] = Counter()
    repeat: Counter[tuple[str, str, str, str, int]] = Counter()
    for row in private_rows:
        key = (row["split"], row["cohort"], row["mask"], row["order"], row["n"])
        groups[key].append(row)
        users[key].add(row["user_key"])
        coverage[key].update(row["ranking"])
        short[key] += len(row["ranking"]) < 5
        repeat[key] += row["known_history_repeats"]
    report: list[dict[str, object]] = []
    for key in sorted(groups):
        rows = groups[key]
        aggregate = _slice(rows)
        if aggregate is None:
            continue
        slices: dict[str, dict[str, dict[str, object]]] = {}
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
    prepared_dir: str | Path,
    prediction_dir: str | Path,
    output_dir: str | Path,
    *,
    split: str,
    expected_prepared_identity: str,
    selection_path: str | Path | None = None,
    expected_selection_identity: str | None = None,
) -> Path:
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
    allowed: set[tuple[str, str, int]] | None = None
    selection_identity: str | None = None
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
    catalog_ids = set(ids)
    label_query = _paired_rows(
        prepared_dir, prep, split, allowed, protocol=protocol, ids=set(catalog["anime"])
    )
    binary = prediction["schema_version"] == 2
    score_rows: Iterator[object]
    if binary:
        score_rows = iter_score_rows(
            prediction_dir / "scores.bin",
            prediction["scores"],
            expected_catalog_ids=ids,
            expected_binding_sha256=sha(
                canonical_bytes({k: v for k, v in prediction.items() if k != "scores"})
            ),
            expected_popularity=tuple(stats["items"][str(item)]["positive_users"] for item in ids),
            popularity_arm=prediction["model_id"] == "popularity",
        )
    else:
        score_rows = stream_jsonl(
            prediction_dir / "scores.jsonl.gz", cast("StreamDescriptor", prediction["scores"])
        )
    stage = open_stage(output_dir)
    try:
        private = StreamWriter(stage / "user-metrics.jsonl.gz")
        test_queries = (
            StreamWriter(stage / "recomputed-test-queries.jsonl.gz") if split == "test" else None
        )
        rows_for_report: list[PrivateMetric] = []
        nrows = 0
        for (query, label), row_value in zip(label_query, score_rows, strict=True):
            if test_queries is not None:
                test_queries.write_json(query)
            if binary:
                binary_row = cast("DecodedScoreRow", row_value)
                require_query_binding(
                    binary_row,
                    query_id=query["query_id"],
                    n=query["n"],
                    popularity_arm=prediction["model_id"] == "popularity",
                )
            else:
                exact(row_value, {"query_id", "scores"})
                row = cast("dict[str, object]", row_value)
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
            controlled_excluded = history - targets
            if binary:
                controlled = _rank_binary(binary_row, ids, controlled_excluded)
                product = _rank_binary(binary_row, ids, seeds)
            else:
                controlled = _rank(
                    row["scores"], ids, controlled_excluded, n0=query["n"] == 0, stats=stats
                )
                product = _rank(row["scores"], ids, seeds, n0=query["n"] == 0, stats=stats)
            for mask, ranking, relevance, excluded in (
                ("controlled", controlled, targets, controlled_excluded),
                ("product", product, positive - seeds, seeds),
            ):
                if (
                    not relevance
                    or len(ranking) != len(set(ranking))
                    or any(item not in catalog_ids or item in excluded for item in ranking)
                ):
                    fail()
                result: PrivateMetric = {
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
