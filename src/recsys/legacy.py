"""A documented, repaired comparator for the original notebook pipeline.

This module deliberately keeps the notebook's metadata serialization and exclusions.
It is not a claim that the lost historical cluster assignments were reproduced.
"""

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass

from .build import DIMENSION


class LegacyError(ValueError):
    """Invalid legacy input or numerical result."""


FEATURES = ("Action", "Military", "Adventure", "Fantasy", "Music", "Romance", "Shoujo",
            "Dementia", "Psychological", "Drama", "Shounen Ai", "Comedy", "Demons", "Ecchi",
            "School", "Parody", "Shounen", "Historical", "Seinen", "Mystery", "Sci-Fi",
            "Space", "Horror", "Martial Arts", "Samurai", "Mecha", "Supernatural",
            "Thriller", "Magic", "Sports", "Harem", "Vampire", "Game", "Super Power",
            "Shoujo Ai", "Kids", "Police", "Slice of Life", "Yaoi", "Josei", "Cars")
FIELDS = ("Name", "Score", "Type", "Episodes", "Source", "Duration", "Rating",
          "Popularity", "Favorites", "sypnopsis", "season", "year", "cluster", "Genres")
ANIME_COLUMNS = ("MAL_ID", "Name", "Score", "Genres", "English name", "Type", "Episodes",
                 "Aired", "Source", "Duration", "Rating", "Popularity", "Favorites")
SYNOPSIS_COLUMNS = ("MAL_ID", "sypnopsis")
REQUIRED = ("Name", "Score", "Genres", "Type", "Episodes", "Aired", "Duration",
            "Rating", "Popularity", "Favorites")
RATINGS = {"G - All Ages": 0, "PG - Children": 7, "PG-13 - Teens 13 or older": 13,
           "R - 17+ (violence & profanity)": 18, "R+ - Mild Nudity": 18}
SEASONS = {"Mar": "spring", "Oct": "autumn", "Apr": "spring", "Jan": "winter",
           "Dec": "winter", "Sep": "autumn", "Jul": "summer", "Aug": "summer",
           "Nov": "autumn", "May": "spring", "Jun": "summer", "Feb": "winter"}
KMEANS = {"n_clusters": 20, "init": "k-means++", "n_init": 10, "max_iter": 300,
          "tol": 1e-4, "verbose": 0, "random_state": 22, "copy_x": True,
          "algorithm": "lloyd"}
RETAINED_SHA256 = "6a621eed1bf9a089ac57f1a4370a3efdcf4e042b56826c632d753e44c062c6cd"


@dataclass
class LegacyDataset:
    records: dict[int, dict]
    parsed_genres: dict[int, set[str]]
    exclusions: dict[int, list[str]]
    counts: dict


def _id(value):
    raw = str(value)
    if not re.fullmatch(r"[1-9][0-9]*", raw) or int(raw) > 2_147_483_647:
        raise LegacyError(f"invalid MAL_ID {raw!r}")
    return int(raw)


def _missing(value):
    # Pandas read_csv's relevant default NA spellings plus the anime-only literal Unknown.
    return value is None or str(value) in {"", "Unknown", "NA", "N/A", "NaN", "nan", "NULL", "null", "None"}


def _index(rows, columns, source):
    indexed = {}
    for item in rows:
        row = item[1] if isinstance(item, tuple) else item
        if not isinstance(row, dict) or any(column not in row for column in columns):
            raise LegacyError(f"{source}: missing consumed column")
        mal_id = _id(row["MAL_ID"])
        if mal_id in indexed:
            raise LegacyError(f"{source}: duplicate MAL_ID {mal_id}")
        indexed[mal_id] = row
    return indexed


def _duration(value):
    parts = value.split()
    seconds = 0
    for i, part in enumerate(parts):
        if part in ("hr.", "min.", "sec."):
            try:
                seconds += int(parts[i - 1]) * {"hr.": 3600, "min.": 60, "sec.": 1}[part]
            except (ValueError, IndexError) as exc:
                raise LegacyError(f"invalid Duration {value!r}") from exc
    return seconds


def _float(value, field, mal_id):
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise LegacyError(f"invalid {field} for MAL_ID {mal_id}") from exc
    if not math.isfinite(result):
        raise LegacyError(f"nonfinite {field} for MAL_ID {mal_id}")
    return result


def prepare_legacy_rows(anime_rows, synopsis_rows) -> LegacyDataset:
    """Apply notebook eligibility with stable IDs and explicit scalar types."""
    anime = _index(anime_rows, ANIME_COLUMNS, "anime")
    synopsis = _index(synopsis_rows, SYNOPSIS_COLUMNS, "synopsis")
    records, genres, exclusions = {}, {}, {}
    missing_counts = Counter()
    metadata_count = 0
    for mal_id, row in sorted(anime.items()):
        reasons = [field for field in REQUIRED if _missing(row[field])]
        # English name replaces Name when present; missing original Name is then harmless.
        name = row["English name"] if not _missing(row["English name"]) else row["Name"]
        if not _missing(name) and "Name" in reasons:
            reasons.remove("Name")
        if _missing(row["Source"]):
            source = "Unknown"
        else:
            source = row["Source"]
        missing_counts.update(reasons)
        if reasons:
            exclusions[mal_id] = sorted(reasons)
            continue
        metadata_count += 1
        syn = synopsis.get(mal_id)
        if syn is None or syn["sypnopsis"] is None or str(syn["sypnopsis"]) in {"", "NA", "N/A", "NaN", "nan", "NULL", "null", "None"}:
            exclusions[mal_id] = ["synopsis_join"]
            continue
        aired = row["Aired"].replace(",", "")
        season = SEASONS.get(aired.split()[0], "Unknown")
        year = None
        for part in aired.split():
            if len(part) == 4:
                try:
                    year = int(part)
                    break
                except ValueError:
                    pass
        if year is None:
            raise LegacyError(f"no year for MAL_ID {mal_id}")
        if row["Rating"] not in RATINGS:
            raise LegacyError(f"unknown Rating for MAL_ID {mal_id}")
        raw_genres = row["Genres"]
        parsed = set(raw_genres.split(", "))
        record = {"Name": str(name), "Score": _float(row["Score"], "Score", mal_id),
                  "Type": str(row["Type"]), "Episodes": _float(row["Episodes"], "Episodes", mal_id),
                  "Source": str(source), "Duration": _duration(row["Duration"]),
                  "Rating": RATINGS[row["Rating"]],
                  "Popularity": _float(row["Popularity"], "Popularity", mal_id),
                  "Favorites": _float(row["Favorites"], "Favorites", mal_id),
                  "sypnopsis": str(syn["sypnopsis"]), "season": season, "year": year,
                  "Genres": raw_genres}
        records[mal_id] = record
        genres[mal_id] = parsed
    counts = {"anime_input": len(anime), "metadata_retained": metadata_count,
              "synopsis_join_retained": len(records), "metadata_missing_by_field": dict(sorted(missing_counts.items())),
              "metadata_excluded": len(anime) - metadata_count,
              "synopsis_join_excluded": metadata_count - len(records)}
    return LegacyDataset(records, genres, exclusions, counts)


def retained_ids_sha(dataset):
    return hashlib.sha256("".join(f"{i}\n" for i in sorted(dataset.records)).encode()).hexdigest()


def cluster_legacy(dataset):
    """Fit the explicit notebook-era KMeans profile on ordered binary genre features."""
    import numpy as np
    from sklearn.cluster import KMeans
    ids = sorted(dataset.records)
    if len(ids) < KMEANS["n_clusters"]:
        raise LegacyError("KMeans requires at least 20 rows")
    matrix = np.ascontiguousarray([[float(feature in dataset.parsed_genres[i]) for feature in FEATURES]
                                   for i in ids], dtype=np.float64)
    fit = KMeans(**KMEANS).fit(matrix)
    labels = {mal_id: int(fit.labels_[index]) for index, mal_id in enumerate(ids)}
    digest = hashlib.sha256((json.dumps([{"mal_id": i, "cluster": labels[i]} for i in ids],
                                      sort_keys=True, separators=(",", ":")) + "\n").encode()).hexdigest()
    return labels, {"feature_order": list(FEATURES), "kmeans_parameters": KMEANS,
                    "cluster_histogram": dict(sorted(Counter(labels.values()).items())),
                    "inertia": float(fit.inertia_), "n_iter": int(fit.n_iter_),
                    "assignment_sha256": digest}


def legacy_tokens(record, cluster: int) -> list[str]:
    if type(cluster) is not int:
        raise LegacyError("cluster must be an integer")
    metadata = {field: cluster if field == "cluster" else record[field] for field in FIELDS}
    return re.sub(r"[^\w\s]", "", str(metadata).lower()).split()


def build_legacy_vectors(dataset, labels, embeddings):
    import numpy as np
    ids = sorted(dataset.records)
    if set(ids) != set(labels):
        raise LegacyError("cluster labels do not cover dataset")
    vectors = np.zeros((len(ids), DIMENSION), dtype=np.float64)
    statuses, counts = {}, Counter()
    texts = []
    for index, mal_id in enumerate(ids):
        record = dataset.records[mal_id]
        cluster = labels[mal_id]
        metadata = {field: cluster if field == "cluster" else record[field] for field in FIELDS}
        texts.append({"mal_id": mal_id, "text": str(metadata)})
        tokens = legacy_tokens(record, cluster)
        matched = 0
        for token in tokens:
            vector = embeddings.get(token)
            if vector is not None:
                if not isinstance(vector, np.ndarray) or vector.shape != (DIMENSION,) or not np.isfinite(vector).all():
                    raise LegacyError(f"invalid GloVe vector {token!r}")
                vectors[index] += vector
                matched += 1
        norm = np.linalg.norm(vectors[index])
        if not np.isfinite(norm):
            raise LegacyError(f"nonfinite vector norm for {mal_id}")
        if norm:
            vectors[index] /= norm
        status = "no_tokens" if not tokens else "oov_only" if not matched else "zero_vector" if not norm else "ready"
        statuses[mal_id] = status
        counts[status] += 1
    vector_hash = hashlib.sha256()
    for i, vector in zip(ids, vectors):
        vector_hash.update(i.to_bytes(4, "little"))
        vector_hash.update(np.asarray(vector, dtype="<f8").tobytes())
    text_hash = hashlib.sha256((json.dumps(texts, sort_keys=True, ensure_ascii=False,
                                           separators=(",", ":")) + "\n").encode()).hexdigest()
    return ids, vectors, {"status_counts": {k: counts[k] for k in ("no_tokens", "oov_only", "zero_vector", "ready")},
                          "nonready_ids": [i for i in ids if statuses[i] != "ready"],
                          "statuses": statuses, "serialized_text_sha256": text_hash,
                          "vector_sha256": vector_hash.hexdigest()}


def legacy_neighbors(dataset, ids, vectors, labels, query_id: int):
    import numpy as np
    if query_id not in dataset.records:
        return {"query_mal_id": query_id, "status": "legacy_exclusion",
                "reasons": dataset.exclusions.get(query_id, ["not_in_legacy_universe"]), "recommendations": []}
    index = ids.index(query_id)
    ready = np.any(vectors != 0, axis=1)
    if not ready[index]:
        return {"query_mal_id": query_id, "status": "zero_vector", "reasons": [], "recommendations": []}
    scores = np.einsum("k,jk->j", vectors[index], vectors, optimize=False)
    if not np.isfinite(scores).all() or np.any(np.abs(scores) > 1 + 1e-12):
        raise LegacyError("invalid cosine similarity")
    np.clip(scores, -1, 1, out=scores)
    order = sorted((j for j, i in enumerate(ids) if i != query_id and ready[j]),
                   key=lambda j: (-float(scores[j]), ids[j]))[:99]
    same_cluster = [j for j in order if labels[ids[j]] == labels[query_id]]
    same_cluster.sort(key=lambda j: (-dataset.records[ids[j]]["Score"],
                                     -dataset.records[ids[j]]["year"], ids[j]))
    recommendations = [{"rank": rank, "mal_id": ids[j], "score": float(scores[j])}
                       for rank, j in enumerate(same_cluster[:5], 1)]
    return {"query_mal_id": query_id,
            "status": "ok" if recommendations else "no_candidates_after_cluster",
            "reasons": [], "recommendations": recommendations}
