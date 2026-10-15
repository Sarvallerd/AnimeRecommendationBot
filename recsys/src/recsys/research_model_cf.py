"""Sparse, train-only collaborative baselines for the ML-02 campaign."""

from __future__ import annotations

import os

from .research_models import ScoreVector, iter_training_rows
from .research_protocol import exact, fail

CODE_FILES = ("research_model_cf.py",)
DEPENDENCIES = ("scipy", "scikit-learn", "threadpoolctl")
_SOURCE = "fit/train-interactions.tsv.gz"
_KNN = {"positive_min_rating": 8, "shrinkage": 10, "source_block": 128, "top_k": 200}
_SVD = {"flip_sign": True, "n_iter": 4, "n_oversamples": 16,
        "power_iteration_normalizer": "LU", "rank": 64, "random_state": 3737,
        "svd_lapack_driver": "gesdd", "transpose": False}
_COMMON_METADATA = {"kind", "source", "items", "train_users", "matrix_users",
                    "train_completed_rows", "positive_rows", "supported_items"}


def _parameters(parameters):
    if type(parameters) is not dict:
        fail("research_model_invalid")
    if set(parameters) == set(_KNN) and all(type(parameters[k]) is type(v) and parameters[k] == v
                                            for k, v in _KNN.items()):
        return "itemknn"
    if set(parameters) == set(_SVD) and all(type(parameters[k]) is type(v) and parameters[k] == v
                                            for k, v in _SVD.items()):
        return "puresvd"
    fail("research_model_invalid")


def _save(path, array):
    import numpy as np
    with open(path, "xb", opener=lambda name, flags: os.open(name, flags, 0o600)) as stream:
        np.save(stream, array, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())


def _training_matrix(inputs):
    """Build binary positive CSR without retaining users or raw interaction rows."""
    import numpy as np
    from scipy.sparse import csr_matrix

    ids = inputs.mal_ids
    stats = inputs.train_stats
    m = len(ids)
    users = stats["train_users"]
    expected_positive = sum(stats["items"][str(item)]["positive_users"] for item in ids)
    limit = np.iinfo(np.int32).max
    if m > limit or users > limit or expected_positive > limit:
        fail("research_resource_limit")
    columns = {item: col for col, item in enumerate(ids)}
    indices = np.empty(expected_positive, dtype=np.int32)
    indptr = np.empty(users + 1, dtype=np.int32)
    indptr[0] = 0
    completed = np.zeros(m, dtype=np.int64)
    positive = np.zeros(m, dtype=np.int64)
    rating_sum = np.zeros(m, dtype=np.int64)
    count = 0
    matrix_users = 0
    current_user = None
    for user, item, rating in iter_training_rows(inputs):
        if user != current_user:
            if current_user is not None:
                indptr[matrix_users] = count
            matrix_users += 1
            if matrix_users > users:
                fail("research_model_invalid")
            current_user = user
        col = columns[item]
        completed[col] += 1
        rating_sum[col] += rating
        if rating >= 8:
            if count >= expected_positive:
                fail("research_model_invalid")
            indices[count] = col
            positive[col] += 1
            count += 1
    if current_user is not None:
        indptr[matrix_users] = count
    if count != expected_positive or (matrix_users == 0) != (stats["train_completed_rows"] == 0):
        fail("research_model_invalid")
    for col, item in enumerate(ids):
        entry = stats["items"][str(item)]
        if (completed[col] != entry["completed_users"] or
                positive[col] != entry["positive_users"] or
                rating_sum[col] != entry["rating_sum"] or
                completed[col] != entry["rating_count"]):
            fail("research_model_invalid")
    data = np.ones(count, dtype=np.float64)
    matrix = csr_matrix((data, indices, indptr[:matrix_users + 1].copy()),
                        shape=(matrix_users, m), dtype=np.float64)
    matrix.sort_indices()
    return matrix, matrix_users, positive


def _metadata(kind, inputs, matrix_users, positive):
    return {"kind": kind, "source": _SOURCE, "items": len(inputs.mal_ids),
            "train_users": inputs.train_stats["train_users"],
            "matrix_users": matrix_users,
            "train_completed_rows": inputs.train_stats["train_completed_rows"],
            "positive_rows": int(positive.sum()),
            "supported_items": int((positive > 0).sum())}


def _fit_knn(matrix, ids, positive, stage):
    import numpy as np
    from threadpoolctl import threadpool_limits

    m = len(ids)
    max_edges = m * _KNN["top_k"]
    if max_edges > np.iinfo(np.int32).max:
        fail("research_resource_limit")
    pointers = np.zeros(m + 1, dtype=np.int32)
    columns = np.empty(max_edges, dtype=np.int32)
    weights = np.empty(max_edges, dtype=np.float64)
    write = 0
    with threadpool_limits(limits=1, user_api="blas"):
        transposed = matrix.T.tocsr()
        transposed.sort_indices()
        for start in range(0, m, _KNN["source_block"]):
            end = min(start + _KNN["source_block"], m)
            block = (transposed[start:end] @ matrix).tocsr()
            block.sum_duplicates()
            block.sort_indices()
            for row in range(start, end):
                lo, hi = block.indptr[row - start:row - start + 2]
                candidate = block.indices[lo:hi]
                cooccurrence = block.data[lo:hi]
                keep = (candidate != row) & (cooccurrence > 0)
                candidate = candidate[keep]
                cooccurrence = cooccurrence[keep]
                if candidate.size:
                    similarity = (cooccurrence / np.sqrt(float(positive[row]) * positive[candidate])
                                  * cooccurrence / (cooccurrence + _KNN["shrinkage"]))
                    # mal_ids are sorted, so column order is numeric MAL_ID order.
                    order = np.lexsort((candidate, -similarity))
                    chosen = order[:_KNN["top_k"]]
                    by_column = np.argsort(candidate[chosen], kind="stable")
                    candidate = candidate[chosen][by_column]
                    similarity = similarity[chosen][by_column]
                    take = len(candidate)
                    columns[write:write + take] = candidate
                    weights[write:write + take] = similarity
                    write += take
                pointers[row + 1] = write
    _save(stage / "indptr.npy", pointers)
    _save(stage / "indices.npy", columns[:write])
    _save(stage / "weights.npy", weights[:write])
    return write


def _fit_svd(matrix, positive, stage):
    import numpy as np
    from sklearn.utils.extmath import randomized_svd
    from threadpoolctl import threadpool_limits

    m = matrix.shape[1]
    rank = max(0, min(_SVD["rank"], matrix.shape[0] - 1, int((positive > 0).sum()) - 1))
    factors = np.zeros((m, rank), dtype=np.float64)
    if rank:
        with threadpool_limits(limits=1, user_api="blas"):
            _, _, right = randomized_svd(
                matrix, n_components=rank, n_oversamples=_SVD["n_oversamples"],
                n_iter=_SVD["n_iter"],
                power_iteration_normalizer=_SVD["power_iteration_normalizer"],
                transpose=_SVD["transpose"], flip_sign=_SVD["flip_sign"],
                random_state=_SVD["random_state"],
                svd_lapack_driver=_SVD["svd_lapack_driver"])
        factors[:, :] = right.T
        factors[positive == 0] = 0
        if not np.isfinite(factors).all():
            fail("research_model_invalid")
    _save(stage / "factors.npy", factors)
    return rank


def fit(inputs, parameters, assets, stage):
    kind = _parameters(parameters)
    if assets != {}:
        fail("research_model_invalid")
    matrix, users, positive = _training_matrix(inputs)
    metadata = _metadata(kind, inputs, users, positive)
    if kind == "itemknn":
        metadata["edges"] = _fit_knn(matrix, inputs.mal_ids, positive, stage)
    else:
        metadata["effective_rank"] = _fit_svd(matrix, positive, stage)
    return metadata


def _validate_metadata(metadata, kind, inputs):
    import numpy as np
    exact(metadata, _COMMON_METADATA | ({"edges"} if kind == "itemknn" else {"effective_rank"}))
    stats = inputs.train_stats
    positive = np.asarray([stats["items"][str(item)]["positive_users"]
                           for item in inputs.mal_ids], dtype=np.int64)
    cmax = max(stats["items"][str(item)]["completed_users"] for item in inputs.mal_ids)
    expected = {"kind": kind, "source": _SOURCE, "items": len(inputs.mal_ids),
                "train_users": stats["train_users"],
                "train_completed_rows": stats["train_completed_rows"],
                "positive_rows": int(positive.sum()),
                "supported_items": int((positive > 0).sum())}
    if any(type(metadata[key]) is not type(value) or metadata[key] != value
           for key, value in expected.items()):
        fail("research_model_invalid")
    users = metadata["matrix_users"]
    if (type(users) is not int or not 0 <= users <= min(stats["train_users"],
                                                        stats["train_completed_rows"]) or
            users < cmax or (users == 0) != (stats["train_completed_rows"] == 0)):
        fail("research_model_invalid")
    metric = metadata["edges" if kind == "itemknn" else "effective_rank"]
    if type(metric) is not int or metric < 0:
        fail("research_model_invalid")
    if kind == "puresvd" and metric != max(0, min(_SVD["rank"], users - 1,
                                                   expected["supported_items"] - 1)):
        fail("research_model_invalid")
    if kind == "itemknn" and metric > len(inputs.mal_ids) * _KNN["top_k"]:
        fail("research_model_invalid")
    return positive


def _array(path, dtype, shape):
    import numpy as np
    try:
        value = np.load(path, allow_pickle=False, mmap_mode="r")
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError("research_model_invalid") from exc
    if value.dtype != np.dtype(dtype) or value.shape != shape:
        fail("research_model_invalid")
    return value


def _seed_columns(seed_mal_ids, columns):
    if (type(seed_mal_ids) is not tuple or len(seed_mal_ids) > 10 or
            any(type(item) is not int or item not in columns for item in seed_mal_ids) or
            len(set(seed_mal_ids)) != len(seed_mal_ids)):
        fail("research_model_invalid")
    return tuple(sorted(columns[item] for item in seed_mal_ids))


def load(model_dir, manifest, inputs):
    import numpy as np
    from threadpoolctl import threadpool_limits

    kind = manifest["variant_id"]
    if kind not in ("itemknn", "puresvd") or _parameters(manifest["parameters"]) != kind:
        fail("research_model_invalid")
    positive = _validate_metadata(manifest["metadata"], kind, inputs)
    m = len(inputs.mal_ids)
    columns = {item: col for col, item in enumerate(inputs.mal_ids)}
    zeros = np.zeros(m, dtype=np.float64)
    unavailable = np.zeros(m, dtype=np.bool_)
    zeros.setflags(write=False)
    unavailable.setflags(write=False)
    if kind == "itemknn":
        if set(manifest["files"]) != {"indptr.npy", "indices.npy", "weights.npy"}:
            fail("research_model_invalid")
        edges = manifest["metadata"]["edges"]
        pointers = _array(model_dir / "indptr.npy", np.int32, (m + 1,))
        targets = _array(model_dir / "indices.npy", np.int32, (edges,))
        weights = _array(model_dir / "weights.npy", np.float64, (edges,))
        if (pointers[0] != 0 or pointers[-1] != edges or
                np.any(np.diff(pointers.astype(np.int64)) < 0) or
                np.any(np.diff(pointers.astype(np.int64)) > _KNN["top_k"]) or
                np.any(targets < 0) or np.any(targets >= m) or
                not np.isfinite(weights).all() or np.any(weights <= 0) or
                np.any(weights >= 1)):
            fail("research_model_invalid")
        for row in range(m):
            start, end = pointers[row:row + 2]
            dest = targets[start:end]
            if (positive[row] == 0 and end > start or
                    np.any(dest == row) or np.any(positive[dest] == 0) or
                    np.any(np.diff(dest.astype(np.int64)) <= 0)):
                fail("research_model_invalid")

        class ItemKNNScorer:
            def score(self, seed_mal_ids):
                seeds = _seed_columns(seed_mal_ids, columns)
                supported = sum(positive[col] > 0 for col in seeds)
                if not seeds or not supported:
                    return ScoreVector(zeros, unavailable, int(supported))
                values = np.zeros(m, dtype=np.float64)
                for col in seeds:
                    lo, hi = pointers[col:col + 2]
                    values[targets[lo:hi]] += weights[lo:hi]
                values /= len(seeds)
                available = values > 0
                return ScoreVector(values, available, int(supported))

        return ItemKNNScorer()

    if set(manifest["files"]) != {"factors.npy"}:
        fail("research_model_invalid")
    rank = manifest["metadata"]["effective_rank"]
    factors = _array(model_dir / "factors.npy", np.float64, (m, rank))
    if not np.isfinite(factors).all() or np.any(factors[positive == 0] != 0):
        fail("research_model_invalid")
    supported_items = positive > 0
    supported_items.setflags(write=False)

    class PureSVDScorer:
        def score(self, seed_mal_ids):
            seeds = _seed_columns(seed_mal_ids, columns)
            supported = sum(positive[col] > 0 for col in seeds)
            if not seeds or not supported or rank == 0:
                return ScoreVector(zeros, unavailable, 0)
            with threadpool_limits(limits=1, user_api="blas"):
                profile = np.asarray(factors[list(seeds)].sum(axis=0) / len(seeds),
                                     dtype=np.float64)
                values = np.asarray(factors @ profile, dtype=np.float64)
            if not np.isfinite(values).all():
                fail("research_model_invalid")
            return ScoreVector(values, supported_items, int(supported))

    return PureSVDScorer()
