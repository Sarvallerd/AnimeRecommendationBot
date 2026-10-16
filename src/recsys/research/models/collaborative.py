"""Sparse, train-only collaborative baselines for the ML-02 campaign."""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypedDict, cast, overload

from recsys.contracts.bundle import JSONValue
from recsys.research.models.inputs import iter_training_rows
from recsys.research.models.manifest import ModelManifest
from recsys.research.models.types import Scorer, ScoreVector, ScoringInputs, TrainingInputs
from recsys.research.protocol import exact, fail

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray
    from scipy.sparse import csr_matrix

    from recsys.research.gpu import CupyDenseDots


class KnnParameters(TypedDict):
    positive_min_rating: int
    shrinkage: int
    source_block: int
    top_k: int


class SvdParameters(TypedDict):
    flip_sign: bool
    n_iter: int
    n_oversamples: int
    power_iteration_normalizer: Literal["auto", "QR", "LU", "none"]
    rank: int
    random_state: int
    svd_lapack_driver: Literal["gesdd", "gesvd"]
    transpose: bool


_SOURCE = "fit/train-interactions.tsv.gz"
_KNN: KnnParameters = {"positive_min_rating": 8, "shrinkage": 10, "source_block": 128, "top_k": 200}
_SVD: SvdParameters = {
    "flip_sign": True,
    "n_iter": 4,
    "n_oversamples": 16,
    "power_iteration_normalizer": "LU",
    "rank": 64,
    "random_state": 3737,
    "svd_lapack_driver": "gesdd",
    "transpose": False,
}
_COMMON_METADATA = {
    "kind",
    "source",
    "items",
    "train_users",
    "matrix_users",
    "train_completed_rows",
    "positive_rows",
    "supported_items",
}


def _parameters(parameters: object) -> Literal["itemknn", "puresvd"]:
    if type(parameters) is not dict:
        fail("research_model_invalid")
    parameters = cast("dict[str, object]", parameters)
    if set(parameters) == set(_KNN) and all(
        type(parameters[k]) is type(v) and parameters[k] == v for k, v in _KNN.items()
    ):
        return "itemknn"
    if set(parameters) == set(_SVD) and all(
        type(parameters[k]) is type(v) and parameters[k] == v for k, v in _SVD.items()
    ):
        return "puresvd"
    fail("research_model_invalid")


def _save(path: Path, array: NDArray[np.float64] | NDArray[np.int32]) -> None:
    import numpy as np

    with open(path, "xb", opener=lambda name, flags: os.open(name, flags, 0o600)) as stream:
        np.save(stream, array, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())


def _training_matrix(
    inputs: TrainingInputs,
) -> tuple[csr_matrix[np.float64], int, NDArray[np.int64]]:
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
        if (
            completed[col] != entry["completed_users"]
            or positive[col] != entry["positive_users"]
            or rating_sum[col] != entry["rating_sum"]
            or completed[col] != entry["rating_count"]
        ):
            fail("research_model_invalid")
    data = np.ones(count, dtype=np.float64)
    matrix = csr_matrix(
        (data, indices, indptr[: matrix_users + 1].copy()),
        shape=(matrix_users, m),
        dtype=np.float64,
    )
    matrix.sort_indices()
    return matrix, matrix_users, positive


def _metadata(
    kind: str, inputs: TrainingInputs, matrix_users: int, positive: NDArray[np.int64]
) -> dict[str, object]:
    return {
        "kind": kind,
        "source": _SOURCE,
        "items": len(inputs.mal_ids),
        "train_users": inputs.train_stats["train_users"],
        "matrix_users": matrix_users,
        "train_completed_rows": inputs.train_stats["train_completed_rows"],
        "positive_rows": int(positive.sum()),
        "supported_items": int((positive > 0).sum()),
    }


def _fit_knn(
    matrix: csr_matrix[np.float64],
    ids: tuple[int, ...],
    positive: NDArray[np.int64],
    stage: Path,
) -> int:
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
                lo, hi = block.indptr[row - start : row - start + 2]
                candidate = block.indices[lo:hi]
                cooccurrence = block.data[lo:hi]
                keep = (candidate != row) & (cooccurrence > 0)
                candidate = candidate[keep]
                cooccurrence = cooccurrence[keep]
                if candidate.size:
                    similarity = (
                        cooccurrence
                        / np.sqrt(float(positive[row]) * positive[candidate])
                        * cooccurrence
                        / (cooccurrence + _KNN["shrinkage"])
                    )
                    # mal_ids are sorted, so column order is numeric MAL_ID order.
                    order = np.lexsort((candidate, -similarity))
                    chosen = order[: _KNN["top_k"]]
                    by_column = np.argsort(candidate[chosen], kind="stable")
                    candidate = candidate[chosen][by_column]
                    similarity = similarity[chosen][by_column]
                    take = len(candidate)
                    columns[write : write + take] = candidate
                    weights[write : write + take] = similarity
                    write += take
                pointers[row + 1] = write
    _save(stage / "indptr.npy", pointers)
    _save(stage / "indices.npy", columns[:write])
    _save(stage / "weights.npy", weights[:write])
    return write


def _fit_svd(matrix: csr_matrix[np.float64], positive: NDArray[np.int64], stage: Path) -> int:
    import numpy as np
    from sklearn.utils.extmath import randomized_svd
    from threadpoolctl import threadpool_limits

    m = matrix.shape[1]
    rank = max(0, min(_SVD["rank"], matrix.shape[0] - 1, int((positive > 0).sum()) - 1))
    factors = np.zeros((m, rank), dtype=np.float64)
    if rank:
        with threadpool_limits(limits=1, user_api="blas"):
            _, _, right = randomized_svd(
                matrix,
                n_components=rank,
                n_oversamples=_SVD["n_oversamples"],
                n_iter=_SVD["n_iter"],
                power_iteration_normalizer=_SVD["power_iteration_normalizer"],
                transpose=_SVD["transpose"],
                flip_sign=_SVD["flip_sign"],
                random_state=_SVD["random_state"],
                svd_lapack_driver=_SVD["svd_lapack_driver"],
            )
        factors[:, :] = right.T
        factors[positive == 0] = 0
        if not np.isfinite(factors).all():
            fail("research_model_invalid")
    _save(stage / "factors.npy", factors)
    return rank


def fit(
    inputs: TrainingInputs,
    parameters: dict[str, JSONValue],
    assets: dict[str, Path],
    stage: Path,
) -> dict[str, object]:
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


def _validate_metadata(
    metadata: dict[str, object], kind: str, inputs: ScoringInputs
) -> NDArray[np.int64]:
    import numpy as np

    exact(metadata, _COMMON_METADATA | ({"edges"} if kind == "itemknn" else {"effective_rank"}))
    stats = inputs.train_stats
    positive = np.asarray(
        [stats["items"][str(item)]["positive_users"] for item in inputs.mal_ids], dtype=np.int64
    )
    cmax = max(stats["items"][str(item)]["completed_users"] for item in inputs.mal_ids)
    expected = {
        "kind": kind,
        "source": _SOURCE,
        "items": len(inputs.mal_ids),
        "train_users": stats["train_users"],
        "train_completed_rows": stats["train_completed_rows"],
        "positive_rows": int(positive.sum()),
        "supported_items": int((positive > 0).sum()),
    }
    if any(
        type(metadata[key]) is not type(value) or metadata[key] != value
        for key, value in expected.items()
    ):
        fail("research_model_invalid")
    users = metadata["matrix_users"]
    if (
        type(users) is not int
        or not 0 <= users <= min(stats["train_users"], stats["train_completed_rows"])
        or users < cmax
        or (users == 0) != (stats["train_completed_rows"] == 0)
    ):
        fail("research_model_invalid")
    metric = metadata["edges" if kind == "itemknn" else "effective_rank"]
    if type(metric) is not int or metric < 0:
        fail("research_model_invalid")
    if kind == "puresvd" and metric != max(
        0, min(_SVD["rank"], users - 1, cast("int", expected["supported_items"]) - 1)
    ):
        fail("research_model_invalid")
    if kind == "itemknn" and metric > len(inputs.mal_ids) * _KNN["top_k"]:
        fail("research_model_invalid")
    return positive


@overload
def _array(path: Path, dtype: type[np.int32], shape: tuple[int, ...]) -> NDArray[np.int32]: ...


@overload
def _array(path: Path, dtype: type[np.float64], shape: tuple[int, ...]) -> NDArray[np.float64]: ...


def _array(
    path: Path,
    dtype: type[np.int32] | type[np.float64],
    shape: tuple[int, ...],
) -> NDArray[np.int32] | NDArray[np.float64]:
    import numpy as np

    try:
        value = np.load(path, allow_pickle=False, mmap_mode="r")
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError("research_model_invalid") from exc
    if value.dtype != np.dtype(dtype) or value.shape != shape:
        fail("research_model_invalid")
    if dtype is np.int32:
        return cast("NDArray[np.int32]", value)
    return cast("NDArray[np.float64]", value)


def _seed_columns(seed_mal_ids: tuple[int, ...], columns: dict[int, int]) -> tuple[int, ...]:
    if (
        type(seed_mal_ids) is not tuple
        or len(seed_mal_ids) > 10
        or any(type(item) is not int or item not in columns for item in seed_mal_ids)
        or len(set(seed_mal_ids)) != len(seed_mal_ids)
    ):
        fail("research_model_invalid")
    return tuple(sorted(columns[item] for item in seed_mal_ids))


def load(
    model_dir: Path,
    manifest: ModelManifest,
    inputs: ScoringInputs,
    *,
    backend: str = "cpu",
    payload_prefix: str = "",
) -> Scorer:
    import numpy as np
    from threadpoolctl import threadpool_limits

    kind = manifest["variant_id"]
    if payload_prefix not in ("", "svd_") or (kind != "puresvd" and payload_prefix):
        fail("research_model_invalid")
    if kind not in ("itemknn", "puresvd") or _parameters(manifest["parameters"]) != kind:
        fail("research_model_invalid")
    if backend not in ("cpu", "gpu") or (backend == "gpu" and kind != "puresvd"):
        fail("research_backend_unsupported")
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
        edges = cast("int", manifest["metadata"]["edges"])
        pointers = _array(model_dir / (payload_prefix + "indptr.npy"), np.int32, (m + 1,))
        targets = _array(model_dir / (payload_prefix + "indices.npy"), np.int32, (edges,))
        weights = _array(model_dir / (payload_prefix + "weights.npy"), np.float64, (edges,))
        if (
            pointers[0] != 0
            or pointers[-1] != edges
            or np.any(np.diff(pointers.astype(np.int64)) < 0)
            or np.any(np.diff(pointers.astype(np.int64)) > _KNN["top_k"])
            or np.any(targets < 0)
            or np.any(targets >= m)
            or not np.isfinite(weights).all()
            or np.any(weights <= 0)
            or np.any(weights >= 1)
        ):
            fail("research_model_invalid")
        for row in range(m):
            start, end = pointers[row : row + 2]
            dest = targets[start:end]
            if (
                positive[row] == 0
                and end > start
                or np.any(dest == row)
                or np.any(positive[dest] == 0)
                or np.any(np.diff(dest.astype(np.int64)) <= 0)
            ):
                fail("research_model_invalid")

        class ItemKNNScorer:
            def score(self, seed_mal_ids: tuple[int, ...]) -> ScoreVector:
                seeds = _seed_columns(seed_mal_ids, columns)
                supported = sum(positive[col] > 0 for col in seeds)
                if not seeds or not supported:
                    return ScoreVector(zeros, unavailable, int(supported))
                values = np.zeros(m, dtype=np.float64)
                for col in seeds:
                    lo, hi = pointers[col : col + 2]
                    values[targets[lo:hi]] += weights[lo:hi]
                values /= len(seeds)
                available = values > 0
                return ScoreVector(values, available, int(supported))

        return ItemKNNScorer()

    if set(manifest["files"]) != {"factors.npy"}:
        fail("research_model_invalid")
    rank = cast("int", manifest["metadata"]["effective_rank"])
    factors = _array(model_dir / (payload_prefix + "factors.npy"), np.float64, (m, rank))
    if not np.isfinite(factors).all() or np.any(factors[positive == 0] != 0):
        fail("research_model_invalid")
    supported_items = positive > 0
    supported_items.setflags(write=False)
    dots: CupyDenseDots | None = None
    if backend == "gpu" and rank:
        from recsys.research.gpu import CupyDenseDots

        dots = CupyDenseDots(np.ascontiguousarray(np.asarray(factors), dtype=np.float64))

    class PureSVDScorer:
        def __init__(self) -> None:
            self.dots = dots
            self.closed = False

        def close(self) -> None:
            if self.closed:
                return
            self.closed = True
            if self.dots is not None:
                owned, self.dots = self.dots, None
                owned.close()

        def score(self, seed_mal_ids: tuple[int, ...]) -> ScoreVector:
            return self.score_many((seed_mal_ids,))[0]

        def supported_seed_ids(self, seed_mal_ids: tuple[int, ...]) -> frozenset[int]:
            seeds = _seed_columns(seed_mal_ids, columns)
            return frozenset(
                int(inputs.mal_ids[col]) for col in seeds if rank and positive[col] > 0
            )

        def score_many(self, seed_sets: tuple[tuple[int, ...], ...]) -> tuple[ScoreVector, ...]:
            if backend == "gpu" and self.closed:
                fail("research_gpu_closed")
            if type(seed_sets) is not tuple or len(seed_sets) > 128:
                fail("research_model_invalid")
            answer: list[ScoreVector | None] = [None] * len(seed_sets)
            positions: list[int] = []
            profiles: list[NDArray[np.float64]] = []
            supports: list[int] = []
            for position, seed_mal_ids in enumerate(seed_sets):
                seeds = _seed_columns(seed_mal_ids, columns)
                supported = sum(positive[col] > 0 for col in seeds)
                if not seeds or not supported or rank == 0:
                    answer[position] = ScoreVector(zeros, unavailable, 0)
                    continue
                # The CPU fold-in and the denominator are identical for both backends.
                profile = np.asarray(
                    factors[list(seeds)].sum(axis=0) / len(seeds), dtype=np.float64
                )
                positions.append(position)
                profiles.append(profile)
                supports.append(int(supported))
            if profiles:
                if backend == "gpu":
                    if self.dots is None:
                        fail("research_gpu_closed")
                    block = np.ascontiguousarray(np.asarray(profiles), dtype=np.float64)
                    rows: Iterable[NDArray[np.float64]] = self.dots.dot(block)
                else:
                    with threadpool_limits(limits=1, user_api="blas"):
                        rows = tuple(
                            np.asarray(factors @ profile, dtype=np.float64) for profile in profiles
                        )
                for position, values, supported in zip(positions, rows, supports, strict=True):
                    if not np.isfinite(values).all():
                        fail("research_model_invalid")
                    answer[position] = ScoreVector(values, supported_items, supported)
            if any(item is None for item in answer):
                fail("research_model_invalid")
            return cast("tuple[ScoreVector, ...]", tuple(answer))

    return PureSVDScorer()
