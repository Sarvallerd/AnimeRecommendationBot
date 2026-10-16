from __future__ import annotations

from typing import Literal, Self

import numpy as np
from numpy.typing import NDArray

type FloatArray = NDArray[np.float64]

class KMeans:
    labels_: NDArray[np.int32 | np.int64]
    inertia_: float
    n_iter_: int
    def __init__(
        self,
        n_clusters: int = ...,
        *,
        init: Literal["k-means++", "random"] = ...,
        n_init: int | Literal["auto"] = ...,
        max_iter: int = ...,
        tol: float = ...,
        verbose: int = ...,
        random_state: int | None = ...,
        copy_x: bool = ...,
        algorithm: Literal["lloyd", "elkan"] = ...,
    ) -> None: ...
    def fit(self, X: FloatArray, y: None = ..., sample_weight: FloatArray | None = ...) -> Self: ...
