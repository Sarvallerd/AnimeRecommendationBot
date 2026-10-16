from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

type FloatArray = NDArray[np.float64]
type SparseFloat = csr_matrix[np.float64]

def randomized_svd(
    M: FloatArray | SparseFloat,
    n_components: int,
    *,
    n_oversamples: int = ...,
    n_iter: int | Literal["auto"] = ...,
    power_iteration_normalizer: Literal["auto", "QR", "LU", "none"] = ...,
    transpose: bool | Literal["auto"] = ...,
    flip_sign: bool = ...,
    random_state: int | None = ...,
    svd_lapack_driver: Literal["gesdd", "gesvd"] = ...,
) -> tuple[FloatArray, FloatArray, FloatArray]: ...
