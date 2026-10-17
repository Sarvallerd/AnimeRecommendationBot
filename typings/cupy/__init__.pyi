from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy import float64 as float64
from numpy.typing import NDArray

from . import cuda as cuda

__version__: str
type Float64HostArray = NDArray[np.float64]

class ndarray:
    @property
    def T(self) -> ndarray: ...
    def __matmul__(self, other: ndarray, /) -> ndarray: ...

def asarray(
    a: Float64HostArray | Sequence[float], dtype: type[np.float64] | None = ...
) -> ndarray: ...
def asnumpy(a: ndarray, *, blocking: bool = ...) -> Float64HostArray: ...

class ElementwiseKernel:
    def __init__(self, in_params: str, out_params: str, operation: str, name: str) -> None: ...
    def __call__(self, x: ndarray) -> ndarray: ...
