from __future__ import annotations

import builtins
from types import TracebackType
from typing import Literal, Self

class threadpool_limits:
    def __init__(
        self, limits: int | None = ..., user_api: Literal["blas", "openmp"] | None = ...
    ) -> None: ...
    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        type: builtins.type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...

def threadpool_info(debugging_info: bool = ...) -> list[dict[str, object]]: ...
