"""Private ML-02 model inputs, fixed variants, and immutable model packages."""

from __future__ import annotations

import importlib
import importlib.metadata
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from recsys.research.protocol import fail

BACKENDS = {
    "popularity": "popularity",
    "content-mean": "content",
    "content-max": "content",
    "content-union": "content",
    "itemknn": "collaborative",
    "puresvd": "collaborative",
    "hybrid": "hybrid",
}


def _backend(variant_id):
    if type(variant_id) is not str or variant_id not in BACKENDS:
        fail("research_model_variant_invalid")
    try:
        return importlib.import_module("." + BACKENDS[variant_id], __package__)
    except ModuleNotFoundError as exc:
        if exc.name == f"{__package__}.{BACKENDS[variant_id]}":
            fail("research_model_unavailable")
        raise
