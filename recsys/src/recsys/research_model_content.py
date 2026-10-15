"""Pinned GloVe content baselines over the entire frozen MAL catalog."""

from collections import OrderedDict
import os
from pathlib import Path

from .build import BuildError, DIMENSION, _tokens, build_vectors, load_embeddings
from . import research_models as models
from .research_protocol import exact, fail

CODE_FILES = ("research_model_content.py", "build.py", "sources.py")
DEPENDENCIES = ()
VECTORIZER = "glove-known-occurrence-mean-l2-v1"
CACHE_BYTES = 512 * 1024 * 1024


def _same_typed(value, expected):
    if type(value) is not type(expected):
        return False
    if type(value) is dict:
        return set(value) == set(expected) and all(
            _same_typed(value[key], expected[key]) for key in expected)
    if type(value) is list:
        return len(value) == len(expected) and all(
            _same_typed(actual, wanted) for actual, wanted in zip(value, expected))
    return value == expected


def _variant(parameters):
    matches = [name for name in ("content-mean", "content-max", "content-union")
               if _same_typed(parameters, models.EXPECTED_VARIANTS[name])]
    if len(matches) != 1:
        fail("research_model_invalid")
    return matches[0]


def _save_array(path, array):
    with open(path, "xb", opener=lambda name, flags: os.open(name, flags, 0o600)) as stream:
        import numpy as np
        np.save(stream, array, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())


def fit(inputs, parameters, assets, stage):
    """Fit only catalog text and the locked embeddings; never read train rows."""
    import numpy as np

    _variant(parameters)
    if (type(assets) is not dict or set(assets) != {"glove"} or
            not isinstance(assets["glove"], Path)):
        fail("research_model_invalid")
    anime = inputs.catalog["anime"]
    if tuple(sorted(map(int, anime))) != inputs.mal_ids:
        fail("research_model_invalid")
    vocabulary = {token for item in anime.values() for token in _tokens(item)}
    locked = models.release_info()["inputs"]["embeddings"]
    try:
        embeddings, _ = load_embeddings(assets["glove"], vocabulary,
                                        {"size": locked["text_size"],
                                         "sha256": locked["text_sha256"]})
        ids, vectors, details = build_vectors(anime, embeddings)
    except (BuildError, OSError, UnicodeError, ValueError) as exc:
        raise ValueError("research_model_invalid") from exc
    if tuple(ids) != inputs.mal_ids or vectors.shape != (len(ids), DIMENSION):
        fail("research_model_invalid")
    ready = np.asarray([details["anime"][str(item)]["vector_status"] == "ready"
                        for item in ids], dtype=np.bool_)
    vectors = np.ascontiguousarray(vectors, dtype="<f8")
    if (not np.isfinite(vectors).all() or
            not np.array_equal(ready, np.any(vectors != 0, axis=1))):
        fail("research_model_invalid")
    _save_array(stage / "vectors.npy", vectors)
    _save_array(stage / "ready.npy", ready)
    return {"source": "model-catalog.json", "vectorizer": VECTORIZER,
            "dimension": DIMENSION, "items": len(ids),
            "ready_items": int(np.count_nonzero(ready))}


class _LRU:
    def __init__(self):
        self.entries = OrderedDict()
        self.bytes = 0

    def get(self, key):
        value = self.entries.get(key)
        if value is not None:
            self.entries.move_to_end(key)
        return value

    def put(self, key, value):
        size = value.nbytes
        if size > CACHE_BYTES:
            return value
        while self.bytes + size > CACHE_BYTES:
            _, old = self.entries.popitem(last=False)
            self.bytes -= old.nbytes
        value.setflags(write=False)
        self.entries[key] = value
        self.bytes += size
        return value


class ContentScorer:
    def __init__(self, ids, vectors, ready, variant):
        import numpy as np
        self.ids = np.asarray(ids, dtype=np.int64)
        self.lookup = {item: index for index, item in enumerate(ids)}
        self.vectors = vectors
        self.ready = ready
        self.variant = variant
        self.cache = _LRU()

    def _row(self, index):
        import numpy as np
        row = np.einsum("ik,jk->ij", self.vectors[index:index + 1],
                        self.vectors, optimize=False)[0]
        if not np.isfinite(row).all() or np.any(row > 1 + 1e-12):
            fail("research_model_invalid")
        np.minimum(row, 1.0, out=row)
        return row

    def _source(self, index):
        import numpy as np
        cached = self.cache.get(index)
        if cached is not None:
            return cached
        row = self._row(index)
        if self.variant != "content-union":
            return self.cache.put(index, row)
        candidates = np.flatnonzero(self.ready & (row > 0))
        if len(candidates) > 110:
            boundary = np.partition(row[candidates], -110)[-110]
            candidates = candidates[row[candidates] >= boundary]
        order = np.lexsort((self.ids[candidates], -row[candidates]))
        top = np.asarray(candidates[order[:110]], dtype=np.int32)
        return self.cache.put(index, top)

    def score(self, seed_mal_ids):
        import numpy as np
        if (type(seed_mal_ids) is not tuple or len(seed_mal_ids) > 10 or
                any(type(item) is not int or item not in self.lookup
                    for item in seed_mal_ids) or
                len(set(seed_mal_ids)) != len(seed_mal_ids)):
            fail("research_model_invalid")
        supported = [self.lookup[item] for item in sorted(seed_mal_ids)
                     if self.ready[self.lookup[item]]]
        values = np.zeros(len(self.ids), dtype=np.float64)
        available = np.zeros(len(self.ids), dtype=np.bool_)
        if not supported:
            return models.ScoreVector(values, available, 0)
        if self.variant == "content-mean":
            for index in supported:
                values += self._source(index)
            values /= len(supported)
            available[:] = self.ready & (values > 0)
        elif self.variant == "content-max":
            values[:] = self._source(supported[0])
            for index in supported[1:]:
                np.maximum(values, self._source(index), out=values)
            available[:] = self.ready & (values > 0)
        else:
            visible = {self.lookup[item] for item in seed_mal_ids}
            for index in supported:
                ranked = [target for target in self._source(index) if target not in visible][:100]
                for rank, target in enumerate(ranked, 1):
                    values[target] += 1.0 / (60 + rank)
            values /= len(supported)
            available[:] = values > 0
        return models.ScoreVector(values, available, len(supported))


def load(model_dir, manifest, inputs):
    """Recheck the payload semantics after the common manifest and hash checks."""
    import numpy as np

    variant = manifest["variant_id"]
    if variant not in ("content-mean", "content-max", "content-union") or not _same_typed(
            manifest["parameters"], models.EXPECTED_VARIANTS[variant]):
        fail("research_model_invalid")
    metadata = manifest["metadata"]
    exact(metadata, {"source", "vectorizer", "dimension", "items", "ready_items"})
    if (metadata["source"] != "model-catalog.json" or
            metadata["vectorizer"] != VECTORIZER or
            type(metadata["dimension"]) is not int or metadata["dimension"] != DIMENSION or
            type(metadata["items"]) is not int or metadata["items"] != len(inputs.mal_ids) or
            type(metadata["ready_items"]) is not int or
            not 0 <= metadata["ready_items"] <= len(inputs.mal_ids) or
            set(manifest["files"]) != {"vectors.npy", "ready.npy"}):
        fail("research_model_invalid")
    try:
        vectors = np.load(model_dir / "vectors.npy", mmap_mode="r", allow_pickle=False)
        ready = np.load(model_dir / "ready.npy", mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError("research_model_invalid") from exc
    if (vectors.dtype != np.dtype("<f8") or vectors.dtype.byteorder not in ("<", "=") or
            vectors.shape != (len(inputs.mal_ids), DIMENSION) or
            not vectors.flags.c_contiguous or vectors.flags.writeable or
            ready.dtype != np.dtype("bool") or ready.shape != (len(inputs.mal_ids),) or
            not ready.flags.c_contiguous or ready.flags.writeable or
            not np.isfinite(vectors).all() or
            not np.array_equal(ready, np.any(vectors != 0, axis=1)) or
            np.any(vectors[~ready] != 0) or
            int(np.count_nonzero(ready)) != metadata["ready_items"] or
            not np.all(np.abs(np.linalg.norm(vectors[ready], axis=1) - 1) <= 1e-12)):
        fail("research_model_invalid")
    return ContentScorer(inputs.mal_ids, vectors, ready, variant)
