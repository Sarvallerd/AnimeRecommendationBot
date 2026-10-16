"""Portable fixed content/SVD reciprocal-rank fusion for ML-02."""

from __future__ import annotations

import os
import shutil

from . import research_models as models
from .bundle import canonical_bytes
from .research_artifacts import read_json, safe_path
from .research_protocol import exact, fail, sha

CODE_FILES = ("research_model_hybrid.py", "research_model_content.py", "build.py",
              "sources.py", "research_model_cf.py")
DEPENDENCIES = ("scipy", "scikit-learn", "threadpoolctl")
_PARTS = (("content", "content_model", "content-mean", "content_"),
          ("svd", "puresvd_model", "puresvd", "svd_"))
_METADATA = {"schema_version", "components", "component_ranking",
             "visible_seed_policy", "support_policy"}


def _parameters(parameters):
    if canonical_bytes(parameters) != canonical_bytes(models.EXPECTED_VARIANTS["hybrid"]):
        fail("research_model_invalid")


def _component_manifest(model_dir, inputs, backend, revision, variant):
    manifest, raw = models._read_model_package(
        model_dir, inputs=inputs, backend=backend, expected_code_revision=revision)
    if manifest["variant_id"] != variant:
        fail("research_model_invalid")
    return manifest, raw


def fit(inputs, parameters, assets, stage, *, code_revision, backend):
    """Embed already fitted packages; this operation never opens train rows."""
    if backend != "cpu":
        fail("research_backend_unsupported")
    _parameters(parameters)
    if type(assets) is not dict or set(assets) != {"content_model", "puresvd_model"}:
        fail("research_model_invalid")
    components = {}
    for name, asset, variant, prefix in _PARTS:
        source = safe_path(assets[asset], directory=True)
        manifest, raw = _component_manifest(source, inputs, backend, code_revision, variant)
        # A source package is semantically checked before its immutable bytes are copied.
        scorer = models._backend(variant).load(source, manifest, inputs, backend="cpu")
        close = getattr(scorer, "close", None)
        if callable(close):
            close()
        target_name = name + "_manifest.json"
        with open(stage / target_name, "xb", opener=lambda path, flags: os.open(path, flags, 0o600)) as out:
            out.write(raw)
            out.flush()
            os.fsync(out.fileno())
        for logical in manifest["files"]:
            original = safe_path(source / logical, directory=False)
            target = stage / (prefix + logical)
            with original.open("rb") as inp, open(
                    target, "xb", opener=lambda path, flags: os.open(path, flags, 0o600)) as out:
                shutil.copyfileobj(inp, out, 1024 * 1024)
                out.flush()
                os.fsync(out.fileno())
            if models._file_descriptor(target) != {
                    key: manifest["files"][logical][key] for key in ("bytes", "sha256")}:
                fail("research_model_invalid")
        components[name] = {"variant_id": variant, "manifest_file": target_name,
                            "payload_prefix": prefix, "identity": "sha256:" + sha(raw)}
    return {"schema_version": 1, "components": components,
            "component_ranking": "raw-float64-before-publisher-quantization",
            "visible_seed_policy": "exclude-before-component-ranking",
            "support_policy": "union-of-component-supported-seeds"}


def _embedded(model_dir, outer, inputs, backend):
    metadata = outer["metadata"]
    exact(metadata, _METADATA)
    if (type(metadata["schema_version"]) is not int or metadata["schema_version"] != 1 or
            metadata["component_ranking"] != "raw-float64-before-publisher-quantization" or
            metadata["visible_seed_policy"] != "exclude-before-component-ranking" or
            metadata["support_policy"] != "union-of-component-supported-seeds" or
            type(metadata["components"]) is not dict or
            set(metadata["components"]) != {"content", "svd"} or
            set(outer["external_inputs"]) != {"content_model", "puresvd_model"}):
        fail("research_model_invalid")
    expected_files = set()
    result = {}
    revision = outer["producer"]["code_revision"]
    execution = outer["producer"]["execution"]
    for name, asset, variant, prefix in _PARTS:
        entry = metadata["components"][name]
        exact(entry, {"variant_id", "manifest_file", "payload_prefix", "identity"})
        logical_name = name + "_manifest.json"
        if (entry["variant_id"] != variant or entry["manifest_file"] != logical_name or
                entry["payload_prefix"] != prefix):
            fail("research_model_invalid")
        expected_files.add(logical_name)
        manifest, raw = read_json(model_dir / logical_name)
        raw_descriptor = models.descriptor(raw)
        if (outer["external_inputs"][asset] != raw_descriptor or
                outer["files"].get(logical_name) != raw_descriptor or
                entry["identity"] != "sha256:" + sha(raw) or
                manifest.get("variant_id") != variant):
            fail("research_model_invalid")
        models._validate_manifest(manifest, inputs=inputs, backend=backend,
                                  expected_code_revision=revision)
        if manifest["producer"]["execution"] != execution:
            fail("research_backend_mismatch")
        for logical, description in manifest["files"].items():
            actual = prefix + logical
            if actual in expected_files or outer["files"].get(actual) != description:
                fail("research_model_invalid")
            expected_files.add(actual)
        result[name] = manifest
    if set(outer["files"]) != expected_files:
        fail("research_model_invalid")
    return result


def _fusion(seed_sets, ids, content, svd):
    import numpy as np

    if type(seed_sets) is not tuple or len(seed_sets) > 128:
        fail("research_model_invalid")
    left = content.score_many(seed_sets)
    right = svd.score_many(seed_sets)
    if len(left) != len(seed_sets) or len(right) != len(seed_sets):
        fail("research_model_invalid")
    id_array = np.asarray(ids, dtype=np.int64)
    positions = {item: index for index, item in enumerate(ids)}
    output = []
    for seeds, a, b in zip(seed_sets, left, right, strict=True):
        if (type(seeds) is not tuple or len(seeds) > 10 or len(seeds) != len(set(seeds)) or
                any(type(item) is not int or item not in positions for item in seeds)):
            fail("research_model_invalid")
        visible = np.zeros(len(ids), dtype=np.bool_)
        visible[[positions[item] for item in seeds]] = True
        available = np.zeros(len(ids), dtype=np.bool_)
        values = np.zeros(len(ids), dtype=np.float64)
        for vector, weight in ((a, .5), (b, .5)):
            if (not isinstance(vector, models.ScoreVector) or
                    vector.values.shape != (len(ids),) or
                    vector.available.shape != (len(ids),) or
                    not np.isfinite(vector.values).all()):
                fail("research_model_invalid")
            candidates = np.flatnonzero(vector.available & ~visible)
            order = np.lexsort((id_array[candidates], -vector.values[candidates]))
            ranked = candidates[order]
            values[ranked] += weight / (60 + np.arange(1, len(ranked) + 1))
            available[ranked] = True
        supported = len(content.supported_seed_ids(seeds) | svd.supported_seed_ids(seeds))
        if not 0 <= supported <= len(seeds):
            fail("research_model_invalid")
        output.append(models.ScoreVector(values, available, supported))
    return tuple(output)


class HybridScorer:
    def __init__(self, ids, content, svd):
        self.ids = ids
        self.content = content
        self.svd = svd

    def close(self):
        try:
            close = getattr(self.svd, "close", None)
            if callable(close):
                close()
        finally:
            close = getattr(self.content, "close", None)
            if callable(close):
                close()

    def score_many(self, seed_sets):
        return _fusion(seed_sets, self.ids, self.content, self.svd)

    def score(self, seed_mal_ids):
        return self.score_many((seed_mal_ids,))[0]


def load(model_dir, manifest, inputs, *, backend="cpu"):
    """Load the two embedded CPU components without their source directories."""
    if backend != "cpu":
        fail("research_backend_unsupported")
    _parameters(manifest["parameters"])
    embedded = _embedded(model_dir, manifest, inputs, backend)
    loaded = []
    try:
        for name, _, variant, prefix in _PARTS:
            scorer = models._backend(variant).load(model_dir, embedded[name], inputs,
                                                    backend="cpu",
                                                    payload_prefix=prefix)
            loaded.append(scorer)
        return HybridScorer(inputs.mal_ids, loaded[0], loaded[1])
    except BaseException:
        for scorer in reversed(loaded):
            close = getattr(scorer, "close", None)
            if callable(close):
                close()
        raise
