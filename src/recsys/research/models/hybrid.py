"""Portable fixed content/SVD reciprocal-rank fusion for ML-02."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import TypedDict, cast

from recsys.contracts.bundle import JSONValue, canonical_bytes
from recsys.research.artifacts import descriptor, file_descriptor, read_json, safe_path
from recsys.research.models.collaborative import load as load_collaborative
from recsys.research.models.content import load as load_content
from recsys.research.models.manifest import ModelManifest, _read_model_package, _validate_manifest
from recsys.research.models.spec import EXPECTED_VARIANTS
from recsys.research.models.types import (
    CloseableScorer,
    ScoreVector,
    ScoringInputs,
    SupportedBatchScorer,
)
from recsys.research.protocol import exact, fail, sha

_PARTS = (
    ("content", "content_model", "content-mean", "content_"),
    ("svd", "puresvd_model", "puresvd", "svd_"),
)
_METADATA = {
    "schema_version",
    "components",
    "component_ranking",
    "visible_seed_policy",
    "support_policy",
}


class EmbeddedComponent(TypedDict):
    variant_id: str
    manifest_file: str
    payload_prefix: str
    identity: str


class HybridMetadata(TypedDict):
    schema_version: int
    components: dict[str, EmbeddedComponent]
    component_ranking: str
    visible_seed_policy: str
    support_policy: str


def _parameters(parameters: dict[str, JSONValue]) -> None:
    if canonical_bytes(parameters) != canonical_bytes(EXPECTED_VARIANTS["hybrid"]):
        fail("research_model_invalid")


def _component_manifest(
    model_dir: Path,
    inputs: ScoringInputs,
    backend: str,
    revision: str,
    variant: str,
) -> tuple[ModelManifest, bytes]:
    manifest, raw = _read_model_package(
        model_dir, inputs=inputs, backend=backend, expected_code_revision=revision
    )
    if manifest["variant_id"] != variant:
        fail("research_model_invalid")
    return manifest, raw


def fit(
    inputs: ScoringInputs,
    parameters: dict[str, JSONValue],
    assets: dict[str, Path],
    stage: Path,
    *,
    code_revision: str,
    backend: str,
) -> dict[str, object]:
    """Embed already fitted packages; this operation never opens train rows."""
    if backend != "cpu":
        fail("research_backend_unsupported")
    _parameters(parameters)
    if type(assets) is not dict or set(assets) != {"content_model", "puresvd_model"}:
        fail("research_model_invalid")
    components: dict[str, EmbeddedComponent] = {}
    for name, asset, variant, prefix in _PARTS:
        source = safe_path(assets[asset], directory=True)
        manifest, raw = _component_manifest(source, inputs, backend, code_revision, variant)
        # A source package is semantically checked before its immutable bytes are copied.
        scorer = (load_content if variant == "content-mean" else load_collaborative)(
            source, manifest, inputs, backend="cpu"
        )
        close = getattr(scorer, "close", None)
        if callable(close):
            close()
        target_name = name + "_manifest.json"
        with open(
            stage / target_name, "xb", opener=lambda path, flags: os.open(path, flags, 0o600)
        ) as out:
            out.write(raw)
            out.flush()
            os.fsync(out.fileno())
        for logical in manifest["files"]:
            original = safe_path(source / logical, directory=False)
            target = stage / (prefix + logical)
            with (
                original.open("rb") as inp,
                open(target, "xb", opener=lambda path, flags: os.open(path, flags, 0o600)) as out,
            ):
                shutil.copyfileobj(inp, out, 1024 * 1024)
                out.flush()
                os.fsync(out.fileno())
            if file_descriptor(target) != {
                key: manifest["files"][logical][key] for key in ("bytes", "sha256")
            }:
                fail("research_model_invalid")
        components[name] = {
            "variant_id": variant,
            "manifest_file": target_name,
            "payload_prefix": prefix,
            "identity": "sha256:" + sha(raw),
        }
    return {
        "schema_version": 1,
        "components": components,
        "component_ranking": "raw-float64-before-publisher-quantization",
        "visible_seed_policy": "exclude-before-component-ranking",
        "support_policy": "union-of-component-supported-seeds",
    }


def _embedded(
    model_dir: Path,
    outer: ModelManifest,
    inputs: ScoringInputs,
    backend: str,
) -> dict[str, ModelManifest]:
    metadata_value = outer["metadata"]
    metadata = cast("HybridMetadata", metadata_value)
    exact(metadata, _METADATA)
    if (
        type(metadata["schema_version"]) is not int
        or metadata["schema_version"] != 1
        or metadata["component_ranking"] != "raw-float64-before-publisher-quantization"
        or metadata["visible_seed_policy"] != "exclude-before-component-ranking"
        or metadata["support_policy"] != "union-of-component-supported-seeds"
        or type(metadata["components"]) is not dict
        or set(metadata["components"]) != {"content", "svd"}
        or set(outer["external_inputs"]) != {"content_model", "puresvd_model"}
    ):
        fail("research_model_invalid")
    expected_files: set[str] = set()
    result: dict[str, ModelManifest] = {}
    revision = outer["producer"]["code_revision"]
    execution = outer["producer"]["execution"]
    for name, asset, variant, prefix in _PARTS:
        entry = metadata["components"][name]
        exact(entry, {"variant_id", "manifest_file", "payload_prefix", "identity"})
        logical_name = name + "_manifest.json"
        if (
            entry["variant_id"] != variant
            or entry["manifest_file"] != logical_name
            or entry["payload_prefix"] != prefix
        ):
            fail("research_model_invalid")
        expected_files.add(logical_name)
        manifest_value, raw = read_json(model_dir / logical_name)
        raw_descriptor = descriptor(raw)
        if (
            outer["external_inputs"][asset] != raw_descriptor
            or outer["files"].get(logical_name) != raw_descriptor
            or entry["identity"] != "sha256:" + sha(raw)
            or not isinstance(manifest_value, dict)
            or manifest_value.get("variant_id") != variant
        ):
            fail("research_model_invalid")
        manifest = _validate_manifest(
            manifest_value, inputs=inputs, backend=backend, expected_code_revision=revision
        )
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


def _fusion(
    seed_sets: tuple[tuple[int, ...], ...],
    ids: tuple[int, ...],
    content: SupportedBatchScorer,
    svd: SupportedBatchScorer,
) -> tuple[ScoreVector, ...]:
    import numpy as np

    if type(seed_sets) is not tuple or len(seed_sets) > 128:
        fail("research_model_invalid")
    left = content.score_many(seed_sets)
    right = svd.score_many(seed_sets)
    if len(left) != len(seed_sets) or len(right) != len(seed_sets):
        fail("research_model_invalid")
    id_array = np.asarray(ids, dtype=np.int64)
    positions = {item: index for index, item in enumerate(ids)}
    output: list[ScoreVector] = []
    for seeds, a, b in zip(seed_sets, left, right, strict=True):
        if (
            type(seeds) is not tuple
            or len(seeds) > 10
            or len(seeds) != len(set(seeds))
            or any(type(item) is not int or item not in positions for item in seeds)
        ):
            fail("research_model_invalid")
        visible = np.zeros(len(ids), dtype=np.bool_)
        visible[[positions[item] for item in seeds]] = True
        available = np.zeros(len(ids), dtype=np.bool_)
        values = np.zeros(len(ids), dtype=np.float64)
        for vector, weight in ((a, 0.5), (b, 0.5)):
            if (
                not isinstance(vector, ScoreVector)
                or vector.values.shape != (len(ids),)
                or vector.available.shape != (len(ids),)
                or not np.isfinite(vector.values).all()
            ):
                fail("research_model_invalid")
            candidates = np.flatnonzero(vector.available & ~visible)
            order = np.lexsort((id_array[candidates], -vector.values[candidates]))
            ranked = candidates[order]
            values[ranked] += weight / (60 + np.arange(1, len(ranked) + 1))
            available[ranked] = True
        supported = len(content.supported_seed_ids(seeds) | svd.supported_seed_ids(seeds))
        if not 0 <= supported <= len(seeds):
            fail("research_model_invalid")
        output.append(ScoreVector(values, available, supported))
    return tuple(output)


class HybridScorer:
    def __init__(
        self, ids: tuple[int, ...], content: SupportedBatchScorer, svd: SupportedBatchScorer
    ) -> None:
        self.ids = ids
        self.content = content
        self.svd = svd

    def close(self) -> None:
        try:
            close = getattr(self.svd, "close", None)
            if callable(close):
                close()
        finally:
            close = getattr(self.content, "close", None)
            if callable(close):
                close()

    def score_many(self, seed_sets: tuple[tuple[int, ...], ...]) -> tuple[ScoreVector, ...]:
        return _fusion(seed_sets, self.ids, self.content, self.svd)

    def score(self, seed_mal_ids: tuple[int, ...]) -> ScoreVector:
        return self.score_many((seed_mal_ids,))[0]


def load(
    model_dir: Path, manifest: ModelManifest, inputs: ScoringInputs, *, backend: str = "cpu"
) -> HybridScorer:
    """Load the two embedded CPU components without their source directories."""
    if backend != "cpu":
        fail("research_backend_unsupported")
    _parameters(manifest["parameters"])
    embedded = _embedded(model_dir, manifest, inputs, backend)
    loaded: list[CloseableScorer] = []
    content_scorer: SupportedBatchScorer | None = None
    svd_scorer: SupportedBatchScorer | None = None
    try:
        for name, _, variant, prefix in _PARTS:
            if variant == "content-mean":
                content_loaded = load_content(
                    model_dir, embedded[name], inputs, backend="cpu", payload_prefix=prefix
                )
                content_scorer = content_loaded
                loaded.append(content_loaded)
            else:
                svd_loaded = cast(
                    "SupportedBatchScorer",
                    load_collaborative(
                        model_dir, embedded[name], inputs, backend="cpu", payload_prefix=prefix
                    ),
                )
                svd_scorer = svd_loaded
                loaded.append(svd_loaded)
        if content_scorer is None or svd_scorer is None:
            fail("research_model_invalid")
        return HybridScorer(inputs.mal_ids, content_scorer, svd_scorer)
    except BaseException:
        for scorer in reversed(loaded):
            close = getattr(scorer, "close", None)
            if callable(close):
                close()
        raise
