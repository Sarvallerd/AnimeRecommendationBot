"""Private ML-02 model inputs, fixed variants, and immutable model packages."""

from __future__ import annotations

import importlib
import importlib.metadata
import platform
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from recsys import __version__
from recsys.contracts.bundle import canonical_bytes
from recsys.contracts.provenance import SOURCE_LAYOUT, source_fingerprints
from recsys.release.lock import release_info
from recsys.research.artifacts import (
    check_descriptor,
    file_descriptor,
    read_json,
    safe_path,
    separate,
)
from recsys.research.models.spec import (
    EXPECTED_VARIANTS,
    GPU_DEPENDENCIES,
    VARIANT_DEPENDENCIES,
    _model_backend,
    campaign,
)
from recsys.research.models.types import ScoringInputs
from recsys.research.protocol import exact, fail

MODEL_FIELDS = {
    "schema_version",
    "kind",
    "variant_id",
    "parameters",
    "campaign_spec_sha256",
    "prepared_identity",
    "protocol_identity",
    "fit_identity",
    "catalog_descriptor",
    "train_interactions",
    "train_stats",
    "external_inputs",
    "producer",
    "files",
    "metadata",
}


HEX40 = re.compile(r"[0-9a-f]{40}\Z")


HEX64 = re.compile(r"[0-9a-f]{64}\Z")


NAME = re.compile(r"[a-z][a-z0-9_-]*\.(?:json|npy)\Z")


def _external(variant_id, assets, prepared_dir, output_dir):
    assets = {} if assets is None else assets
    if type(assets) is not dict:
        fail("research_model_invalid")
    required = (
        {"glove"}
        if variant_id.startswith("content-")
        else {"content_model", "puresvd_model"}
        if variant_id == "hybrid"
        else set()
    )
    if set(assets) != required or any(not isinstance(path, Path) for path in assets.values()):
        fail("research_model_invalid")
    entries = {}
    for name, path in assets.items():
        separate(path, prepared_dir, output_dir)
        desc = file_descriptor(path / "model-manifest.json" if variant_id == "hybrid" else path)
        locked = release_info()["inputs"]["embeddings"]
        if name == "glove" and (
            desc != {"bytes": locked["text_size"], "sha256": locked["text_sha256"]}
        ):
            fail("research_model_invalid")
        entries[name] = desc
    return entries


def _producer(code_revision, variant_id, *, model_backend="cpu", scoring=None):
    import numpy as np

    if type(code_revision) is not str or HEX40.fullmatch(code_revision) is None:
        fail("research_model_invalid")
    if variant_id not in VARIANT_DEPENDENCIES:
        fail("research_model_variant_invalid")
    dependencies = {"numpy": np.__version__}
    for name in VARIANT_DEPENDENCIES[variant_id]:
        if name in dependencies or type(name) is not str:
            fail("research_model_invalid")
        dependencies[name] = importlib.metadata.version(name)
    if model_backend == "gpu":
        if type(scoring) is not dict:
            fail("research_backend_mismatch")
        for name in GPU_DEPENDENCIES:
            try:
                dependencies[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                fail("research_gpu_unavailable")
    elif model_backend != "cpu" or scoring is not None:
        fail("research_backend_unsupported")
    execution = {"backend": "cpu", "dtype": "float64", "blas_threads": 1}
    if scoring is not None:
        execution["scoring"] = scoring
    return {
        "code_revision": code_revision,
        "code_files_sha256": source_fingerprints(),
        "source_layout": SOURCE_LAYOUT,
        "python_version": platform.python_version(),
        "package_version": __version__,
        "dependency_versions": dependencies,
        "execution": execution,
    }


def _payloads(stage):
    import numpy as np

    if any(
        entry.is_symlink() or not entry.is_file() or NAME.fullmatch(entry.name) is None
        for entry in stage.iterdir()
    ):
        fail("research_model_invalid")
    answer = {}
    for entry in sorted(stage.iterdir()):
        base = file_descriptor(entry)
        if entry.suffix == ".json":
            read_json(entry, base)
        else:
            try:
                array = np.load(entry, allow_pickle=False, mmap_mode="r")
            except (OSError, ValueError, TypeError) as exc:
                raise ValueError("research_model_invalid") from exc
            if not isinstance(array, np.ndarray) or array.dtype.hasobject or array.dtype.fields:
                fail("research_model_invalid")
            base = {**base, "dtype": array.dtype.str, "shape": list(array.shape)}
        answer[entry.name] = base
    return answer


def _check_payloads(model_dir, files):
    import numpy as np

    if type(files) is not dict or any(
        type(name) is not str or NAME.fullmatch(name) is None for name in files
    ):
        fail("research_model_invalid")
    observed = {entry.name for entry in model_dir.iterdir()}
    if observed != {"model-manifest.json", *files}:
        fail("research_model_invalid")
    for name, desc in files.items():
        entry = safe_path(model_dir / name, directory=False)
        if entry.suffix == ".json":
            exact(desc, {"bytes", "sha256"})
            check_descriptor(desc)
            read_json(entry, desc)
        else:
            exact(desc, {"bytes", "sha256", "dtype", "shape"})
            check_descriptor({key: desc[key] for key in ("bytes", "sha256")})
            if (
                type(desc["dtype"]) is not str
                or type(desc["shape"]) is not list
                or any(type(n) is not int or n < 0 for n in desc["shape"])
            ):
                fail("research_model_invalid")
        if file_descriptor(entry) != {key: desc[key] for key in ("bytes", "sha256")}:
            fail("research_model_invalid")
        if entry.suffix == ".npy":
            try:
                array = np.load(entry, allow_pickle=False, mmap_mode="r")
            except (OSError, ValueError, TypeError) as exc:
                raise ValueError("research_model_invalid") from exc
            if (
                not isinstance(array, np.ndarray)
                or array.dtype.hasobject
                or array.dtype.fields
                or array.dtype.str != desc["dtype"]
                or list(array.shape) != desc["shape"]
            ):
                fail("research_model_invalid")


def _validate_manifest(
    manifest, *, inputs: ScoringInputs, backend="cpu", expected_code_revision=None
):
    exact(manifest, MODEL_FIELDS)
    variant = manifest["variant_id"]
    _model_backend(variant, backend)
    if type(variant) is not str or variant not in EXPECTED_VARIANTS:
        fail("research_model_variant_invalid")
    spec, campaign_hash = campaign()
    if (
        type(manifest["producer"]) is not dict
        or type(manifest["producer"].get("code_revision")) is not str
    ):
        fail("research_model_invalid")
    producer = manifest["producer"]
    if manifest.get("schema_version") != 2 or producer.get("source_layout") != SOURCE_LAYOUT:
        fail("research_source_layout_unsupported")
    if expected_code_revision is not None and producer["code_revision"] != expected_code_revision:
        fail("research_model_invalid")
    execution = producer.get("execution")
    if type(execution) is not dict or set(execution) != (
        {"backend", "dtype", "blas_threads", "scoring"}
        if backend == "gpu"
        else {"backend", "dtype", "blas_threads"}
    ):
        fail("research_backend_mismatch")
    scoring = execution.get("scoring") if backend == "gpu" else None
    if backend == "gpu":
        from recsys.research.gpu import gpu_execution

        with gpu_execution(expected_scoring=scoring):
            pass
    expected_producer = _producer(
        producer["code_revision"], variant, model_backend=backend, scoring=scoring
    )
    if (
        type(manifest["schema_version"]) is not int
        or manifest["schema_version"] != 2
        or manifest["kind"] != "arb-ml02-model"
        or canonical_bytes(manifest["parameters"]) != canonical_bytes(spec["variants"][variant])
        or manifest["campaign_spec_sha256"] != campaign_hash
        or manifest["prepared_identity"] != inputs.prepared_identity
        or manifest["protocol_identity"] != inputs.protocol_identity
        or manifest["fit_identity"] != inputs.fit_identity
        or manifest["catalog_descriptor"] != inputs.catalog_descriptor
        or manifest["train_interactions"] != inputs.train_interactions_descriptor
        or manifest["train_stats"] != inputs.train_stats_descriptor
        or type(manifest["external_inputs"]) is not dict
        or canonical_bytes(producer) != canonical_bytes(expected_producer)
    ):
        fail("research_model_invalid")
    if variant in ("popularity", "itemknn", "puresvd") and manifest["external_inputs"]:
        fail("research_model_invalid")
    if variant.startswith("content-"):
        locked = release_info()["inputs"]["embeddings"]
        if manifest["external_inputs"] != {
            "glove": {"bytes": locked["text_size"], "sha256": locked["text_sha256"]}
        }:
            fail("research_model_invalid")
    if variant == "hybrid":
        if set(manifest["external_inputs"]) != {"content_model", "puresvd_model"}:
            fail("research_model_invalid")
    for desc in manifest["external_inputs"].values():
        check_descriptor(desc)


def _read_model_package(
    model_dir, *, inputs: ScoringInputs, backend="cpu", expected_code_revision=None
):
    model_dir = safe_path(model_dir, directory=True)
    manifest, raw = read_json(model_dir / "model-manifest.json")
    _validate_manifest(
        manifest, inputs=inputs, backend=backend, expected_code_revision=expected_code_revision
    )
    _check_payloads(model_dir, manifest["files"])
    return manifest, raw
