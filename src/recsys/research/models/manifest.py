"""Private ML-02 model inputs, fixed variants, and immutable model packages."""

from __future__ import annotations

import importlib
import importlib.metadata
import platform
import re
from pathlib import Path
from typing import TypedDict, cast

from recsys import __version__
from recsys.contracts.bundle import JSONValue, canonical_bytes
from recsys.contracts.provenance import SOURCE_LAYOUT, source_fingerprints
from recsys.release.lock import release_info
from recsys.research.artifacts import (
    FileDescriptor,
    StreamDescriptor,
    check_descriptor,
    file_descriptor,
    read_json,
    safe_path,
    separate,
)
from recsys.research.gpu import GpuMetadata
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


class ArrayDescriptor(FileDescriptor):
    dtype: str
    shape: list[int]


class ModelProducer(TypedDict):
    code_revision: str
    code_files_sha256: dict[str, str]
    source_layout: str
    python_version: str
    package_version: str
    dependency_versions: dict[str, str]
    execution: dict[str, object]


class ModelManifest(TypedDict):
    schema_version: int
    kind: str
    variant_id: str
    parameters: dict[str, JSONValue]
    campaign_spec_sha256: str
    prepared_identity: str
    protocol_identity: str
    fit_identity: str
    catalog_descriptor: FileDescriptor
    train_interactions: StreamDescriptor
    train_stats: FileDescriptor
    external_inputs: dict[str, FileDescriptor]
    producer: ModelProducer
    files: dict[str, FileDescriptor | ArrayDescriptor]
    metadata: dict[str, object]


HEX40 = re.compile(r"[0-9a-f]{40}\Z")


HEX64 = re.compile(r"[0-9a-f]{64}\Z")


NAME = re.compile(r"[a-z][a-z0-9_-]*\.(?:json|npy)\Z")


def _external(
    variant_id: str,
    assets: dict[str, Path] | None,
    prepared_dir: Path,
    output_dir: Path,
) -> dict[str, FileDescriptor]:
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
    entries: dict[str, FileDescriptor] = {}
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


def _producer(
    code_revision: object,
    variant_id: str,
    *,
    model_backend: str = "cpu",
    scoring: GpuMetadata | None = None,
) -> ModelProducer:
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
    execution: dict[str, object] = {"backend": "cpu", "dtype": "float64", "blas_threads": 1}
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


def _payloads(stage: Path) -> dict[str, FileDescriptor | ArrayDescriptor]:
    import numpy as np

    if any(
        entry.is_symlink() or not entry.is_file() or NAME.fullmatch(entry.name) is None
        for entry in stage.iterdir()
    ):
        fail("research_model_invalid")
    answer: dict[str, FileDescriptor | ArrayDescriptor] = {}
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
            base = cast(
                "ArrayDescriptor", {**base, "dtype": array.dtype.str, "shape": list(array.shape)}
            )
        answer[entry.name] = base
    return answer


def _check_payloads(model_dir: Path, files: object) -> None:
    import numpy as np

    if type(files) is not dict or any(
        type(name) is not str or NAME.fullmatch(name) is None for name in files
    ):
        fail("research_model_invalid")
    files = cast("dict[str, dict[str, object]]", files)
    observed = {entry.name for entry in model_dir.iterdir()}
    if observed != {"model-manifest.json", *files}:
        fail("research_model_invalid")
    for name, desc in files.items():
        entry = safe_path(model_dir / name, directory=False)
        if entry.suffix == ".json":
            exact(desc, {"bytes", "sha256"})
            check_descriptor(desc)
            read_json(entry, cast("FileDescriptor", desc))
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
    manifest: object,
    *,
    inputs: ScoringInputs,
    backend: str = "cpu",
    expected_code_revision: str | None = None,
) -> ModelManifest:
    exact(manifest, MODEL_FIELDS)
    manifest = cast("dict[str, object]", manifest)
    variant = manifest["variant_id"]
    if type(variant) is not str or variant not in EXPECTED_VARIANTS:
        fail("research_model_variant_invalid")
    _model_backend(variant, backend)
    spec, campaign_hash = campaign()
    producer_value = manifest["producer"]
    if type(producer_value) is not dict or type(producer_value.get("code_revision")) is not str:
        fail("research_model_invalid")
    producer = cast("dict[str, object]", producer_value)
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
        producer["code_revision"],
        variant,
        model_backend=backend,
        scoring=cast("GpuMetadata | None", scoring),
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
    check_descriptor(manifest["catalog_descriptor"])
    check_descriptor(manifest["train_interactions"], stream=True)
    check_descriptor(manifest["train_stats"])
    external_inputs = cast("dict[str, object]", manifest["external_inputs"])
    if variant in ("popularity", "itemknn", "puresvd") and external_inputs:
        fail("research_model_invalid")
    if variant.startswith("content-"):
        locked = release_info()["inputs"]["embeddings"]
        if external_inputs != {
            "glove": {"bytes": locked["text_size"], "sha256": locked["text_sha256"]}
        }:
            fail("research_model_invalid")
    if variant == "hybrid":
        if set(external_inputs) != {"content_model", "puresvd_model"}:
            fail("research_model_invalid")
    for desc in external_inputs.values():
        check_descriptor(desc)
    files = manifest["files"]
    metadata = manifest["metadata"]
    if type(files) is not dict or type(metadata) is not dict:
        fail("research_model_invalid")
    for name, desc in files.items():
        if type(name) is not str or NAME.fullmatch(name) is None:
            fail("research_model_invalid")
        if name.endswith(".json"):
            check_descriptor(desc)
        else:
            exact(desc, {"bytes", "sha256", "dtype", "shape"})
            if (
                type(desc["dtype"]) is not str
                or type(desc["shape"]) is not list
                or any(type(n) is not int or n < 0 for n in desc["shape"])
            ):
                fail("research_model_invalid")
            check_descriptor({key: desc[key] for key in ("bytes", "sha256")})
    return cast("ModelManifest", manifest)


def _read_model_package(
    model_dir: str | Path,
    *,
    inputs: ScoringInputs,
    backend: str = "cpu",
    expected_code_revision: str | None = None,
) -> tuple[ModelManifest, bytes]:
    model_dir = safe_path(model_dir, directory=True)
    value, raw = read_json(model_dir / "model-manifest.json")
    manifest = _validate_manifest(
        value, inputs=inputs, backend=backend, expected_code_revision=expected_code_revision
    )
    _check_payloads(model_dir, manifest["files"])
    return manifest, raw
