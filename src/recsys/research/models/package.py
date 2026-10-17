"""Private ML-02 model inputs, fixed variants, and immutable model packages."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, cast, overload

from recsys.research.artifacts import cleanup, open_stage, publish, safe_path, separate, write_json
from recsys.research.models.dispatch import (
    HybridBackend,
    PopularityBackend,
    StandardBackend,
    _backend,
)
from recsys.research.models.inputs import read_scoring_inputs, read_training_inputs
from recsys.research.models.manifest import (
    ModelManifest,
    _external,
    _payloads,
    _producer,
    _read_model_package,
)
from recsys.research.models.spec import _model_backend, campaign
from recsys.research.models.types import (
    CloseableScorer,
    Scorer,
    ScoringInputs,
    SupportedBatchScorer,
    TrainingInputs,
)
from recsys.research.protocol import fail


def fit_model(
    prepared_dir: str | Path,
    output_dir: str | Path,
    *,
    expected_prepared_identity: str,
    variant_id: str,
    code_revision: str,
    assets: dict[str, Path] | None = None,
    backend: str = "cpu",
) -> Path:
    prepared_dir, output_dir = Path(prepared_dir), Path(output_dir)
    _model_backend(variant_id, backend)
    separate(prepared_dir, output_dir)
    inputs = (
        read_scoring_inputs(prepared_dir, expected_prepared_identity=expected_prepared_identity)
        if variant_id == "hybrid"
        else read_training_inputs(
            prepared_dir, expected_prepared_identity=expected_prepared_identity
        )
    )
    variant_module = _backend(variant_id)
    spec, campaign_hash = campaign()
    external = _external(variant_id, assets, prepared_dir, output_dir)
    if backend == "gpu":
        from recsys.research.gpu import probe_gpu

        scoring = probe_gpu()
    else:
        scoring = None
    producer = _producer(code_revision, variant_id, model_backend=backend, scoring=scoring)
    parameters = spec["variants"][variant_id]
    stage = open_stage(output_dir)
    try:
        if variant_id == "hybrid":
            metadata = cast("HybridBackend", variant_module).fit(
                inputs,
                parameters,
                {} if assets is None else assets,
                stage,
                code_revision=code_revision,
                backend=backend,
            )
        else:
            metadata = cast("PopularityBackend | StandardBackend", variant_module).fit(
                cast("TrainingInputs", inputs), parameters, {} if assets is None else assets, stage
            )
        if type(metadata) is not dict:
            fail("research_model_invalid")
        files = _payloads(stage)
        manifest: ModelManifest = {
            "schema_version": 2,
            "kind": "arb-ml02-model",
            "variant_id": variant_id,
            "parameters": parameters,
            "campaign_spec_sha256": campaign_hash,
            "prepared_identity": inputs.prepared_identity,
            "protocol_identity": inputs.protocol_identity,
            "fit_identity": inputs.fit_identity,
            "catalog_descriptor": inputs.catalog_descriptor,
            "train_interactions": inputs.train_interactions_descriptor,
            "train_stats": inputs.train_stats_descriptor,
            "external_inputs": external,
            "producer": producer,
            "files": files,
            "metadata": metadata,
        }
        write_json(stage / "model-manifest.json", manifest)
        publish(stage, output_dir)
        return output_dir / "model-manifest.json"
    except BaseException:
        cleanup(stage)
        raise


@overload
def load_model(
    model_dir: str | Path, *, inputs: ScoringInputs, backend: Literal["gpu"]
) -> tuple[ModelManifest, SupportedBatchScorer]: ...


@overload
def load_model(
    model_dir: str | Path, *, inputs: ScoringInputs, backend: str = "cpu"
) -> tuple[ModelManifest, Scorer]: ...


def load_model(
    model_dir: str | Path, *, inputs: ScoringInputs, backend: str = "cpu"
) -> tuple[ModelManifest, Scorer]:
    model_dir = safe_path(model_dir, directory=True)
    manifest, raw = _read_model_package(model_dir, inputs=inputs, backend=backend)
    variant_module = _backend(manifest["variant_id"])
    if manifest["variant_id"] == "popularity":
        scorer = cast("PopularityBackend", variant_module).load(model_dir, manifest, inputs)
    elif manifest["variant_id"] == "hybrid":
        scorer = cast("HybridBackend", variant_module).load(
            model_dir, manifest, inputs, backend=backend
        )
    else:
        scorer = cast("StandardBackend", variant_module).load(
            model_dir, manifest, inputs, backend=backend
        )
    if not callable(getattr(scorer, "score", None)):
        if isinstance(scorer, CloseableScorer):
            scorer.close()
        fail("research_model_invalid")
    if backend == "gpu" and not isinstance(scorer, SupportedBatchScorer):
        if isinstance(scorer, CloseableScorer):
            scorer.close()
        fail("research_model_invalid")
    return manifest, scorer
