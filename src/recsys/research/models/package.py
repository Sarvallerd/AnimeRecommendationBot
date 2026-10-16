"""Private ML-02 model inputs, fixed variants, and immutable model packages."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from recsys.research.artifacts import cleanup, open_stage, publish, safe_path, separate, write_json
from recsys.research.models.dispatch import _backend
from recsys.research.models.inputs import read_scoring_inputs, read_training_inputs
from recsys.research.models.manifest import _external, _payloads, _producer, _read_model_package
from recsys.research.models.spec import _model_backend, campaign
from recsys.research.models.types import ScoringInputs
from recsys.research.protocol import fail


def fit_model(
    prepared_dir,
    output_dir,
    *,
    expected_prepared_identity,
    variant_id,
    code_revision,
    assets: dict[str, Path] | None = None,
    backend="cpu",
):
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
        metadata = (
            variant_module.fit(
                inputs,
                parameters,
                {} if assets is None else assets,
                stage,
                code_revision=code_revision,
                backend=backend,
            )
            if variant_id == "hybrid"
            else variant_module.fit(inputs, parameters, {} if assets is None else assets, stage)
        )
        if type(metadata) is not dict:
            fail("research_model_invalid")
        files = _payloads(stage)
        manifest = {
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


def load_model(model_dir, *, inputs: ScoringInputs, backend="cpu"):
    model_dir = safe_path(model_dir, directory=True)
    manifest, raw = _read_model_package(model_dir, inputs=inputs, backend=backend)
    variant_module = _backend(manifest["variant_id"])
    scorer = (
        variant_module.load(model_dir, manifest, inputs, backend="gpu")
        if backend == "gpu"
        else variant_module.load(model_dir, manifest, inputs)
    )
    if not callable(getattr(scorer, "score", None)):
        close = getattr(scorer, "close", None)
        if callable(close):
            close()
        fail("research_model_invalid")
    return manifest, scorer
