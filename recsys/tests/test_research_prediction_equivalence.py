"""The seven CPU scorers retain their decoded scores and private metrics."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Literal, cast

from research_model_support import ModelFixture, patch_release_info
from test_research_model_content import _glove

from recsys.contracts.bundle import canonical_bytes
from recsys.research.artifacts import StreamDescriptor, read_json, stream_jsonl
from recsys.research.campaign.common import VARIANTS
from recsys.research.evaluation.evaluate import evaluate_predictions
from recsys.research.models.inputs import read_scoring_inputs
from recsys.research.models.package import fit_model
from recsys.research.predict import predict_validation
from recsys.research.protocol import sha
from recsys.research.score_blocks import iter_score_rows


class PredictionEquivalenceTests(unittest.TestCase):
    def test_all_cpu_arms_legacy_and_binary_decoded_scores_and_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ModelFixture(temporary)
            glove = Path(temporary) / "glove.txt"
            fixture.lock["inputs"]["embeddings"] = _glove(glove)
            prepared, identity = fixture.make_prepared()
            root = Path(temporary)
            fitted: dict[str, Path] = {}
            with patch_release_info(fixture.lock):
                inputs = read_scoring_inputs(prepared, expected_prepared_identity=identity)
                for variant in VARIANTS:
                    model = root / f"model-{variant}"
                    assets = (
                        {"glove": glove}
                        if variant.startswith("content-")
                        else {
                            "content_model": fitted["content-mean"],
                            "puresvd_model": fitted["puresvd"],
                        }
                        if variant == "hybrid"
                        else None
                    )
                    fit_model(
                        prepared,
                        model,
                        expected_prepared_identity=identity,
                        variant_id=variant,
                        code_revision="a" * 40,
                        assets=assets,
                    )
                    fitted[variant] = model
                    legacy = root / f"legacy-{variant}"
                    binary = root / f"binary-{variant}"
                    for target, fmt in cast(
                        "tuple[tuple[Path, Literal['jsonl-v1', 'binary-v1']], ...]",
                        ((legacy, "jsonl-v1"), (binary, "binary-v1")),
                    ):
                        predict_validation(
                            prepared,
                            model,
                            target,
                            expected_prepared_identity=identity,
                            score_format=fmt,
                        )
                    old_manifest, _ = read_json(legacy / "prediction-manifest.json")
                    new_manifest, _ = read_json(binary / "prediction-manifest.json")
                    assert isinstance(old_manifest, dict) and isinstance(new_manifest, dict)
                    old_rows = list(
                        stream_jsonl(
                            legacy / "scores.jsonl.gz",
                            cast("StreamDescriptor", old_manifest["scores"]),
                        )
                    )
                    new_rows = list(
                        iter_score_rows(
                            binary / "scores.bin",
                            new_manifest["scores"],
                            expected_catalog_ids=inputs.mal_ids,
                            expected_binding_sha256=sha(
                                canonical_bytes(
                                    {
                                        key: value
                                        for key, value in new_manifest.items()
                                        if key != "scores"
                                    }
                                )
                            ),
                            expected_popularity=tuple(
                                inputs.train_stats["items"][str(item)]["positive_users"]
                                for item in inputs.mal_ids
                            ),
                            popularity_arm=variant == "popularity",
                        )
                    )
                    decoded = [
                        {
                            "query_id": row.query_id,
                            "scores": [
                                [mal_id, int(score) if available else None]
                                for mal_id, score, available in zip(
                                    inputs.mal_ids, row.values, row.available, strict=True
                                )
                            ],
                        }
                        for row in new_rows
                    ]
                    self.assertEqual(old_rows, decoded, variant)
                    outputs = []
                    for target, name in ((legacy, "legacy-eval"), (binary, "binary-eval")):
                        output = root / f"{name}-{variant}"
                        evaluate_predictions(
                            prepared,
                            target,
                            output,
                            split="validation",
                            expected_prepared_identity=identity,
                        )
                        outputs.append(output)
                    self.assertEqual(
                        (outputs[0] / "user-metrics.jsonl.gz").read_bytes(),
                        (outputs[1] / "user-metrics.jsonl.gz").read_bytes(),
                        variant,
                    )
                    replay = root / f"binary-replay-{variant}"
                    evaluate_predictions(
                        prepared,
                        binary,
                        replay,
                        split="validation",
                        expected_prepared_identity=identity,
                    )
                    for filename in (
                        "user-metrics.jsonl.gz",
                        "report.json",
                        "evaluation-manifest.json",
                    ):
                        self.assertEqual(
                            (outputs[1] / filename).read_bytes(),
                            (replay / filename).read_bytes(),
                            (variant, filename),
                        )
