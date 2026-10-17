"""Retired CuPy scoring rejects requests before observing paths or science inputs."""

import tempfile
import unittest
from pathlib import Path
from typing import cast
from unittest.mock import patch

from research_model_support import ModelFixture, patch_release_info

from recsys.contracts.bundle import canonical_bytes
from recsys.research.artifacts import read_json
from recsys.research.campaign.runner import run_campaign
from recsys.research.models.package import fit_model, load_model, read_scoring_inputs
from recsys.research.models.spec import (
    GPU_SCORING_RECOVERY_COMMIT,
    require_cpu_backend,
)
from recsys.research.models.types import ScoringInputs
from recsys.research.predict import predict_validation
from recsys.research.protocol import ResearchError


class _OtherString(str):
    pass


class _ForbiddenPath:
    def __fspath__(self) -> str:
        raise AssertionError("path accessed before backend rejection")


class BackendRetirementTests(unittest.TestCase):
    def test_shared_guard_accepts_only_exact_cpu_and_retires_exact_gpu(self) -> None:
        require_cpu_backend("cpu")
        self.assertEqual(GPU_SCORING_RECOVERY_COMMIT, "fa983eaeeeeb41b41ac893b7de9827a40c25d27d")
        for value, code in (
            ("gpu", "research_gpu_scoring_retired"),
            ("CUDA", "research_backend_unsupported"),
            (None, "research_backend_unsupported"),
            (True, "research_backend_unsupported"),
            (_OtherString("cpu"), "research_backend_unsupported"),
            (_OtherString("gpu"), "research_backend_unsupported"),
        ):
            with self.subTest(value=value), self.assertRaises(ResearchError) as raised:
                require_cpu_backend(value)
            self.assertEqual(str(raised.exception), code)

    def test_four_public_apis_reject_before_paths_or_measurements(self) -> None:
        forbidden = cast("str | Path", _ForbiddenPath())
        for backend, code in (
            ("gpu", "research_gpu_scoring_retired"),
            ("unknown", "research_backend_unsupported"),
            (None, "research_backend_unsupported"),
            (False, "research_backend_unsupported"),
            (_OtherString("cpu"), "research_backend_unsupported"),
        ):
            requested = cast("str", backend)
            measurements: dict[str, object] = {}
            actions = (
                lambda: fit_model(
                    forbidden,
                    forbidden,
                    expected_prepared_identity="unused",
                    variant_id="popularity",
                    code_revision="a" * 40,
                    backend=requested,
                ),
                lambda: load_model(
                    forbidden, inputs=cast("ScoringInputs", None), backend=requested
                ),
                lambda: predict_validation(
                    forbidden,
                    forbidden,
                    forbidden,
                    expected_prepared_identity="unused",
                    backend=requested,
                    measurements=measurements,
                    measurement_output=forbidden,
                ),
                lambda: run_campaign(
                    forbidden,
                    forbidden,
                    expected_prepared_identity="unused",
                    code_revision="a" * 40,
                    glove=cast("Path", forbidden),
                    backend=requested,
                ),
            )
            for action in actions:
                with (
                    self.subTest(backend=backend, action=action),
                    self.assertRaises(ResearchError) as raised,
                ):
                    action()
                self.assertEqual(str(raised.exception), code)
                self.assertEqual(measurements, {})

    def test_historical_scoring_manifest_rejected_before_payload_access(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = ModelFixture(directory)
            prepared, identity = fixture.make_prepared()
            model = fixture.fit_popularity(prepared, identity)
            manifest_path = model / "model-manifest.json"
            manifest, _ = read_json(manifest_path)
            assert isinstance(manifest, dict)
            producer = manifest["producer"]
            assert isinstance(producer, dict)
            execution = producer["execution"]
            assert isinstance(execution, dict)
            execution["scoring"] = {"backend": "cupy"}
            manifest_path.write_bytes(canonical_bytes(manifest))
            with (
                patch_release_info(fixture.lock),
                patch(
                    "recsys.research.models.manifest._check_payloads",
                    side_effect=AssertionError("payload touched"),
                ),
            ):
                inputs = read_scoring_inputs(prepared, expected_prepared_identity=identity)
                with self.assertRaises(ResearchError) as raised:
                    load_model(model, inputs=inputs)
            self.assertEqual(str(raised.exception), "research_gpu_scoring_retired")
