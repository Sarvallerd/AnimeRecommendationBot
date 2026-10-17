"""Explicit CUDA micro-training and separate CPU-only weight loading."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from research_model_support import patch_release_info
from training_support import SeedTrainingFixture

from recsys.contracts.bundle import canonical_bytes
from recsys.research.artifacts import identity
from recsys.research.models.inputs import read_training_inputs
from recsys.research.training.artifacts import publish_training_artifact, verify_training_artifact
from recsys.research.training.candidate import load_candidate
from recsys.research.training.spec import DeviceSpec
from recsys.research.training.torch_model import fit_native, load_native

_CPU_LOADER = """
import json
import sys
import torch
sys.path.insert(0, sys.argv[5])
from research_model_support import patch_release_info
from recsys.research.models.inputs import read_scoring_inputs
from recsys.research.training.spec import DeviceSpec
from recsys.research.training.torch_model import load_native
if torch.__version__ != '2.13.0+cpu' or torch.version.cuda is not None:
    raise SystemExit('cpu-only wheel required')
with patch_release_info(json.loads(sys.argv[4])):
    inputs = read_scoring_inputs(sys.argv[1], expected_prepared_identity=sys.argv[2])
    scorer = load_native(
        sys.argv[3], expected_identity=sys.argv[6], inputs=inputs,
        inference_device=DeviceSpec('cpu', None),
    )
    scores = scorer.score((1, 2))
    print(json.dumps({
        'values': scores.values.tolist(),
        'available': scores.available.tolist(),
        'supported_seeds': scores.supported_seeds,
        'device': scorer.runtime.device.wire(),
        'cuda_peak_allocated_bytes': scorer.runtime.cuda_peak_allocated_bytes,
    }))
    scorer.close()
"""


@unittest.skipUnless(os.environ.get("ARB_TEST_TORCH_CUDA") == "1", "opt-in CUDA smoke")
class TorchCudaTests(unittest.TestCase):
    def test_cuda_train_save_and_cpu_only_loader(self) -> None:
        import numpy as np
        import safetensors
        import torch

        self.assertEqual(torch.__version__, "2.13.0+cu129")
        self.assertEqual(np.__version__, "2.5.3")
        self.assertEqual(safetensors.__version__, "0.8.0")
        self.assertTrue(torch.cuda.is_available(), "opt-in CUDA device unavailable")
        cpu_python = os.environ.get("ARB_TEST_CPU_PYTHON")
        self.assertIsNotNone(cpu_python, "ARB_TEST_CPU_PYTHON must name a CPU-only interpreter")
        assert cpu_python is not None
        self.assertTrue(Path(cpu_python).is_file())
        code_revision = os.environ.get("ARB_TEST_CODE_REVISION", "a" * 40)
        self.assertRegex(code_revision, r"\A[0-9a-f]{40}\Z")
        if "ARB_TEST_CODE_REVISION" in os.environ:
            repository = Path(__file__).resolve().parents[2]
            head = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repository,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            status = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repository,
                capture_output=True,
                text=True,
                check=True,
            ).stdout
            self.assertEqual(head, code_revision)
            self.assertEqual(status, "", "post-commit source checkout must be clean")

        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        fixture = SeedTrainingFixture(temporary.name)
        prepared, prepared_identity = fixture.make_prepared()
        with patch_release_info(fixture.lock):
            inputs = read_training_inputs(prepared, expected_prepared_identity=prepared_identity)
        spec = load_candidate(
            train_device=DeviceSpec("cuda", 0), inference_device=DeviceSpec("cpu", None)
        )
        raw = fixture.root / "raw-cuda.safetensors"
        fit = fit_native(inputs, spec, raw)
        self.assertEqual(fit.supported_mal_ids, (1, 2, 3, 4, 5, 6))
        self.assertEqual(sum(epoch.examples for epoch in fit.epochs), 9)
        self.assertEqual(sum(epoch.optimizer_steps for epoch in fit.epochs), 3)
        self.assertTrue(all(epoch.max_abs_gradient > 0 for epoch in fit.epochs))
        self.assertNotEqual(fit.initial_weights_sha256, fit.final_weights_sha256)
        self.assertEqual(fit.runtime.device, DeviceSpec("cuda", 0))
        self.assertTrue(fit.runtime.torch_cuda_version)
        self.assertTrue(fit.runtime.device_name)
        self.assertIsNotNone(fit.runtime.cuda_peak_allocated_bytes)
        self.assertIsNotNone(fit.runtime.cuda_peak_reserved_bytes)
        assert fit.runtime.cuda_peak_allocated_bytes is not None
        self.assertGreater(fit.runtime.cuda_peak_allocated_bytes, 0)
        from safetensors.torch import load_file

        tensor = load_file(raw)["item_embedding.weight"]
        self.assertEqual(tuple(tensor.shape), (len(inputs.mal_ids), 32))
        raw_tensor = np.asarray(tensor.numpy(), dtype="<f4", order="C").tobytes(order="C")
        self.assertEqual(len(raw_tensor), len(inputs.mal_ids) * 32 * 4)
        self.assertEqual(hashlib.sha256(raw_tensor).hexdigest(), fit.final_weights_sha256)
        from_numpy = torch.from_numpy(
            np.frombuffer(raw_tensor, dtype="<f4").copy().reshape(tensor.shape)
        )
        torch.testing.assert_close(from_numpy, tensor, rtol=0, atol=0)
        np.testing.assert_array_equal(from_numpy.to("cuda:0").to("cpu").numpy(), tensor.numpy())
        model = fixture.root / "model"
        with patch_release_info(fixture.lock):
            manifest = publish_training_artifact(
                model,
                spec=spec,
                inputs=inputs,
                weights_path=raw,
                supported_mal_ids=fit.supported_mal_ids,
                code_revision=code_revision,
                framework_version="2.13.0+cu129",
            )
            model_identity = identity(manifest)
            verified = verify_training_artifact(
                model, expected_identity=model_identity, inputs=inputs
            )
            self.assertEqual(verified.manifest["producer"]["code_revision"], code_revision)
            self.assertEqual(
                verified.manifest["source_identity"],
                "sha256:"
                + hashlib.sha256(canonical_bytes(verified.manifest["producer"])).hexdigest(),
            )
            gpu_scorer = load_native(
                model,
                expected_identity=model_identity,
                inputs=inputs,
                inference_device=DeviceSpec("cuda", 0),
            )
        try:
            gpu_scores = gpu_scorer.score((1, 2))
        finally:
            gpu_scorer.close()

        result = subprocess.run(
            [
                cpu_python,
                "-c",
                _CPU_LOADER,
                str(prepared),
                prepared_identity,
                str(model),
                json.dumps(fixture.lock, sort_keys=True),
                str(Path(__file__).parent),
                model_identity,
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "ARB_TEST_TORCH_CUDA": ""},
        )
        loaded = json.loads(result.stdout)
        self.assertEqual(loaded["device"], {"kind": "cpu", "index": None})
        self.assertIsNone(loaded["cuda_peak_allocated_bytes"])
        self.assertEqual(loaded["supported_seeds"], gpu_scores.supported_seeds)
        np.testing.assert_array_equal(loaded["available"], gpu_scores.available)
        np.testing.assert_allclose(loaded["values"], gpu_scores.values, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
