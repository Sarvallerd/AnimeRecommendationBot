"""Opt-in native CPU training on the small declared seed fixture."""

from __future__ import annotations

import builtins
import copy
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import DEFAULT, patch

from research_model_support import patch_release_info
from training_support import SeedTrainingFixture

from recsys.contracts.bundle import canonical_bytes
from recsys.data.dataset import HEADER
from recsys.research.artifacts import StreamDescriptor, StreamWriter, identity
from recsys.research.models.inputs import read_training_inputs
from recsys.research.models.types import TrainingInputs, TrainStats
from recsys.research.protocol import ResearchError
from recsys.research.training.artifacts import publish_training_artifact, verify_training_artifact
from recsys.research.training.candidate import load_candidate
from recsys.research.training.spec import DeviceSpec
from recsys.research.training.torch_model import fit_native, load_native
from recsys.research.training.types import NativeFitResult

_ABSENT_DIR_TRAIN = """
import json
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[5])
from research_model_support import patch_release_info
from recsys.research.models.inputs import read_training_inputs
from recsys.research.training.candidate import load_candidate
from recsys.research.training.spec import DeviceSpec
from recsys.research.training.torch_model import fit_native
with patch_release_info(json.loads(sys.argv[4])):
    inputs = read_training_inputs(sys.argv[1], expected_prepared_identity=sys.argv[2])
    spec = load_candidate(
        train_device=DeviceSpec('cpu', None), inference_device=DeviceSpec('cpu', None)
    )
    result = fit_native(inputs, spec, Path(sys.argv[3]))
    print(json.dumps({
        'supported': result.supported_mal_ids,
        'examples': sum(row.examples for row in result.epochs),
        'steps': sum(row.optimizer_steps for row in result.epochs),
    }))
"""


@unittest.skipUnless(os.environ.get("ARB_TEST_TORCH") == "1", "opt-in native CPU training")
class TorchTrainingTests(unittest.TestCase):
    def setUp(self) -> None:
        # Imports belong to the enabled test body, never collection or the ordinary CLI.
        import numpy as np
        import safetensors
        import torch

        self.assertEqual(torch.__version__, "2.13.0+cpu")
        self.assertEqual(np.__version__, "2.5.3")
        self.assertEqual(safetensors.__version__, "0.8.0")
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.fixture = SeedTrainingFixture(temporary.name)
        self.prepared, self.prepared_identity = self.fixture.make_prepared()
        with patch_release_info(self.fixture.lock):
            self.inputs = read_training_inputs(
                self.prepared, expected_prepared_identity=self.prepared_identity
            )
        self.spec = load_candidate(
            train_device=DeviceSpec("cpu", None), inference_device=DeviceSpec("cpu", None)
        )
        self.code_revision = os.environ.get("ARB_TEST_CODE_REVISION", "a" * 40)
        self.assertRegex(self.code_revision, r"\A[0-9a-f]{40}\Z")
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
            self.assertEqual(head, self.code_revision)
            self.assertEqual(status, "", "post-commit source checkout must be clean")

    def _fit(self, filename: str) -> tuple[Path, NativeFitResult]:
        path = self.fixture.root / filename
        result = fit_native(self.inputs, self.spec, path)
        self.assertEqual(path, result.weights_path)
        self.assertTrue(path.is_file())
        return path, result

    def _publish(self, weights: Path, target: str = "model") -> tuple[Path, str]:
        output = self.fixture.root / target
        with patch_release_info(self.fixture.lock):
            manifest = publish_training_artifact(
                output,
                spec=self.spec,
                inputs=self.inputs,
                weights_path=weights,
                supported_mal_ids=(1, 2, 3, 4, 5, 6),
                code_revision=self.code_revision,
                framework_version="2.13.0+cpu",
            )
        return output, identity(manifest)

    def _inputs_with_profiles(
        self, profiles: dict[str, dict[int, int]], name: str
    ) -> TrainingInputs:
        path = self.fixture.root / f"{name}.tsv.gz"
        writer = StreamWriter(path)
        writer.write(HEADER)
        stats: TrainStats = copy.deepcopy(self.inputs.train_stats)
        for item in stats["items"].values():
            item["completed_users"] = 0
            item["positive_users"] = 0
            item["rating_sum"] = 0
            item["rating_count"] = 0
        rows = 0
        for user, profile in sorted(profiles.items()):
            for mal_id, rating in sorted(profile.items()):
                writer.write(f"{user}\t{mal_id}\t{rating}\tcompleted\n".encode("ascii"))
                item = stats["items"][str(mal_id)]
                item["completed_users"] += 1
                item["positive_users"] += int(rating >= 8)
                item["rating_sum"] += rating
                item["rating_count"] += 1
                rows += 1
        descriptor = writer.close()
        stats["train_users"] = len(profiles)
        stats["train_completed_rows"] = rows
        return replace(
            self.inputs,
            train_interactions_path=path,
            train_interactions_descriptor=descriptor,
            train_stats=stats,
        )

    def test_actual_steps_hashes_support_repeat_and_roundtrip(self) -> None:
        import numpy as np
        import torch
        from safetensors.torch import load_file

        weights, result = self._fit("raw-a.safetensors")
        self.assertEqual(result.supported_mal_ids, (1, 2, 3, 4, 5, 6))
        self.assertEqual([row.epoch for row in result.epochs], [0, 1, 2])
        self.assertEqual([row.examples for row in result.epochs], [3, 3, 3])
        self.assertEqual([row.optimizer_steps for row in result.epochs], [1, 1, 1])
        for row in result.epochs:
            self.assertTrue(np.isfinite(row.mean_loss) and row.mean_loss > 0.0)
            self.assertTrue(np.isfinite(row.max_abs_gradient) and row.max_abs_gradient > 0.0)
        self.assertRegex(result.initial_weights_sha256, r"\A[0-9a-f]{64}\Z")
        self.assertRegex(result.final_weights_sha256, r"\A[0-9a-f]{64}\Z")
        self.assertNotEqual(result.initial_weights_sha256, result.final_weights_sha256)
        tensor = load_file(weights)["item_embedding.weight"]
        self.assertEqual(tuple(tensor.shape), (len(self.inputs.mal_ids), 32))
        raw = np.asarray(tensor.numpy(), dtype="<f4", order="C").tobytes(order="C")
        self.assertEqual(len(raw), len(self.inputs.mal_ids) * 32 * 4)
        self.assertEqual(hashlib.sha256(raw).hexdigest(), result.final_weights_sha256)
        np.testing.assert_array_equal(
            torch.from_numpy(np.frombuffer(raw, dtype="<f4").copy().reshape(tensor.shape)).numpy(),
            tensor.numpy(),
        )

        second, repeated = self._fit("raw-b.safetensors")
        self.assertEqual(result.initial_weights_sha256, repeated.initial_weights_sha256)
        self.assertEqual(result.final_weights_sha256, repeated.final_weights_sha256)
        self.assertEqual(weights.read_bytes(), second.read_bytes())

        model, model_identity = self._publish(weights)
        with patch_release_info(self.fixture.lock):
            scorer = load_native(model, expected_identity=model_identity, inputs=self.inputs)
            verified = verify_training_artifact(
                model, expected_identity=model_identity, inputs=self.inputs
            )
        self.assertEqual(verified.manifest["producer"]["code_revision"], self.code_revision)
        self.assertEqual(
            verified.manifest["source_identity"],
            "sha256:" + hashlib.sha256(canonical_bytes(verified.manifest["producer"])).hexdigest(),
        )
        try:
            self.assertEqual(scorer.runtime.device, DeviceSpec("cpu", None))
            self.assertIsNone(scorer.runtime.torch_cuda_version)
            self.assertIsNone(scorer.runtime.device_name)
            self.assertIsNone(scorer.runtime.cuda_peak_allocated_bytes)
            self.assertIsNone(scorer.runtime.cuda_peak_reserved_bytes)
            self.assertEqual(scorer.supported_seed_ids((1, 7, 9)), frozenset({1}))
            vector = scorer.score((1, 2))
            self.assertEqual(vector.supported_seeds, 2)
            self.assertEqual(
                np.flatnonzero(vector.available).tolist(),
                [self.inputs.mal_ids.index(item) for item in range(1, 7)],
            )
            expected = tensor.numpy() @ tensor.numpy()[[0, 1]].mean(axis=0)
            np.testing.assert_allclose(vector.values, expected, rtol=1e-5, atol=1e-6)
            mixed = scorer.score((1, 9))
            np.testing.assert_array_equal(mixed.values, scorer.score((1,)).values)
            all_cold = scorer.score((9,))
            self.assertEqual(all_cold.supported_seeds, 0)
            self.assertFalse(all_cold.available.any())
            self.assertFalse(scorer.score(()).available.any())
            self.assertEqual(len(scorer.score_many(((1,), (1, 2)))), 2)
            for bad in (tuple(range(1, 12)), (1, 1), (True,), (22,)):
                with (
                    self.subTest(bad=bad),
                    self.assertRaisesRegex(ResearchError, "training_seed_invalid"),
                ):
                    scorer.score(bad)
        finally:
            scorer.close()

    def test_cpu_training_and_loading_do_not_probe_cuda(self) -> None:
        import torch

        with (
            patch.object(torch.cuda, "is_available", side_effect=AssertionError("CUDA probe")),
            patch.object(torch.cuda, "device_count", side_effect=AssertionError("CUDA probe")),
            patch.object(torch.cuda, "mem_get_info", side_effect=AssertionError("CUDA probe")),
        ):
            raw, _result = self._fit("cpu-raw.safetensors")
            model, model_identity = self._publish(raw)
            with patch_release_info(self.fixture.lock):
                scorer = load_native(model, expected_identity=model_identity, inputs=self.inputs)
            self.assertTrue(scorer.score((1,)).available.any())
            scorer.close()

    def test_wrong_tensors_fail_after_valid_envelope(self) -> None:
        import torch
        from safetensors.torch import save

        for name, tensors in (
            ("key", {"other": torch.zeros((21, 32), dtype=torch.float32)}),
            ("shape", {"item_embedding.weight": torch.zeros((20, 32), dtype=torch.float32)}),
            ("dtype", {"item_embedding.weight": torch.zeros((21, 32), dtype=torch.float64)}),
            (
                "nonfinite",
                {"item_embedding.weight": torch.full((21, 32), float("nan"))},
            ),
        ):
            with self.subTest(name=name):
                raw = self.fixture.root / f"{name}.safetensors"
                raw.write_bytes(save(tensors))
                model, model_identity = self._publish(raw, f"model-{name}")
                with (
                    patch_release_info(self.fixture.lock),
                    self.assertRaisesRegex(ResearchError, "training_artifact_invalid"),
                ):
                    load_native(model, expected_identity=model_identity, inputs=self.inputs)

    def test_train_only_open_guard_and_absent_heldout_directories(self) -> None:
        from recsys.research.training import torch_model

        banned = (
            self.prepared / "development",
            self.prepared / "evaluator",
        )

        def check_path(value: object) -> None:
            if isinstance(value, (str, bytes, os.PathLike)):
                path = Path(os.fsdecode(os.fspath(value))).resolve(strict=False)
                if any(path == root or path.is_relative_to(root) for root in banned):
                    raise AssertionError("held-out preparation file opened")

        original_builtin = builtins.open
        original_io = io.open
        original_os = os.open

        def guarded(path: object, *args: object, **kwargs: object) -> object:
            check_path(path)
            return DEFAULT

        with (
            patch("builtins.open", wraps=original_builtin, side_effect=guarded),
            patch("io.open", wraps=original_io, side_effect=guarded),
            patch("os.open", wraps=original_os, side_effect=guarded),
            patch_release_info(self.fixture.lock),
        ):
            inputs = read_training_inputs(
                self.prepared, expected_prepared_identity=self.prepared_identity
            )
            result = torch_model.fit_native(
                inputs, self.spec, self.fixture.root / "guarded.safetensors"
            )
        self.assertEqual(sum(row.optimizer_steps for row in result.epochs), 3)

        copied = self.fixture.root / "train-only-copy"
        shutil.copytree(self.prepared, copied)
        shutil.rmtree(copied / "development")
        shutil.rmtree(copied / "evaluator")
        self.assertFalse((copied / "development").exists())
        self.assertFalse((copied / "evaluator").exists())
        raw = self.fixture.root / "absent-dirs.safetensors"
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                _ABSENT_DIR_TRAIN,
                str(copied),
                self.prepared_identity,
                str(raw),
                json.dumps(self.fixture.lock, sort_keys=True),
                str(Path(__file__).parent),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(
            json.loads(completed.stdout),
            {
                "supported": [1, 2, 3, 4, 5, 6],
                "examples": 9,
                "steps": 3,
            },
        )
        self.assertTrue(raw.is_file())

    def test_zero_examples_and_first_optimizer_failure_leave_no_weights(self) -> None:
        import torch

        for name, profiles in (
            (
                "singleton",
                {self.fixture.training_users[0]: {1: 10}, self.fixture.training_users[1]: {2: 9}},
            ),
            (
                "all-warm",
                {
                    self.fixture.training_users[0]: {1: 10, 2: 9},
                    self.fixture.training_users[1]: {1: 9, 2: 10},
                },
            ),
        ):
            with self.subTest(name=name):
                inputs = self._inputs_with_profiles(profiles, name)
                target = self.fixture.root / f"{name}.safetensors"
                with (
                    patch.object(
                        torch.optim.AdamW, "step", side_effect=AssertionError("optimizer ran")
                    ) as step,
                    self.assertRaisesRegex(ResearchError, "training_no_examples"),
                ):
                    fit_native(inputs, self.spec, target)
                step.assert_not_called()
                self.assertFalse(target.exists())
        target = self.fixture.root / "failed-first-step.safetensors"
        with (
            patch.object(
                torch.optim.AdamW, "step", side_effect=RuntimeError("first step failed")
            ) as step,
            self.assertRaisesRegex(RuntimeError, "first step failed"),
        ):
            fit_native(self.inputs, self.spec, target)
        step.assert_called_once()
        self.assertFalse(target.exists())

    def test_late_gzip_truncation_is_found_before_optimizer(self) -> None:
        import torch

        original = self.inputs.train_interactions_path.read_bytes()
        damaged = original[:-4]  # Truncate only the trailer, after all compressed rows.
        path = self.fixture.root / "late-truncated.tsv.gz"
        path.write_bytes(damaged)
        descriptor: StreamDescriptor = {
            **self.inputs.train_interactions_descriptor,
            "bytes": len(damaged),
            "sha256": hashlib.sha256(damaged).hexdigest(),
        }
        inputs = replace(
            self.inputs, train_interactions_path=path, train_interactions_descriptor=descriptor
        )
        target = self.fixture.root / "late-truncated.safetensors"
        with (
            patch.object(
                torch.optim.AdamW, "__init__", side_effect=AssertionError("optimizer created")
            ) as optimizer,
            self.assertRaises(ResearchError),
        ):
            fit_native(inputs, self.spec, target)
        optimizer.assert_not_called()
        self.assertFalse(target.exists())

    def test_truncated_hash_race_and_symlink_weights_fail_closed(self) -> None:
        import numpy as np
        import torch
        from safetensors.torch import load as decode

        weights, fit = self._fit("valid-raw.safetensors")
        model, model_identity = self._publish(weights)
        model_weights = model / "weights.safetensors"
        original = model_weights.read_bytes()

        truncated = self.fixture.root / "truncated.safetensors"
        truncated.write_bytes(original[:-4])
        truncated_model, truncated_identity = self._publish(truncated, "truncated-model")
        with (
            patch_release_info(self.fixture.lock),
            self.assertRaisesRegex(ResearchError, "training_artifact_invalid"),
        ):
            load_native(truncated_model, expected_identity=truncated_identity, inputs=self.inputs)

        modified = bytearray(original)
        modified[-1] ^= 1
        model_weights.write_bytes(modified)
        with (
            patch_release_info(self.fixture.lock),
            patch("safetensors.torch.load", side_effect=AssertionError("decoded bad hash")),
            self.assertRaisesRegex(ResearchError, "training_artifact_invalid"),
        ):
            load_native(model, expected_identity=model_identity, inputs=self.inputs)
        model_weights.write_bytes(original)

        model_weights.unlink()
        model_weights.symlink_to(weights)
        with (
            patch_release_info(self.fixture.lock),
            self.assertRaisesRegex(ResearchError, "research_unsafe_path"),
        ):
            load_native(model, expected_identity=model_identity, inputs=self.inputs)
        model_weights.unlink()
        model_weights.write_bytes(original)

        with patch_release_info(self.fixture.lock):
            original_scorer = load_native(
                model, expected_identity=model_identity, inputs=self.inputs
            )
        original_scores = original_scorer.score((1, 2)).values.copy()
        original_scorer.close()

        from recsys.research.training import torch_model

        checked_read = torch_model._read_verified_weights

        def changed_before_read(path: Path, *, expected_size: int, expected_hash: str) -> bytes:
            path.write_bytes(modified)
            return checked_read(path, expected_size=expected_size, expected_hash=expected_hash)

        with (
            patch_release_info(self.fixture.lock),
            patch.object(torch_model, "_read_verified_weights", side_effect=changed_before_read),
            patch("safetensors.torch.load", side_effect=AssertionError("decoded raced bytes")),
            self.assertRaisesRegex(ResearchError, "training_artifact_invalid"),
        ):
            load_native(model, expected_identity=model_identity, inputs=self.inputs)
        model_weights.write_bytes(original)

        decoded_bytes: list[bytes] = []

        def changed_after_read(raw: bytes) -> dict[str, torch.Tensor]:
            decoded_bytes.append(raw)
            model_weights.write_bytes(modified)
            return decode(raw)

        with (
            patch_release_info(self.fixture.lock),
            patch("safetensors.torch.load", side_effect=changed_after_read),
        ):
            scorer = load_native(model, expected_identity=model_identity, inputs=self.inputs)
        self.assertEqual(decoded_bytes, [original])
        self.assertEqual(scorer.supported_seed_ids((1, 7)), frozenset({1}))
        self.assertRegex(fit.final_weights_sha256, r"\A[0-9a-f]{64}\Z")
        np.testing.assert_array_equal(scorer.score((1, 2)).values, original_scores)
        scorer.close()


if __name__ == "__main__":
    unittest.main()
