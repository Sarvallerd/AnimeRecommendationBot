"""Small, opt-in native fit and full-catalog validation scoring acceptance."""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from typing import cast

from native_campaign_support import replace_validation_queries
from research_model_support import patch_release_info
from training_support import SeedTrainingFixture

from recsys.research.artifacts import identity, read_json
from recsys.research.campaign.artifacts import (
    _read_receipt,
    _verify_native_prediction_measurement,
    _verify_native_runtime,
)
from recsys.research.campaign.common import _canonical, _digest, _private_file
from recsys.research.campaign.native import fit_native_campaign_arm, predict_native_validation
from recsys.research.campaign.plan import CampaignPlan
from recsys.research.campaign.spec import NATIVE_VARIANT
from recsys.research.models.inputs import read_scoring_inputs
from recsys.research.protocol import ResearchError
from recsys.research.training.candidate import load_candidate
from recsys.research.training.spec import DeviceSpec


@unittest.skipUnless(os.environ.get("ARB_TEST_TORCH") == "1", "opt-in native CPU campaign")
class NativeCampaignTests(unittest.TestCase):
    def test_real_fit_full_catalog_masks_encoding_and_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = SeedTrainingFixture(directory)
            prepared, _ = fixture.make_prepared()
            prepared_identity, queries = replace_validation_queries(prepared)
            root = Path(directory)
            train_only = root / "train-only-prepared"
            shutil.copytree(prepared, train_only)
            shutil.rmtree(train_only / "development")
            shutil.rmtree(train_only / "evaluator")
            fit_attempt = root / "fit-attempt"
            fit_attempt.mkdir(mode=0o700)
            fit_resources = fit_attempt / "resources"
            fit_resources.mkdir(mode=0o700)
            model = fit_attempt / "artifact"
            fit_evidence = fit_resources / "measurement.json"
            spec = load_candidate(
                train_device=DeviceSpec("cpu", None),
                inference_device=DeviceSpec("cpu", None),
            )
            with patch_release_info(fixture.lock):
                manifest_path = fit_native_campaign_arm(
                    train_only,
                    model,
                    fit_evidence,
                    expected_prepared_identity=prepared_identity,
                    code_revision="a" * 40,
                    spec=spec,
                    plan_identity="sha256:" + "b" * 64,
                    request_identity="sha256:" + "c" * 64,
                )
                model_identity = identity(manifest_path)
                inputs = read_scoring_inputs(prepared, expected_prepared_identity=prepared_identity)
                prediction_resources = root / "prediction-resources"
                prediction_resources.mkdir(mode=0o700)
                guarded_emits: list[int] = []
                prediction = root / "prediction"
                prediction_manifest = predict_native_validation(
                    prepared,
                    model,
                    prediction,
                    expected_prepared_identity=prepared_identity,
                    expected_model_identity=model_identity,
                    code_revision="a" * 40,
                    measurement_output=prediction_resources / "measurement.json",
                    storage_guard=guarded_emits.append,
                )
                score_only = root / "score-only-prepared"
                shutil.copytree(prepared, score_only)
                (score_only / "fit" / "train-interactions.tsv.gz").unlink()
                shutil.rmtree(score_only / "evaluator")
                score_only_resources = root / "score-only-resources"
                score_only_resources.mkdir(mode=0o700)
                score_only_output = root / "score-only-prediction"
                predict_native_validation(
                    score_only,
                    model,
                    score_only_output,
                    expected_prepared_identity=prepared_identity,
                    expected_model_identity=model_identity,
                    code_revision="a" * 40,
                    measurement_output=score_only_resources / "measurement.json",
                    storage_guard=lambda _next_bytes: None,
                )
                low_resources = root / "low-resources"
                low_resources.mkdir(mode=0o700)

                def low_space(_next_bytes: int) -> None:
                    raise ResearchError("research_campaign_storage_low")

                with self.assertRaisesRegex(ResearchError, "research_campaign_storage_low"):
                    predict_native_validation(
                        prepared,
                        model,
                        root / "prediction-low",
                        expected_prepared_identity=prepared_identity,
                        expected_model_identity=model_identity,
                        code_revision="a" * 40,
                        measurement_output=low_resources / "measurement.json",
                        storage_guard=low_space,
                    )
                self.assertFalse((root / "prediction-low").exists())
                self.assertFalse(list(root.glob(".prediction-low.*")))
            self.assertEqual(
                {path.name for path in model.iterdir()},
                {"training-manifest.json", "items.json", "weights.safetensors"},
            )
            fit_info, _ = read_json(fit_evidence)
            assert isinstance(fit_info, dict)
            training = cast("dict[str, object]", fit_info["training"])
            self.assertEqual(fit_info["model_identity"], model_identity)
            self.assertEqual(training["examples"], 9)
            self.assertEqual(training["optimizer_steps"], 3)
            self.assertEqual(training["supported_items"], 6)
            self.assertNotEqual(
                training["initial_weights_sha256"],
                training["final_weights_sha256"],
            )
            self.assertTrue(guarded_emits)
            self.assertTrue(all(value >= 0 for value in guarded_emits))
            self.assertEqual(
                (prediction / "scores.bin").read_bytes(),
                (score_only_output / "scores.bin").read_bytes(),
            )
            manifest, _ = read_json(prediction_manifest)
            assert isinstance(manifest, dict)
            self.assertEqual(manifest["model_id"], NATIVE_VARIANT)
            self.assertEqual(manifest["config_sha256"], model_identity.removeprefix("sha256:"))
            self.assertEqual(manifest["scores"]["rows"], len(queries))
            from recsys.contracts.bundle import canonical_bytes
            from recsys.research.protocol import sha
            from recsys.research.score_blocks import iter_score_rows

            decoded = list(
                iter_score_rows(
                    prediction / "scores.bin",
                    manifest["scores"],
                    expected_catalog_ids=inputs.mal_ids,
                    expected_binding_sha256=sha(
                        canonical_bytes({k: v for k, v in manifest.items() if k != "scores"})
                    ),
                    expected_popularity=tuple(
                        inputs.train_stats["items"][str(item)]["positive_users"]
                        for item in inputs.mal_ids
                    ),
                    popularity_arm=False,
                )
            )
            rows = [
                {
                    "query_id": row.query_id,
                    "scores": [
                        [item, int(value) if available else None]
                        for item, value, available in zip(
                            inputs.mal_ids, row.values, row.available, strict=True
                        )
                    ],
                }
                for row in decoded
            ]
            self.assertEqual([row["query_id"] for row in rows], [q["query_id"] for q in queries])
            by_case: dict[tuple[int, ...], list[list[int | None]]] = {
                tuple(q["seed_mal_ids"]): cast("list[list[int | None]]", row["scores"])
                for q, row in zip(queries, rows)
            }
            ids = inputs.mal_ids
            self.assertEqual(
                by_case[()],
                [[item, inputs.train_stats["items"][str(item)]["positive_users"]] for item in ids],
            )
            self.assertEqual(by_case[()][-1], [ids[-1], 0])
            self.assertTrue(all(score is None for _, score in by_case[(9,)]))
            self.assertTrue(all(score is None for _, score in by_case[(7,)]))
            self.assertEqual(by_case[(1, 9)], by_case[(1,)])
            self.assertEqual(
                [item for item, score in by_case[(2, 3)] if score is not None],
                [1, 2, 3, 4, 5, 6],
            )
            measurement, _ = read_json(prediction_resources / "measurement.json")
            assert isinstance(measurement, dict)
            self.assertEqual(measurement["runtime"]["device"], {"kind": "cpu", "index": None})
            self.assertIsNone(measurement["runtime"]["cuda_peak_allocated_bytes"])
            self.assertEqual(measurement["score_pairs"], len(queries) * len(ids))

    def test_guard_rejection_stops_bounded_writer(self) -> None:
        from recsys.research.predict import GzipOneWriter

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scores.jsonl.gz"
            calls = 0

            def guard(_next_bytes: int) -> None:
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise ResearchError("research_campaign_storage_low")

            writer = GzipOneWriter(path, guard)
            with self.assertRaisesRegex(ResearchError, "research_campaign_storage_low"):
                writer.write_json({"query_id": "q1_" + "a" * 64, "scores": [[1, 1]]})
                writer.close()
            writer.abort()
            self.assertGreaterEqual(calls, 2)


class NativeReceiptTests(unittest.TestCase):
    def test_request_token_plan_and_measurement_are_bound(self) -> None:
        from recsys.research.artifacts import file_descriptor

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            root.mkdir(mode=0o700)
            attempt = root / "variants" / NATIVE_VARIANT / "attempts" / "fit" / "1"
            resources = attempt / "resources"
            resources.mkdir(parents=True, mode=0o700)
            plan_raw = _canonical({"schema_version": 4})
            _private_file(root / "campaign-plan.json", plan_raw)
            _private_file(resources / "measurement.json", _canonical({"count": 9}))
            request = {
                "schema_version": 3,
                "kind": "arb-ml02-campaign-worker",
                "root": str(root),
                "plan_identity": _digest(plan_raw),
                "generation": "g",
                "variant": NATIVE_VARIANT,
                "phase": "fit",
                "attempt": 1,
                "prepared_dir": str(root / "prepared"),
                "glove": str(root / "glove"),
                "model_dir": None,
                "prediction_dir": None,
                "content_model": None,
                "svd_model": None,
                "output_dir": str(attempt / "artifact"),
                "campaign_id": "ml02-native-v3",
                "request_token": "e" * 64,
            }
            request_path = attempt / "request.json"
            _private_file(request_path, _canonical(request))
            result = {
                "schema_version": 3,
                "kind": "arb-ml02-campaign-worker-receipt",
                "phase": "fit",
                "variant": NATIVE_VARIANT,
                "attempt": 1,
                "generation": "g",
                "status": "ok",
                "manifest_sha256": "sha256:" + "a" * 64,
                "plan_identity": _digest(plan_raw),
                "request_identity": _digest(_canonical(request)),
                "measurement_descriptor": file_descriptor(resources / "measurement.json"),
            }
            result_path = attempt / "result.json"
            _private_file(result_path, _canonical(result))
            self.assertEqual(
                _read_receipt(attempt, NATIVE_VARIANT, "fit", 1, "g", native_campaign=True),
                result,
            )
            request["request_token"] = "short"
            request_path.write_bytes(_canonical(request))
            result["request_identity"] = _digest(_canonical(request))
            result_path.write_bytes(_canonical(result))
            with self.assertRaisesRegex(ResearchError, "research_campaign_receipt_invalid"):
                _read_receipt(attempt, NATIVE_VARIANT, "fit", 1, "g", native_campaign=True)
            request["request_token"] = "e" * 64
            request_path.write_bytes(_canonical(request))
            result["request_identity"] = _digest(_canonical(request))
            result["measurement_descriptor"] = {"bytes": 1, "sha256": "b" * 64}
            result_path.write_bytes(_canonical(result))
            with self.assertRaisesRegex(ResearchError, "research_campaign_receipt_invalid"):
                _read_receipt(attempt, NATIVE_VARIANT, "fit", 1, "g", native_campaign=True)


class NativeRuntimeVerificationTests(unittest.TestCase):
    def _plan(self, framework: str) -> CampaignPlan:
        return cast(
            "CampaignPlan",
            {
                "producer": {
                    "python_version": "3.12.3",
                    "package_versions": {
                        "torch": framework,
                        "numpy": "2.5.3",
                        "safetensors": "0.8.0",
                    },
                }
            },
        )

    def _runtime(self, *, cuda: bool, framework: str) -> dict[str, object]:
        return {
            "framework_version": framework,
            "python_version": "3.12.3",
            "numpy_version": "2.5.3",
            "safetensors_version": "0.8.0",
            "device": {"kind": "cuda", "index": 0} if cuda else {"kind": "cpu", "index": None},
            "torch_cuda_version": "12.9" if cuda else None,
            "device_name": "fixture CUDA device" if cuda else None,
            "cuda_peak_allocated_bytes": 1024 if cuda else None,
            "cuda_peak_reserved_bytes": 2048 if cuda else None,
        }

    def test_runtime_versions_and_cuda_evidence_are_exact(self) -> None:
        for cuda, framework in ((False, "2.13.0+cpu"), (True, "2.13.0+cu129")):
            with self.subTest(cuda=cuda):
                plan = self._plan(framework)
                device = DeviceSpec("cuda", 0) if cuda else DeviceSpec("cpu", None)
                valid = self._runtime(cuda=cuda, framework=framework)
                _verify_native_runtime(
                    valid,
                    plan,
                    device=device,
                    framework_version=framework,
                    code="research_campaign_native_fit_invalid",
                )
                wrong_plan = self._plan("2.13.0+cpu" if cuda else "2.13.0+cu129")
                with self.assertRaisesRegex(ResearchError, "research_campaign_native_fit_invalid"):
                    _verify_native_runtime(
                        valid,
                        wrong_plan,
                        device=device,
                        framework_version=framework,
                        code="research_campaign_native_fit_invalid",
                    )
                mutations: list[tuple[str, object]] = [
                    ("python_version", "3.12.99"),
                    ("numpy_version", "9.9.9"),
                    ("safetensors_version", "9.9.9"),
                    ("framework_version", "2.13.0+other"),
                    ("device", {"kind": "cuda", "index": 1}),
                ]
                if cuda:
                    mutations.extend(
                        [
                            ("torch_cuda_version", None),
                            ("device_name", None),
                            ("cuda_peak_allocated_bytes", None),
                            ("cuda_peak_reserved_bytes", None),
                            ("cuda_peak_allocated_bytes", -1),
                            ("cuda_peak_reserved_bytes", True),
                        ]
                    )
                else:
                    mutations.extend(
                        [
                            ("torch_cuda_version", "12.9"),
                            ("device_name", "GPU"),
                            ("cuda_peak_allocated_bytes", 0),
                            ("cuda_peak_reserved_bytes", 0),
                        ]
                    )
                for field, replacement in mutations:
                    with self.subTest(cuda=cuda, field=field, replacement=replacement):
                        altered = deepcopy(valid)
                        altered[field] = replacement
                        with self.assertRaisesRegex(
                            ResearchError, "research_campaign_native_fit_invalid"
                        ):
                            _verify_native_runtime(
                                altered,
                                plan,
                                device=device,
                                framework_version=framework,
                                code="research_campaign_native_fit_invalid",
                            )

    def test_prediction_measurement_allowlist_and_finite_timings(self) -> None:
        framework = "2.13.0+cu129"
        plan = self._plan(framework)
        model_identity = "sha256:" + "a" * 64
        valid = {
            "schema_version": 2,
            "kind": "arb-ml02-native-prediction-measurement",
            "model_identity": model_identity,
            "stored_inference_device": {"kind": "cpu", "index": None},
            "effective_inference_device": {"kind": "cpu", "index": None},
            "runtime": self._runtime(cuda=False, framework=framework),
            "backend": "cpu",
            "status": "published",
            "queries": 5,
            "score_pairs": 105,
            "synchronized_scoring_seconds": 0.1,
            "encoding_write_seconds": 0.2,
            "score_format": "arb-score-blocks-v1",
            "staging_wall_seconds": 0.4,
            "publication_wall_seconds": 0.5,
        }
        _verify_native_prediction_measurement(
            valid,
            plan,
            model_identity=model_identity,
            framework_version=framework,
            query_count=5,
            catalog_count=21,
        )
        mutations: list[tuple[str, object]] = [
            ("unexpected", "private path"),
            ("backend", "gpu"),
            ("score_pairs", 104),
            ("synchronized_scoring_seconds", float("nan")),
            ("encoding_write_seconds", -0.1),
            ("staging_wall_seconds", True),
            ("publication_wall_seconds", float("inf")),
            ("publication_wall_seconds", 0.3),
        ]
        for field, replacement in mutations:
            with self.subTest(field=field):
                altered = deepcopy(valid)
                altered[field] = replacement
                with self.assertRaisesRegex(ResearchError, "research_campaign_prediction_invalid"):
                    _verify_native_prediction_measurement(
                        altered,
                        plan,
                        model_identity=model_identity,
                        framework_version=framework,
                        query_count=5,
                        catalog_count=21,
                    )
        for field, replacement in (
            ("python_version", "3.12.99"),
            ("numpy_version", "9.9.9"),
            ("safetensors_version", "9.9.9"),
            ("torch_cuda_version", "12.9"),
            ("cuda_peak_allocated_bytes", 0),
        ):
            with self.subTest(runtime_field=field):
                altered = deepcopy(valid)
                runtime = cast("dict[str, object]", altered["runtime"])
                runtime[field] = replacement
                with self.assertRaisesRegex(ResearchError, "research_campaign_prediction_invalid"):
                    _verify_native_prediction_measurement(
                        altered,
                        plan,
                        model_identity=model_identity,
                        framework_version=framework,
                        query_count=5,
                        catalog_count=21,
                    )
