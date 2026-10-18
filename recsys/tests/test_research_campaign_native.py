"""Small, opt-in native fit and full-catalog validation scoring acceptance."""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import cast

from native_campaign_support import replace_validation_queries
from research_model_support import patch_release_info
from training_support import SeedTrainingFixture

from recsys.research.artifacts import identity, read_json, stream_jsonl
from recsys.research.campaign.artifacts import _read_receipt
from recsys.research.campaign.common import _canonical, _digest, _private_file
from recsys.research.campaign.native import fit_native_campaign_arm, predict_native_validation
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
                (prediction / "scores.jsonl.gz").read_bytes(),
                (score_only_output / "scores.jsonl.gz").read_bytes(),
            )
            manifest, _ = read_json(prediction_manifest)
            assert isinstance(manifest, dict)
            self.assertEqual(manifest["model_id"], NATIVE_VARIANT)
            self.assertEqual(manifest["config_sha256"], model_identity.removeprefix("sha256:"))
            self.assertEqual(manifest["scores"]["rows"], len(queries))
            rows = cast(
                "list[dict[str, object]]",
                list(stream_jsonl(prediction / "scores.jsonl.gz", manifest["scores"])),
            )
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
                "schema_version": 2,
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
                "campaign_id": "ml02-native-v2",
                "request_token": "e" * 64,
            }
            request_path = attempt / "request.json"
            _private_file(request_path, _canonical(request))
            result = {
                "schema_version": 2,
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
