import importlib
import tempfile
import unittest
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from recsys.bundle import canonical_bytes
from recsys.research_artifacts import read_json
from recsys.research_models import (campaign, fit_model, iter_training_rows, load_model,
                                    read_scoring_inputs, read_training_inputs, _external)
from recsys.research_protocol import ResearchError
from research_model_support import ModelFixture


class ModelTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.fixture = ModelFixture(tmp.name)
        self.prepared, self.prepared_identity = self.fixture.make_prepared()

    def inputs(self):
        with patch("recsys.research_models.release_info", return_value=self.fixture.lock):
            return read_scoring_inputs(self.prepared, expected_prepared_identity=self.prepared_identity)

    def test_popularity_model_deterministic_and_train_stats_only(self):
        source_open = Path.open
        def guarded(path, *args, **kwargs):
            if path.name in {"train-interactions.tsv.gz", "validation-labels.jsonl.gz",
                             "test-labels.jsonl.gz", "splits.tsv.gz"}:
                raise AssertionError("private stream opened")
            return source_open(path, *args, **kwargs)
        with patch.object(Path, "open", guarded):
            first = self.fixture.fit_popularity(self.prepared, self.prepared_identity, "first")
            second = self.fixture.fit_popularity(self.prepared, self.prepared_identity, "second")
            inputs = self.inputs()
            manifest, scorer = load_model(first, inputs=inputs)
        self.assertEqual((first / "model-manifest.json").read_bytes(),
                         (second / "model-manifest.json").read_bytes())
        self.assertEqual(set(manifest), {"schema_version", "kind", "variant_id", "parameters",
                                         "campaign_spec_sha256", "prepared_identity", "protocol_identity",
                                         "fit_identity", "catalog_descriptor", "train_interactions",
                                         "train_stats", "external_inputs", "producer", "files", "metadata"})
        answer = scorer.score((1, 2))
        self.assertEqual(answer.supported_seeds, 2)
        self.assertTrue(answer.available.all())
        self.assertEqual(answer.values.tolist(), [inputs.train_stats["items"][str(item)]["positive_users"]
                                                   for item in inputs.mal_ids])
        self.assertEqual(set(campaign()[0]["variants"]),
                         {"popularity", "content-mean", "content-max", "content-union",
                          "itemknn", "puresvd", "hybrid"})

    def test_train_stream_full_validation_and_lazy_unavailable_backend(self):
        with patch("recsys.research_models.release_info", return_value=self.fixture.lock):
            inputs = read_training_inputs(self.prepared, expected_prepared_identity=self.prepared_identity)
        rows = list(iter_training_rows(inputs))
        self.assertEqual(len(rows), inputs.train_stats["train_completed_rows"])
        self.assertTrue(all(1 <= rating <= 10 for _, _, rating in rows))
        real_import = importlib.import_module
        def missing_cf(name, package=None):
            if name == ".research_model_cf" and package == "recsys":
                raise ModuleNotFoundError(name="recsys.research_model_cf")
            return real_import(name, package)
        with self.assertRaisesRegex(ResearchError, "research_model_unavailable"):
            with patch("recsys.research_models.release_info", return_value=self.fixture.lock), \
                 patch("recsys.research_models.importlib.import_module", side_effect=missing_cf):
                fit_model(self.prepared, self.fixture.root / "missing",
                          expected_prepared_identity=self.prepared_identity,
                          variant_id="itemknn", code_revision="a" * 40)
        self.assertFalse((self.fixture.root / "missing").exists())

    def test_tampered_model_manifest_or_prepared_identity_fails_closed(self):
        model = self.fixture.fit_popularity(self.prepared, self.prepared_identity)
        path = model / "model-manifest.json"
        original = path.read_bytes()
        for field, value in (("parameters", {"extra": 1}),
                             ("campaign_spec_sha256", "0" * 64),
                             ("catalog_descriptor", {"bytes": 0, "sha256": "0" * 64})):
            with self.subTest(field=field):
                manifest, _ = read_json(path)
                manifest[field] = value
                path.write_bytes(canonical_bytes(manifest))
                with self.assertRaises(ResearchError):
                    load_model(model, inputs=self.inputs())
                path.write_bytes(original)
        manifest, _ = read_json(path)
        manifest["producer"]["code_files_sha256"]["research_models.py"] = "0" * 64
        path.write_bytes(canonical_bytes(manifest))
        with self.assertRaises(ResearchError):
            load_model(model, inputs=self.inputs())
        path.write_bytes(original)
        with self.assertRaises(ResearchError):
            with patch("recsys.research_models.release_info", return_value=self.fixture.lock):
                read_scoring_inputs(self.prepared, expected_prepared_identity="sha256:" + "0" * 64)
        (model / "extra.json").write_bytes(canonical_bytes({}))
        with self.assertRaises(ResearchError):
            load_model(model, inputs=self.inputs())

    def test_no_overlap_no_replace_and_private_mode(self):
        model = self.fixture.fit_popularity(self.prepared, self.prepared_identity)
        self.assertEqual(model.stat().st_mode & 0o777, 0o700)
        self.assertEqual((model / "model-manifest.json").stat().st_mode & 0o777, 0o600)
        with self.assertRaises(ResearchError):
            self.fixture.fit_popularity(self.prepared, self.prepared_identity)
        with self.assertRaises(ResearchError):
            with patch("recsys.research_models.release_info", return_value=self.fixture.lock):
                fit_model(self.prepared, self.prepared / "nested",
                          expected_prepared_identity=self.prepared_identity,
                          variant_id="popularity", code_revision="a" * 40)

    def test_common_publisher_protects_backend_numpy_payload(self):
        def fit_stub(inputs, parameters, assets, stage):
            path = stage / "vector.npy"
            np.save(path, np.array([1.0], dtype=np.float64))
            os.chmod(path, 0o644)
            return {"fixture": True}
        backend = SimpleNamespace(CODE_FILES=("research_model_popularity.py",), fit=fit_stub,
                                  load=lambda model_dir, manifest, inputs:
                                      SimpleNamespace(score=lambda seeds: None))
        for variant in ("itemknn", "puresvd"):
            with self.subTest(variant=variant):
                output = self.fixture.root / f"payload-{variant}"
                with patch("recsys.research_models.release_info", return_value=self.fixture.lock), \
                     patch("recsys.research_models._backend", return_value=backend):
                    fit_model(self.prepared, output, expected_prepared_identity=self.prepared_identity,
                              variant_id=variant, code_revision="a" * 40)
                    loaded, _ = load_model(output, inputs=self.inputs())
                    self.assertEqual(loaded["external_inputs"], {})
                self.assertEqual((output / "vector.npy").stat().st_mode & 0o777, 0o600)
                self.assertEqual(output.stat().st_mode & 0o777, 0o700)
                path = output / "model-manifest.json"
                original = path.read_bytes()
                changed, _ = read_json(path)
                changed["external_inputs"] = {"glove": {"bytes": 0, "sha256": "0" * 64}}
                path.write_bytes(canonical_bytes(changed))
                with patch("recsys.research_models.release_info", return_value=self.fixture.lock), \
                     patch("recsys.research_models._backend", return_value=backend):
                    with self.assertRaises(ResearchError):
                        load_model(output, inputs=self.inputs())
                path.write_bytes(original)
                with (output / "vector.npy").open("ab") as stream:
                    stream.write(b"tamper")
                with patch("recsys.research_models.release_info", return_value=self.fixture.lock), \
                     patch("recsys.research_models._backend", return_value=backend):
                    with self.assertRaises(ResearchError):
                        load_model(output, inputs=self.inputs())

    def test_content_asset_accepts_concrete_path_and_checks_lock(self):
        glove = self.fixture.root / "glove.txt"
        glove.write_bytes(b"tiny test embedding")
        from recsys.research_protocol import sha
        lock = {**self.fixture.lock, "inputs": {**self.fixture.lock["inputs"],
            "embeddings": {"text_size": glove.stat().st_size,
                           "text_sha256": sha(glove.read_bytes())}}}
        with patch("recsys.research_models.release_info", return_value=lock):
            self.assertEqual(_external("content-mean", {"glove": Path(glove)},
                                       self.prepared, self.fixture.root / "future-model"),
                             {"glove": {"bytes": glove.stat().st_size,
                                        "sha256": sha(glove.read_bytes())}})
