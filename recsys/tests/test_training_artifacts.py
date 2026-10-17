"""Container integrity and atomicity on tiny framework-free fixtures."""

import errno
import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from test_training_spec import valid_spec

from recsys.contracts.bundle import canonical_bytes
from recsys.research.artifacts import FileDescriptor, StreamDescriptor
from recsys.research.models.types import ScoringInputs, TrainStats
from recsys.research.protocol import ResearchError
from recsys.research.training.artifacts import publish_training_artifact, verify_training_artifact
from recsys.research.training.spec import TrainingSpec, parse_training_spec


def _identity(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _inputs() -> ScoringInputs:
    file: FileDescriptor = {"bytes": 1, "sha256": "a" * 64}
    stream: StreamDescriptor = {
        **file,
        "rows": 1,
        "uncompressed_bytes": 1,
        "canonical_sha256": "b" * 64,
    }
    stats: TrainStats = {
        "schema_version": 1,
        "train_users": 1,
        "train_completed_rows": 1,
        "items": {
            "1": {
                "completed_users": 1,
                "positive_users": 1,
                "rating_sum": 8,
                "rating_count": 1,
                "band": "head",
            },
            "2": {
                "completed_users": 0,
                "positive_users": 0,
                "rating_sum": 0,
                "rating_count": 0,
                "band": "cold",
            },
        },
    }
    return ScoringInputs(
        _identity(b"prep"),
        _identity(b"protocol"),
        file,
        _identity(b"fit"),
        stream,
        file,
        (1, 2),
        {"schema_version": 1, "anime": {}},
        stats,
    )


def _tiny_safetensors() -> bytes:
    # Valid safetensors v1 container: one finite F32 value at byte offset 0.
    header = json.dumps(
        {"embedding": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}, separators=(",", ":")
    ).encode()
    header += b" " * (-len(header) % 8)
    return struct.pack("<Q", len(header)) + header + struct.pack("<f", 0.25)


class TrainingArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.weights = self.root / "original.safetensors"
        self.weights.write_bytes(_tiny_safetensors())
        self.inputs = _inputs()
        self.spec: TrainingSpec = parse_training_spec(valid_spec())
        self.target = self.root / "model"

    def publish(self) -> str:
        manifest = publish_training_artifact(
            self.target,
            spec=self.spec,
            inputs=self.inputs,
            weights_path=self.weights,
            supported_mal_ids=(1,),
            code_revision="a" * 40,
            framework_version="2.0.0",
        )
        return _identity(manifest.read_bytes())

    def rewrite_manifest(self, manifest: dict[str, object]) -> str:
        raw = canonical_bytes(manifest)
        (self.target / "training-manifest.json").write_bytes(raw)
        return _identity(raw)

    def test_publish_verify_and_no_native_import(self) -> None:
        identity = self.publish()
        verified = verify_training_artifact(
            self.target, expected_identity=identity, inputs=self.inputs
        )
        self.assertEqual(verified.identity, identity)
        self.assertEqual(verified.mal_ids, (1, 2))
        self.assertEqual(verified.supported_mal_ids, (1,))
        self.assertEqual(verified.spec.train_device.kind, "cuda")
        self.assertEqual(verified.spec.inference_device.kind, "cpu")
        self.assertEqual(
            {p.name for p in self.target.iterdir()},
            {"training-manifest.json", "items.json", "weights.safetensors"},
        )
        self.assertEqual(self.weights.read_bytes(), verified.weights_path.read_bytes())
        self.assertEqual(self.target.stat().st_mode & 0o777, 0o700)
        self.assertEqual(verified.weights_path.stat().st_mode & 0o777, 0o600)

    def test_reject_non_safetensors_source_names_before_staging(self) -> None:
        forbidden = ("pt", "pth", "pkl", "pickle", "zip", "npz")
        names = ["arbitrary.pkl", "weights", "weights.bin", "weights.SAFETENSORS"]
        for extension in forbidden:
            names.extend(
                (
                    f"weights.{extension}.safetensors",
                    f"weights.{extension.upper()}.safetensors",
                    f"weights.{extension}",
                )
            )
        names.append(".pkl.safetensors")
        for name in names:
            with self.subTest(name=name):
                source = self.root / name
                original = (
                    b"not-a-safe-container" if name == "arbitrary.pkl" else _tiny_safetensors()
                )
                source.write_bytes(original)
                parent_entries = {entry.name for entry in self.root.iterdir()}
                with (
                    patch(
                        "recsys.research.training.artifacts.open_stage",
                        side_effect=AssertionError("stage creation attempted"),
                    ),
                    self.assertRaisesRegex(ResearchError, "^training_artifact_invalid$"),
                ):
                    publish_training_artifact(
                        self.target,
                        spec=self.spec,
                        inputs=self.inputs,
                        weights_path=source,
                        supported_mal_ids=(1,),
                        code_revision="a" * 40,
                        framework_version="2.0.0",
                    )
                self.assertEqual(source.read_bytes(), original)
                self.assertEqual({entry.name for entry in self.root.iterdir()}, parent_entries)
                self.assertFalse(self.target.exists())
                source.unlink()

    def test_tamper_extra_symlink_and_input_mismatch(self) -> None:
        identity = self.publish()
        wrong = _identity(b"wrong")
        with self.assertRaisesRegex(ResearchError, "training_identity_mismatch"):
            verify_training_artifact(self.target, expected_identity=wrong)
        with self.assertRaisesRegex(ResearchError, "training_data_mismatch"):
            verify_training_artifact(
                self.target,
                expected_identity=identity,
                inputs=ScoringInputs(**{**self.inputs.__dict__, "prepared_identity": wrong}),
            )
        extra = self.target / "extra"
        extra.write_text("x")
        with self.assertRaisesRegex(ResearchError, "training_artifact_invalid"):
            verify_training_artifact(self.target, expected_identity=identity)
        extra.unlink()
        (self.target / "weights.safetensors").write_bytes(b"bad")
        with self.assertRaisesRegex(ResearchError, "training_artifact_invalid"):
            verify_training_artifact(self.target, expected_identity=identity)

    def test_reject_bad_items_even_with_updated_descriptors(self) -> None:
        identity = self.publish()
        manifest_path = self.target / "training-manifest.json"
        manifest = json.loads(manifest_path.read_bytes())
        item_path = self.target / "items.json"
        for bad in (
            {"schema_version": 1, "mal_ids": [2, 1], "supported": [True, False]},
            {"schema_version": 1, "mal_ids": [1, 2], "supported": [1, False]},
        ):
            with self.subTest(bad=bad):
                raw = canonical_bytes(bad)
                item_path.write_bytes(raw)
                manifest["files"]["items.json"] = {
                    "bytes": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
                manifest_raw = canonical_bytes(manifest)
                manifest_path.write_bytes(manifest_raw)
                with self.assertRaisesRegex(ResearchError, "training_artifact_invalid"):
                    verify_training_artifact(self.target, expected_identity=_identity(manifest_raw))
        self.assertNotEqual(identity, _identity(manifest_path.read_bytes()))

    def test_rehashed_data_and_source_are_checked(self) -> None:
        self.publish()
        original = (self.target / "training-manifest.json").read_bytes()
        manifest = json.loads(original)
        manifest["data"]["prepared_identity"] = _identity(b"foreign")
        manifest["data_identity"] = _identity(canonical_bytes(manifest["data"]))
        identity = self.rewrite_manifest(manifest)
        # No inputs are read in CLI mode; only declared provenance can be checked.
        self.assertEqual(
            verify_training_artifact(self.target, expected_identity=identity).identity, identity
        )
        with self.assertRaisesRegex(ResearchError, "training_data_mismatch"):
            verify_training_artifact(self.target, expected_identity=identity, inputs=self.inputs)
        manifest = json.loads(original)
        manifest["producer"]["code_files_sha256"]["__init__.py"] = "0" * 64
        manifest["source_identity"] = _identity(canonical_bytes(manifest["producer"]))
        identity = self.rewrite_manifest(manifest)
        with self.assertRaisesRegex(ResearchError, "training_source_mismatch"):
            verify_training_artifact(self.target, expected_identity=identity)

    def test_symlink_oversize_and_supported_subset(self) -> None:
        identity = self.publish()
        weight = self.target / "weights.safetensors"
        weight.unlink()
        weight.symlink_to(self.weights)
        with self.assertRaisesRegex(ResearchError, "research_unsafe_path"):
            verify_training_artifact(self.target, expected_identity=identity)
        weight.unlink()
        weight.write_bytes(_tiny_safetensors())
        with (self.target / "items.json").open("wb") as out:
            out.truncate(8 * 1024 * 1024 + 1)
        with self.assertRaisesRegex(ResearchError, "training_resource_limit"):
            verify_training_artifact(self.target, expected_identity=identity)
        other = self.root / "other"
        with self.assertRaisesRegex(ResearchError, "training_data_mismatch"):
            publish_training_artifact(
                other,
                spec=self.spec,
                inputs=self.inputs,
                weights_path=self.weights,
                supported_mal_ids=(2,),
                code_revision="a" * 40,
                framework_version="2",
            )
        self.assertFalse(other.exists())

    def test_source_overlap_existing_target_and_reserved_adapter(self) -> None:
        with self.assertRaisesRegex(ResearchError, "research_unsafe_path"):
            publish_training_artifact(
                self.root,
                spec=self.spec,
                inputs=self.inputs,
                weights_path=self.weights,
                supported_mal_ids=(1,),
                code_revision="a" * 40,
                framework_version="2",
            )
        self.publish()
        with self.assertRaisesRegex(ResearchError, "training_output_exists"):
            self.publish()
        reserved = valid_spec()
        reserved.update(
            family="boosted-ranker", framework="catboost", weights_format="catboost-cbm-v1"
        )
        with self.assertRaisesRegex(ResearchError, "training_adapter_unsupported"):
            publish_training_artifact(
                self.root / "other",
                spec=parse_training_spec(reserved),
                inputs=self.inputs,
                weights_path=self.weights,
                supported_mal_ids=(1,),
                code_revision="a" * 40,
                framework_version="1",
            )

    def test_failure_cleans_only_own_stage(self) -> None:
        original = self.weights.read_bytes()
        with patch(
            "recsys.research.training.artifacts.verify_training_artifact",
            side_effect=RuntimeError("injected"),
        ):
            with self.assertRaisesRegex(RuntimeError, "injected"):
                self.publish()
        self.assertEqual(self.weights.read_bytes(), original)
        self.assertFalse(self.target.exists())
        self.assertEqual({p.name for p in self.root.iterdir()}, {"original.safetensors"})
        with patch(
            "recsys.research.training.artifacts.write_json", side_effect=OSError("injected")
        ):
            with self.assertRaisesRegex(ResearchError, "training_artifact_invalid"):
                self.publish()
        self.assertFalse(self.target.exists())
        self.assertEqual({p.name for p in self.root.iterdir()}, {"original.safetensors"})
        with patch("recsys.research.training.artifacts.publish", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.publish()
        self.assertFalse(self.target.exists())
        self.assertEqual({p.name for p in self.root.iterdir()}, {"original.safetensors"})
        with patch(
            "recsys.research.training.artifacts.publish",
            side_effect=FileExistsError(errno.EEXIST, "competing target"),
        ):
            with self.assertRaisesRegex(ResearchError, "training_output_exists"):
                self.publish()
        self.assertFalse(self.target.exists())
        self.assertEqual({p.name for p in self.root.iterdir()}, {"original.safetensors"})


if __name__ == "__main__":
    unittest.main()
