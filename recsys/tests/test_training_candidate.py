"""The native candidate is declared once and only its devices may vary."""

import json
import unittest
from dataclasses import replace
from unittest.mock import patch

from recsys.research.protocol import ResearchError
from recsys.research.training.candidate import (
    candidate_identity,
    load_candidate,
    validate_candidate,
)
from recsys.research.training.spec import DeviceSpec


class CandidateTests(unittest.TestCase):
    def test_exact_resource_and_device_override(self) -> None:
        spec = load_candidate(
            train_device=DeviceSpec("cpu", None), inference_device=DeviceSpec("cpu", None)
        )
        self.assertEqual(
            candidate_identity(),
            "sha256:f13001ab517ceffa67e88ab23c614fae172e6eacc0b7f34bf9b22f88929690d4",
        )
        self.assertEqual(spec.seed, 6060)
        self.assertEqual(spec.parameters["init_mean"], 0.0)
        self.assertIs(type(spec.parameters["init_mean"]), float)
        self.assertEqual(len(spec.parameters), 32)
        self.assertEqual(validate_candidate(spec), spec)
        cuda = load_candidate(
            train_device=DeviceSpec("cuda", 0), inference_device=DeviceSpec("cpu", None)
        )
        self.assertEqual(cuda.train_device, DeviceSpec("cuda", 0))

    def test_fixed_values_and_types_cannot_change(self) -> None:
        spec = load_candidate(
            train_device=DeviceSpec("cpu", None), inference_device=DeviceSpec("cpu", None)
        )
        for key, value in (("init_mean", 0), ("seed", 6061), ("epochs", 4)):
            if key == "seed":
                modified = replace(spec, seed=value)
            else:
                modified = replace(spec, parameters={**spec.parameters, key: value})
            with self.subTest(key=key), self.assertRaises(ResearchError):
                validate_candidate(modified)
        missing = {key: value for key, value in spec.parameters.items() if key != "batch_size"}
        with self.assertRaises(ResearchError):
            validate_candidate(replace(spec, parameters=missing))
        extra = {**missing, "new_parameter": 128}
        with self.assertRaises(ResearchError):
            validate_candidate(replace(spec, parameters=extra))
        for device in (DeviceSpec("cuda", True), DeviceSpec("cpu", 0)):
            with self.subTest(device=device), self.assertRaises(ResearchError):
                load_candidate(train_device=device, inference_device=DeviceSpec("cpu", None))

    def test_raw_hash_rejects_reformatted_resource(self) -> None:
        with patch("recsys.research.training.candidate.resource_bytes") as read:
            read.return_value = json.dumps({"candidate": "item-seed-mean-bpr-v1"}).encode()
            with self.assertRaises(ResearchError):
                candidate_identity()


if __name__ == "__main__":
    unittest.main()
