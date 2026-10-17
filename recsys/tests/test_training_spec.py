"""Validation of independent train and inference device metadata."""

import copy
import unittest

from recsys.research.protocol import ResearchError
from recsys.research.training.spec import parse_device, parse_training_spec


class WireString(str):
    """A string-like Python value that is not an exact JSON string type."""


def valid_spec() -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": "arb-training-spec",
        "family": "seed-embedding",
        "framework": "pytorch",
        "weights_format": "safetensors-v1",
        "train_device": {"kind": "cuda", "index": 0},
        "inference_device": {"kind": "cpu", "index": None},
        "precision": "float32",
        "seed": 42,
        "parameters": {"epochs": 2, "learning_rate": 0.01, "normalize": True},
    }


class TrainingSpecTests(unittest.TestCase):
    def test_independent_devices_and_canonical_round_trip(self) -> None:
        spec = parse_training_spec(valid_spec())
        self.assertEqual(spec.train_device, parse_device("cuda:0"))
        self.assertEqual(spec.inference_device, parse_device("cpu"))
        self.assertEqual(parse_training_spec(spec.wire()).wire(), spec.wire())
        self.assertEqual(parse_device("cuda:31").index, 31)

    def test_reject_noncanonical_devices(self) -> None:
        for value in (
            "cuda",
            "cuda:01",
            "cuda:-1",
            "cuda:32",
            "gpu",
            "auto",
            " cpu",
            WireString("cpu"),
            WireString("cuda:0"),
            True,
            0,
        ):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(ResearchError, "training_device_invalid"),
            ):
                parse_device(value)
        for bad_wire in (
            {"kind": "cpu", "index": 0},
            {"kind": "cuda", "index": True},
            {"kind": WireString("cpu"), "index": None},
            {"kind": WireString("cuda"), "index": 0},
            {"kind": None, "index": None},
            {"kind": 0, "index": None},
        ):
            config = valid_spec()
            config["inference_device"] = bad_wire
            with self.assertRaisesRegex(ResearchError, "training_device_invalid"):
                parse_training_spec(config)

    def test_reject_shapes_combinations_and_scalars(self) -> None:
        cases: list[tuple[str, object, str]] = [
            ("schema_version", True, "training_spec_invalid"),
            ("seed", True, "training_spec_invalid"),
            ("seed", 2**32, "training_spec_invalid"),
            ("precision", "float64", "training_spec_invalid"),
            ("framework", "catboost", "training_combination_unsupported"),
            ("weights_format", "pickle", "training_combination_unsupported"),
            ("parameters", {"bad-key": 1}, "training_spec_invalid"),
            ("parameters", {"epochs": float("nan")}, "training_spec_invalid"),
            ("parameters", {"epochs": None}, "training_spec_invalid"),
            ("parameters", {"epochs": [1]}, "training_spec_invalid"),
            ("parameters", {"epochs": 2**63}, "training_spec_invalid"),
            ("parameters", {"note": "x" * 257}, "training_spec_invalid"),
        ]
        for field, value, code in cases:
            with self.subTest(field=field, value=repr(value)[:40]):
                config = copy.deepcopy(valid_spec())
                config[field] = value
                with self.assertRaisesRegex(ResearchError, code):
                    parse_training_spec(config)
        config = valid_spec()
        config["extra"] = True
        with self.assertRaisesRegex(ResearchError, "training_spec_invalid"):
            parse_training_spec(config)
        config = valid_spec()
        config["parameters"] = {f"p{n}": n for n in range(33)}
        with self.assertRaisesRegex(ResearchError, "training_spec_invalid"):
            parse_training_spec(config)

    def test_reject_nonexact_string_literal_fields(self) -> None:
        for field in ("kind", "family", "framework", "weights_format", "precision"):
            original = valid_spec()[field]
            assert isinstance(original, str)
            for replacement in (WireString(original), None, 0):
                with self.subTest(field=field, value=repr(replacement)):
                    config = valid_spec()
                    config[field] = replacement
                    with self.assertRaisesRegex(ResearchError, "training_spec_invalid"):
                        parse_training_spec(config)

    def test_reserved_catboost_combination_parses(self) -> None:
        config = valid_spec()
        config.update(
            family="boosted-ranker", framework="catboost", weights_format="catboost-cbm-v1"
        )
        self.assertEqual(parse_training_spec(config).framework, "catboost")


if __name__ == "__main__":
    unittest.main()
