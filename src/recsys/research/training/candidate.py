"""The predeclared, immutable seed-embedding candidate."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from typing import cast

from recsys.contracts.resources import resource_bytes
from recsys.research.protocol import ResearchError
from recsys.research.training.spec import (
    DeviceSpec,
    TrainingSpec,
    parse_training_spec,
)

_NAME = "research_seed_embedding_v1.json"
_SHA256 = "f13001ab517ceffa67e88ab23c614fae172e6eacc0b7f34bf9b22f88929690d4"
_IDENTITY = "sha256:" + _SHA256


def _resource() -> dict[str, object]:
    try:
        raw = resource_bytes("campaigns", _NAME)
    except (OSError, ValueError) as exc:
        raise ResearchError("training_candidate_invalid") from exc
    if hashlib.sha256(raw).hexdigest() != _SHA256:
        raise ResearchError("training_candidate_invalid")
    try:
        value = json.loads(raw)
    except (UnicodeError, ValueError) as exc:
        raise ResearchError("training_candidate_invalid") from exc
    if type(value) is not dict or set(value) != {
        "schema_version",
        "kind",
        "candidate",
        "training_spec",
        "execution_limits",
        "prediction",
        "runtime_versions",
        "verification_tolerance",
    }:
        raise ResearchError("training_candidate_invalid")
    result = cast("dict[str, object]", value)
    if (
        type(result["schema_version"]) is not int
        or result["schema_version"] != 1
        or result["kind"] != "arb-native-training-candidate"
        or result["candidate"] != "item-seed-mean-bpr-v1"
    ):
        raise ResearchError("training_candidate_invalid")
    return result


def _same_exact(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        left_map = cast("dict[str, object]", left)
        right_map = cast("dict[str, object]", right)
        return left_map.keys() == right_map.keys() and all(
            _same_exact(left_map[key], right_map[key]) for key in left_map
        )
    if type(left) is list:
        left_list = cast("list[object]", left)
        right_list = cast("list[object]", right)
        return len(left_list) == len(right_list) and all(
            _same_exact(a, b) for a, b in zip(left_list, right_list)
        )
    return left == right


def candidate_identity() -> str:
    _resource()
    return _IDENTITY


def validate_candidate(spec: TrainingSpec) -> TrainingSpec:
    """Reparse and compare every fixed field, allowing only device overrides."""
    parsed = parse_training_spec(spec.wire())
    expected = _resource()["training_spec"]
    if type(expected) is not dict:
        raise ResearchError("training_candidate_invalid")
    expected_wire = cast("dict[str, object]", expected)
    observed = cast("dict[str, object]", parsed.wire())
    for key in expected_wire:
        if key not in ("train_device", "inference_device") and not _same_exact(
            observed[key], expected_wire[key]
        ):
            raise ResearchError("training_candidate_invalid")
    return parsed


def load_candidate(*, train_device: DeviceSpec, inference_device: DeviceSpec) -> TrainingSpec:
    wire = _resource()["training_spec"]
    if type(wire) is not dict:
        raise ResearchError("training_candidate_invalid")
    baseline = parse_training_spec(wire)
    return validate_candidate(
        replace(baseline, train_device=train_device, inference_device=inference_device)
    )
