"""Small framework-independent, versioned training configuration."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, TypedDict, cast

from recsys.contracts.bundle import canonical_bytes
from recsys.research.protocol import ResearchError

type Parameter = bool | int | float | str
type Family = Literal["seed-embedding", "boosted-ranker"]
type Framework = Literal["pytorch", "catboost"]
type WeightsFormat = Literal["safetensors-v1", "catboost-cbm-v1"]


class DeviceWire(TypedDict):
    kind: Literal["cpu", "cuda"]
    index: int | None


class TrainingSpecWire(TypedDict):
    schema_version: int
    kind: str
    family: Family
    framework: Framework
    weights_format: WeightsFormat
    train_device: DeviceWire
    inference_device: DeviceWire
    precision: str
    seed: int
    parameters: dict[str, Parameter]


@dataclass(frozen=True)
class DeviceSpec:
    kind: Literal["cpu", "cuda"]
    index: int | None

    def wire(self) -> DeviceWire:
        return {"kind": self.kind, "index": self.index}


@dataclass(frozen=True)
class TrainingSpec:
    family: Family
    framework: Framework
    weights_format: WeightsFormat
    train_device: DeviceSpec
    inference_device: DeviceSpec
    seed: int
    parameters: Mapping[str, Parameter]
    precision: Literal["float32"] = "float32"

    def wire(self) -> TrainingSpecWire:
        return {
            "schema_version": 1,
            "kind": "arb-training-spec",
            "family": self.family,
            "framework": self.framework,
            "weights_format": self.weights_format,
            "train_device": self.train_device.wire(),
            "inference_device": self.inference_device.wire(),
            "precision": self.precision,
            "seed": self.seed,
            "parameters": dict(self.parameters),
        }


def parse_device(value: object) -> DeviceSpec:
    if type(value) is not str:
        raise ResearchError("training_device_invalid")
    if value == "cpu":
        return DeviceSpec("cpu", None)
    if re.fullmatch(r"cuda:(?:[0-9]|[12][0-9]|3[01])", value) is None:
        raise ResearchError("training_device_invalid")
    return DeviceSpec("cuda", int(value[5:]))


def _parse_device_wire(value: object) -> DeviceSpec:
    if type(value) is not dict or set(value) != {"kind", "index"}:
        raise ResearchError("training_device_invalid")
    wire = cast("dict[str, object]", value)
    kind, index = wire["kind"], wire["index"]
    if kind == "cpu" and index is None:
        return DeviceSpec("cpu", None)
    if kind == "cuda" and type(index) is int and 0 <= index <= 31:
        return DeviceSpec("cuda", index)
    raise ResearchError("training_device_invalid")


def parse_training_spec(value: object) -> TrainingSpec:
    """Validate exactly the wire fields; this does not load a native framework."""
    fields = {
        "schema_version",
        "kind",
        "family",
        "framework",
        "weights_format",
        "train_device",
        "inference_device",
        "precision",
        "seed",
        "parameters",
    }
    if type(value) is not dict or set(value) != fields:
        raise ResearchError("training_spec_invalid")
    raw = cast("dict[str, object]", value)
    if (
        type(raw["schema_version"]) is not int
        or raw["schema_version"] != 1
        or raw["kind"] != "arb-training-spec"
    ):
        raise ResearchError("training_spec_invalid")
    train = _parse_device_wire(raw["train_device"])
    inference = _parse_device_wire(raw["inference_device"])
    if (raw["family"], raw["framework"], raw["weights_format"]) not in (
        ("seed-embedding", "pytorch", "safetensors-v1"),
        ("boosted-ranker", "catboost", "catboost-cbm-v1"),
    ):
        raise ResearchError("training_combination_unsupported")
    if (
        raw["precision"] != "float32"
        or type(raw["seed"]) is not int
        or not 0 <= raw["seed"] <= 2**32 - 1
    ):
        raise ResearchError("training_spec_invalid")
    params = raw["parameters"]
    if type(params) is not dict or len(params) > 32:
        raise ResearchError("training_spec_invalid")
    parsed: dict[str, Parameter] = {}
    for key, item in params.items():
        if type(key) is not str or re.fullmatch(r"[a-z][a-z0-9_]{0,63}", key) is None:
            raise ResearchError("training_spec_invalid")
        if type(item) is bool:
            parsed[key] = item
        elif type(item) is int and -(2**63) <= item < 2**63:
            parsed[key] = item
        elif type(item) is float and math.isfinite(item):
            parsed[key] = item
        elif type(item) is str and len(item) <= 256:
            parsed[key] = item
        else:
            raise ResearchError("training_spec_invalid")
    spec = TrainingSpec(
        cast("Family", raw["family"]),
        cast("Framework", raw["framework"]),
        cast("WeightsFormat", raw["weights_format"]),
        train,
        inference,
        raw["seed"],
        MappingProxyType(parsed),
    )
    try:
        if len(canonical_bytes(spec.wire())) > 65536:
            raise ResearchError("training_resource_limit")
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise ResearchError("training_spec_invalid") from exc
    return spec
