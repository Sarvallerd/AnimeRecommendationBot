"""Framework independent values and exact wire shapes for native training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict

from recsys.research.artifacts import FileDescriptor, PreparationProducer
from recsys.research.training.spec import DeviceSpec, DeviceWire, TrainingSpecWire

type CandidateId = Literal["item-seed-mean-bpr-v1"]


@dataclass(frozen=True)
class TrainingExample:
    seed_mal_ids: tuple[int, ...]
    positive_mal_id: int
    negative_mal_ids: tuple[int, ...]


@dataclass(frozen=True)
class TrainingScan:
    completed_rows: int
    observed_users: int
    positive_rows: int
    warm_mal_ids: tuple[int, ...]


@dataclass(frozen=True)
class NativeRuntime:
    framework_version: str
    python_version: str
    numpy_version: str
    safetensors_version: str
    device: DeviceSpec
    torch_cuda_version: str | None
    device_name: str | None
    cuda_peak_allocated_bytes: int | None
    cuda_peak_reserved_bytes: int | None


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    examples: int
    optimizer_steps: int
    mean_loss: float
    max_abs_gradient: float


@dataclass(frozen=True)
class NativeFitResult:
    weights_path: Path
    supported_mal_ids: tuple[int, ...]
    initial_weights_sha256: str
    final_weights_sha256: str
    epochs: tuple[EpochMetrics, ...]
    runtime: NativeRuntime


@dataclass(frozen=True)
class PredictionRow:
    mal_id: int
    score: float


class EpochMetricsWire(TypedDict):
    epoch: int
    examples: int
    optimizer_steps: int
    mean_loss: float
    max_abs_gradient: float


class TrainingMetricsWire(TypedDict):
    epochs: list[EpochMetricsWire]
    examples: int
    optimizer_steps: int
    supported_items: int
    initial_weights_sha256: str
    final_weights_sha256: str
    changed_weights: Literal[True]


class NativeRuntimeWire(TypedDict):
    framework_version: str
    python_version: str
    numpy_version: str
    safetensors_version: str
    device: DeviceWire
    torch_cuda_version: str | None
    device_name: str | None
    cuda_peak_allocated_bytes: int | None
    cuda_peak_reserved_bytes: int | None


class ExecutionLimitsWire(TypedDict):
    workers: Literal[1]
    timeout_seconds: Literal[300]
    rss_limit_bytes: Literal[4294967296]
    artifact_max_bytes: Literal[67108864]
    cuda_min_free_bytes: Literal[2147483648]
    request_max_bytes: Literal[65536]
    worker_result_max_bytes: Literal[65536]
    receipt_max_bytes: Literal[65536]
    prediction_max_bytes: Literal[32768]


class ProcessResultWire(TypedDict):
    outcome: str
    returncode: int | None
    elapsed_seconds: float
    sampled_peak_rss_bytes: int
    wait4_peak_rss_bytes: int
    observed_peak_rss_bytes: int
    rss_samples: int
    max_sample_gap_seconds: float
    cleanup_complete: bool


class ModelReference(TypedDict):
    relative_path: Literal["model"]
    identity: str


class PredictionReference(TypedDict):
    relative_path: Literal["prediction.json"]
    descriptor: FileDescriptor
    identity: str


class FitWorkerRequest(TypedDict):
    schema_version: Literal[1]
    kind: Literal["arb-training-worker-request"]
    request_token: str
    candidate: CandidateId
    candidate_resource_identity: str
    prepared_dir: str
    expected_prepared_identity: str
    code_revision: str
    operation: Literal["fit"]
    spec: TrainingSpecWire


class PredictWorkerRequest(TypedDict):
    schema_version: Literal[1]
    kind: Literal["arb-training-worker-request"]
    request_token: str
    candidate: CandidateId
    candidate_resource_identity: str
    prepared_dir: str
    expected_prepared_identity: str
    code_revision: str
    operation: Literal["predict"]
    model_dir: str
    expected_model_identity: str
    seed_mal_ids: list[int]
    limit: int
    inference_device: DeviceWire | None


class FitWorkerResult(TypedDict):
    schema_version: Literal[1]
    kind: Literal["arb-training-worker-result"]
    operation: Literal["fit"]
    status: Literal["success"]
    candidate: CandidateId
    candidate_resource_identity: str
    request_identity: str
    prepared_identity: str
    fit_identity: str
    data_identity: str
    source_identity: str
    runtime: NativeRuntimeWire
    model: ModelReference
    config_identity: str
    training: TrainingMetricsWire


class PredictWorkerResult(TypedDict):
    schema_version: Literal[1]
    kind: Literal["arb-training-worker-result"]
    operation: Literal["predict"]
    status: Literal["success"]
    candidate: CandidateId
    candidate_resource_identity: str
    request_identity: str
    prepared_identity: str
    fit_identity: str
    data_identity: str
    source_identity: str
    runtime: NativeRuntimeWire
    model_identity: str
    inference_producer: PreparationProducer
    stored_inference_device: DeviceWire
    requested_inference_device: DeviceWire | None
    effective_inference_device: DeviceWire
    prediction: PredictionReference


class WorkerFailure(TypedDict):
    schema_version: Literal[1]
    kind: Literal["arb-training-worker-result"]
    operation: Literal["fit", "predict"]
    status: Literal["failure"]
    request_identity: str
    code: str


class TrainingRunReceipt(TypedDict):
    schema_version: Literal[1]
    kind: Literal["arb-native-training-run"]
    candidate: CandidateId
    candidate_resource_identity: str
    request_identity: str
    prepared_identity: str
    fit_identity: str
    data_identity: str
    source_identity: str
    runtime: NativeRuntimeWire
    model: ModelReference
    config_identity: str
    training: TrainingMetricsWire
    limits: ExecutionLimitsWire
    process: ProcessResultWire


class PredictionRunReceipt(TypedDict):
    schema_version: Literal[1]
    kind: Literal["arb-native-prediction-run"]
    candidate: CandidateId
    candidate_resource_identity: str
    request_identity: str
    prepared_identity: str
    fit_identity: str
    data_identity: str
    source_identity: str
    runtime: NativeRuntimeWire
    model_identity: str
    inference_producer: PreparationProducer
    stored_inference_device: DeviceWire
    requested_inference_device: DeviceWire | None
    effective_inference_device: DeviceWire
    prediction: PredictionReference
    limits: ExecutionLimitsWire
    process: ProcessResultWire
