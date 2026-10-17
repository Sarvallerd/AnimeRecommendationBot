"""Native, fixed PyTorch item embedding candidate for train-only seed recommendation."""

from __future__ import annotations

import hashlib
import math
import os
import platform
import stat
from pathlib import Path
from typing import TYPE_CHECKING

from recsys.research.artifacts import safe_path
from recsys.research.models.types import ScoreVector, ScoringInputs, TrainingInputs
from recsys.research.protocol import ResearchError
from recsys.research.training.artifacts import (
    ITEMS,
    MANIFEST,
    WEIGHTS,
    verify_training_artifact,
)
from recsys.research.training.spec import DeviceSpec, TrainingSpec

if TYPE_CHECKING:
    import torch

    from recsys.research.training.types import NativeFitResult, NativeRuntime, TrainingExample

_DIMENSION = 32
_MAX_ARTIFACT_BYTES = 64 * 1024 * 1024
_MIN_CUDA_FREE_BYTES = 2 * 1024 * 1024 * 1024
_FRAMEWORK_VERSIONS = frozenset({"2.13.0+cpu", "2.13.0+cu129"})


def _versions() -> None:
    """Keep optional framework imports off the ordinary CLI and CPU baselines."""
    try:
        import numpy
        import safetensors
        import torch
    except ImportError as exc:
        raise ResearchError("training_dependency_unavailable") from exc

    if (
        torch.__version__ not in _FRAMEWORK_VERSIONS
        or numpy.__version__ != "2.5.3"
        or safetensors.__version__ != "0.8.0"
    ):
        raise ResearchError("training_framework_unsupported")


def _device(requested: DeviceSpec) -> tuple[str, str | None, str | None]:
    """Admit CUDA explicitly; a CPU request makes no CUDA API calls."""
    import torch

    version = str(torch.__version__)
    if requested.kind == "cpu" and requested.index is None:
        return "cpu", None, None
    if requested.kind != "cuda" or requested.index is None:
        raise ResearchError("training_device_invalid")
    if version != "2.13.0+cu129":
        raise ResearchError("training_device_unavailable")
    if not torch.cuda.is_available() or requested.index >= torch.cuda.device_count():
        raise ResearchError("training_device_unavailable")
    device = f"cuda:{requested.index}"
    free, _total = torch.cuda.mem_get_info(device)
    if free < _MIN_CUDA_FREE_BYTES:
        raise ResearchError("training_resource_limit")
    name = torch.cuda.get_device_name(requested.index)
    cuda_version = torch.version.cuda
    if cuda_version is None or not name:
        raise ResearchError("training_device_unavailable")
    return device, cuda_version, name


def _runtime(
    requested: DeviceSpec, cuda_version: str | None, device_name: str | None
) -> NativeRuntime:
    import numpy
    import safetensors
    import torch

    from recsys.research.training.types import NativeRuntime

    allocated: int | None = None
    reserved: int | None = None
    if requested.kind == "cuda":
        allocated = int(torch.cuda.max_memory_allocated(requested.index))
        reserved = int(torch.cuda.max_memory_reserved(requested.index))
    return NativeRuntime(
        str(torch.__version__),
        platform.python_version(),
        str(numpy.__version__),
        str(safetensors.__version__),
        requested,
        cuda_version,
        device_name,
        allocated,
        reserved,
    )


def _weight_bytes(weight: torch.Tensor) -> bytes:
    """Hash the exact dense little-endian F32 row-major tensor, with no header."""
    import numpy as np
    import torch

    if weight.ndim != 2 or weight.shape[1] != _DIMENSION or weight.dtype != torch.float32:
        raise ResearchError("training_native_failed")
    cpu = weight.detach().to(device="cpu", dtype=weight.dtype).contiguous()
    if not cpu.isfinite().all().item():
        raise ResearchError("training_nonfinite")
    array = np.asarray(cpu.numpy(), dtype="<f4", order="C")
    if array.shape != (weight.shape[0], _DIMENSION) or array.dtype != np.dtype("<f4"):
        raise ResearchError("training_native_failed")
    raw = array.tobytes(order="C")
    if len(raw) != weight.shape[0] * _DIMENSION * 4:
        raise ResearchError("training_native_failed")
    return raw


def _hash_weight(weight: torch.Tensor) -> str:
    return hashlib.sha256(_weight_bytes(weight)).hexdigest()


def _example_loss(
    example: TrainingExample,
    embedding: torch.nn.Embedding,
    index: dict[int, int],
    device: str,
) -> torch.Tensor:
    import torch

    seeds = torch.tensor([index[item] for item in example.seed_mal_ids], device=device)
    positive = torch.tensor([index[example.positive_mal_id]], device=device)
    negatives = torch.tensor([index[item] for item in example.negative_mal_ids], device=device)
    context = embedding.forward(seeds).mean(dim=0)
    positive_score = (context * embedding.forward(positive)[0]).sum()
    negative_scores = (embedding.forward(negatives) * context).sum(dim=1)
    return torch.nn.functional.softplus(-(positive_score - negative_scores)).mean()


def fit_native(
    inputs: TrainingInputs, spec: TrainingSpec, raw_weights_path: Path
) -> NativeFitResult:
    """Perform three fixed train-only epochs and write one private safetensors payload."""
    from recsys.research.training.candidate import validate_candidate
    from recsys.research.training.examples import iter_training_examples, validate_training_rows
    from recsys.research.training.types import EpochMetrics, NativeFitResult

    validated = validate_candidate(spec)
    scan = validate_training_rows(inputs)  # Complete EOF/descriptors/stats before optimizer.
    if not inputs.mal_ids or len(inputs.mal_ids) > 100000:
        raise ResearchError("training_resource_limit")
    safe_path(raw_weights_path.parent, directory=True)
    if (
        raw_weights_path.suffix != ".safetensors"
        or raw_weights_path.exists()
        or raw_weights_path.is_symlink()
    ):
        raise ResearchError("training_artifact_invalid")
    if validated.train_device.kind == "cuda":
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    _versions()
    import torch

    device, cuda_version, device_name = _device(validated.train_device)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if validated.train_device.kind == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats(validated.train_device.index)

    index = {mal_id: offset for offset, mal_id in enumerate(inputs.mal_ids)}
    generator = torch.Generator(device="cpu")
    generator.manual_seed(6060)
    embedding = torch.nn.Embedding(len(index), _DIMENSION, device="cpu", dtype=torch.float32)
    with torch.no_grad():
        embedding.weight.normal_(mean=0.0, std=0.01, generator=generator)
    initial_hash = _hash_weight(embedding.weight)
    embedding = embedding.to(device=device)
    optimizer = torch.optim.AdamW(
        embedding.parameters(),
        lr=0.01,
        weight_decay=0.0001,
        betas=(0.9, 0.999),
        eps=1e-8,
        foreach=False,
        fused=False,
    )
    epochs: list[EpochMetrics] = []
    supported: set[int] = set()
    total_steps = 0
    for epoch in range(3):
        batch: list[TrainingExample] = []
        examples = 0
        steps = 0
        losses_sum = 0.0
        max_gradient = 0.0
        for example in iter_training_examples(inputs, scan=scan, epoch=epoch):
            batch.append(example)
            if len(batch) < 128:
                continue
            loss, gradient = _optimizer_step(batch, embedding, optimizer, index, device)
            examples += len(batch)
            losses_sum += loss * len(batch)
            max_gradient = max(max_gradient, gradient)
            supported.update(item for sample in batch for item in sample.seed_mal_ids)
            supported.update(sample.positive_mal_id for sample in batch)
            steps += 1
            total_steps += 1
            batch.clear()
        if batch:
            loss, gradient = _optimizer_step(batch, embedding, optimizer, index, device)
            examples += len(batch)
            losses_sum += loss * len(batch)
            max_gradient = max(max_gradient, gradient)
            supported.update(item for sample in batch for item in sample.seed_mal_ids)
            supported.update(sample.positive_mal_id for sample in batch)
            steps += 1
            total_steps += 1
        epochs.append(
            EpochMetrics(
                epoch, examples, steps, losses_sum / examples if examples else 0.0, max_gradient
            )
        )
    if total_steps == 0 or len(supported) < 2:
        raise ResearchError("training_no_examples")
    final_cpu = embedding.weight.detach().to(device="cpu", dtype=torch.float32).contiguous()
    final_hash = _hash_weight(final_cpu)
    if final_hash == initial_hash:
        raise ResearchError("training_unchanged_weights")
    from safetensors.torch import save

    raw = save({"item_embedding.weight": final_cpu})
    if len(raw) > _MAX_ARTIFACT_BYTES:
        raise ResearchError("training_resource_limit")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    with os.fdopen(os.open(raw_weights_path, flags, 0o600), "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    runtime = _runtime(validated.train_device, cuda_version, device_name)
    return NativeFitResult(
        raw_weights_path,
        tuple(sorted(supported)),
        initial_hash,
        final_hash,
        tuple(epochs),
        runtime,
    )


def _optimizer_step(
    batch: list[TrainingExample],
    embedding: torch.nn.Embedding,
    optimizer: torch.optim.AdamW,
    index: dict[int, int],
    device: str,
) -> tuple[float, float]:
    import torch

    optimizer.zero_grad(set_to_none=True)
    losses = [_example_loss(sample, embedding, index, device) for sample in batch]
    loss_tensor = torch.stack(losses).mean()
    if not loss_tensor.isfinite().item():
        raise ResearchError("training_nonfinite")
    torch.autograd.backward(loss_tensor)
    gradient_tensor = embedding.weight.grad
    if gradient_tensor is None or not gradient_tensor.isfinite().all().item():
        raise ResearchError("training_nonfinite")
    max_gradient = float(gradient_tensor.detach().abs().max().item())
    optimizer.step()
    if not embedding.weight.isfinite().all().item():
        raise ResearchError("training_nonfinite")
    loss = float(loss_tensor.detach().item())
    if not math.isfinite(loss) or not math.isfinite(max_gradient):
        raise ResearchError("training_nonfinite")
    return loss, max_gradient


def _read_verified_weights(path: Path, *, expected_size: int, expected_hash: str) -> bytes:
    """One bounded no-follow read; decode these same bytes after descriptor recheck."""
    if path.is_symlink():
        raise ResearchError("research_unsafe_path")
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            metadata = os.fstat(fd)
            if not stat.S_ISREG(metadata.st_mode):
                raise ResearchError("research_unsafe_path")
            if metadata.st_size != expected_size or not 1 <= expected_size <= _MAX_ARTIFACT_BYTES:
                raise ResearchError("training_resource_limit")
            with os.fdopen(fd, "rb", closefd=False) as stream:
                raw = stream.read(expected_size + 1)
            if len(raw) != expected_size or hashlib.sha256(raw).hexdigest() != expected_hash:
                raise ResearchError("training_artifact_invalid")
            return raw
        finally:
            os.close(fd)
    except OSError as exc:
        raise ResearchError("training_artifact_invalid") from exc


class TorchSeedScorer:
    """Score known seed sets against the verified support mask."""

    def __init__(
        self,
        *,
        mal_ids: tuple[int, ...],
        supported_mal_ids: tuple[int, ...],
        weights: torch.Tensor,
        device: str,
        runtime: NativeRuntime,
    ) -> None:
        self.mal_ids = mal_ids
        self._index = {item: offset for offset, item in enumerate(mal_ids)}
        self._supported = frozenset(supported_mal_ids)
        self._weights: torch.Tensor | None = weights
        self._device = device
        self.runtime = runtime

    def supported_seed_ids(self, seed_mal_ids: tuple[int, ...]) -> frozenset[int]:
        return frozenset(seed_mal_ids) & self._supported

    def score(self, seed_mal_ids: tuple[int, ...]) -> ScoreVector:
        import numpy as np
        import torch

        if (
            len(seed_mal_ids) > 10
            or len(set(seed_mal_ids)) != len(seed_mal_ids)
            or any(type(item) is not int or item not in self._index for item in seed_mal_ids)
        ):
            raise ResearchError("training_seed_invalid")
        chosen = sorted(self.supported_seed_ids(seed_mal_ids))
        available = np.asarray([item in self._supported for item in self.mal_ids], dtype=np.bool_)
        if not chosen:
            return ScoreVector(
                np.zeros(len(self.mal_ids), dtype=np.float64), np.zeros_like(available), 0
            )
        weights = self._weights
        if weights is None:
            raise ResearchError("training_scorer_closed")
        with torch.inference_mode():
            seed_rows = torch.tensor([self._index[item] for item in chosen], device=self._device)
            context = weights.index_select(0, seed_rows).mean(dim=0)
            scores_tensor = torch.mv(weights, context)
            values = np.asarray(scores_tensor.to(device="cpu").numpy(), dtype=np.float64)
        if not np.isfinite(values).all():
            raise ResearchError("training_nonfinite")
        values.setflags(write=False)
        available.setflags(write=False)
        return ScoreVector(values, available, len(chosen))

    def score_many(self, seeds: tuple[tuple[int, ...], ...]) -> tuple[ScoreVector, ...]:
        if len(seeds) > 128:
            raise ResearchError("training_resource_limit")
        return tuple(self.score(query) for query in seeds)

    def close(self) -> None:
        self._weights = None


def load_native(
    model_dir: Path,
    *,
    expected_identity: str,
    inputs: ScoringInputs,
    inference_device: DeviceSpec | None = None,
) -> TorchSeedScorer:
    """Verify 058 provenance and safely decode one framework-owned F32 tensor."""
    from recsys.research.training.candidate import validate_candidate

    verified = verify_training_artifact(
        model_dir, expected_identity=expected_identity, inputs=inputs
    )
    validate_candidate(verified.spec)
    if not verified.mal_ids or len(verified.supported_mal_ids) < 2:
        raise ResearchError("training_artifact_invalid")
    descriptors = verified.manifest["files"]
    total = (
        descriptors[ITEMS]["bytes"]
        + descriptors[WEIGHTS]["bytes"]
        + (verified.weights_path.parent / MANIFEST).stat().st_size
    )
    if total > _MAX_ARTIFACT_BYTES:
        raise ResearchError("training_resource_limit")
    if verified.manifest["framework_version"] not in _FRAMEWORK_VERSIONS:
        raise ResearchError("training_framework_unsupported")
    requested = inference_device if inference_device is not None else verified.spec.inference_device
    if requested.kind == "cuda":
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    _versions()
    import torch

    device, cuda_version, device_name = _device(requested)
    raw = _read_verified_weights(
        verified.weights_path,
        expected_size=descriptors[WEIGHTS]["bytes"],
        expected_hash=descriptors[WEIGHTS]["sha256"],
    )
    from safetensors import SafetensorError
    from safetensors.torch import load

    try:
        tensors = load(raw)
    except (ValueError, RuntimeError, SafetensorError) as exc:
        raise ResearchError("training_artifact_invalid") from exc
    if set(tensors) != {"item_embedding.weight"}:
        raise ResearchError("training_artifact_invalid")
    tensor = tensors["item_embedding.weight"]
    if (
        tensor.layout != torch.strided
        or tensor.dtype != torch.float32
        or tensor.shape != (len(verified.mal_ids), _DIMENSION)
        or not tensor.is_contiguous()
        or not tensor.isfinite().all().item()
    ):
        raise ResearchError("training_artifact_invalid")
    embedding = torch.nn.Embedding(len(verified.mal_ids), _DIMENSION)
    try:
        embedding.load_state_dict({"weight": tensor}, strict=True)
    except (RuntimeError, ValueError) as exc:
        raise ResearchError("training_artifact_invalid") from exc
    loaded = embedding.weight.detach().to(device=device, dtype=torch.float32)
    runtime = _runtime(requested, cuda_version, device_name)
    return TorchSeedScorer(
        mal_ids=verified.mal_ids,
        supported_mal_ids=verified.supported_mal_ids,
        weights=loaded,
        device=device,
        runtime=runtime,
    )
