"""Versioned training configuration and immutable model artifact contracts."""

from recsys.research.training.artifacts import (
    VerifiedTrainingArtifact,
    publish_training_artifact,
    verify_training_artifact,
)
from recsys.research.training.spec import (
    DeviceSpec,
    TrainingSpec,
    parse_device,
    parse_training_spec,
)

__all__ = (
    "DeviceSpec",
    "TrainingSpec",
    "VerifiedTrainingArtifact",
    "parse_device",
    "parse_training_spec",
    "publish_training_artifact",
    "verify_training_artifact",
)
