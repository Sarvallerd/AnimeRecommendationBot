"""Versioned authority for every installed Python producer and immutable asset."""

import hashlib
from importlib import resources
from pathlib import PurePosixPath

SOURCE_LAYOUT = "arb054-recsys-layout-v2"
LEGACY_V1_PACKAGE_VERSION = "0.1.0"
LEGACY_V1_SOURCE_SHA256 = {
    "dataset.py": "b61da6809c45ac8099e78fceb64515bdcc0952d3b59697bf1d0a851114cce79f",
    "dataset_metadata.py": "09a47627fce76e6ad72aeff5656ff011eeb90ad0d0ef97f4e96f022d3124a97d",
    "dataset_interactions.py": "fcb6a274ff4c619f20c888021b0a6f56e7ee55ffad7ccd161754108b77004c75",
    "aliases.py": "64b702f8d4a6c9ec86b24db16bc290adfb111dbe528e53c9e4566116994de6b6",
    "release.py": "de185c16eafb884680de7e602691c28d6fc69f69ebcd995eae377710c3d61351",
    "build.py": "51106b320099c557f464f3122960f2c0dfe95972f9f313cda80c12e3d3d51745",
    "bundle.py": "641315298c96313e6f16732d5661204a7b50f9006fad4db8fc2803a6dbf51f0a",
    "export.py": "a766b67c1c8d495775e35e24ef919316e81f2e2344eecd952edb9400394f76c8",
    "release_build.py": "ca87a079e5f45a25e131613d514d627e07e2025f1c12d002edf7149212447385",
}

SOURCE_FILES = (
    "__init__.py",
    "__main__.py",
    "cli/__init__.py",
    "cli/main.py",
    "cli/parser.py",
    "content/__init__.py",
    "content/glove.py",
    "contracts/__init__.py",
    "contracts/bundle.py",
    "contracts/io.py",
    "contracts/provenance.py",
    "contracts/resources.py",
    "data/__init__.py",
    "data/aliases.py",
    "data/dataset.py",
    "data/interactions.py",
    "data/mal2020.py",
    "data/metadata.py",
    "data/obtain.py",
    "data/sources.py",
    "historical/__init__.py",
    "historical/assessment.py",
    "historical/comparison.py",
    "historical/genre_baseline.py",
    "historical/legacy_vectors.py",
    "release/__init__.py",
    "release/build.py",
    "release/catalog.py",
    "release/export.py",
    "release/lock.py",
    "release/prepare.py",
    "research/__init__.py",
    "research/artifacts.py",
    "research/campaign/__init__.py",
    "research/campaign/artifacts.py",
    "research/campaign/common.py",
    "research/campaign/journal.py",
    "research/campaign/plan.py",
    "research/campaign/report.py",
    "research/campaign/runner.py",
    "research/campaign/worker.py",
    "research/evaluation/__init__.py",
    "research/evaluation/compare.py",
    "research/evaluation/evaluate.py",
    "research/evaluation/finalize.py",
    "research/evaluation/selection.py",
    "research/gpu.py",
    "research/models/__init__.py",
    "research/models/collaborative.py",
    "research/models/content.py",
    "research/models/dispatch.py",
    "research/models/hybrid.py",
    "research/models/inputs.py",
    "research/models/manifest.py",
    "research/models/package.py",
    "research/models/popularity.py",
    "research/models/spec.py",
    "research/models/types.py",
    "research/predict.py",
    "research/prepare.py",
    "research/protocol.py",
    "research/resources.py",
    "resources/campaigns/research_campaign_v1.json",
    "resources/expectations/nafiul_2026_v1_alias_expectations.json",
    "resources/expectations/neelagiri_2025_v1_expectations.json",
    "resources/prompts/assessment_prompt_v1.md",
    "resources/protocols/research_protocol_v1.json",
    "resources/queries/quality_queries_v1.json",
    "resources/registries/source_registry.json",
    "resources/registries/source_registry_glove_6b_300d_v1.json",
    "resources/registries/source_registry_nafiul_2026_v1.json",
    "resources/registries/source_registry_neelagiri_2025_v1.json",
    "resources/releases/release_neelagiri_2025_nafiul_2026_v1.json",
)


def _source_path(name: str) -> PurePosixPath:
    if not isinstance(name, str) or not name or "\\" in name:
        raise ValueError("source_path_invalid")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in ("", ".", "..") for part in name.split("/")):
        raise ValueError("source_path_invalid")
    if path.suffix not in (".py", ".json", ".md"):
        raise ValueError("source_path_invalid")
    return path


def source_fingerprints() -> dict[str, str]:
    """Fingerprint the current installed raw bytes under their actual nested paths."""
    if len(SOURCE_FILES) != len(set(SOURCE_FILES)) or tuple(sorted(SOURCE_FILES)) != SOURCE_FILES:
        raise ValueError("source_inventory_invalid")
    package = resources.files("recsys")
    return {
        name: hashlib.sha256(package.joinpath(*_source_path(name).parts).read_bytes()).hexdigest()
        for name in SOURCE_FILES
    }
