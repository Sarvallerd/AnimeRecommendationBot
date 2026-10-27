"""Finite access to immutable package assets by responsibility."""

from importlib import resources
from pathlib import PurePosixPath

_RESOURCE_NAMES = {
    "registries": frozenset(
        (
            "source_registry.json",
            "source_registry_neelagiri_2025_v1.json",
            "source_registry_nafiul_2026_v1.json",
            "source_registry_glove_6b_300d_v1.json",
        )
    ),
    "expectations": frozenset(
        ("neelagiri_2025_v1_expectations.json", "nafiul_2026_v1_alias_expectations.json")
    ),
    "releases": frozenset(("release_neelagiri_2025_nafiul_2026_v1.json",)),
    "protocols": frozenset(("research_protocol_v1.json",)),
    "campaigns": frozenset(
        (
            "research_campaign_v1.json",
            "research_campaign_native_v2.json",
            "research_campaign_native_v3.json",
            "research_seed_embedding_v1.json",
        )
    ),
    "queries": frozenset(("quality_queries_v1.json",)),
    "prompts": frozenset(("assessment_prompt_v1.md",)),
}


def resource_path(group: str, filename: str) -> str:
    """Return the installed relative path for a known asset."""
    if not isinstance(group, str) or group not in _RESOURCE_NAMES:
        raise ValueError("resource_unknown")
    if not isinstance(filename, str) or filename not in _RESOURCE_NAMES[group]:
        raise ValueError("resource_unknown")
    path = PurePosixPath(filename)
    if (
        path.is_absolute()
        or len(path.parts) != 1
        or path.parts[0] in ("", ".", "..")
        or "\\" in filename
    ):
        raise ValueError("resource_invalid")
    return f"resources/{group}/{filename}"


def resource_bytes(group: str, filename: str) -> bytes:
    """Read only an approved embedded asset, including from an installed wheel."""
    path = PurePosixPath(resource_path(group, filename))
    return resources.files("recsys").joinpath(*path.parts).read_bytes()
