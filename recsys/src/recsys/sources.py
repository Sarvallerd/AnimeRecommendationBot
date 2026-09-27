"""Pinned input descriptions; loading them performs no work at import time."""

import hashlib
import json
import re
from importlib import resources
from pathlib import PurePath


class SourceError(Exception):
    """An input source could not be validated or installed."""


def load_registry() -> tuple[dict, str]:
    """Load the registry packaged with the installed distribution."""
    raw = resources.files("recsys").joinpath("source_registry.json").read_bytes()
    try:
        registry = json.loads(raw)
    except (ValueError, UnicodeError) as exc:
        raise SourceError(f"invalid packaged source registry: {exc}") from exc
    validate_registry(registry)
    return registry, hashlib.sha256(raw).hexdigest()


def validate_registry(registry: dict) -> None:
    """Reject unsafe or incomplete descriptors before touching the data directory."""
    if not isinstance(registry, dict) or registry.get("schema_version") != 1:
        raise SourceError("source registry must have schema_version 1")
    if not isinstance(registry.get("provenance"), dict):
        raise SourceError("source registry needs provenance")
    sources = registry.get("sources")
    if not isinstance(sources, list) or not sources:
        raise SourceError("source registry needs sources")
    seen_ids: set[str] = set()
    seen_names: set[str] = set()
    for source in sources:
        if not isinstance(source, dict):
            raise SourceError("invalid source descriptor")
        source_id = source.get("id")
        filename = source.get("filename")
        size = source.get("size")
        digest = source.get("sha256")
        if not isinstance(source_id, str) or not source_id or source_id in seen_ids:
            raise SourceError("source ids must be unique nonempty strings")
        if (not isinstance(filename, str) or filename in ("", ".", "..")
                or PurePath(filename).name != filename or "/" in filename or "\\" in filename
                or filename in seen_names):
            raise SourceError(f"unsafe or duplicate source filename: {filename!r}")
        if type(size) is not int or size < 0:
            raise SourceError(f"invalid size for {source_id}")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise SourceError(f"invalid SHA256 for {source_id}")
        remote = "url" in source
        archive = "archive_id" in source or "member" in source
        if remote == archive:
            raise SourceError(f"source {source_id} needs exactly one origin")
        if remote:
            if not isinstance(source["url"], str) or not source["url"].startswith(("https://", "http://")):
                raise SourceError(f"invalid URL for {source_id}")
        elif (source.get("archive_id") not in seen_ids
              or not isinstance(source.get("member"), str)
              or not source["member"]):
            raise SourceError(f"invalid archive reference for {source_id}")
        seen_ids.add(source_id)
        seen_names.add(filename)
