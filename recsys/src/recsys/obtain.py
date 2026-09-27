"""Verified, atomic acquisition of the pinned recommendation inputs."""

import hashlib
import http.client
import json
import os
import stat
import tempfile
import urllib.error
import urllib.request
import zipfile
import zlib
from pathlib import Path

from .sources import SourceError, load_registry, validate_registry

CHUNK_SIZE = 1024 * 1024
TIMEOUT_SECONDS = 60
MANIFEST_NAME = "source-manifest.json"


def _digest_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(CHUNK_SIZE):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def _verify_existing(path: Path, source: dict) -> bool:
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise SourceError(f"unsafe source target: {path}")
    if not path.exists():
        return False
    actual_size, actual_hash = _digest_file(path)
    if actual_size != source["size"] or actual_hash != source["sha256"]:
        raise SourceError(
            f"corrupt source {path}: expected size {source['size']} and SHA256 "
            f"{source['sha256']}; actual size {actual_size} and SHA256 {actual_hash}. "
            "Remove this file and retry."
        )
    return True


def _write_verified(path: Path, source: dict, chunks) -> None:
    """Stream to a sibling temp file, then install only the exact expected bytes."""
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(prefix=f".{path.name}.", suffix=".tmp",
                                         dir=path.parent, delete=False) as output:
            temp_path = Path(output.name)
            digest = hashlib.sha256()
            size = 0
            for chunk in chunks:
                size += len(chunk)
                if size > source["size"]:
                    raise SourceError(f"oversized source {path}: expected {source['size']} bytes, received more")
                digest.update(chunk)
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        actual_hash = digest.hexdigest()
        if size != source["size"] or actual_hash != source["sha256"]:
            raise SourceError(
                f"invalid source {path}: expected size {source['size']} and SHA256 "
                f"{source['sha256']}; actual size {size} and SHA256 {actual_hash}"
            )
        temp_path.replace(path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _chunks(stream):
    while chunk := stream.read(CHUNK_SIZE):
        yield chunk


def _download(path: Path, source: dict) -> None:
    try:
        with urllib.request.urlopen(source["url"], timeout=TIMEOUT_SECONDS) as stream:
            _write_verified(path, source, _chunks(stream))
    except (urllib.error.URLError, TimeoutError, OSError, http.client.HTTPException) as exc:
        raise SourceError(f"download failed for {path} from {source['url']}: {exc}") from exc


def _extract(path: Path, source: dict, archive_path: Path) -> None:
    try:
        with zipfile.ZipFile(archive_path) as archive:
            matches = [item for item in archive.infolist() if item.filename == source["member"]]
            if len(matches) != 1:
                raise SourceError(
                    f"archive {archive_path} must contain exactly one {source['member']}; found {len(matches)}"
                )
            member = matches[0]
            mode = member.external_attr >> 16
            if (member.is_dir() or stat.S_IFMT(mode) in (stat.S_IFDIR, stat.S_IFLNK)
                    or member.file_size != source["size"]):
                raise SourceError(f"unsafe or incorrect member {source['member']} in {archive_path}")
            with archive.open(member) as stream:
                _write_verified(path, source, _chunks(stream))
    except (zipfile.BadZipFile, zipfile.LargeZipFile, RuntimeError, EOFError, OSError,
            NotImplementedError, zlib.error) as exc:
        raise SourceError(f"cannot extract {source['member']} from {archive_path}: {exc}") from exc


def obtain(data_dir: Path, *, offline: bool = False, registry: dict | None = None,
           registry_sha256: str | None = None) -> Path:
    """Install all registry sources and return the verified manifest path.

    The registry arguments allow small local fixtures in tests; the CLI always uses
    the packaged registry.
    """
    if registry is None:
        registry, registry_sha256 = load_registry()
    else:
        validate_registry(registry)
        if registry_sha256 is None:
            registry_sha256 = hashlib.sha256(
                json.dumps(registry, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
    data_dir = Path(data_dir)
    if data_dir.is_symlink() or (data_dir.exists() and not data_dir.is_dir()):
        raise SourceError(f"unsafe data directory: {data_dir}")
    data_dir.mkdir(parents=True, exist_ok=True)
    manifest = data_dir / MANIFEST_NAME
    if manifest.is_dir():
        raise SourceError(f"unsafe manifest target: {manifest}")
    manifest.unlink(missing_ok=True)
    by_id = {source["id"]: source for source in registry["sources"]}
    for source in registry["sources"]:
        path = data_dir / source["filename"]
        if _verify_existing(path, source):
            continue
        if "url" in source:
            if offline:
                raise SourceError(f"missing source {path}; retry without --offline")
            _download(path, source)
        else:
            archive = by_id[source["archive_id"]]
            _extract(path, source, data_dir / archive["filename"])
    payload = {
        "schema_version": 1,
        "registry_sha256": registry_sha256,
        "provenance": registry["provenance"],
        "verified": [
            {key: source[key] for key in ("filename", "size", "sha256")}
            for source in registry["sources"]
        ],
    }
    manifest_source = {
        "size": 0,
        "sha256": "",
    }
    content = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    manifest_source["size"] = len(content)
    manifest_source["sha256"] = hashlib.sha256(content).hexdigest()
    _write_verified(manifest, manifest_source, (content,))
    return manifest
