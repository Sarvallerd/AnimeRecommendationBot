"""Shared bounded gzip reading and atomic, no-replace publication."""

import ctypes
import os
import zlib
from pathlib import Path


class ArtifactIOError(ValueError):
    """Safe fixed-code failure for an invalid compressed artifact."""


def fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def publish_no_replace(stage: Path, target: Path) -> None:
    # Linux renameat2 keeps directory publication atomic and refuses a competing target.
    libc = ctypes.CDLL(None, use_errno=True)
    result = libc.renameat2(-100, os.fsencode(stage), -100, os.fsencode(target), 1)
    if result != 0:
        raise OSError(ctypes.get_errno(), "atomic dataset publication failed")
    fsync_directory(target.parent)


def iter_single_member_gzip(path: Path):
    """Yield one gzip member's lines and refuse trailing members or bytes."""
    decompressor = zlib.decompressobj(31)
    pending = b""
    with path.open("rb") as stream:
        if stream.read(10) != bytes.fromhex("1f8b08000000000000ff"):
            raise ArtifactIOError("artifact_gzip_invalid")
        stream.seek(0)
        while chunk := stream.read(64 * 1024):
            if decompressor.eof:
                raise ArtifactIOError("artifact_gzip_invalid")
            remaining = chunk
            while remaining:
                try:
                    decoded = decompressor.decompress(remaining, 1024 * 1024)
                except zlib.error as exc:
                    raise ArtifactIOError("artifact_gzip_invalid") from exc
                if decompressor.unused_data:
                    raise ArtifactIOError("artifact_gzip_invalid")
                remaining = decompressor.unconsumed_tail
                pending += decoded
                parts = pending.split(b"\n")
                pending = parts.pop()
                if len(pending) > 16 * 1024 * 1024:
                    raise ArtifactIOError("artifact_gzip_invalid")
                for part in parts:
                    yield part + b"\n"
    if not decompressor.eof or pending:
        raise ArtifactIOError("artifact_gzip_invalid")
