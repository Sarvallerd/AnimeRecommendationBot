"""Strict, bounded, private research artifacts and atomic publication."""

import gzip
from .dataset import _stream_lines
import hashlib
import io
import json
import os
import platform
import re
import shutil
import tempfile
from importlib import resources
from pathlib import Path

from .bundle import canonical_bytes, parse_canonical
from .dataset import _publish_no_replace
from .research_protocol import ResearchError, exact, fail, sha

MAX_LINE = 4 * 1024 * 1024
MAX_STREAM = 64 * 1024 ** 3
CODE_FILES = ("research_protocol.py", "research_artifacts.py", "research_prepare.py",
              "research_evaluate.py", "research_compare.py", "research_finalize.py",
              "research_protocol_v1.json")


def safe_path(path, *, directory=None):
    path = Path(path)
    if any(part.is_symlink() for part in (path, *path.parents)):
        fail("research_unsafe_path")
    if directory is True and not path.is_dir():
        fail("research_unsafe_path")
    if directory is False and (not path.is_file() or path.stat().st_size > MAX_STREAM):
        fail("research_unsafe_path")
    return path


def separate(*paths):
    resolved = [safe_path(path).resolve(strict=False) for path in paths]
    for i, a in enumerate(resolved):
        for b in resolved[i + 1:]:
            if a == b or a.is_relative_to(b) or b.is_relative_to(a):
                fail("research_unsafe_path")


def descriptor(raw):
    return {"bytes": len(raw), "sha256": sha(raw)}


def check_descriptor(value, *, stream=False):
    fields = {"bytes", "sha256"}
    if stream:
        fields |= {"rows", "uncompressed_bytes", "canonical_sha256"}
    exact(value, fields)
    for key in ("bytes", "rows", "uncompressed_bytes"):
        if key in value and (type(value[key]) is not int or value[key] < 0 or value[key] > MAX_STREAM):
            fail()
    for key in ("sha256", "canonical_sha256"):
        if key in value and (type(value[key]) is not str or
                             re.fullmatch(r"[0-9a-f]{64}", value[key]) is None):
            fail()


def read_json(path, expected=None):
    path = safe_path(path, directory=False)
    if path.stat().st_size > 128 * 1024 * 1024:
        fail()
    raw = path.read_bytes()
    try:
        value = parse_canonical(raw, path.name)
    except (ValueError, UnicodeError) as exc:
        raise ResearchError("research_integrity_error") from exc
    if expected is not None:
        check_descriptor(expected)
    if expected is not None and descriptor(raw) != expected:
        fail()
    return value, raw


def write_json(path, value):
    raw = canonical_bytes(value)
    with open(path, "xb", opener=lambda name, flags: os.open(name, flags, 0o600)) as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return descriptor(raw)


def stream_jsonl(path, expected=None, *, max_total=MAX_STREAM):
    """Check both byte layers while yielding canonical rows; no full decompression."""
    if expected is not None:
        check_descriptor(expected, stream=True)
    path = safe_path(path, directory=False)
    h = hashlib.sha256()
    n = 0
    with path.open("rb") as inp:
        while block := inp.read(1024 * 1024):
            n += len(block)
            if n > max_total:
                fail("research_resource_limit")
            h.update(block)
    if expected is not None and (expected["bytes"] != n or expected["sha256"] != h.hexdigest()):
        fail()
    rows = count = 0
    plain = hashlib.sha256()
    try:
        for raw in _stream_lines(path):
            if len(raw) > MAX_LINE or not raw.endswith(b"\n"):
                fail("research_resource_limit")
            count += len(raw)
            if count > max_total:
                fail("research_resource_limit")
            rows += 1
            plain.update(raw)
            yield parse_canonical(raw, "research-row")
    except (OSError, EOFError, ValueError) as exc:
        raise ResearchError("research_integrity_error") from exc
    if expected is not None and (expected["rows"] != rows or
                                  expected["uncompressed_bytes"] != count or
                                  expected["canonical_sha256"] != plain.hexdigest()):
        fail()


def stream_tsv(path, expected=None):
    if expected is not None:
        check_descriptor(expected, stream=True)
    path = safe_path(path, directory=False)
    compressed_hash = hashlib.sha256()
    n = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            n += len(chunk)
            if n > MAX_STREAM:
                fail("research_resource_limit")
            compressed_hash.update(chunk)
    if expected is not None and (expected["bytes"] != n or expected["sha256"] != compressed_hash.hexdigest()):
        fail()
    rows = count = 0
    plain = hashlib.sha256()
    for raw in _stream_lines(path):
        if len(raw) > MAX_LINE or not raw.endswith(b"\n"):
            fail("research_resource_limit")
        count += len(raw)
        if count > MAX_STREAM:
            fail("research_resource_limit")
        rows += 1
        plain.update(raw)
        yield raw
    if expected is not None and (expected["rows"] != rows or expected["uncompressed_bytes"] != count or
                                  expected["canonical_sha256"] != plain.hexdigest()):
        fail()


class StreamWriter:
    def __init__(self, path):
        self.path = path
        self.file = open(path, "xb", opener=lambda name, flags: os.open(name, flags, 0o600))
        self.gzip = gzip.GzipFile(fileobj=self.file, mode="wb", filename="", mtime=0, compresslevel=6)
        self.rows = 0
        self.bytes = 0
        self.hash = hashlib.sha256()

    def write(self, raw):
        if len(raw) > MAX_LINE or self.bytes + len(raw) > MAX_STREAM:
            fail("research_resource_limit")
        self.gzip.write(raw)
        self.rows += 1
        self.bytes += len(raw)
        self.hash.update(raw)

    def write_json(self, value):
        self.write(canonical_bytes(value))

    def abort(self):
        if not self.gzip.closed:
            self.gzip.close()
        if not self.file.closed:
            self.file.close()

    def __del__(self):
        try:
            self.abort()
        except (AttributeError, OSError):
            pass

    def close(self):
        self.gzip.close()
        self.file.flush()
        os.fsync(self.file.fileno())
        self.file.close()
        compressed = self.path.read_bytes() if self.path.stat().st_size <= MAX_LINE else None
        if compressed is not None:
            base = descriptor(compressed)
        else:
            h = hashlib.sha256()
            n = 0
            with self.path.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    n += len(chunk)
                    h.update(chunk)
            base = {"bytes": n, "sha256": h.hexdigest()}
        return {**base, "rows": self.rows, "uncompressed_bytes": self.bytes,
                "canonical_sha256": self.hash.hexdigest()}


def open_stage(target):
    target = safe_path(target)
    if target.exists() or not target.parent.is_dir():
        fail("research_unsafe_path")
    stage = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    os.chmod(stage, 0o700)
    return stage


def publish(stage, target):
    for root, dirs, files in os.walk(stage):
        for name in dirs:
            os.chmod(Path(root) / name, 0o700)
        for name in files:
            os.chmod(Path(root) / name, 0o600)
        fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    _publish_no_replace(stage, target)


def cleanup(stage):
    if stage.exists():
        shutil.rmtree(stage)


def module_fingerprints():
    package = resources.files("recsys")
    return {name: sha(package.joinpath(name).read_bytes()) for name in CODE_FILES}


def producer(code_revision):
    import numpy as np
    from . import __version__
    if type(code_revision) is not str or len(code_revision) != 40 or any(
            char not in "0123456789abcdef" for char in code_revision):
        fail()
    return {"code_revision": code_revision, "code_files_sha256": module_fingerprints(),
            "python_version": platform.python_version(), "package_version": __version__,
            "numpy_version": np.__version__}


def identity(path):
    return "sha256:" + sha(safe_path(path, directory=False).read_bytes())


def check_identity(value):
    if type(value) is not str or len(value) != 71 or not value.startswith("sha256:"):
        fail()
    try:
        bytes.fromhex(value[7:])
    except ValueError:
        fail()
