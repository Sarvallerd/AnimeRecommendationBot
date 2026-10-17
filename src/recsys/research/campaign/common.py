"""Fixed, supervised and resumable validation-only ML-02 campaign.

The worker entry uses only the standard library until the generation gate has
been acquired and checked. All expensive scientific work runs in workers.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Never

VARIANTS = (
    "popularity",
    "content-mean",
    "content-max",
    "content-union",
    "itemknn",
    "puresvd",
    "hybrid",
)


PHASES = ("fit", "predict", "evaluate", "replay")


HYBRID_COMPONENT_VARIANTS = ("content-mean", "puresvd")


FIT_SECONDS = 7200.0


EVALUATION_SECONDS = 3600.0


CAMPAIGN_SECONDS = 172800.0


RSS_LIMIT_BYTES = 8 * 1024**3


_SAFE_CODE = re.compile(r"research_[a-z0-9_]+\Z")


def _fail(code: str = "research_campaign_invalid") -> Never:
    from recsys.research.protocol import fail

    fail(code)


def _canonical(value: object) -> bytes:
    from recsys.contracts.bundle import canonical_bytes

    return canonical_bytes(value)


def _digest(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime) -> str:
    return value.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _parse_iso(value: object) -> datetime:
    if type(value) is not str or not value.endswith("Z"):
        _fail()
    try:
        return datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        _fail()


def _private_file(path: str | Path, raw: bytes) -> None:
    path = Path(path)
    with open(path, "xb", opener=lambda name, flags: os.open(name, flags, 0o600)) as out:
        out.write(raw)
        out.flush()
        os.fsync(out.fileno())
    _sync_dir(path.parent)


def _replace_private(path: str | Path, raw: bytes) -> None:
    path = Path(path)
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex)
    _private_file(temporary, raw)
    os.replace(temporary, path)
    _sync_dir(path.parent)


def _sync_dir(directory: str | Path) -> None:
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _read_private(path: str | Path, *, limit: int = 16 * 1024 * 1024) -> tuple[object, bytes]:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        _fail()
    info = path.stat()
    if info.st_size > limit or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600:
        _fail()
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (ValueError, UnicodeError):
        _fail()
    if raw != _canonical(value):
        _fail()
    return value, raw


def _touch_lock(path: Path) -> int:
    if path.is_symlink():
        _fail()
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    info = os.fstat(fd)
    if stat.S_IMODE(info.st_mode) != 0o600 or info.st_uid != os.getuid():
        os.close(fd)
        _fail()
    return fd


def _ensure_private(directory: str | Path) -> None:
    directory = Path(directory)
    if directory.exists():
        if directory.is_symlink() or not directory.is_dir() or directory.stat().st_mode & 0o077:
            _fail()
        return
    _ensure_private(directory.parent)
    directory.mkdir(mode=0o700)
    _sync_dir(directory.parent)


def _bytes_equal(left: str | Path, right: str | Path) -> bool:
    with Path(left).open("rb") as one, Path(right).open("rb") as two:
        while True:
            a, b = one.read(1024 * 1024), two.read(1024 * 1024)
            if a != b:
                return False
            if not a:
                return True
