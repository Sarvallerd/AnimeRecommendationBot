"""Configure or verify private, worktree-independent DVC storage."""

from __future__ import annotations

import argparse
import configparser
import os
import stat
import subprocess
import sys
from pathlib import Path


class StorageError(Exception):
    pass


def _run(argv: list[str], repo: Path) -> str:
    completed = subprocess.run(argv, cwd=repo, text=True, capture_output=True, check=False)
    if completed.returncode:
        detail = (completed.stderr or completed.stdout).strip().splitlines()
        raise StorageError(
            f"{argv[0]} {argv[1]} failed: {detail[-1] if detail else completed.returncode}"
        )
    return completed.stdout.strip()


def _repo_root(repo: Path) -> Path:
    root = Path(_run(["git", "rev-parse", "--show-toplevel"], repo))
    if not root.is_absolute() or root.resolve() != repo.resolve():
        raise StorageError("run from the repository root")
    _run(["git", "ls-files", "--error-unmatch", ".dvc/config"], repo)
    if not (repo / ".dvc/config").is_file():
        raise StorageError("tracked .dvc/config is missing")
    _components_without_links(repo / ".dvc")
    return root


def _components_without_links(path: Path) -> None:
    if not path.is_absolute() or ".." in path.parts:
        raise StorageError("storage root must be an absolute normalized filesystem path")
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        if current.is_symlink():
            raise StorageError(f"symlink in storage path: {current}")


def _contains(parent: Path, child: Path) -> bool:
    return child == parent or parent in child.parents


def _storage_root(repo: Path, requested: Path | None) -> Path:
    if requested is None:
        xdg = os.environ.get("XDG_DATA_HOME")
        base = Path(xdg) if xdg else Path.home() / ".local/share"
        requested = base / "anime-recommendation-bot/dvc"
    _components_without_links(requested)
    root = requested.resolve(strict=False)
    listings = _run(["git", "worktree", "list", "--porcelain"], repo)
    forbidden = [
        Path(line[9:]).resolve() for line in listings.splitlines() if line.startswith("worktree ")
    ]
    common = Path(
        _run(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"], repo)
    ).resolve()
    forbidden.append(common)
    for path in forbidden:
        if _contains(root, path) or _contains(path, root):
            raise StorageError(f"storage root overlaps a Git worktree or common directory: {path}")
        if root.exists() and path.exists() and root.samefile(path):
            raise StorageError(f"storage root aliases a Git worktree or common directory: {path}")
    return root


def _private_dir(path: Path, *, create: bool) -> None:
    if path.is_symlink():
        raise StorageError(f"storage directory must not be a symlink: {path}")
    if path.exists():
        info = path.stat()
        if (
            not stat.S_ISDIR(info.st_mode)
            or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) != 0o700
        ):
            raise StorageError(
                f"storage directory must be owned by this user and mode 0700: {path}"
            )
    elif create:
        path.mkdir(parents=True, mode=0o700)
    else:
        raise StorageError(f"storage directory is missing: {path}")
    if stat.S_IMODE(path.stat().st_mode) != 0o700:
        raise StorageError(f"storage directory must have mode 0700: {path}")


def _local_config(repo: Path) -> configparser.ConfigParser | None:
    path = repo / ".dvc/config.local"
    if not path.exists():
        return None
    if path.is_symlink():
        raise StorageError(".dvc/config.local must be a regular private file")
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise StorageError(".dvc/config.local must be owned by this user and mode 0600")
    config = configparser.ConfigParser(interpolation=None)
    try:
        with path.open(encoding="utf-8") as stream:
            config.read_file(stream)
    except (OSError, configparser.Error) as exc:
        raise StorageError(f"invalid .dvc/config.local: {exc}") from exc
    return config


def _expected(root: Path) -> dict[tuple[str, str], str]:
    return {
        ("cache", "dir"): str(root / "cache"),
        ("cache", "type"): "copy",
        ("core", "remote"): "local",
        ('remote "local"', "url"): str(root / "remote"),
        ('remote "local"', "jobs"): "2",
        ('remote "local"', "verify"): "true",
    }


def _verify_config(config: configparser.ConfigParser | None, root: Path) -> None:
    if config is None:
        raise StorageError("missing .dvc/config.local; run configure")
    for (section, key), value in _expected(root).items():
        # DVC serializes remote sections as ['remote "local"'].
        serialized_section = f"'{section}'" if section.startswith("remote ") else section
        actual = config.get(serialized_section, key, fallback=None)
        if actual != value:
            raise StorageError(
                f"conflicting .dvc/config.local [{section}] {key}; expected {value!r}"
            )


def _verify_effective(repo: Path, root: Path) -> None:
    for (section, key), expected in _expected(root).items():
        setting = f"remote.local.{key}" if section.startswith("remote ") else f"{section}.{key}"
        actual = _run([sys.executable, "-m", "dvc", "config", setting], repo)
        if actual != expected:
            raise StorageError(
                f"effective DVC setting {setting} is {actual!r}, expected {expected!r}"
            )


def _verify_ignores(repo: Path) -> None:
    files = [
        ".dvc/config.local",
        ".dvc/cache/example",
        ".dvc/tmp/example",
        "data/snapshots/fixture/payload.gz",
    ]
    for item in files:
        if not _run(["git", "check-ignore", "--no-index", item], repo):
            raise StorageError(f"Git must ignore {item}")
    for item in ["data/snapshots/fixture/example.dvc", "data/snapshots/fixture/.gitignore"]:
        result = subprocess.run(["git", "check-ignore", "-q", "--no-index", item], cwd=repo)
        if result.returncode == 0:
            raise StorageError(f"Git must allow DVC metadata: {item}")
    for item in [".tools/fixture", ".venv/fixture", "target/fixture"]:
        if _run([sys.executable, "-m", "dvc", "check-ignore", item], repo) == "":
            raise StorageError(f"DVC must ignore {item}")
    for item in (
        "data/snapshots/sample.csv",
        "data/snapshots/sample.gz",
        "data/snapshots/sample.npz",
        "data/snapshots/glove.6B.300d.txt",
    ):
        payload_check = subprocess.run(
            [sys.executable, "-m", "dvc", "check-ignore", item],
            cwd=repo,
            capture_output=True,
            text=True,
        )
        if payload_check.returncode != 1 or payload_check.stdout.strip():
            raise StorageError(f"DVC must allow snapshot payloads: {item}")


def configure(repo: Path, requested: Path | None = None, *, check_only: bool = False) -> Path:
    repo = _repo_root(repo)
    prior = _local_config(repo)
    if check_only and requested is None and prior is not None:
        cache = prior.get("cache", "dir", fallback="")
        if not cache:
            raise StorageError("missing local cache.dir")
        requested = Path(cache).parent
    root = _storage_root(repo, requested)
    if root.exists():
        _private_dir(root, create=False)
    for path in (root, root / "cache", root / "remote"):
        _components_without_links(path)
    if prior is not None:
        _verify_config(prior, root)
    elif check_only:
        raise StorageError("missing .dvc/config.local; run configure")
    else:
        previous_umask = os.umask(0o077)
        try:
            _private_dir(root, create=True)
            _private_dir(root / "cache", create=True)
            _private_dir(root / "remote", create=True)
            dvc = [sys.executable, "-m", "dvc"]
            _run(dvc + ["config", "--local", "cache.dir", str(root / "cache")], repo)
            _run(dvc + ["config", "--local", "cache.type", "copy"], repo)
            _run(
                dvc + ["remote", "add", "--local", "--default", "local", str(root / "remote")], repo
            )
            _run(dvc + ["remote", "modify", "--local", "local", "jobs", "2"], repo)
            _run(dvc + ["remote", "modify", "--local", "local", "verify", "true"], repo)
        finally:
            os.umask(previous_umask)
    _verify_config(_local_config(repo), root)
    _verify_effective(repo, root)
    for path in (root, root / "cache", root / "remote"):
        _components_without_links(path)
        _private_dir(path, create=False)
    _verify_ignores(repo)
    return root


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("configure", "check"))
    parser.add_argument("--storage-root", type=Path)
    args = parser.parse_args()
    try:
        root = configure(Path.cwd(), args.storage_root, check_only=args.action == "check")
    except StorageError as exc:
        print(f"DVC local storage: {exc}", file=sys.stderr)
        return 1
    print(f"DVC local storage {args.action}: OK ({root})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
