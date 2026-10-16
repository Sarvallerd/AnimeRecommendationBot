"""Exercise local DVC storage with tiny disposable A/B/A snapshots."""

from __future__ import annotations

import gzip
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from dvc_local import StorageError, configure

ROOT = Path(__file__).resolve().parents[1]
DVC = [sys.executable, "-m", "dvc"]


def run(argv: list[str], repo: Path, *, succeeds: bool = True) -> str:
    result = subprocess.run(
        argv,
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
        env={**os.environ, "DVC_NO_ANALYTICS": "1"},
    )
    if (result.returncode == 0) != succeeds:
        raise AssertionError(
            f"unexpected exit {result.returncode}: {' '.join(argv[:3])}\n{result.stdout}\n{result.stderr}"
        )
    return (result.stdout + result.stderr).strip() if not succeeds else result.stdout.strip()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory(directory: Path) -> dict[str, str]:
    return {
        str(path.relative_to(directory)): digest(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def files_snapshot(directory: Path) -> dict[str, str]:
    return inventory(directory) if directory.exists() else {}


def assert_private(path: Path, directory: bool) -> None:
    mode = path.stat().st_mode & 0o777
    assert mode == (0o700 if directory else 0o600), (path, oct(mode))


def local_config(repo: Path, cache: Path, remote: Path) -> None:
    run(DVC + ["config", "--local", "cache.dir", str(cache)], repo)
    run(DVC + ["config", "--local", "cache.type", "copy"], repo)
    run(DVC + ["remote", "add", "--local", "--default", "local", str(remote)], repo)
    run(DVC + ["remote", "modify", "--local", "local", "jobs", "2"], repo)
    run(DVC + ["remote", "modify", "--local", "local", "verify", "true"], repo)


def assert_bad(repo: Path, location: Path | None, expected: str) -> None:
    try:
        configure(repo, location)
    except StorageError as exc:
        assert expected in str(exc), (expected, str(exc))
    else:
        raise AssertionError(f"configuration unexpectedly accepted {expected}")


def main() -> None:
    assert version("dvc") == "3.67.1"
    for package in ("anime-recommendation-recsys", "maturin", "numpy"):
        try:
            version(package)
        except PackageNotFoundError:
            continue
        raise AssertionError(f"data-only environment installed {package}")

    previous_umask = os.umask(0o077)
    try:
        with tempfile.TemporaryDirectory(prefix="arb047-dvc-") as scratch:
            base = Path(scratch)
            repo, source, storage = (base / name for name in ("repo", "source", "storage"))
            repo.mkdir()
            source.mkdir()
            run(["git", "init", "-q", str(repo)], base)
            run(["git", "config", "user.name", "DVC fixture"], repo)
            run(["git", "config", "user.email", "fixture@example.invalid"], repo)
            (repo / ".dvc").mkdir()
            for name in (".dvc/config", ".dvc/.gitignore", ".dvcignore", ".gitignore"):
                target = repo / name
                target.write_bytes((ROOT / name).read_bytes())
            run(["git", "add", ".dvc/config", ".dvc/.gitignore", ".dvcignore", ".gitignore"], repo)
            run(["git", "commit", "-qm", "Fixture setup"], repo)

            assert_bad(repo, repo / "storage", "overlaps")
            (base / "source-link").symlink_to(source, target_is_directory=True)
            assert_bad(repo, base / "source-link" / "storage", "symlink")
            alias_storage = base / "alias-storage"
            alias_storage.mkdir()
            (alias_storage / "cache").symlink_to(source, target_is_directory=True)
            assert_bad(repo, alias_storage, "symlink")
            assert not (repo / ".dvc/config.local").exists()
            configure(repo, storage)
            configure(repo, storage)
            before_check = files_snapshot(repo / ".dvc")
            assert configure(repo, None, check_only=True) == storage
            assert files_snapshot(repo / ".dvc") == before_check
            assert_bad(repo, base / "other-storage", "conflicting")
            local_file = repo / ".dvc/config.local"
            local_file.chmod(0o644)
            try:
                assert_bad(repo, storage, "mode 0600")
            finally:
                local_file.chmod(0o600)
            for directory in (storage, storage / "cache", storage / "remote"):
                for wrong_mode in (0o500, 0o600, 0o755):
                    directory.chmod(wrong_mode)
                    try:
                        assert_bad(repo, storage, "mode 0700")
                        try:
                            configure(repo, storage, check_only=True)
                        except StorageError as exc:
                            assert "mode 0700" in str(exc), str(exc)
                        else:
                            raise AssertionError(f"check accepted {oct(wrong_mode)} on {directory}")
                    finally:
                        directory.chmod(0o700)
            for path in (storage, storage / "cache", storage / "remote"):
                assert_private(path, True)
            assert_private(repo / ".dvc/config.local", False)
            assert run(["git", "status", "--porcelain"], repo) == ""

            payload_a = source / "a"
            payload_a.mkdir()
            (payload_a / "item.json").write_bytes(b'{"id":1,"name":"fixture A"}\n')
            (payload_a / "bytes.bin").write_bytes(bytes(range(32)))
            with (payload_a / "packed.gz").open("wb") as raw:
                with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as stream:
                    stream.write(b"tiny fixture\n")
            source_a = inventory(payload_a)
            dest = repo / "data/snapshots/fixtures/example"
            dest.parent.mkdir(parents=True)
            run(DVC + ["add", str(payload_a), "--out", str(dest.relative_to(repo))], repo)
            assert inventory(dest) == source_a
            pointer = repo / "data/snapshots/fixtures/example.dvc"
            assert pointer.is_file()
            run(
                [
                    "git",
                    "add",
                    "data/snapshots/fixtures/example.dvc",
                    "data/snapshots/fixtures/.gitignore",
                ],
                repo,
            )
            staged = run(["git", "diff", "--cached", "--name-only"], repo).splitlines()
            assert staged == [
                "data/snapshots/fixtures/.gitignore",
                "data/snapshots/fixtures/example.dvc",
            ]
            metadata = (
                pointer.read_text() + (repo / "data/snapshots/fixtures/.gitignore").read_text()
            )
            assert str(source) not in metadata and str(storage) not in metadata
            assert "deps:" not in metadata and "url:" not in metadata
            run(["git", "commit", "-qm", "Snapshot A"], repo)
            commit_a = run(["git", "rev-parse", "HEAD"], repo)
            run(DVC + ["push", "-r", "local", str(pointer.relative_to(repo))], repo)
            remote_a = files_snapshot(storage / "remote")
            cache_a = files_snapshot(storage / "cache")
            shutil.rmtree(dest)
            run(DVC + ["checkout", str(pointer.relative_to(repo))], repo)
            assert inventory(dest) == source_a
            for rel in source_a:
                src, out = payload_a / rel, dest / rel
                assert (
                    src.stat().st_ino != out.stat().st_ino or src.stat().st_dev != out.stat().st_dev
                )
                assert not out.is_symlink()

            clone = base / "clone"
            run(["git", "clone", "-q", str(repo), str(clone)], base)
            cache_clone = base / "clone-cache"
            cache_clone.mkdir()
            local_config(clone, cache_clone, storage / "remote")
            assert files_snapshot(cache_clone) == {}
            run(DVC + ["pull", "-r", "local", "data/snapshots/fixtures/example.dvc"], clone)
            assert inventory(clone / "data/snapshots/fixtures/example") == source_a
            assert files_snapshot(cache_clone)

            payload_b = source / "b"
            shutil.copytree(payload_a, payload_b)
            (payload_b / "item.json").write_bytes(b'{"id":1,"name":"fixture B"}\n')
            source_b = inventory(payload_b)
            shutil.copytree(payload_b, dest, dirs_exist_ok=True)
            run(DVC + ["add", str(dest.relative_to(repo))], repo)
            run(["git", "add", "data/snapshots/fixtures/example.dvc"], repo)
            run(["git", "commit", "-qm", "Snapshot B"], repo)
            commit_b = run(["git", "rev-parse", "HEAD"], repo)
            run(DVC + ["push", "-r", "local", str(pointer.relative_to(repo))], repo)
            assert inventory(dest) == source_b
            for commit, expected in (
                (commit_a, source_a),
                (commit_b, source_b),
                (commit_a, source_a),
            ):
                run(["git", "switch", "-q", "--detach", commit], repo)
                run(DVC + ["checkout", "data/snapshots/fixtures/example.dvc"], repo)
                assert inventory(dest) == expected

            remote_after = files_snapshot(storage / "remote")
            cache_after = files_snapshot(storage / "cache")
            (dest / "item.json").write_bytes(b"modified working copy\n")
            assert inventory(payload_a) == source_a
            assert files_snapshot(storage / "cache") == cache_after
            assert files_snapshot(storage / "remote") == remote_after
            assert (
                remote_a.items() <= remote_after.items() and cache_a.items() <= cache_after.items()
            )

            # A new cache must refuse the deliberately damaged remote object.
            object_id = hashlib.md5((payload_a / "item.json").read_bytes()).hexdigest()
            remote_object = storage / "remote" / "files/md5" / object_id[:2] / object_id[2:]
            assert remote_object.is_file()
            remote_object.chmod(0o600)
            remote_object.write_bytes(b"corrupt fixture object")
            corrupt_clone = base / "corrupt-clone"
            run(["git", "clone", "-q", str(repo), str(corrupt_clone)], base)
            corrupt_cache = base / "corrupt-cache"
            corrupt_cache.mkdir()
            local_config(corrupt_clone, corrupt_cache, storage / "remote")
            failure = run(
                DVC + ["pull", "-r", "local", "data/snapshots/fixtures/example.dvc"],
                corrupt_clone,
                succeeds=False,
            ).lower()
            assert "checkout failed" in failure and "example" in failure, failure
            assert not (corrupt_clone / "data/snapshots/fixtures/example/item.json").exists()

            linked = base / "linked"
            run(["git", "worktree", "add", "-q", "--detach", str(linked), commit_b], repo)
            assert (linked / ".git").is_file()
            linked_storage = base / "linked-storage"
            configure(linked, linked_storage)
            run(["git", "worktree", "remove", "--force", str(linked)], repo)
            assert linked_storage.is_dir() and (linked_storage / "cache").is_dir()

            # The repository's build context allowlist excludes the new data surface.
            dockerignore = (ROOT / ".dockerignore").read_text().splitlines()
            assert dockerignore[0] == "**"
            assert not any(
                line.startswith("!data/") or line.startswith("!.dvc") or line.startswith("!docs/")
                for line in dockerignore
            )
            assert inventory(payload_b) == source_b
    finally:
        os.umask(previous_umask)
    print(
        "DVC fixture: configuration, A/B/A, fresh pull, corruption, linked worktree, and hygiene OK"
    )


if __name__ == "__main__":
    main()
