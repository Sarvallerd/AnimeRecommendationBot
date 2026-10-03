"""Smoke-test the root Python/Rust package in a disposable source copy."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[2]
LOCKS = ("uv.lock", "Cargo.lock")
OBSOLETE = ("recsys/pyproject.toml", "recsys/uv.lock", "bot/pyproject.toml",
            "bot/Cargo.toml", "bot/Cargo.lock")
RESOURCES = ("source_registry.json", "quality_queries_v1.json", "assessment_prompt_v1.md")
MARKER = "arb-021-rebuilt"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy_source(destination: Path) -> None:
    files = ("pyproject.toml", "uv.lock", "Cargo.toml", "Cargo.lock", "rust-toolchain.toml",
             "packaging/scripts/recsys", "recsys/README.md", "contracts/check_bundle.py")
    trees = ("recsys/src", "bot/src", "bot/migrations", "bot/tests", "tests/fixtures/bundle")
    for name in files:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    for name in trees:
        shutil.copytree(ROOT / name, destination / name)


def run(command: list[str], cwd: Path, env: dict[str, str], *, timeout: int = 600) -> str:
    result = subprocess.run(command, cwd=cwd, env=env, text=True, capture_output=True,
                            timeout=timeout)
    if result.returncode:
        raise AssertionError(f"{' '.join(command)} exited {result.returncode}\n"
                             f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")
    return result.stdout.strip()


def uv(root: Path, env: dict[str, str], *args: str, timeout: int = 600) -> str:
    return run(["uv", *args], root, env, timeout=timeout)


def assert_manifest(root: Path, env: dict[str, str]) -> None:
    assert all((root / name).is_file() for name in ("pyproject.toml", "uv.lock", "Cargo.toml", "Cargo.lock"))
    assert all(not (root / name).exists() for name in OBSOLETE), OBSOLETE
    metadata = json.loads(run(["cargo", "metadata", "--locked", "--no-deps", "--format-version", "1"], root, env))
    packages = metadata["packages"]
    assert len(packages) == 1 and packages[0]["name"] == "bot", packages
    package = packages[0]
    assert Path(package["manifest_path"]).resolve() == root / "Cargo.toml"
    targets = {(target["name"], tuple(target["kind"])): Path(target["src_path"]).resolve()
               for target in package["targets"]}
    assert targets[("bot", ("lib",))] == root / "bot/src/lib.rs", targets
    assert targets[("bot", ("bin",))] == root / "bot/src/main.rs", targets
    assert targets[("check_bundle", ("bin",))] == root / "bot/src/bin/check_bundle.rs", targets
    tests = {name: path for (name, kind), path in targets.items() if kind == ("test",)}
    expected = {path.stem: path.resolve() for path in (root / "bot/tests").glob("*.rs")}
    assert len(tests) == 7 and tests == expected, (tests, expected)
    assert len(targets) == 10, targets


def assert_install(root: Path, env: dict[str, str], *, editable: bool, extra: bool = False) -> None:
    script = r"""
import importlib.metadata as metadata
from importlib import resources
from pathlib import Path
import recsys
import shutil
import sys

root = Path.cwd().resolve()
venv = (root / '.venv').resolve()
assert Path(sys.prefix).resolve() == venv, sys.prefix
scripts = venv / 'bin'
for name in ('recsys', 'bot', 'check_bundle'):
    command = shutil.which(name)
    assert command is not None and Path(command).resolve().parent == scripts, command
installed = Path(recsys.__file__).resolve()
if EDITABLE:
    assert installed.is_relative_to(root / 'recsys' / 'src'), installed
else:
    assert installed.is_relative_to(venv), installed
for name in ('source_registry.json', 'quality_queries_v1.json', 'assessment_prompt_v1.md'):
    assert resources.files('recsys').joinpath(name).read_bytes() == (root / 'recsys/src/recsys' / name).read_bytes(), name
assert metadata.version('anime-recommendation-recsys') == '0.1.0'
files = [str(path) for path in metadata.distribution('anime-recommendation-recsys').files]
for name in ('recsys', 'bot', 'check_bundle'):
    assert any(path.endswith('/bin/' + name) for path in files), (name, files)
assert not any(path.endswith(('.so', '.pyd', '.dylib')) for path in files), files
installed_script = scripts / 'recsys'
assert installed_script.stat().st_mode & 0o111
assert Path(installed_script.read_bytes().splitlines()[0][2:].decode()).resolve() == (scripts / 'python').resolve()
""".replace("EDITABLE", "True" if editable else "False")
    args = ["run", "--locked"]
    if not editable:
        args.append("--no-editable")
    if extra:
        args += ["--extra", "quality"]
    uv(root, env, *args, "python", "-I", "-c", script)


def check_fixture(root: Path, env: dict[str, str], *, extra: bool = False,
                  no_editable: bool = False) -> str:
    fixture = "tests/fixtures/bundle"
    expected = "sha256:" + digest(root / fixture / "manifest.json") + " records=8"
    uv_args = ["--locked"]
    if no_editable:
        uv_args.append("--no-editable")
    if extra:
        uv_args.extend(("--extra", "quality"))
    outputs = (
        run([sys.executable, "contracts/check_bundle.py", fixture], root, env),
        uv(root, env, "run", *uv_args, "recsys", "validate", fixture),
        uv(root, env, "run", *uv_args, "check_bundle", fixture),
    )
    assert outputs == (expected,) * 3, outputs
    return expected


def assert_direct_script(root: Path, base: Path, env: dict[str, str], expected: str) -> None:
    script = root / ".venv/bin/recsys"
    outside_env = env.copy()
    outside_env.pop("PYTHONPATH", None)
    outside_env.pop("VIRTUAL_ENV", None)
    outside_env["PATH"] = str(base / "empty-path")
    assert run([str(script), "--version"], base, outside_env, timeout=20) == "recsys 0.1.0"
    assert run([str(script), "validate", str(root / "tests/fixtures/bundle")],
               base, outside_env, timeout=20) == expected


def main() -> None:
    assert sys.version_info[:2] == (3, 12), "Python 3.12 is required"
    before = {name: digest(ROOT / name) for name in LOCKS}
    with tempfile.TemporaryDirectory(prefix="arb-021-env-") as temporary:
        base = Path(temporary)
        root = base / "source"
        root.mkdir()
        copy_source(root)
        env = os.environ.copy()
        for name in ("VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME", "UV_PROJECT_ENVIRONMENT",
                     "UV_ENV_FILE", "TELOXIDE_TOKEN", "DATABASE_URL", "ARTIFACTS_DIR",
                     "RUST_LOG", "ARB_TEST_DATABASE_URL"):
            env.pop(name, None)
        env["UV_CACHE_DIR"] = str(base / "uv-cache")
        env["CARGO_TARGET_DIR"] = str(base / "cargo-target")
        assert_manifest(root, env)
        uv(root, env, "sync", "--locked")
        assert_install(root, env, editable=True)
        assert uv(root, env, "run", "--locked", "recsys", "--version") == "recsys 0.1.0"
        expected = check_fixture(root, env)
        assert_direct_script(root, base, env, expected)

        config_env = env | {
            "TELOXIDE_TOKEN": "1:arb_021_smoke",
            "DATABASE_URL": "postgresql://arb_021:arb_021@127.0.0.1:1/arb_021?sslmode=disable",
            "ARTIFACTS_DIR": "tests/fixtures/bundle", "RUST_LOG": "off",
        }
        assert uv(root, config_env, "run", "--locked", "bot", "--check-config", timeout=20) == "Configuration is valid."
        uv(root, env, "sync", "--locked")
        assert_install(root, env, editable=True)
        assert check_fixture(root, env) == expected

        source = root / "bot/src/bin/check_bundle.rs"
        original = source.read_text()
        needle = 'println!("{} records={}", bundle.identity(), bundle.catalog().len())'
        assert original.count(needle) == 1
        source.write_text(original.replace(needle, 'println!("{} records={} ' + MARKER + '", bundle.identity(), bundle.catalog().len())'))
        fixture = "tests/fixtures/bundle"
        assert uv(root, env, "run", "--locked", "check_bundle", fixture) == expected + " " + MARKER
        uv(root, env, "sync", "--locked")
        assert uv(root, env, "run", "--locked", "check_bundle", fixture) == expected + " " + MARKER
        source.write_text(original)
        assert uv(root, env, "run", "--locked", "check_bundle", fixture) == expected

        uv(root, env, "sync", "--locked", "--extra", "quality")
        quality_script = ("import numpy,pandas,sklearn; "
                          "assert (numpy.__version__,pandas.__version__,sklearn.__version__) "
                          "== ('2.5.3','3.0.6','1.9.1')")
        uv(root, env, "run", "--locked", "--extra", "quality", "python", "-I", "-c", quality_script)
        assert uv(root, env, "run", "--locked", "--extra", "quality", "recsys", "quality", "--help")
        assert_install(root, env, editable=True, extra=True)
        assert check_fixture(root, env, extra=True) == expected

        uv(root, env, "sync", "--locked", "--no-editable", "--extra", "quality")
        assert_install(root, env, editable=False, extra=True)
        assert_direct_script(root, base, env, expected)
        assert check_fixture(root, env, extra=True, no_editable=True) == expected
        assert {name: digest(root / name) for name in LOCKS} == before
    assert {name: digest(ROOT / name) for name in LOCKS} == before
    print("Root environment: commands, resources, fixture, config, rebuild/restore, quality, wheel, locks OK")


if __name__ == "__main__":
    main()
