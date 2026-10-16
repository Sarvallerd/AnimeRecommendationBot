# One local development environment

Run commands from the repository root. The root `pyproject.toml`, `uv.lock`, `Cargo.toml`, and `Cargo.lock` define the Python package, its two Rust commands, and the Rust crate. Python 3.12, Rust 1.95.0, a C compiler, `pkg-config`, and OpenSSL development headers are required. Install pinned uv 0.11.12. Each worktree gets its own `.venv`; do not point concurrent worktrees at one shared environment.

```sh
uv sync --locked
uv run --locked recsys --help
uv run --locked recsys validate tests/fixtures/bundle
uv run --locked check_bundle tests/fixtures/bundle
```

The `recsys` command is installed as a Python wheel script. Its active sources are in [`src/recsys/`](../src/README.md); tests, guides and reports remain in `recsys/`. The `bot` and `check_bundle` commands are compiled Rust binaries in the same `.venv/bin`. The package is editable by default, so Python source edits are visible immediately. uv checks watched Rust and package inputs before `uv run` or `uv sync` and rebuilds when they change. Running `.venv/bin/bot` or another installed command directly skips that check.

For the optional quality dependencies, include the extra on both sync and later runs:

```sh
uv sync --locked --extra quality
uv run --locked --extra quality recsys quality --help
uv run --locked --extra quality python -m unittest discover -s recsys/tests -p 'test_*.py' -v
```

To install an ordinary wheel instead of editable sources, use `uv sync --locked --no-editable --extra quality`. Both modes install all three commands and packaged Python resources. The [layout migration](../docs/migrations/arb053-python-src-v1.md) records the source move and historical recovery. The environment checker compares every installed `.py`, `.json` and `.md` package file to the source tree in both modes and tests wheel rebuild and restore after a temporary Python edit.

The bot reads its configuration from environment variables. It never loads `.env` on its own. Keep local secrets in an existing `.env` and opt in explicitly when using uv:

```sh
uv run --locked --env-file .env bot --check-config
uv run --locked --env-file .env bot
```

`--check-config` verifies token and database URL syntax and that the artifact directory exists; it does not inspect bundle contents or contact Telegram or PostgreSQL. Use `check_bundle` to validate a bundle. `bot --prepare` connects to PostgreSQL and is outside the local smoke check.

Cargo commands also work from the repository root:

```sh
cargo build --locked --all-targets
cargo test --locked
cargo run --locked --bin check_bundle -- tests/fixtures/bundle
```

The root `uv.lock` preserves the pinned Python and quality packages. The root `Cargo.lock` pins Rust dependencies. Maturin 1.15.0 is pinned in the build requirements and runs in uv's isolated build environment. Rust binaries use the local machine's development profile; use the Docker builder for its release build.

The uv cache keys watch ordinary source, migration SQL, manifests, packaging script, and relevant build environment changes. uv 0.11.12 uses the latest ctime on Unix and mtime elsewhere. If an older watched file is deleted, or an external Rust toolchain or OpenSSL installation changes, force a reinstall:

```sh
uv sync --locked --reinstall-package anime-recommendation-recsys
uv sync --locked --extra quality --reinstall-package anime-recommendation-recsys
```

Run `python3 dev/tests/check_environment.py` to verify a fresh copy without reading local secrets or data. It builds in a temporary directory and uses only the small synthetic bundle fixture. By default it uses a disposable uv cache. For an offline local run, set `ARB_DEV_UV_CACHE_DIR` to the absolute path of an existing cache with the pinned dependency and build artifacts, and set `UV_OFFLINE=1` explicitly. uv may read and write that cache; the source copy, virtual environment and Cargo target stay disposable, and every check still runs. Such a run does not verify a cold network download or alter the existing scientific environment and lock.
