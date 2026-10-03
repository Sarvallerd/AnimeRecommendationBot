FROM ghcr.io/astral-sh/uv:0.11.12@sha256:3a59a3cdd5f7c217faa36e32dbc7fddbb0412889c2a0a5229f6d790e5a019dd7 AS uv
FROM rust:1.95.0-slim-bookworm@sha256:d7482085ff5b415f84dba5647ae71606650bdef00db7aeb69f4b3d170c3e4082 AS rust
FROM python:3.12.14-slim-bookworm@sha256:392307d22300de8b5986851a12d9176dfc0fc073e65bf6523ebd7dcbeb23564e AS build
COPY --from=uv /uv /usr/local/bin/uv
COPY --from=rust /usr/local/cargo /usr/local/cargo
COPY --from=rust /usr/local/rustup /usr/local/rustup
RUN apt-get update && apt-get install -y --no-install-recommends build-essential pkg-config libssl-dev && rm -rf /var/lib/apt/lists/*
ENV CARGO_HOME=/usr/local/cargo RUSTUP_HOME=/usr/local/rustup UV_PYTHON_DOWNLOADS=0 UV_PROJECT_ENVIRONMENT=/opt/venv PATH=/usr/local/cargo/bin:/opt/venv/bin:$PATH
WORKDIR /src
COPY pyproject.toml uv.lock Cargo.toml Cargo.lock rust-toolchain.toml ./
COPY packaging/scripts/recsys packaging/scripts/recsys
COPY recsys/README.md recsys/README.md
COPY recsys/src/ recsys/src/
COPY bot/src/ bot/src/
COPY bot/migrations/ bot/migrations/
COPY bot/tests/ bot/tests/
RUN MATURIN_PEP517_ARGS="--profile release" uv sync --locked --no-editable --no-dev && chmod -R a+rX /opt/venv

FROM python:3.12.14-slim-bookworm@sha256:392307d22300de8b5986851a12d9176dfc0fc073e65bf6523ebd7dcbeb23564e
COPY --from=uv /uv /usr/local/bin/uv
COPY --from=build /opt/venv /opt/venv
ENV UV_PYTHON_DOWNLOADS=0 UV_PROJECT_ENVIRONMENT=/opt/venv PATH=/opt/venv/bin:$PATH
COPY deploy/prepare-artifacts.sh /usr/local/bin/prepare-artifacts
RUN chmod 755 /usr/local/bin/prepare-artifacts
WORKDIR /data
ENTRYPOINT ["recsys"]
CMD ["--help"]
