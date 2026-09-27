FROM ghcr.io/astral-sh/uv:0.11.12@sha256:3a59a3cdd5f7c217faa36e32dbc7fddbb0412889c2a0a5229f6d790e5a019dd7 AS uv
FROM python:3.12.14-slim-bookworm@sha256:392307d22300de8b5986851a12d9176dfc0fc073e65bf6523ebd7dcbeb23564e
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_PYTHON_DOWNLOADS=0 UV_PROJECT_ENVIRONMENT=/opt/venv PATH=/opt/venv/bin:$PATH
WORKDIR /src/recsys
COPY recsys/pyproject.toml recsys/uv.lock recsys/README.md ./
COPY recsys/src/ ./src/
RUN uv sync --locked --no-editable --no-dev && chmod -R a+rX /opt/venv
COPY deploy/prepare-artifacts.sh /usr/local/bin/prepare-artifacts
RUN chmod 755 /usr/local/bin/prepare-artifacts
WORKDIR /data
ENTRYPOINT ["recsys"]
CMD ["--help"]
