# Anime recommendation preparation

This Python package will prepare recommendation artifacts offline for the Rust bot. It currently provides only the command-line entry point and a version command. Dataset loading, normalization, artifact building, validation, and evaluation will be added in later tasks.

Requires Python 3.12. From this directory, install the locked environment with:

```sh
uv sync --locked --no-editable
uv run --locked --no-editable recsys --help
uv run --locked --no-editable recsys --version
```

Run the package checks with:

```sh
uv run --locked --no-editable python -m unittest discover -s tests -v
```
