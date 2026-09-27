# Anime recommendation preparation

This Python package prepares recommendation inputs offline for the Rust bot. It can acquire the pinned MAL 2020 CSV files and GloVe 6B 300-dimensional vectors. Normalization, artifact building, validation, and evaluation will be added in later tasks.

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

Acquire the inputs into `data/raw` relative to the current directory:

```sh
uv run --locked --no-editable recsys obtain
uv run --locked --no-editable recsys obtain --data-dir /path/to/cache --offline
```

The first command downloads about 875 MB and extracts a 1.04 GB text file, so allow at least 2 GB of storage plus temporary download space. Every run verifies all four files by size and SHA256, then writes `source-manifest.json`. Offline mode never downloads; it can extract the text file if the verified ZIP is cached. If a file fails verification, the command reports its path and observed values. Remove that file and rerun to fetch or extract it again. Partial downloads are removed automatically.

The MAL files are pinned to [repository commit `9a1d7f5`](https://github.com/Hernan4444/MyAnimeList-Database/tree/9a1d7f56482accbde99e24cd7cfcad65e2d24b1f). The [author's Kaggle listing](https://www.kaggle.com/datasets/hernan4444/anime-recommendation-database-2020) reports CC0: Public Domain and dataset version 7; the repository commit is a separate version pin. The [GloVe project](https://nlp.stanford.edu/projects/glove/) lists PDDL 1.0 for its pretrained vectors and Apache 2.0 for its code. The ZIP URL redirects to Stanford's downloads host. Exact URLs, sizes, hashes, and provenance are in the packaged source registry.

The MAL source uses the column name `sypnopsis` in `anime_with_synopsis.csv`. Its author excluded Hentai genres from that synopsis file. These source facts matter for later normalization and joins by `MAL_ID`.
