# Anime recommendation preparation

This Python package prepares recommendation inputs offline for the Rust bot. It can acquire the pinned MAL 2020 CSV files and GloVe 6B 300-dimensional vectors. Normalization of the MAL CSV files, pinned GloVe cosine neighbor building, and a diagnostic genre baseline are included. Immutable bundle export and the fixed 20-query quality comparison are complete. The [ARB-018 report](reports/arb018-v1/README.md) documents methods, ratings, timings, and limits.

Requires Python 3.12, Rust 1.95.0, a C compiler, pkg-config, and OpenSSL development headers. From the repository root, install the locked environment with:

```sh
uv sync --locked --no-editable
uv run --locked --no-editable recsys --help
uv run --locked --no-editable recsys --version
```

Run the package checks with:

```sh
uv run --locked --no-editable --extra quality python -m unittest discover -s recsys/tests -p 'test_*.py' -v
```

Acquire the unchanged MAL 2020 and GloVe default inputs into `data/raw` relative to the current directory:

```sh
uv run --locked --no-editable recsys obtain
uv run --locked --no-editable recsys obtain --data-dir /path/to/cache --offline
```

The first command downloads about 875 MB and extracts a 1.04 GB text file, so allow at least 2 GB of storage plus temporary download space. Every run verifies all four files by size and SHA256, then writes `source-manifest.json`. Offline mode never downloads; it can extract the text file if the verified ZIP is cached. If a file fails verification, the command reports its path and observed values. Remove that file and rerun to fetch or extract it again. Partial downloads are removed automatically.

For the separately pinned 2025 research snapshot, run:

```sh
uv run --locked --no-editable recsys obtain --snapshot neelagiri-2025-v1
uv run --locked --no-editable recsys obtain --snapshot neelagiri-2025-v1 --offline
uv run --locked --no-editable recsys obtain --snapshot neelagiri-2025-v1 --data-dir /path/to/private/cache --offline
```

Without `--data-dir`, this snapshot uses `data/raw/neelagiri-2025-v1`; an explicit path is used exactly. Keep snapshots in separate cache directories: an existing manifest from another snapshot is preserved and rejected, so select another `--data-dir` instead of mixing their files. Its [dataset audit](reports/dataset-audit-v1/README.md) pins Kaggle version 1: `details.csv` (20,178,926 bytes) and `ratings.csv` (4,504,586,658 bytes), both verified by full size and SHA-256 before an atomic `source-manifest.json` is written. Allow more than 4.52 GB plus temporary space for a new download. Each run rehashes the cached files; `--offline` succeeds only when both are present and verified. A damaged cached file is left for inspection and must be removed explicitly before retrying. The source declares CC BY-NC-SA 4.0; this intake is for noncommercial research, with exact rating collection date unknown. Commercial use and underlying MAL-derived rights remain to be resolved. This command only obtains data: it does not normalize the new CSVs, replace the default runtime bundle, or change the pinned GloVe input. That migration follows later DATA-03/04 tasks.

The MAL files are pinned to [repository commit `9a1d7f5`](https://github.com/Hernan4444/MyAnimeList-Database/tree/9a1d7f56482accbde99e24cd7cfcad65e2d24b1f). The [author's Kaggle listing](https://www.kaggle.com/datasets/hernan4444/anime-recommendation-database-2020) reports CC0: Public Domain and dataset version 7; the repository commit is a separate version pin. The [GloVe project](https://nlp.stanford.edu/projects/glove/) lists PDDL 1.0 for its pretrained vectors and Apache 2.0 for its code. The ZIP URL redirects to Stanford's downloads host. Exact URLs, sizes, hashes, and provenance are in the packaged source registry.

The MAL source uses the column name `sypnopsis` in `anime_with_synopsis.csv`. Its author excluded Hentai genres from that synopsis file. These source facts matter for later normalization and joins by `MAL_ID`.

Normalize the verified CSV files with:

```sh
uv run --locked --no-editable recsys normalize \
  --anime-csv data/raw/anime.csv \
  --synopsis-csv data/raw/anime_with_synopsis.csv \
  --output-dir data/normalized
```

This command verifies both files against the packaged size and SHA-256 pins. It writes canonical UTF-8 `catalog.json`, `durations.json`, and `normalization-report.json`. The catalog uses the v1 contract in `contracts/`; durations contain seconds and a `per_episode` or `unspecified` scope for every retained ID. The report lists source provenance, output hashes, counts, row exclusions, field exceptions, unmatched IDs, and duplicate title groups. It contains no timestamp or machine-specific path. No GloVe data is read during normalization.

Run the pinned 20-query genre baseline after normalization:

```sh
uv run --locked --no-editable recsys quality baseline \
  --catalog data/normalized/catalog.json \
  --output-dir data/quality
```

It writes a deterministic genre baseline and an unjudged relevance template. See [the quality protocol](QUALITY.md) for the query set and rubric, and the [completed comparison](reports/arb018-v1/README.md) for assessor coverage, unknown grades, and measured scope. Install the `quality` extra for comparison tools; the base locked environment is enough to obtain, normalize, build, export, and validate.

Build pinned GloVe cosine neighbors after normalization with `recsys build`; see [the build guide](BUILD.md) for the command, validation, algorithm, report, and output contract.

Export a validated immutable runtime bundle with `recsys export`; see [the export guide](EXPORT.md) for the command, verification, and publication contract.
