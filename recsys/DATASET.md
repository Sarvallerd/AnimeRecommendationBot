# Neelagiri 2025 research dataset

This workflow produces a separate, immutable research dataset from the pinned Kaggle version 1 `details.csv` and `ratings.csv`. It does not change the current MAL 2020 command, GloVe builder, bot runtime, or database. The source declares CC BY-NC-SA 4.0; its use here is limited to noncommercial research. The precise date when ratings were collected and individual event times are unknown.

## Commands

Install the locked package, obtain the two pinned CSVs, and normalize into a path that does not yet exist:

```sh
uv sync --locked --no-editable
uv run --locked --no-editable recsys obtain --snapshot neelagiri-2025-v1 --data-dir /private/cache --offline
uv run --locked --no-editable recsys dataset normalize \
  --snapshot neelagiri-2025-v1 --data-dir /private/cache \
  --output-dir /private/research/2025-a --scratch-dir /private/research/scratch \
  --memory-limit-mib 2048 --threads 2 --spill-limit-mib 32768
uv run --locked --no-editable recsys dataset validate /private/research/2025-a
```

The cache needs a matching `source-manifest.json`. Each normalization independently copies and hashes both source files, including the 4.5 GB ratings CSV. Scratch and staging directories are private (0700); copied CSVs and artifacts are private (0600). The output path is published atomically only after parsing, pinned aggregate checks, and streaming validation succeed. Preserve output directories as immutable snapshots. A second run uses a new output directory and its own scratch directory. DuckDB's 2 GiB setting is a query memory limit, not a process RSS cap. Allow space for verified copies, DuckDB tables and spill files, and two outputs when testing reproducibility.

The ratings CSV must begin with exactly `username,anime_id,status,score,is_rewatching,num_watched_episodes` in that order, followed by LF or CRLF. The parser rejects renamed, reordered, or duplicate header fields before DuckDB reads any rating rows.

## Contracts

The output directory contains exactly `catalog.json`, `item-facts.json`, `interactions.tsv.gz`, `dataset-qa.json`, and `dataset-manifest.json`. JSON is canonical UTF-8 with one final LF. The catalog preserves the existing v1 fields and stable MAL_ID keys. `item-facts.json` adds first release date, airing status, and seasonal year per catalog item; recent cohorts use the validated first release date and aired/airing status.

`interactions.tsv.gz` has a single gzip member with empty filename, zero modification time, and compression level 6. Its uncompressed header is:

```text
dataset_user_key\tmal_id\trating\twatch_status
```

Each following LF-terminated line has four tab-separated ASCII fields. Rows are sorted by pseudonymous key and then numeric MAL_ID, with each pair appearing once. Explicit integer scores 1–10 are ratings; zero or missing score is unrated. Groups with conflicting explicit scores, any invalid score, or explicit and unrated states together are excluded. Repeated matching scores do not add weight. If statuses disagree while the score agrees, status is `unknown`. A pseudonymous key is a SHA-256 digest over an exact source username and a fixed dataset namespace; it is not anonymous and must be treated as private data. No interaction timestamp exists.

A username is blank only when null or made entirely of these 29 Unicode whitespace codepoints: U+0009–U+000D, U+001C–U+001F, U+0020, U+0085, U+00A0, U+1680, U+2000–U+200A, U+2028, U+2029, U+202F, U+205F, and U+3000. Accepted nonblank usernames retain their exact bytes for grouping and key derivation; they are never stripped, case-folded, or Unicode-normalized. Score whitespace uses the same set. After trimming it, empty values and zero forms such as `+000.000` are unrated; explicit values must be unsigned 1–10 with optional leading plus, leading zeroes, and `.0` digits, such as `+0010.000`. Scientific notation (`1e1`, `1e-999`), `.0`, `1.`, fractions, `NaN`, `Inf`, and any minus sign (`-0`, `-0.0`) are invalid. Classification uses exact strings, not floating-point conversion.

`dataset-qa.json` records aggregate counts, loss reasons, histograms, and release cohorts. The packaged expectations are frozen from the [source audit](reports/dataset-audit-v1/README.md); a mismatch stops publication. The manifest pins both source hashes, normalizer policy and code fingerprints, producer versions, and output hashes. Its SHA-256 bytes are the dataset identity. The validator checks those pins, checksums, catalog/fact contracts, sorted unique interaction rows, and independently recomputes output-derived QA. A caller may pin `expected_identity` when integrating a trusted frozen dataset.

The research dataset contains no English-title or synonym enrichment. A separately pinned alias source and an integration task are required before replacing the runtime catalog. The [normalization report](reports/dataset-normalization-v1/README.md) records full-data results and reproducibility checks.
