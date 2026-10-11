# Pinned title aliases for the 2025 research catalog

`recsys aliases` produces a separate title overlay from Nafiul's `anime_info.csv` and the accepted Neelagiri research dataset. It never changes the source dataset, legacy GloVe inputs, or the running bot. The Nafiul source is Kaggle version 1, with an exact 22,314,208-byte SHA-256 pin. Its declared license is CC BY-NC-SA 4.0; this intake is limited to noncommercial research. The exact collection date is unknown, so the source must not be used to claim a temporal holdout.

The accepted base dataset identity is `sha256:78bb29bfe18a00519b5dcd06e9682e1c4c84528166f0c51a80d24ea28c7a62b6`. Keep the base and metadata cache in separate private directories. The full base validator checks the 69-million-pair archive before either operation; allow several minutes per command. The metadata source alone is small and does not include a replacement ratings cohort.

```sh
uv run --locked --no-editable recsys obtain --snapshot nafiul-2026-v1 \
  --data-dir /private/nafiul-cache --offline
uv run --locked --no-editable recsys aliases normalize \
  --snapshot nafiul-2026-v1 \
  --data-dir /private/nafiul-cache \
  --base-dir /private/accepted-neelagiri-dataset \
  --base-identity sha256:78bb29bfe18a00519b5dcd06e9682e1c4c84528166f0c51a80d24ea28c7a62b6 \
  --output-dir /private/aliases-a
uv run --locked --no-editable recsys aliases validate /private/aliases-a \
  --data-dir /private/nafiul-cache \
  --base-dir /private/accepted-neelagiri-dataset \
  --base-identity sha256:78bb29bfe18a00519b5dcd06e9682e1c4c84528166f0c51a80d24ea28c7a62b6
```

For a new cache, omit `--offline` on the first `obtain` call. The command verifies every cached source byte and writes its source manifest. Normalization reads only verified bytes and publishes a new directory atomically. Reusing an output path fails without replacing it. `validate` repeats the complete base check and derives the aliases again from the pinned metadata bytes. Its optional `--expected-identity sha256:…` binds the result to a previously accepted alias release.

The three files are canonical UTF-8 JSON with sorted keys and a trailing newline: `aliases.json`, `alias-qa.json`, and `alias-manifest.json`. A MAL_ID maps to one or two typed title strings. A valid English title comes first, including when it equals the base title or Japanese alias. A source main title follows only if it differs from the base title, all base aliases, and that English title. Duplicate source MAL_ID groups are removed in full. Only a canonical MAL_ID, an exact MyAnimeList URL, and a base catalog match may emit aliases. The output keeps spelling, punctuation, case, and internal whitespace; it trims edges, applies NFC, and quarantines a field containing replacement or invisible control characters. Synonyms are counted but never imported. No Nafiul ratings, synopsis, Japanese title, or source status enters the overlay.

The future catalog composition step must retain ordered alias texts, then append existing base aliases with exact-value deduplication. It must keep the English alias even if equal to the base primary title. The current Rust display path checks only the first alias for Latin eligibility; a valid alias in another script remains searchable but may not be the displayed recommendation title. The research overlay and full QA are documented in [the alias report](reports/alias-enrichment-v1/README.md).
