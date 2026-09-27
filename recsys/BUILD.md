# Pinned GloVe neighbor build

Run the offline builder after normalization:

```sh
uv run --locked --no-editable recsys build \
  --catalog data/normalized/catalog.json \
  --normalization-report data/normalized/normalization-report.json \
  --glove data/raw/glove.6B.300d.txt \
  --output-dir data/build
```

The builder reads canonical v1 catalog and normalization report bytes, checks their hashes and MAL source pins against the packaged registry, then streams the full pinned GloVe 6B 300d text file once. It hashes every byte and checks the pinned size and SHA-256 before writing. Only words used by the catalog are stored. For those words, it requires exactly 300 finite numbers and rejects duplicate entries. Other GloVe rows are checked for UTF-8 and a nonempty word/vector structure; their numbers are not parsed.

For each anime, the builder tokenizes the title, aliases in their existing order, genres in their existing order, and synopsis when present. The frozen expression is `[a-z]+(?:'[a-z]+)?` after Python lowercase conversion. Every occurrence of a known word counts once, including repeats across fields. Unknown words do not enter the mean. It calculates a float64 arithmetic mean of known 300D vectors and L2 normalizes it. A record with no matched words, no tokens, or an exact zero mean has a zero vector and an empty recommendation list. Score, year, type, and episodes do not affect recommendations.

Similarity is the float64 dot product of normalized vectors. The builder keeps at most five distinct positive candidates per source, excludes the source itself and zero vectors, and orders equal serialized scores by numeric `MAL_ID`. Tiny cosine overshoots up to `1 + 1e-12` are clamped to 1. A score block contains at most 256 rows and 64 MiB of scores. `--block-size` accepts 1 through 256 and changes only the requested block row count recorded in the report; neighbor bytes are invariant across block sizes in the locked environment.

The output directory receives canonical UTF-8 `neighbors.json` and `build-report.json`, with the report installed last. The report records the reusable v1 manifest `algorithm`, `sources`, and `files` objects, input hashes, code and package versions, block settings, token coverage, zero-vector IDs, per-anime counts, and a neighbor-count histogram. It contains no timestamp, machine path, timing, or memory measurement. A future bundle exporter must verify the report's catalog and neighbor hashes before using those objects in `manifest.json`; this task does not produce a complete runtime bundle.

Input hashes and provenance checks detect accidental substitution or corruption. They do not authenticate the source. Repeated builds with identical inputs, configuration, and locked Python/NumPy environment produce identical output bytes. Cross-platform floating-point byte identity is not promised.
