# Immutable bundle export

After `recsys normalize` and `recsys build`, publish the verified catalog and neighbors:

Run from the repository root with the root locked environment:

```sh
uv run --locked recsys export \
  --catalog data/normalized/catalog.json \
  --neighbors data/build/neighbors.json \
  --build-report data/build/build-report.json \
  --normalization-report data/normalized/normalization-report.json \
  --output-dir data/bundles
uv run --locked recsys validate data/bundles/sha256-<manifest-digest>
```

Export prints the published path, full `sha256:` identity, and record count. Validate prints exactly `sha256:<digest> records=<count>`. Both commands work from any current directory with the installed package. For a checkout-only contract check, run `python3 contracts/check_bundle.py BUNDLE_DIR` from the repository root.

The exporter reads each input once, checks canonical JSON and the three-file bundle contract, and verifies the build report against the catalog, neighbors, normalization report, packaged registry, and the supported `arb009-v1` builder. It checks the pinned source descriptors, GloVe size and hash recorded by the builder, exact algorithm profile, installed builder source hashes and versions, token and neighbor diagnostics, and block settings. It does not reopen GloVe or the MAL CSV files or recompute embeddings. Input hashes provide integrity checks, not authentication.

`data/bundles/sha256-<manifest-digest>/` contains exactly `catalog.json`, `neighbors.json`, and `manifest.json`. The catalog and neighbor bytes are preserved. The canonical manifest includes the build report hash and exporter source provenance, so an input or exporter code change produces a new identity. Reports stay outside the runtime bundle. The same inputs, exporter source, and supported environment produce the same three bytes.

Publication requires POSIX file locks and an operator-owned store. Cooperating exporters hold a per-store `.publication.lock`, write and sync a private staging directory, validate it, then rename it into its final identity directory and sync the store. An existing valid directory with identical bytes is reused unchanged. An incomplete, altered, or unsafe target is rejected and left for the operator to inspect. A killed process may leave a hidden staging directory; it is never an active bundle. This command does not update a runtime pointer. Supported producer checks bind this exporter to the current builder; future builder versions need an explicit profile update.
