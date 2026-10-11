# Composite research catalog release

ARB-031 composes the accepted Neelagiri dataset with the accepted Nafiul title-alias artifact. A release binds both immutable upstream identities and contains only `catalog.json` and `release-manifest.json`. The overlay prepends stored alias texts in their existing order, removes exact duplicates, then retains the base aliases. Every other catalog field and every MAL_ID stays unchanged. This work does not import ratings or profiles into the bot database.

Use a verified base, aliases, the pinned Nafiul cache, and explicit trusted identities:

```sh
recsys release compose --base-dir BASE --base-identity sha256:BASE_DIGEST --aliases-dir ALIASES --alias-identity sha256:ALIAS_DIGEST --alias-data-dir NAFIUL_CACHE --output-dir NEW_RELEASE
recsys release validate NEW_RELEASE --release-identity sha256:RELEASE_DIGEST --base-dir BASE --aliases-dir ALIASES --alias-data-dir NAFIUL_CACHE
```

Each operation performs the full alias and base validation. The manifest records both upstream manifests and consumed file descriptors, the overlay policy, exact counts, and producer fingerprints. Publication uses a new directory and rejects a repeated or unsafe target. Old producer fingerprints remain valid only when a trusted identity and independently derived bytes agree.

The GloVe model is separate from the catalog release. Its `glove-6b-300d-v1` selector pins only the legacy ZIP and text descriptors. Build and export with the installed package:

```sh
recsys release build --release-dir NEW_RELEASE --release-identity sha256:RELEASE_DIGEST --base-dir BASE --aliases-dir ALIASES --alias-data-dir NAFIUL_CACHE --glove GLOVE_TEXT --output-dir NEW_BUILD --block-size 256
recsys release export --release-dir NEW_RELEASE --release-identity sha256:RELEASE_DIGEST --base-dir BASE --aliases-dir ALIASES --alias-data-dir NAFIUL_CACHE --neighbors NEW_BUILD/neighbors.json --build-report NEW_BUILD/build-report.json --output-dir BUNDLE_STORE
recsys validate BUNDLE_STORE/sha256-BUNDLE_DIGEST
```

The v2 build report pins the release, GloVe registry and file, unchanged `arb009-v1` vector algorithm, producer versions, output hashes, execution block size, and exact diagnostics. Export verifies that report and publishes the existing three-JSON v1 runtime contract. Block size can change report and bundle identities while preserving catalog and neighbor bytes.

Small synthetic producer fixtures test the contract. The pinned full-input offline reference passed with the builder image using Python 3.12.14 and NumPy 2.5.3, a fixed reviewed source commit, A/A composition, 256/128 builds, Python and Rust bundle checks, and measured resource use. The [release report](reports/dataset-release-v1/README.md) records the aggregate evidence. Live Telegram dialogue, update, and rollback acceptance remain later tasks.
