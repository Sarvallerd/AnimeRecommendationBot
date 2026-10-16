# Composite research catalog release

New offline preparations use the immutable `neelagiri-2025-nafiul-2026-v1` lock. `current` is a local alias for that release; it does not fetch or select a newer release. Place the verified frozen base, aliases, Nafiul cache, and GloVe text under one data directory, then run:

```sh
recsys release info
recsys release prepare --data-dir DATA
recsys release prepare --release mal-2020-glove-v1 --data-dir DATA
```

The [source layout migration](../docs/migrations/arb053-python-src-v1.md) moved package files to `src/recsys/` byte for byte, preserving package-relative producer names and the existing release lock. A new build still records its actual source revision.

The current selector requires `DATA/datasets/sha256-BASE`, `DATA/aliases/sha256-ALIASES`, `DATA/raw/nafiul-2026-v1`, and `DATA/raw/glove-6b-300d-v1/glove.6B.300d.txt`. It validates the release and either reuses a checked immutable bundle or builds against the exact accepted Python 3.12.14, NumPy 2.5.3, package and five producer files. A different producer requires a new reference or the original fixed producer. The historical selector uses its own `DATA/raw/mal-2020-glove-v1` cache and legacy pipeline. Both print a verified path and `ARB_BUNDLE_DIR` suggestion; neither changes bot settings or PostgreSQL. A selected live bundle stays selected until a separate rollout.

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

ARB-032 `current` preparation also passed with the full frozen inputs in the installed builder image from reviewed commit `3292c78851dc2ecfb3e34af834c7616ceeb1dec5`. Its fresh run reproduced the accepted ARB-031 A256 build report (`sha256:1aa472f9764b82033d9a31a7d472814c5c0df06cd3833837e68894a3057d741e`) and published bundle `sha256:77d171f9ee8c30f606957afb7037276a66a543582491344661638d486d94f728` with 28,955 records. A second prepare returned the same three JSON files byte for byte; installed Python and Rust validators accepted them, and all 13 original input files retained their hashes. The private aggregate evidence has SHA-256 `a56719ebf553016440f69848105c18e0b4f7ba6c7887afb02ceeffea4bee1743`. [Synthetic current and historical Compose acceptance](https://github.com/Sarvallerd/AnimeRecommendationBot/actions/runs/38158265951) passed on that commit. Live Telegram dialogue and production update/rollback acceptance remain `PENDING`.
