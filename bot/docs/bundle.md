# Runtime recommendation bundle

The bot loads `manifest.json`, `catalog.json`, and `neighbors.json` from `ARTIFACTS_DIR` before creating the Telegram bot or connecting to PostgreSQL. An invalid file stops startup with a file and field error. `bot --check-config` checks environment configuration only. To inspect artifacts without credentials or a database, run:

```sh
cargo run --locked --bin check_bundle -- ../tests/fixtures/bundle
```

Success prints `sha256:<manifest digest> records=<catalog count>`. The digest identifies the exact manifest bytes, including its artifact and provenance hashes. The loader verifies the raw artifact hashes, validates all v1 fields and cross-file references, and rejects duplicate JSON keys at every depth. The Python producer and checker enforce exact canonical serialization; the Rust loader accepts valid JSON bytes with the required UTF-8 and final LF rules and uses raw hashes without trying to reproduce Python float formatting.

Applications can call `bot::catalog::Bundle::load(path)` and keep the returned bundle in an `Arc`. `identity()` returns the full bundle identity, `catalog().get(mal_id)` and `catalog().iter()` give immutable catalog views in numeric ID order, and `neighbors(mal_id)` returns an ordered slice. A known source with no recommendations returns `Some(&[])`; an unknown ID returns `None`. The first slice entry has rank 1. `Anime::episodes` uses `EpisodeCount::as_str()` to preserve arbitrarily large positive integer tokens.
