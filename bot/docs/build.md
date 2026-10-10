# Building and checking the Rust bot

The repository pins Rust 1.95.0 and commits `Cargo.lock`. On Linux, install a C compiler, `pkg-config`, and the OpenSSL development package before building.

Run these commands from the repository root:

```bash
cargo build --locked
cargo test --locked
cargo run --locked --bin bot
```

The bot reads environment variables directly. It does not load `.env` automatically. Use `.env.example` as a reference; preserve any existing `.env` and load it in your shell or Compose environment. These variables are required:

- `TELOXIDE_TOKEN`: Telegram BotFather API token.
- `DATABASE_URL`: PostgreSQL URI with a user, database, and host (for example `postgresql://anime_bot:password@localhost:5432/anime_bot?sslmode=disable`). This replaces the old lowercase `db_user`, `db_password`, `db_port`, and `db_name` variables.
- `ARTIFACTS_DIR`: existing directory for the recommendation bundle. A relative path resolves from the process working directory, which is the repository root in the commands above.

`RUST_LOG` is optional and defaults to `info`. It accepts `off`, `error`, `warn`, `info`, `debug`, or `trace`, optionally followed by comma-separated module overrides such as `info,teloxide=warn,bot::db=debug`. Regex filters are unsupported.

`COVERS_ENABLED` is optional and accepts exactly `true` or `false` (default `true`). Normal startup constructs a shared Jikan cover client when enabled. Configuration checking and `--prepare` do not contact or construct the cover service. A cover lookup failure does not stop text recommendations.

To validate configuration without contacting PostgreSQL or Telegram, run:

```bash
cargo run --locked --bin bot -- --check-config
```

The current PostgreSQL connection uses `NoTls`; `sslmode=require` is rejected. Database migration 1 is applied at startup under a transaction lock and preserves old `test_*` rows. The active handlers use the versioned API, search the loaded bundle, and store user actions with stable IDs. Normal startup validates the entire bundle, connects and migrates PostgreSQL, then checks Telegram's webhook and starts polling. `--prepare` performs the same bundle/database gate and exits before creating a Telegram client:

```bash
cargo run --locked --bin bot -- --prepare
```

`--check-config` checks only environment values and the artifact directory; it does not parse the bundle or contact PostgreSQL. See [bundle.md](bundle.md), [dialogue.md](dialogue.md), and [database.md](database.md) for the active contracts. In-memory dialogue and buttons are lost on restart; committed history remains in PostgreSQL.
