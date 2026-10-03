# Building and checking the Rust bot

The repository pins Rust 1.95.0 and commits `Cargo.lock`. On Linux, install a C compiler, `pkg-config`, and the OpenSSL development package before building.

Run these commands from the repository root:

```bash
cargo build --locked
cargo test --locked
cargo run --locked
```

The bot reads environment variables directly. It does not load `.env` automatically. Use `.env.example` as a reference; preserve any existing `.env` and load it in your shell or Compose environment. These variables are required:

- `TELOXIDE_TOKEN`: Telegram BotFather API token.
- `DATABASE_URL`: PostgreSQL URI with a user, database, and host (for example `postgresql://anime_bot:password@localhost:5432/anime_bot?sslmode=disable`). This replaces the old lowercase `db_user`, `db_password`, `db_port`, and `db_name` variables.
- `ARTIFACTS_DIR`: existing directory for the recommendation bundle. A relative path resolves from the process working directory, which is the repository root in the commands above.

`RUST_LOG` is optional and defaults to `info`. It accepts `off`, `error`, `warn`, `info`, `debug`, or `trace`, optionally followed by comma-separated module overrides such as `info,teloxide=warn,bot::db=debug`. Regex filters are unsupported.

To validate configuration without contacting PostgreSQL or Telegram, run:

```bash
cargo run --locked -- --check-config
```

The current PostgreSQL connection uses `NoTls`; `sslmode=require` is rejected. Database schema migration 1 is available through `Db::create`; the current runtime still uses its legacy handlers and tables. Recommendation bundle loading arrives in ARB-011. See [database.md](database.md) for the schema and PostgreSQL tests.
