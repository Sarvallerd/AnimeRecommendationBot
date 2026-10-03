# Docker Compose deployment

Docker Engine and Compose 2.30.0 or newer are required. One project may poll a Telegram token at a time. The Rust image uses UID `10001:10001`, a read-only root filesystem and a read-only immutable bundle mount. PostgreSQL has a named volume and no published host port. The offline builder runs with the operator UID/GID configured in the settings file.

Use an ignored deployment directory outside a task worktree, for example `.arb/my-bot/`, with `compose.env`, `secrets/bot.env`, `secrets/postgres.env`, `data/` and `evidence/`. Restrict secrets and evidence directories to mode `0700` and their files to `0600`; the published bundle directory/files need `0755`/`0644` for the runtime UID. Keep `.env` intact. Copy the examples in this directory and fill these literal raw values:

```text
bot.env:       TELOXIDE_TOKEN=<BotFather API token>
               DATABASE_URL=postgresql://anime_bot:<URL-encoded-password>@postgres:5432/anime_bot?sslmode=disable
postgres.env:  POSTGRES_PASSWORD=<literal password>
compose.env:   ARB_DATA_DIR=/absolute/path/to/data
               ARB_BUNDLE_DIR=/absolute/path/to/data/bundles/sha256-...
               ARB_BOT_ENV=/absolute/path/to/secrets/bot.env
               ARB_POSTGRES_ENV=/absolute/path/to/secrets/postgres.env
               ARB_UID=<operator id -u>
               ARB_GID=<operator id -g>
               ARB_LOG=info
```

Raw Compose env files preserve dollar signs and quotes literally. Do not add shell quoting around their values. Percent encode reserved URL characters in the database password (`$` → `%24`, `@` → `%40`, `/` → `%2F`). The bot accepts `NoTls`; use `sslmode=disable` for this Compose network. The builder never receives the token or database credentials. Its build stage uses the root uv/maturin package and a Rust toolchain to compile the installed commands. The final Python builder image contains those commands, without a Rust compiler or the full dataset; the polling bot image is Rust-only. For normal operation, use the [root setup guide](../README.md); `prepare-artifacts` verifies cached sources, normalizes, builds, exports, validates, and publishes one bundle. `bot --prepare` checks the selected full bundle and applies migration 1 without contacting Telegram. `bot --check-config` checks only configuration and the artifact directory. Normal bot startup repeats the bundle/database checks before polling.

Pass the settings file on **every** Compose command. From the checkout root, set `PROJECT`, `SETTINGS`, and `COMPOSE_FILE` to the chosen stable paths and use:

```bash
compose() { docker compose --env-file "$SETTINGS" -p "$PROJECT" -f "$COMPOSE_FILE" "$@"; }
compose --profile tools build builder bot
compose --profile tools run --rm builder obtain --data-dir /data/raw
compose --profile tools run --rm --no-deps --entrypoint /usr/local/bin/prepare-artifacts builder
# Copy the validated printed basename into ARB_BUNDLE_DIR in SETTINGS.
compose up -d bot
compose ps
```

The `prepare` one-shot service waits for healthy PostgreSQL, loads the selected bundle with placeholder token `0:prepare`, and applies migrations. The bot starts only after `prepare` exits zero. PostgreSQL keeps all request, delivery, rating, feedback, and legacy history in its named volume. `compose down` keeps this volume; **do not use `down -v`** on a real project. A normal bot restart is `compose restart bot`; this clears in-memory dialogue/buttons and preserves committed rows.

## Immutable bundle update and rollback

Build a new bundle alongside the old one. Stop the bot first. Change only `ARB_BUNDLE_DIR` in the settings file using an atomic same-directory replacement that preserves its owner and mode. Recreate `prepare`, require exit code zero and the expected loaded `sha256:` identity, then recreate the bot and confirm the same identity in its current startup logs. Keep the previous bundle directory for rollback.

```bash
compose stop bot
# Atomically select the new ARB_BUNDLE_DIR in SETTINGS.
compose up -d --no-deps --force-recreate prepare
compose ps -a prepare
compose logs --since 5m prepare
compose up -d --no-deps --force-recreate bot
compose ps bot
compose logs --since 5m bot
```

Rollback repeats the same sequence with the old immutable path. Historical requests retain their original bundle identities. The full-data acceptance harness in [`tests/full-catalog-acceptance.sh`](tests/full-catalog-acceptance.sh) performs and checks an A→B→A prepare cycle, leaving PostgreSQL running and the bot stopped; the [live runbook](ACCEPTANCE.md) continues with genuine Telegram actions.

If Docker is installed in WSL and only root can access its socket, use the authorized per-command form without changing host socket or group membership:

```bash
wsl.exe --user root --exec docker \
  --config /home/wozata/projects/AnimeRecommendationBot/.tools/docker/config \
  compose --env-file "$SETTINGS" -p "$PROJECT" -f "$COMPOSE_FILE" ps
```

For a script that calls `docker` internally, use `wsl.exe --user root --exec env DOCKER_CONFIG=/home/wozata/projects/AnimeRecommendationBot/.tools/docker/config bash ...`. Keep `SETTINGS` and `COMPOSE_FILE` as absolute paths in that invocation.

The small synthetic `bash deploy/tests/compose-acceptance.sh` checks image build, migration/persistence, startup guards, and scale refusal without a real token or full dataset. It does not establish live acceptance.
