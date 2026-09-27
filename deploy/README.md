# Docker Compose deployment

Requires Docker Engine and Compose 2.30.0 or newer. The bot uses long polling, so run one active project and one bot process per Telegram token.

Copy `bot.env.example` to `deploy/.env.bot` and `postgres.env.example` to `deploy/.env.postgres`. These files are ignored by Git. Set a real token and a PostgreSQL URL with user `anime_bot`, host `postgres`, port `5432`, and database `anime_bot`. Generate a password, for example with `openssl rand -hex 32`, and put the literal value in `POSTGRES_PASSWORD`. Percent encode reserved characters in the URL password (`$` becomes `%24`, `@` becomes `%40`, `/` becomes `%2F`). Raw Compose env files retain quotes and dollar signs literally; do not add shell quotes around values. The migration service uses the syntactically valid placeholder token `0:prepare`; it never contacts Telegram.

Copy `compose.env.example` to a local deployment settings file if you need different paths. Every command below passes the settings file explicitly. Set `ARB_UID` and `ARB_GID` to the host operator's `id -u` and `id -g`, and create `.arb/data` owned by that user. Set `ARB_DATA_DIR` to its host path. Only that directory is writable by the offline builder. The Python image contains the locked recommendation package and no full dataset.

```sh
mkdir -p .arb/data
id -u
id -g
docker compose --env-file deploy/compose.env.example -p animebot --profile tools build builder bot
docker compose --env-file deploy/compose.env.example -p animebot --profile tools run --rm builder obtain --data-dir /data/raw
```

The obtain command downloads pinned MAL CSV files and GloVe into the persistent data directory. Once present, bundle preparation verifies cached sources offline, normalizes, builds, exports, and validates in the same builder image:

```sh
docker compose --env-file deploy/compose.env.example -p animebot --profile tools run --rm --entrypoint /usr/local/bin/prepare-artifacts builder
```

The last line gives the bundle basename. The wrapper opens traversal on that validated bundle directory and read access on its three JSON files so the separate nonroot runtime user can load it. It does not change cache or secret permissions. Set `ARB_BUNDLE_DIR` to the resulting host path, such as `./.arb/data/bundles/sha256-...`, in your deployment settings. The bind mount refuses a missing path. Start the stack:

```sh
docker compose --env-file deploy/compose.env.example -p animebot up -d bot
```

Compose waits for healthy PostgreSQL, then the one-shot `bot --prepare` command loads the full bundle and applies the database migration. Only after it succeeds does the polling bot start. Normal bot startup also applies migrations, so direct restarts remain safe. `bot --check-config` still checks only configuration and the artifact directory; it does not inspect bundle contents or contact PostgreSQL. `bot --prepare` checks token syntax, loads the full bundle, and applies migrations without Telegram access. Both services mount the selected bundle read-only; the runtime uses a nonroot UID and a read-only root filesystem. PostgreSQL uses its own named volume and has no published host port.

For a bundle update, retain the old immutable directory, point `ARB_BUNDLE_DIR` to the new published directory, and recreate `prepare` and `bot`. To roll back, point it to the old directory and recreate those services. Never replace files in a mounted published bundle. `docker compose --env-file deploy/compose.env.example -p animebot down` preserves PostgreSQL data. `down -v` deletes it and is appropriate only for disposable test projects.

The synthetic Docker acceptance runs with `bash deploy/tests/compose-acceptance.sh`. It builds both images, generates a tiny bundle with installed pipeline code, and tests actual Compose startup and database persistence under a unique disposable project. It does not use a real Telegram token or full data.
