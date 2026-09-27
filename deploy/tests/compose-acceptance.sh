#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
command -v docker >/dev/null
command -v python3 >/dev/null
docker info >/dev/null
compose_version=$(docker compose version --short)
python3 - "$compose_version" <<'PY'
import re, sys
match = re.match(r'^v?(\d+)\.(\d+)\.(\d+)', sys.argv[1])
if not match or tuple(map(int, match.groups())) < (2, 30, 0):
    raise SystemExit('Docker Compose 2.30.0 or newer is required')
PY
project="arb019-ci-$(date +%s)-$RANDOM"
temp=$(mktemp -d)
chmod 755 "$temp"
export ARB_DATA_DIR="$temp/data" ARB_BOT_ENV="$temp/bot.env" ARB_POSTGRES_ENV="$temp/postgres.env"
mkdir -p "$ARB_DATA_DIR"
cat > "$ARB_BOT_ENV" <<'ENV'
TELOXIDE_TOKEN=0:ci
DATABASE_URL=postgresql://anime_bot:a%24%40%2Fb@postgres:5432/anime_bot?sslmode=disable
ENV
cat > "$ARB_POSTGRES_ENV" <<'ENV'
POSTGRES_PASSWORD=a$@/b
ENV
chmod 600 "$ARB_BOT_ENV" "$ARB_POSTGRES_ENV"
compose() { docker compose --env-file deploy/compose.env.example -p "$project" -f compose.yaml -f deploy/tests/compose.ci.yaml "$@"; }
cleanup() {
    compose down --volumes --remove-orphans >/dev/null 2>&1 || true
    rm -rf "$temp"
}
trap cleanup EXIT
wait_exit_zero() {
    local service="$1" id state code
    for _ in {1..120}; do
        id=$(compose ps -a -q "$service")
        if [[ -n "$id" ]]; then
            read -r state code < <(docker inspect --format '{{.State.Status}} {{.State.ExitCode}}' "$id")
            if [[ "$state" == exited ]]; then
                if [[ "$code" == 0 ]]; then return 0; fi
                echo "$service exited $code" >&2
                compose logs --tail=30 "$service" >&2
                return 1
            fi
        fi
        sleep 2
    done
    echo "Timed out waiting for $service" >&2
    return 1
}
assert_failure_gate() {
    local label="$1" bot_id prepare_id state code
    compose rm -sf bot prepare >/dev/null
    if compose up -d bot >/dev/null 2>&1; then
        echo "$label unexpectedly started" >&2
        return 1
    fi
    prepare_id=$(compose ps -a -q prepare)
    [[ -n "$prepare_id" ]]
    read -r state code < <(docker inspect --format '{{.State.Status}} {{.State.ExitCode}}' "$prepare_id")
    [[ "$state" == exited && "$code" != 0 ]]
    bot_id=$(compose ps -a -q bot)
    if [[ -n "$bot_id" ]]; then
        state=$(docker inspect --format '{{.State.Status}}' "$bot_id")
        [[ "$state" == created ]]
    fi
    echo "$label blocked bot startup"
}
echo 'Building runtime and offline builder images'
compose --profile tools build bot builder
bundle=$(compose --profile tools run --rm --no-deps --entrypoint python builder /fixture/build-fixture.py | tail -n 1)
[[ "$bundle" == /data/bundles/sha256-* ]]
export ARB_BUNDLE_DIR="$ARB_DATA_DIR/bundles/${bundle##*/}"
[[ -d "$ARB_BUNDLE_DIR" ]]
compose --profile tools run --rm --no-deps builder validate "$bundle" >/dev/null
compose run --rm --no-deps --entrypoint /usr/local/bin/check_bundle prepare /artifacts >/dev/null
compose up --wait -d postgres >/dev/null
# The old unversioned tables must survive the first migration.
compose exec -T postgres psql -v ON_ERROR_STOP=1 -U anime_bot anime_bot < deploy/tests/legacy.sql >/dev/null
compose up -d bot >/dev/null
wait_exit_zero prepare
wait_exit_zero bot
[[ "$(compose exec -T postgres psql -At -U anime_bot anime_bot -c 'SELECT string_agg(version::text, chr(44) ORDER BY version) FROM arb_schema_migrations')" == 1 ]]
compose exec -T postgres psql -v ON_ERROR_STOP=1 -U anime_bot anime_bot < deploy/tests/history.sql >/dev/null
compose exec -T postgres psql -At -U anime_bot anime_bot < deploy/tests/snapshot.sql > "$temp/before.json"
# Inspect the actual mount and process credentials, then attempt a write as the runtime user.
bot_id=$(compose ps -a -q bot)
[[ "$(docker inspect --format '{{.Config.User}}' "$bot_id")" == 10001:10001 ]]
[[ "$(docker inspect --format '{{range .Mounts}}{{if eq .Destination "/artifacts"}}{{.RW}}{{end}}{{end}}' "$bot_id")" == false ]]
if compose run --rm --no-deps --entrypoint /bin/sh prepare -c 'echo bad > /artifacts/forbidden' >/dev/null 2>&1; then
    echo 'Bundle mount was writable' >&2
    exit 1
fi
compose down >/dev/null
compose up -d bot >/dev/null
wait_exit_zero prepare
wait_exit_zero bot
compose exec -T postgres psql -At -U anime_bot anime_bot < deploy/tests/snapshot.sql > "$temp/after.json"
diff -u "$temp/before.json" "$temp/after.json"
[[ "$(compose exec -T postgres psql -At -U anime_bot anime_bot -c 'SELECT count(*) FROM arb_schema_migrations')" == 1 ]]
# Fail closed on a corrupt bundle before a database or Telegram connection is attempted.
cp -a "$ARB_BUNDLE_DIR" "$temp/corrupt"
chmod u+w "$temp/corrupt/manifest.json"
printf 'corrupt' >> "$temp/corrupt/manifest.json"
valid_bundle="$ARB_BUNDLE_DIR"
export ARB_BUNDLE_DIR="$temp/corrupt"
assert_failure_gate corrupt-bundle
export ARB_BUNDLE_DIR="$valid_bundle"
compose exec -T postgres psql -v ON_ERROR_STOP=1 -U anime_bot anime_bot -c 'INSERT INTO arb_schema_migrations(version) VALUES (99)' >/dev/null
assert_failure_gate unknown-migration
compose exec -T postgres psql -v ON_ERROR_STOP=1 -U anime_bot anime_bot -c 'DELETE FROM arb_schema_migrations WHERE version=99' >/dev/null
compose rm -sf bot prepare >/dev/null
compose up -d bot >/dev/null
wait_exit_zero prepare
wait_exit_zero bot
compose exec -T postgres psql -At -U anime_bot anime_bot < deploy/tests/snapshot.sql > "$temp/recovered.json"
diff -u "$temp/before.json" "$temp/recovered.json"
if compose up -d --scale bot=2 bot >/dev/null 2>&1; then
    echo 'Scaling the polling bot unexpectedly succeeded' >&2
    exit 1
fi
echo 'Compose acceptance passed: offline build, migrations, persistence, guards, and scale refusal'
