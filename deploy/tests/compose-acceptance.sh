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
output="${ARB_LEVEL1_OUTPUT:-$temp}"
mkdir -p "$output"
export ARB_DATA_DIR="$temp/data" ARB_BOT_ENV="$temp/bot.env" ARB_POSTGRES_ENV="$temp/postgres.env"
export ARB_UID="$(id -u)" ARB_GID="$(id -g)"
export ARB_BUNDLE_DIR="$temp/not-selected"
mkdir -p "$ARB_DATA_DIR" "$ARB_BUNDLE_DIR"
cat > "$ARB_BOT_ENV" <<'ENV'
TELOXIDE_TOKEN=0:ci
DATABASE_URL=postgresql://anime_bot:a%24%40%2Fb@postgres:5432/anime_bot?sslmode=disable
ENV
cat > "$ARB_POSTGRES_ENV" <<'ENV'
POSTGRES_PASSWORD=a$@/b
ENV
chmod 600 "$ARB_BOT_ENV" "$ARB_POSTGRES_ENV"
compose() {
    # The scoped WSL Docker wrapper does not inherit this shell's exported values.
    # Pass only explicit disposable paths through Compose's own env file.
    cat > "$temp/compose.env.next" <<ENV
ARB_DATA_DIR=$ARB_DATA_DIR
ARB_BUNDLE_DIR=$ARB_BUNDLE_DIR
ARB_BOT_ENV=$ARB_BOT_ENV
ARB_POSTGRES_ENV=$ARB_POSTGRES_ENV
ARB_UID=$ARB_UID
ARB_GID=$ARB_GID
ARB_LOG=info
ENV
    chmod 600 "$temp/compose.env.next"
    mv "$temp/compose.env.next" "$temp/compose.env"
    docker compose --env-file "$temp/compose.env" -p "$project" -f compose.yaml -f deploy/tests/compose.ci.yaml "$@"
}
cleanup() {
    local original=$?
    trap - EXIT INT TERM
    set +e
    compose --profile tools down --volumes --remove-orphans --rmi local --timeout 10 >/dev/null 2>&1
    local down=$?
    local remaining
    remaining=$(compose ps -a -q 2>/dev/null)
    docker image rm "${project}-bot:local" "${project}-builder:local" >/dev/null 2>&1 || true
    local volume=0
    docker volume inspect "${project}_postgres_data" >/dev/null 2>&1 && volume=1
    local network=0 images=0
    docker network inspect "${project}_level1" >/dev/null 2>&1 && network=1
    docker image inspect "${project}-bot:local" >/dev/null 2>&1 && images=1
    docker image inspect "${project}-builder:local" >/dev/null 2>&1 && images=1
    if [[ $down -ne 0 || -n "$remaining" || $volume -ne 0 || $network -ne 0 || $images -ne 0 ]]; then
        echo 'Owned synthetic Compose cleanup failed' >&2
        exit 1
    fi
    rm -rf "$temp"
    if [[ $original -eq 0 ]]; then
        echo LEVEL1_COMPOSE_CLEANUP_PASS
    fi
    exit "$original"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
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
echo LEVEL1_COMPOSE_BUILD_PASS
fixture_output=$(compose --profile tools run --rm --no-deps --entrypoint python builder /fixture/build-fixture.py)
legacy_bundle=$(printf '%s\n' "$fixture_output" | sed -n 's/^LEGACY_BUNDLE=//p')
bundle=$(printf '%s\n' "$fixture_output" | sed -n 's/^CURRENT_BUNDLE=//p')
[[ "$legacy_bundle" == /data/bundles/sha256-* ]]
[[ "$bundle" == /data/bundles/sha256-* ]]
legacy_dir="$ARB_DATA_DIR/bundles/${legacy_bundle##*/}"
current_dir="$ARB_DATA_DIR/bundles/${bundle##*/}"
export ARB_BUNDLE_DIR="$legacy_dir"
[[ -d "$legacy_dir" && -d "$current_dir" && "$legacy_dir" != "$current_dir" ]]
compose --profile tools run --rm --no-deps builder validate "$bundle" >/dev/null
compose --profile tools run --rm --no-deps builder validate "$legacy_bundle" >/dev/null
compose --profile tools run --rm --no-deps --entrypoint /opt/venv/bin/check_bundle builder "$legacy_bundle" >/dev/null
compose --profile tools run --rm --no-deps --entrypoint /opt/venv/bin/check_bundle builder "$bundle" >/dev/null
compose run --rm --no-deps --entrypoint /usr/local/bin/check_bundle prepare /artifacts >/dev/null
compose up --wait -d postgres >/dev/null
# The old unversioned tables must survive the first migration.
compose exec -T postgres psql -v ON_ERROR_STOP=1 -U anime_bot anime_bot < deploy/tests/legacy.sql >/dev/null
compose up -d bot >/dev/null
wait_exit_zero prepare
wait_exit_zero bot
assert_selection() {
    local expected="$1" bot_id prepare_id source command
    expected="sha256:${expected##*sha256-}"
    [[ "$(docker network inspect "${project}_level1" --format '{{.Internal}}')" == true ]]
    bot_id=$(compose ps -a -q bot)
    prepare_id=$(compose ps -a -q prepare)
    [[ -n "$bot_id" && -n "$prepare_id" ]]
    source=$(docker inspect --format '{{range .Mounts}}{{if eq .Destination "/artifacts"}}{{.Source}}{{end}}{{end}}' "$bot_id")
    [[ "$source" == "$ARB_BUNDLE_DIR" ]]
    [[ "$(docker inspect --format '{{range .Mounts}}{{if eq .Destination "/artifacts"}}{{.RW}}{{end}}{{end}}' "$bot_id")" == false ]]
    [[ "$(docker inspect --format '{{.Config.User}}' "$bot_id")" == 10001:10001 ]]
    command=$(docker inspect --format '{{json .Config.Cmd}}' "$bot_id")
    [[ "$command" == '["--prepare"]' ]]
    compose logs --no-color prepare | grep -F "Loaded recommendation bundle $expected (" >/dev/null
    compose logs --no-color bot | grep -F "Loaded recommendation bundle $expected (" >/dev/null
}
assert_selection "$legacy_bundle"
[[ "$(compose exec -T postgres psql -At -U anime_bot anime_bot -c 'SELECT string_agg(version::text, chr(44) ORDER BY version) FROM arb_schema_migrations')" == 1 ]]
compose exec -T postgres psql -v ON_ERROR_STOP=1 -U anime_bot anime_bot < deploy/tests/history.sql >/dev/null
compose exec -T postgres psql -At -U anime_bot anime_bot < deploy/tests/snapshot.sql > "$output/before.json"
echo LEVEL1_COMPOSE_LEGACY_A_PASS
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
assert_selection "$legacy_bundle"
compose exec -T postgres psql -At -U anime_bot anime_bot < deploy/tests/snapshot.sql > "$output/after-restart.json"
diff -u "$output/before.json" "$output/after-restart.json"
[[ "$(compose exec -T postgres psql -At -U anime_bot anime_bot -c 'SELECT count(*) FROM arb_schema_migrations')" == 1 ]]
postgres_id=$(compose ps -a -q postgres)
postgres_volume=$(docker inspect --format '{{range .Mounts}}{{if eq .Destination "/var/lib/postgresql/data"}}{{.Name}}{{end}}{{end}}' "$postgres_id")
[[ -n "$postgres_id" && -n "$postgres_volume" ]]
echo LEVEL1_COMPOSE_PERSISTENCE_PASS
switch_bundle() {
    local target="$1" expected="$2" label="$3"
    compose stop bot >/dev/null
    export ARB_BUNDLE_DIR="$target"
    compose up -d --no-deps --no-build --force-recreate prepare >/dev/null
    wait_exit_zero prepare
    compose up -d --no-deps --no-build --force-recreate bot >/dev/null
    wait_exit_zero bot
    assert_selection "$expected"
    [[ "$(compose ps -a -q postgres)" == "$postgres_id" ]]
    [[ "$(docker inspect --format '{{range .Mounts}}{{if eq .Destination "/var/lib/postgresql/data"}}{{.Name}}{{end}}{{end}}' "$postgres_id")" == "$postgres_volume" ]]
    compose exec -T postgres psql -At -U anime_bot anime_bot < deploy/tests/snapshot.sql > "$output/$label.json"
    diff -u "$output/before.json" "$output/$label.json"
}
switch_bundle "$current_dir" "$bundle" current-B
echo LEVEL1_COMPOSE_CURRENT_B_PASS
switch_bundle "$legacy_dir" "$legacy_bundle" rollback-A
echo LEVEL1_COMPOSE_ROLLBACK_A_PASS
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
assert_selection "$legacy_bundle"
compose exec -T postgres psql -At -U anime_bot anime_bot < deploy/tests/snapshot.sql > "$output/recovered.json"
diff -u "$output/before.json" "$output/recovered.json"
if compose up -d --scale bot=2 bot >/dev/null 2>&1; then
    echo 'Scaling the polling bot unexpectedly succeeded' >&2
    exit 1
fi
echo LEVEL1_COMPOSE_GUARDS_PASS
