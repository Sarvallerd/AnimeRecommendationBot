# ARB-020 full catalog and live acceptance

This is an opt-in release check on the genuine pinned MAL 2020 catalog, GloVe 6B 300d, one private test bot, and one persistent Compose project. It records actual database and container state. The coordinator keeps raw snapshots, context, settings, credentials, and user observations under ignored private `.arb/` storage. Commit only small code and the public aggregate result after removing any identifying information. An unobserved action stays `PENDING`.

## 1. Prepare the full catalog

Use a clean checkout and a stable ignored directory outside `.worktrees`, with an operator-owned writable `data/raw` cache. The parent of each new evidence directory must be private (`0700`). Copy the six pinned source-cache files into `data/raw` before `--offline`; the obtain step verifies the four registered files and regenerates `source-manifest.json`. Set the seven documented absolute `ARB_*` values in `compose.env`, including `ARB_LOG=info`. Provide raw token and database env files separately; preserve root `.env`. Do not run another poller for the same token.

```bash
ROOT=/home/wozata/projects/AnimeRecommendationBot
CHECKOUT="$ROOT/.worktrees/arb-020" # until the task merges; use ROOT afterward
PROJECT=arb020-live-20261003
SETTINGS="$ROOT/.arb/arb020-20261003/compose.env"
DATA="$ROOT/.arb/arb020-20261003/data"
EVIDENCE="$ROOT/.arb/arb020-20261003/evidence"
COMPOSE_FILE="$CHECKOUT/compose.yaml"
wsl.exe --user root --exec env DOCKER_CONFIG="$ROOT/.tools/docker/config" bash \
  "$CHECKOUT/deploy/tests/full-catalog-acceptance.sh" \
  --project "$PROJECT" --settings "$SETTINGS" --data-dir "$DATA" \
  --evidence-dir "$EVIDENCE/preparation-001" --offline
```

The harness requires a fresh evidence directory and clean Git checkout. It builds actual images; verifies pinned inputs, normalization and both block sizes; exports A and B; checks Python/Rust bundle identity, 17,562 records, the 17,561 full neighbor lists and MAL 40089's empty list; then runs real `prepare` against A→B→A and compares all PostgreSQL rows and migration timestamps. It leaves bundle A selected, PostgreSQL healthy, `prepare` exited zero and no polling bot. The private `build-report.json`, report copies, logs and snapshots stay in the evidence directory. On failure, stop and inspect the private files; the harness retains data and the volume.

## 2. Bind one real private-chat actor

Start only one bot with the real token after preparation. Use the same settings/project for every command. When WSL requires root Docker access, prefix each `docker` call with `wsl.exe --user root --exec docker --config "$ROOT/.tools/docker/config"` and pass the exact settings and compose path. For scripts, use the `wsl.exe --user root --exec env DOCKER_CONFIG=...` form above.

```bash
compose() { wsl.exe --user root --exec docker --config "$ROOT/.tools/docker/config" \
  compose --env-file "$SETTINGS" -p "$PROJECT" -f "$COMPOSE_FILE" "$@"; }
EVIDENCE_TOOL="$CHECKOUT/deploy/tests/acceptance-evidence.py"
evidence() { wsl.exe --user root --exec env DOCKER_CONFIG="$ROOT/.tools/docker/config" \
  python3 "$EVIDENCE_TOOL" "$@"; }
mkdir -m 700 "$EVIDENCE/live-001" "$EVIDENCE/live-001/snapshots" \
  "$EVIDENCE/live-001/results" "$EVIDENCE/live-001/results/inputs"
compose up -d bot
evidence snapshot --project "$PROJECT" --settings "$SETTINGS" \
  --output "$EVIDENCE/live-001/snapshots/recommendations-before.json"
```

For a standard local Docker installation, define `compose` with plain `docker compose` and `evidence` with `python3` instead. The examples here use the authorized WSL root Docker prefix. Files created by root in the evidence directory remain readable to that same root invocation; pass their paths to `evidence` without displaying their contents.

The test user privately sends `/recommend`, then exactly `Cowboy Bebop`, explicitly selects MAL ID **1**, and sees five ordered cards. Capture `recommendations-after.json` before any rating. Confirm this interaction with the user, then bind the actor from the one new resolved request:

```bash
evidence bind-actor \
  --before "$EVIDENCE/live-001/snapshots/recommendations-before.json" \
  --after "$EVIDENCE/live-001/snapshots/recommendations-after.json" \
  --build-report "$EVIDENCE/preparation-001/build-report.json" \
  --output "$EVIDENCE/live-001/context.json"
evidence verify --case recommendations \
  --context "$EVIDENCE/live-001/context.json" \
  --before "$EVIDENCE/live-001/snapshots/recommendations-before.json" \
  --after "$EVIDENCE/live-001/snapshots/recommendations-after.json" \
  --bundle "$(wsl.exe --user root --exec cat "$EVIDENCE/preparation-001/bundle-A.path")" \
  --output "$EVIDENCE/live-001/results/recommendations.json"
```

`context.json` contains a real Telegram actor and must stay private. The machine verifier checks ownership, request/seed/bundle, exact ordered MAL IDs, chat/message coordinates, immutable old rows and timestamps. It emits only fixed reason codes and counts. `snapshot` captures the entire versioned and legacy database in one read-only transaction; never print it or attach it to a PR.

For each phase below, capture a full `before` snapshot **after setup actions** and a full `after` snapshot immediately after the listed user action. Use `snapshot --project ... --settings ... --output NEW_FILE` as above, then `verify --case CASE --context ... --before ... --after ... --bundle A_OR_B --output "$EVIDENCE/live-001/results/CASE.json"`. Snapshot output filenames are private and must be new. The results directory must contain one file per explicit case. `verify` saves hashed copies of its inputs in `results/inputs/` and writes a sanitized `PASS` or `FAIL` result; a failure exits nonzero. Never overwrite a failure result to hide it: use a new evidence run after fixing the cause.

## 3. Human actions and machine windows

| Machine case | Human action and snapshot window | Required proof |
| --- | --- | --- |
| `recommendation-scores` | Score anchor cards at ranks 1 and 2 with 0 and 5. | Exactly those two immutable position scores. |
| `recommendation-repeat` | Before another command, press a consumed score control again. | No new request, delivery, score or feedback. |
| `empty` | `/recommend`, exact `Pittanko!! Nekozakana`, select MAL 40089. | One A request, no delivery; user sees the empty-result notice. |
| `anime-score-1` | `/rate`, exact `Cowboy Bebop`, select MAL 1, score 1. | New request/event and latest pointer. |
| `anime-score-10` | Repeat `/rate` on MAL 1, score 10. | Second event, pointer moves to 10; first event remains. |
| `anime-repeat` | Before another command, press a consumed anime-score control. | No database effect. |
| `feedback-unicode` | `/feedback`, send the **exact** `canary --name unicode` output as one Telegram text message. | One complete body, with no truncation or Unicode normalization. |
| `navigation-search` | After a search setup, test `/start`, `/help`, `/cancel` and stale buttons. | No new business rows in the measured window; human checks menu/help/cancel/stale response. |
| `navigation-choice` | After a candidate-keyboard setup, exercise the same global commands and stale choice. | Same no-effect proof and observed UI behavior. |
| `navigation-anime-score` | After an unused anime-score keyboard setup, exercise the commands and stale score. | Same no-effect proof; setup queries are outside the window. |
| `navigation-feedback` | After entering feedback, exercise `/help`, `/cancel`, `/feedback`, and old controls as documented. | No feedback write until a real text/retry action; human checks retained pending text where applicable. |
| `restart` | Leave an unused keyboard, full snapshot, `compose restart bot`, press old button, full snapshot. | Same container, later start time, A identity, history unchanged; human sees stale notice. |
| `db-outage` | Enter `/feedback`, full before; stop **only this project's** PostgreSQL, runtime-only during; send retry canary; restart PG, full after **before retry**. | Bot ID/start/restart count unchanged, PG stopped during, no canary row yet; human sees honest failure/retry. |
| `feedback-retry` | Press the existing retry control for the exact original canary. | One complete retry body, same bot process. |
| `feedback-retry-repeat` | Press consumed retry control again. | No duplicate row. |
| `update` | Stop bot, select B, recreate successful prepare and bot, make genuine new Cowboy Bebop/MAL 1 flow. | New B request and five B deliveries, old A rows unchanged. |
| `rollback` | Repeat the switch with original A and a further genuine flow. | New A request and five A deliveries, prior A/B rows unchanged. |

For the canaries, `evidence canary --name unicode` and `--name retry` print fixed public text. Copy the entire output exactly, including the final `Конец.` and no trailing newline. Each body exceeds 255 characters and contains Cyrillic `ё`, CJK, emoji, an `e` plus combining accent, quotes, backslash, semicolon, percent, underscore, angle brackets, ampersand and twelve repeated phrases. A changed character fails verification. The retry canary is sent once while the database is down; the user must **not** send a fresh text after recovery. Press the existing button. `/help` preserves pending feedback, `/cancel` closes the dialogue and `/feedback` reopens the same pending body.

Capture outage state with `snapshot --runtime-only`; pass that file through `verify --case db-outage --during FILE`. A runtime-only capture cannot stand in for either full endpoint. Stop/restart only this project's database:

```bash
compose stop postgres
# Capture runtime-only during.json; the user sends the retry canary and sees the failure.
compose up -d --wait postgres
# Capture full db-outage-after.json BEFORE pressing retry.
```

For update/rollback, stop the bot, atomically change only `ARB_BUNDLE_DIR` in `SETTINGS`, run `compose up -d --no-deps --force-recreate prepare`, require its exit code zero and current loaded identity, then `compose up -d --no-deps --force-recreate bot`. Inspect current-start identity, UID, read-only mount and the same database volume before the new request. The verifier checks the selected identity again. Do not edit a published bundle in place.

## 4. UI observations and final status

Create a **private** JSON file with `schema_version: 1`, `context_sha256` equal to the SHA-256 of `context.json`, and `scenarios`. Include exactly these IDs once: `navigation`, `selection-cards`, `score-ranges`, `confirmations`, `empty-notice`, `consumed-controls`, `stale-controls`, `restart`, `outage-failure`, `retry-success`, `update`, `rollback`. Each entry has `id`, `status` (`PENDING`, `PASS`, or `FAIL`), `observed_at` (UTC when PASS), and a short non-identifying `observation` (≤240 characters). Record only what the user actually observed. Database rows do not prove a displayed message. Keep names, IDs, profiles, action keys and arbitrary review text out of this file.

```bash
evidence summarize \
  --context "$EVIDENCE/live-001/context.json" \
  --results-dir "$EVIDENCE/live-001/results" \
  --observations "$EVIDENCE/live-001/ui-observations.json" \
  --output "$EVIDENCE/live-001/live-summary.json"
```

The summary recomputes every `PASS` from its hashed private input snapshots and selected bundle. Duplicate or conflicting case keys fail. One missing machine case or human observation yields `PENDING`; a recorded failure yields `FAIL`; all 18 machine cases and 12 human observations must pass for `live_acceptance=PASS`. This is separate from code review, CI and integrated-master verification.

After rollback, leave A selected, one bot running, PostgreSQL healthy and the same named volume retained. Following squash merge and green master CI/Compose checks, rebuild and restart this **same** project from root `master`; require loaded A identity, retained history and one poller. A Telegram send and a PostgreSQL write are separate operations: a crash or ambiguous transport reply can repeat a card, and in-memory controls disappear on restart. Bundle B differs by build-report provenance from A; its catalog and neighbor bytes are identical.
