# Offline release compatibility and historical live acceptance

The current automated local workflow is [level-1 acceptance](LEVEL1_ACCEPTANCE.md): `python3 deploy/tests/level1-acceptance.py`. It exercises FakeTelegram, a disposable PostgreSQL, and synthetic A→B→A Compose without manual Telegram actions. This is separate from the full pinned catalog and the opt-in live service procedure below; unobserved live UI remains `PENDING`.

For the named release's actor-bound DATA-06 v2 procedure on the existing service, use
[the live release runbook](LIVE_RELEASE_ACCEPTANCE.md). It preserves the original
bundle/image/settings and PostgreSQL volume through original → target → original →
target, recomputes private operation and case evidence, and keeps unobserved UI
steps `PENDING`. The historical schema-1 commands below remain for ARB-020 only.

## Named Neelagiri 2025 + Nafiul 2026 release (ARB-033)

The opt-in `neelagiri-2025-nafiul-2026-v1` mode uses the committed release lock (SHA-256 `46f3ffd60239675b802026025016ee6719cb0525325e088fbfa248012db3de9b`). It checks the actual 28,955-item catalog, 28,872 five-neighbor lists, 83 empty lists, MAL 1 `Cowboy Bebop` → `37991, 24439, 26443, 8425, 23133`, and empty MAL 29740 `Notteke Explet's`. The builder reports must account for the 83 non-ready vectors. New A matches the accepted lock's reference identities; new B differs only by requested/effective block size 128. The legacy profile remains separate, with 17,562 records and the historical MAL 1/40089 seeds.

The full frozen-input offline run and its separate read-only evidence verification **passed** on 2026-10-12 at source commit `adfe084acb8845b16609ebd62b34076c9919154a` (private preparation report SHA-256 `8fa4d7e97e4ff992e10e1b7b37f3169b69c0ba35f6f55a9e9313c680ebecdb82`). Fresh A matched the lock; B had a distinct bundle identity with byte-identical catalog and neighbors. All five old→A→B→A→old prepare phases exited zero and preserved complete synthetic database rows and migration timestamps. The original old selection, settings, and healthy PostgreSQL volume were retained, with no bot poller. This is an offline compatibility result; live Telegram acceptance and production user-history preservation remain `PENDING` for DATA-06 v2.

Use a clean checkout and a new ignored, operator-owned private directory outside every worktree. The acceptance Compose project needs its **own** credentials, volume, and placeholder bot token. Keep the bot stopped. Before starting the runner, copy the saved old three-file immutable bundle into `DATA/bundles/sha256-...`, set `ARB_BUNDLE_DIR` in the original seven-field settings file to that old path, start only this project's PostgreSQL, apply migration 1 with an old prepare, and seed nonempty synthetic history. The initial PostgreSQL container must be healthy. Do not use a production token, production volume, or production user rows. Keep the preserved old bundle immutable.

Copy the complete pinned inputs into the exact frozen layout under `DATA`: `datasets/sha256-<base identity>` with its five generated files, `aliases/sha256-<alias identity>` with three files, `releases/sha256-<release identity>` with `catalog.json` and `release-manifest.json`, `raw/nafiul-2026-v1` with its CSV and source manifest, and `raw/glove-6b-300d-v1/glove.6B.300d.txt`. Use the identities in the lock and retain the raw upstream files unchanged. The runner validates the full release and installed builder before fresh A/B builds; it does not normalize these inputs. Keep private input, settings, credential, snapshot, probe, and report files out of Git and issue/PR text.

```bash
CHECKOUT=/absolute/clean/checkout
PROJECT=arb033-accept-unique
SETTINGS=/absolute/private/compose.env
DATA=/absolute/private/data
EVIDENCE=/absolute/private/evidence/new-run
OLD_IDENTITY=sha256:<exact-preserved-old-bundle-identity>
"$CHECKOUT/deploy/tests/full-catalog-acceptance.sh" \
  --release neelagiri-2025-nafiul-2026-v1 \
  --project "$PROJECT" --settings "$SETTINGS" --data-dir "$DATA" \
  --evidence-dir "$EVIDENCE" --old-identity "$OLD_IDENTITY" --offline
python3 "$CHECKOUT/deploy/tests/release-acceptance.py" verify \
  --evidence-dir "$EVIDENCE"
```

The controller must run as the settings/data owner. Where WSL Docker requires privileged API access, prepend the private argv-preserving Docker wrapper to `PATH` for this command only; do not run the whole Python controller as root. The runner builds the pinned images, probes the installed Python/NumPy/package/source bytes, validates the frozen release, builds/exports A and B afresh, runs both Python and Rust bundle checks, and executes old→A→B→A→old using only `prepare`. It saves a full baseline and five full snapshots, compares every database row and timestamp, verifies the retained PostgreSQL container and volume, and checks the original settings bytes, owner and mode. It never starts a poller or rewrites the original settings. On ordinary failure after switching, it attempts an old prepare if the project is quiescent; `failure-status.json` records the fixed failure/restoration status. A power loss or `SIGKILL` needs manual inspection of the private saved original settings and the old bundle before recovery.

The read-only `verify` command reopens the saved source, profile, lock, installed-probe, settings copies, bundle files, reports, and snapshots and recomputes the phase/database checks. It does not use Docker, rebuild, export, or require today's checkout HEAD to equal the historical run SHA. A `PASS` from this mode is scoped to **full-input offline compatibility with synthetic acceptance-project history**. `live_acceptance` and `production_history_preservation` remain `PENDING`; the new actor-bound UI verifier belongs to DATA-06 v2. Do not feed this schema-2 preparation report to the schema-1 `bind-actor` command below. If a new full catalog ever has no empty list, retain the synthetic empty-case test and mark the live empty observation conditional in DATA-06 rather than changing the catalog.

## Historical MAL 2020 live procedure (ARB-020)

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

The harness requires a fresh evidence directory and clean Git checkout. It pins the starting commit and checks for a changed HEAD or dirty worktree after each build phase and immediately before PASS. If source changes during a run, discard that preparation evidence and start again in a new directory. It builds actual images; verifies pinned inputs, normalization and both block sizes; exports A and B; checks Python/Rust bundle identity, 17,562 records, the 17,561 full neighbor lists and MAL 40089's empty list; then runs real `prepare` against A→B→A and compares all PostgreSQL rows and migration timestamps. It leaves bundle A selected, PostgreSQL healthy, `prepare` exited zero and no polling bot. The private `build-report.json`, report copies, logs and snapshots stay in the evidence directory. On failure, stop and inspect the private files; the harness retains data and the volume.

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

For update/rollback, take the full `before` snapshot while the previous A or B bot is still running. Then stop the bot, atomically change only `ARB_BUNDLE_DIR` in `SETTINGS`, run `compose up -d --no-deps --force-recreate prepare`, require its exit code zero and current loaded identity, then `compose up -d --no-deps --force-recreate bot`. Inspect current-start identity, UID, read-only mount and the same database volume before the new request. Take the full `after` snapshot only after the new bot starts and the new request is completed. The verifier requires a new bot container, a fresh successful `prepare` for the selected bundle, and the same PostgreSQL volume. Do not edit a published bundle in place.

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
