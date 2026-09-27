# Feedback database

The bot uses PostgreSQL. Call `Db::new` with a `tokio_postgres::Config`, then
`Db::migrate` (or its compatibility alias `Db::create`) before using the API.
Migrations run in one transaction under a fixed advisory lock. Version 1 is
recorded in `arb_schema_migrations`; an unknown version stops startup.

The existing `test_users`, `test_request`, and `test_feedback` tables are
unversioned legacy history. Migration 1 creates them if absent and widens
their `tg_id` columns to `BIGINT` without rewriting their rows. The runtime handlers use the version 1 API. The `insert_user` and `insert_msg`
adapters remain for legacy callers and write only to legacy tables. They do not
invent request or bundle provenance.

## Version 1 records

| Table | Purpose |
| --- | --- |
| `arb_users` | Telegram user profile keyed by positive `tg_id` |
| `arb_requests` | Complete raw query, stable action key, and optional immutable MAL seed and bundle |
| `arb_delivered_positions` | Successful Telegram deliveries at ranks 1–5, with target MAL ID and chat/message coordinates |
| `arb_anime_rating_events` | One immutable score event per resolved request, with its seed MAL ID |
| `arb_anime_ratings` | Current event pointer for each user and MAL ID |
| `arb_recommendation_ratings` | One immutable 0–5 score per delivered position |
| `arb_feedback` | Complete feedback body keyed by user and action |

All internal IDs are `BIGINT`. Foreign keys restrict deletion. A request
resolution stores a positive seed MAL ID and a bundle ID in the form
`sha256:` followed by 64 lowercase hexadecimal characters. A delivery must
belong to the requesting user, follow resolution, and target a different MAL
ID from the seed. Telegram delivery coordinates may be shared by several
positions when the message presents several titles.

## Writing and replaying actions

Use a stable action key derived from the Telegram message or callback
identity for `record_query` and `save_feedback`. An exact repeat returns
`WriteOutcome::AlreadyRecorded`; a changed payload with the same key returns
`DbError::Conflict`. Resolution, delivery, and recommendation ratings follow
the same exact-replay rule. Record a delivery only after Telegram confirms the
send. All values are SQL parameters, including legacy adapter text.

`rate_anime` derives the title from the resolved request; callers provide
only the user, request ID, and a score from 1–10. Every new request may record
a new event, even if its score matches an older event. The current rating
points to the event with the greatest event ID. Replaying an older request
cannot move the pointer or its timestamps. A changed score on the same
request conflicts. Each event links through its request to the exact query,
seed, bundle, and timestamps used when the rating was made.

`DbError` exposes only safe error categories. A lost connection fails the
interrupted operation without replaying it. A later operation reconnects,
and its stable action key makes an explicit retry safe. See [feedback.md](feedback.md)
for feedback-specific retry behavior.

## PostgreSQL tests

Default unit tests do not need a database. Integration tests are ignored by
default and require an explicit `ARB_TEST_DATABASE_URL`. The database name
must match `arb_<three digits>_test` or `arb_ci_test`; the suite never reads
`DATABASE_URL` or `.env`. Every test creates a separate schema. For the
ARB-005 local test database:

```bash
ARB_TEST_DATABASE_URL='postgresql://wozata@127.0.0.1:55432/arb_005_test?sslmode=disable' \
  cargo test --locked --manifest-path bot/Cargo.toml db::tests -- --ignored
```
