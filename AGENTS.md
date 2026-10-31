# AnimeRecommendationBot agent guide

## Where things are

Start with [README.md](README.md). Task specs, statuses and the current project status live in Notion: the [project root](https://app.notion.com/p/3e83908d6b1f81adadb0c2231ef2f6b0) and the [task tracker](https://app.notion.com/p/f12792319d6746b09cb3bc822c737871). Read the task card before editing; the guides for data and releases, research, the bot and operations are linked from the root page. A chat transcript is not a specification: record decisions and results in the task card.

## Product and data

- The bot is a Russian-language, private-chat anime recommendation bot. The runtime is Rust; one process polls each Telegram token.
- Recommendation preparation runs offline in Python from the pinned Neelagiri 2025 base, Nafiul 2026 aliases, and GloVe through the immutable named release lock. The historical MAL 2020 preparation remains explicitly selectable. The bot reads generated `catalog.json`, `neighbors.json`, and `manifest.json` keyed by stable `MAL_ID`; the served bundle comes from `recsys release itemknn`, which derives ItemKNN neighbors from that release and its ratings, with GloVe neighbors as the fallback, and shows each franchise once using the pinned AniList relations snapshot `raw/anilist-relations-2026-v1`.
- PostgreSQL stores requests, delivered positions, anime scores 1–10, recommendation feedback 0–5, and full text. Preserve legacy rows and migration history. Do not add personalization unless a task explicitly authorizes it.
- Prepared artifacts bind source files and raw bytes. Changed producer code needs a fresh preparation; never rewrite a historical hash. Fresh reconstruction of the accepted v1 release uses the pinned `1441573fbbbffc4693c7e1100aaae4458d9e88d4` checkout and Python 3.12.14 (see README).
- Keep generated data, credentials, private evidence, and local tools out of Git. Use small fixtures for routine checks.

## Workflow

1. One task is one Notion card, one branch, and one pull request into `master`. The card holds the spec and acceptance criteria.
2. Branch from current green `master`. Use a separate worktree under `.worktrees/` when tasks run in parallel, with separate databases, Compose projects and output directories. Check the branch and a clean status before editing.
3. Run focused checks for what you changed, then open the pull request. Required CI: `repo-checks`, `rust`, `python`, `compose`, `dev-environment`.
4. Review before merge. A careful self-review is enough for ordinary changes. Changes to the PostgreSQL schema or migrations, the recommendation bundle contract, data inputs or the release lock need an independent review by a second agent or the owner.
5. Squash merge after review and green CI; the owner has authorized this. If `master` is red, fix it before the next merge.
6. Update the card: stage, PR link, merged SHA and a short «Итог» section. Overwrite the status on the Notion root page instead of appending logs, and write in plain language.

## Verification and operation

The standard first-level stand is `python3 deploy/tests/level1-acceptance.py`; it uses FakeTelegram, disposable PostgreSQL and small-fixture A→B→A Compose. This does not prove the full catalog or live Telegram dialogue. For release acceptance, follow the [private release-acceptance procedure in Notion](https://app.notion.com/p/3fb3908d6b1f8165a819fb91f377e4b6); a missing live observation is `PENDING`. Never run two polling processes with one Telegram token. Preserve the selected immutable bundle and PostgreSQL volume through rollout, and do not publish the token, real user IDs, action keys, profiles, or feedback bodies in Git, PRs, Notion, or tool output.
