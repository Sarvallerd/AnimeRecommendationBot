# AnimeRecommendationBot agent guide

## Reading order and durable context

Start with [README.md](README.md) for task routes, then [Architecture and Development in Notion](https://app.notion.com/p/3f03908d6b1f81a1863bee31d0e66ec9) for active boundaries and the verification map. Read the task specification and status in Notion before editing. For data, release, research, deployment and bundle contracts, follow the canonical guides linked from those pages. A chat transcript is not a durable specification or shared project memory; record accepted decisions, dependencies, reviewed SHA and evidence in the task and repository documentation. Existing artifacts bind source filenames and raw bytes, so a code move or regenerated manifest needs an explicit versioned migration, not a rewritten historical hash. The v0.2 source layout and old v0.1 producer fingerprints are recorded in [the machine-readable migration record](docs/migrations/arb054-recsys-layout-v2.json). Fresh reconstruction of the accepted v1 release uses the separate pinned `1441573fbbbffc4693c7e1100aaae4458d9e88d4` checkout and Python 3.12.14 before current-tree Compose preparation.

## Product and data

- The bot is a Russian-language, private-chat anime recommendation bot. The runtime is Rust; one process polls each Telegram token.
- New recommendation preparation runs offline in Python from the pinned Neelagiri 2025 base, Nafiul 2026 aliases, and GloVe through the immutable named release lock. The historical MAL 2020 preparation remains explicitly selectable. Runtime recommendations come from generated JSON `catalog.json`, `neighbors.json`, and `manifest.json` keyed by stable `MAL_ID`.
- PostgreSQL stores requests, delivered positions, anime scores 1–10, recommendation feedback 0–5, and full text. Preserve legacy rows and migration history. Do not add personalization unless a task explicitly authorizes it.
- Keep generated data, credentials, private acceptance evidence, and local tools out of commits. Use small fixtures for routine checks.

## One task, one reviewable change

1. The coordinator keeps the specification, dependencies, acceptance criteria, and task status in Notion. Each task has one branch, one worktree, and one pull request. Branch and integration operations are sequential. Create the worktree from current green `master` only after its dependencies merge.
2. Before any edit, verify the absolute repository root, branch, baseline SHA, clean status, exact scope, and file ownership. Edit only assigned files in that worktree. Shared contracts, migrations, manifests, and lockfiles have one named owner; resolve overlaps before work.
3. Run the task sequence with one active role at a time: specification → architect's exact design → implementer → independent reviewer → implementer fixes or completeness check → fresh review of the exact commit SHA. Architect and reviewer read without editing. Implementers do not edit the reviewer verdict. Child agents do not delegate.
4. The coordinator's handoff gives each role the task specification, architect decision, absolute worktree, baseline and final SHA, diff, test evidence, and relevant neighboring changes. The coordinator records role reports and reviewed SHA in Notion. A completed stage releases its thread; at most three child agents are open beside the coordinator.
5. Independent ready tasks may run in parallel without waiting for their whole wave. Give tasks disjoint ownership, separate databases, Compose projects, and output directories. Never use one Telegram token with two polling processes.
6. Implementers commit only when the coordinator explicitly authorizes it. The coordinator owns pushes, pull requests, CI, merges, Notion, and final acceptance. Before merge, update the branch against `master`; any changed SHA requires checks and another independent review. A blocked review, red CI, or PR head different from the reviewed SHA stops integration.
7. The user has already authorized automatic squash merge after a passing exact-SHA review and required green CI. After each merge, verify integrated `master` and its required checks. If `master` is red, stop later merges and repair it first. Report commands, results, blockers, and concrete risks.

## Agent configuration

Project configuration is in `.codex/config.toml` and separate `.codex/agents/*.toml` roles: architect `gpt-6-astra`/`ultra`, implementer `gpt-6-sol`/`high`, reviewer `gpt-5.6-sol`/`high`. Nested agent creation is disabled for roles. `python3 scripts/check_agent_setup.py` checks static files and ignore behavior; a compatible Codex client and an actual role invocation are needed to verify effective model loading. Do not silently lower a model or reasoning level.

## Verification and operation

Run focused checks for edited behavior and the required CI jobs. The standard first-level stand is `python3 deploy/tests/level1-acceptance.py`; it uses FakeTelegram, disposable PostgreSQL and small-fixture A→B→A Compose. This does not prove the full catalog or live Telegram dialogue. For opt-in release acceptance, use the [private release-acceptance procedure in Notion](https://app.notion.com/p/3fb3908d6b1f8165a819fb91f377e4b6): full pinned inputs, actual A/B/A bundle preparation, a single test poller, database snapshots, observed UI steps, update and rollback. A missing live observation is `PENDING`. Preserve the selected immutable bundle and PostgreSQL volume through rollout; do not publish token, real user IDs, action keys, profiles, or feedback bodies in Git, PRs, Notion, or tool output.
