# AnimeRecommendationBot agent guide

## Product and data

- The bot is a Russian-language, private-chat anime recommendation bot. The runtime is Rust; one process polls each Telegram token.
- Recommendation preparation runs offline in Python using the pinned MAL 2020 dataset and GloVe. Runtime recommendations come from generated JSON `catalog.json`, `neighbors.json`, and `manifest.json` keyed by stable `MAL_ID`.
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

Run focused checks for edited behavior and the required CI jobs. The small bundle fixture and synthetic Compose acceptance do not prove the full catalog or live Telegram dialogue. For release acceptance, use the private, opt-in procedure in `deploy/ACCEPTANCE.md`: full pinned inputs, actual A/B/A bundle preparation, a single test poller, database snapshots, observed UI steps, update and rollback. A missing live observation is `PENDING`. Preserve the selected immutable bundle and PostgreSQL volume through rollout; do not publish token, real user IDs, action keys, profiles, or feedback bodies in Git, PRs, Notion, or tool output.
