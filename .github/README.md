# Pull request workflow

Each task uses one branch, one worktree, and one pull request. The coordinator records the task specification, obtains the architect's exact design, assigns implementation, obtains independent review, and tracks any implementation fixes. Final review must cover the exact commit that will be merged; a new commit or rebase requires another review.

The pull request template captures the task, design, check evidence, reviewer verdict, and full reviewed head SHA. Its fields and checkboxes are prompts for contributors and the coordinator; GitHub does not enforce them. The coordinator verifies that required CI passed and that the pull request head still matches the reviewed SHA before performing the authorized squash merge.

The `CI` workflow runs on pull requests, pushes to `master`, and manual dispatch. It checks agent setup, Rust and PostgreSQL tests, Python tests, and agreement between the two bundle contract checkers. The Compose and development environment workflows provide two further required jobs.

## Master merge policy

[merge-policy.json](merge-policy.json) is the exact repository ruleset declaration for `master`. GitHub enforces a pull request, squash merge, current base before merge, resolved review threads, and five named GitHub Actions checks tied to app ID `15368`: `agent-setup`, `rust`, `python`, `compose`, and `dev-environment`. It blocks deletion, force push, and merge commits, with no active bypass actor. Because the repository has one owner who cannot approve their own pull request, required GitHub approval count is zero. GitHub rules do **not** verify independent agent review, its verdict or reviewed commit SHA. The coordinator checks those facts and the five exact `success` conclusions before the already authorized squash merge. GitHub itself can treat `neutral` or `skipped` required checks as passing; the coordinator's gate is stricter.

The standard-library [read-only checker](../scripts/github_merge_policy.py) has three commands. From the repository root, with a working `gh` login that can read repository administration, run:

```bash
python3 scripts/github_merge_policy.py validate
python3 scripts/github_merge_policy.py plan --expected-master-sha <full-current-master-SHA>
python3 scripts/github_merge_policy.py check --expected-master-sha <full-current-master-SHA>
```

`validate` reads only the tracked declaration. `plan` reads current GitHub settings, effective branch rules, latest check runs and `master` twice. It returns `create` only when there is no classic protection, ruleset or effective rule; `noop` means the single active ruleset and all effective rules match. `check` requires the applied policy. Exit `0` means a valid declaration or matching plan/readback, `1` means a confirmed mismatch, and `2` means incomplete or invalid evidence. Output contains no credentials. A `create` plan includes the exact payload, but changes no settings. A clean plan does not assert that GitHub already enforces the policy.

The coordinator applies the policy only after an independently reviewed exact PR head and all five green jobs. Save the before snapshot and plan in private evidence, confirm `master` and the reviewed head are unchanged, then create once with the exact tracked JSON:

```bash
gh api --hostname github.com --method POST -H 'Accept: application/vnd.github+json' -H 'X-GitHub-Api-Version: 2026-03-10' repos/Sarvallerd/AnimeRecommendationBot/rulesets --input .github/merge-policy.json
```

Store the response and created ruleset ID privately. Run `check` against the same master SHA before merging the task PR, then perform the normal squash merge with the reviewed head. After the merge, require all five integrated `master` jobs and rerun `check` with the new master SHA. If the POST outcome is uncertain, read settings first; never repeat an uncertain create blindly. On `403`, `422`, missing evidence or mismatch, stop integration and resolve the discrepancy without weakening the policy or bypassing a red check. A settings write and Git merge are separate operations. Emergency rollback needs a separate review of the task-created, unchanged ruleset ID and may disable that ruleset while preserving its body and history; it is never a way to bypass review or red CI.

For a positive enforcement witness, inspect the GitHub [rule suites](https://docs.github.com/en/rest/repos/rule-suites) for `refs/heads/master` and the merge's before/after SHAs. Record an active passing evaluation of this ruleset with no bypass in private evidence. A missing suite is `PENDING`; the successful readback and normal protected merge still stand, but do not claim every rejection path was experimentally exercised. Do not probe by deleting or force pushing `master`.

Ignored Cargo tests are reserved for guarded PostgreSQL integration tests. CI runs all ignored tests with an explicit test database URL; live Telegram and real catalog checks use separate manual harnesses.
