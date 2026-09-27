# Pull request workflow

Each task uses one branch, one worktree, and one pull request. The coordinator records the task specification, obtains the architect's exact design, assigns implementation, obtains independent review, and tracks any implementation fixes. Final review must cover the exact commit that will be merged; a new commit or rebase requires another review.

The pull request template captures the task, design, check evidence, reviewer verdict, and full reviewed head SHA. Its fields and checkboxes are prompts for contributors and the coordinator; GitHub does not enforce them. The coordinator verifies that required CI passed and that the pull request head still matches the reviewed SHA before performing the authorized squash merge.

The `CI` workflow runs on pull requests, pushes to `master`, and manual dispatch. It checks agent setup, Rust and PostgreSQL tests, Python tests, and agreement between the two bundle contract checkers. Repository branch protection, including which CI checks are required, is configured separately in GitHub settings. This repository document does not change those settings.

Ignored Cargo tests are reserved for guarded PostgreSQL integration tests. CI runs all ignored tests with an explicit test database URL; live Telegram and real catalog checks use separate manual harnesses.
