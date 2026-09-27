# AnimeRecommendationBot agent guide

## Product boundaries

- The bot is a Russian-language, private-chat anime recommendation bot. The runtime is Rust.
- Recommendation preparation runs offline in Python using the MAL 2020 dataset and GloVe. Runtime recommendations come from generated JSON catalog, neighbor, and manifest artifacts keyed by `MAL_ID`.
- PostgreSQL stores feedback: anime scores are 1–10 and recommendation feedback is 0–5. Do not add personalization unless a task explicitly authorizes it.
- Keep generated data, credentials, and local tools out of commits. Use small fixtures for checks.

## Task workflow

- Use one task branch, worktree, and pull request per task. Before work, verify the repository root, branch, expected baseline commit, task scope, and clean status. Edit only assigned files in the assigned worktree.
- The root coordinator owns task planning, Notion updates, pushes, pull requests, CI checks, and merges. Implementers may commit only when the root explicitly authorizes a commit. Child agents do not delegate.
- Run the sequence: task specification → architect's exact design → implementation → independent review → implementation fixes → final review of the exact commit SHA. A new commit or rebase invalidates the previous review.
- The root may squash merge only after final review passes, required CI is green, and the pull request head still matches the reviewed SHA. The user has authorized this automatic merge workflow.
- At most three child agents may be open beside the root. Close completed stage threads. Parallelize only independent tasks with disjoint file ownership; shared files have one owner. Give each task its own database and test paths.
- Report verification commands and results. Surface blockers and risks with concrete evidence.
