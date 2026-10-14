# Local level-1 acceptance

Run from the repository root:

```bash
python3 deploy/tests/level1-acceptance.py
```

This one command runs the ordinary Rust suite, all nine existing PostgreSQL integration tests, ten named dialogue scenarios through the actual Rust router and FakeTelegram, and synthetic Compose acceptance. It starts a uniquely named, disposable PostgreSQL project with a random `127.0.0.1` port and test-only credentials so host Rust tests can connect. The Compose phase uses a separate internal network, small offline bundles and `bot --prepare` with a placeholder token. No Telegram poller or user account is involved. Docker Engine, Compose 2.30+, Rust 1.95.0 and Python 3 are required.

The dialogue phase checks rendered callback coordinates, alias selection, five ordered cards, empty lists, both score ranges and stored history, stale controls, global navigation, description pages, complete Unicode feedback, interrupted database write and retry, context resets and transport faults. The Compose phase builds the production images with a small fixture, selects legacy A, current B, then legacy A again, compares full database snapshots including migration timestamps, and exercises persistence and startup guards. Any missing required test or phase marker fails the run.

The runner writes logs and a compact `arb-level1` JSON report under ignored `target/level1/<run-id>/`. Its `limits` section marks real Telegram, human UI, the full pinned catalog and poller process restart as `NOT_RUN`. It always removes only its own projects; a cleanup failure makes the run fail. To isolate a phase during development, pass `--scope dialogue` or `--scope compose`. The default `all` is the review and CI entry point.

This local result is scoped to the synthetic stand. The historical [full-input offline preparation](ACCEPTANCE.md) and [opt-in live Telegram runbook](LIVE_RELEASE_ACCEPTANCE.md) remain separate; missing live observations remain `PENDING`.
