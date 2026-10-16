# Archived research notebooks

These notebooks record the original exploration, preprocessing, and embeddings work. Active Python sources live in [`src/recsys/`](../../src/README.md); the bot runs from [`bot/`](../../bot/docs/build.md). The repaired legacy comparison uses the archived notebooks as provenance and documents its repairs in the [ARB-018 report](../../recsys/reports/arb018-v1/README.md). Preserve the notebook bytes so those source hashes remain meaningful.

`create_db.ipynb` imports the deleted `src/app/db` and `src/config` modules. It is historical provenance, not a supported current database setup path. Inspect or reproduce it in the separate historical checkout described by the [layout migration](../../docs/migrations/arb053-python-src-v1.md); current database schema changes use [`bot/migrations/`](../../bot/docs/database.md).
