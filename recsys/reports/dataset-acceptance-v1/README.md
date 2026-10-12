# ARB-033 offline release compatibility

Status: **PASS for full-input offline compatibility** on 2026-10-12 at source commit `adfe084acb8845b16609ebd62b34076c9919154a`. The separate read-only evidence verifier also passed. This result covers the complete frozen inputs and synthetic history in an isolated PostgreSQL project. Live Telegram acceptance and preservation of production user history remain **PENDING** for DATA-06 v2.

| Resource | Accepted identity or aggregate |
| --- | --- |
| Named release | `neelagiri-2025-nafiul-2026-v1` |
| Release lock SHA-256 | `46f3ffd60239675b802026025016ee6719cb0525325e088fbfa248012db3de9b` |
| Base identity | `sha256:78bb29bfe18a00519b5dcd06e9682e1c4c84528166f0c51a80d24ea28c7a62b6` |
| Alias identity | `sha256:ffa80087fc9d8bace4fe5517d7aa0fa7ba98bfed365988396b32828ec75a2403` |
| Catalog release identity | `sha256:8da879636b476ffc9d72bf3a08447b778ef047232fc02cd69b1ed0ca02168b5c` |
| Reference A bundle | `sha256:77d171f9ee8c30f606957afb7037276a66a543582491344661638d486d94f728` |
| Reference A report SHA-256 | `1aa472f9764b82033d9a31a7d472814c5c0df06cd3833837e68894a3057d741e` |
| Fresh B bundle | `sha256:d8b773b50791a68ae30ec14f3bf38805df79350ec71edc4322e83113fbc8f5e8` |
| Fresh B report SHA-256 | `cf10d521425c1775deeb219b06a298eff0baffc2c22a3c299f31f2510da56dfe` |
| Private preparation report SHA-256 | `8fa4d7e97e4ff992e10e1b7b37f3169b69c0ba35f6f55a9e9313c680ebecdb82` |
| Catalog / neighbors SHA-256 | `70abba18facb58bf77865c4dadb07e86ebde0191f5b4fe94cd70714a1d2e156c` / `19f05e5a083a675675dbf734ba841ea1193aaa538839f74759e4d38f1e341be7` |
| Catalog and neighbors | 28,955 records; histogram 0:83, 1–4:0, 5:28,872 |
| Builder producer | Python 3.12.14, NumPy 2.5.3, package 0.1.0; five code-file hashes in the lock |
| Accepted reference producer commit | `15d17a3e344a91b57536dad672f1698c3854c166` |
| Actual builder image ID | `sha256:ec8e472d800bd856bafa467271e81c72de4047c05695b7d67fa4800273cf1fb1` |
| Actual prepare image ID | `sha256:2e7d268e83f7eea9f57b0b214ea4e51733462b602adafc03741f535fc6b11abd` |
| Actual PostgreSQL image ID | `sha256:efedf3595f1d6f415c08568ba171029bf54052e754cc9f030e3f2412b21f3d67` |

The exact new smoke seed is `Cowboy Bebop` (MAL 1) with ordered targets `37991, 24439, 26443, 8425, 23133`. The empty seed is `Notteke Explet's` (MAL 29740). The retained historical release uses a separate legacy profile; do not substitute its counts or seeds for this release.

Fresh A used block size 256 and matched every locked reference hash. Fresh B used block size 128; its catalog and neighbors were byte-identical to A, and its build report differed only in the execution block. The actual Rust checker accepted both bundles. The old→A→B→A→old prepare sequence produced five exit-zero starts. Complete database rows and migration timestamps matched the nonempty synthetic baseline after every phase. The original settings remained byte- and metadata-identical, PostgreSQL retained its container and volume and stayed healthy, and the old 17,562-record bundle was selected again. No bot poller ran.

The accepted reference producer commit above identifies the earlier source used for the lock; this acceptance execution used the checked-out `adfe084a` commit and verified the installed producer bytes against the lock. An earlier isolated attempt failed when its read-only release-validation container was stopped before any new bundle or prepare switch; its failed private evidence was retained. The successful run used a fresh evidence directory. Offline PASS does not change the two `PENDING` live and production-history statuses.

Use the private, opt-in commands in [deploy/ACCEPTANCE.md](../../../deploy/ACCEPTANCE.md) with a clean checkout, frozen full inputs, isolated healthy PostgreSQL, placeholder token, saved old bundle and nonempty synthetic history. The runner writes `preparation-report.json` and private saved evidence under a new ignored directory. After the run, use `release-acceptance.py verify --evidence-dir PRIVATE_DIR` to independently recompute the offline result. Publish only aggregate statuses, versions and hashes; never publish paths, settings, credentials, DB rows, profiles, feedback bodies, or raw evidence.
