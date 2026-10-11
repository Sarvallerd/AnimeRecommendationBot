# ARB-033 offline release compatibility

Status: **PENDING full opt-in run**. This page contains only public source identities and procedure. Small synthetic tests and code review cannot establish full-input compatibility. Live Telegram acceptance and preservation of production user history remain **PENDING** for DATA-06 v2, even after an offline run passes.

| Resource | Accepted identity or aggregate |
| --- | --- |
| Named release | `neelagiri-2025-nafiul-2026-v1` |
| Release lock SHA-256 | `46f3ffd60239675b802026025016ee6719cb0525325e088fbfa248012db3de9b` |
| Base identity | `sha256:78bb29bfe18a00519b5dcd06e9682e1c4c84528166f0c51a80d24ea28c7a62b6` |
| Alias identity | `sha256:ffa80087fc9d8bace4fe5517d7aa0fa7ba98bfed365988396b32828ec75a2403` |
| Catalog release identity | `sha256:8da879636b476ffc9d72bf3a08447b778ef047232fc02cd69b1ed0ca02168b5c` |
| Reference A bundle | `sha256:77d171f9ee8c30f606957afb7037276a66a543582491344661638d486d94f728` |
| Reference A report SHA-256 | `1aa472f9764b82033d9a31a7d472814c5c0df06cd3833837e68894a3057d741e` |
| Catalog / neighbors SHA-256 | `70abba18facb58bf77865c4dadb07e86ebde0191f5b4fe94cd70714a1d2e156c` / `19f05e5a083a675675dbf734ba841ea1193aaa538839f74759e4d38f1e341be7` |
| Catalog and neighbors | 28,955 records; histogram 0:83, 1–4:0, 5:28,872 |
| Builder producer | Python 3.12.14, NumPy 2.5.3, package 0.1.0; five code-file hashes in the lock |

The exact new smoke seed is `Cowboy Bebop` (MAL 1) with ordered targets `37991, 24439, 26443, 8425, 23133`. The empty seed is `Notteke Explet's` (MAL 29740). The retained historical release uses a separate legacy profile; do not substitute its counts or seeds for this release.

Use the private, opt-in commands in [deploy/ACCEPTANCE.md](../../../deploy/ACCEPTANCE.md) with a clean checkout, frozen full inputs, isolated healthy PostgreSQL, placeholder token, saved old bundle and nonempty synthetic history. The runner writes `preparation-report.json` and private saved evidence under a new ignored directory. After the run, use `release-acceptance.py verify --evidence-dir PRIVATE_DIR` to independently recompute the offline result. Publish only aggregate statuses, versions and hashes; never publish paths, settings, credentials, DB rows, profiles, feedback bodies, or raw evidence.
