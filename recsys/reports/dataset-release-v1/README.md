# Dataset release v1 reference

Status: **PASS for full pinned-input offline reference**. This records the ARB-031 catalog release and two reference bundles. It does not establish live Telegram, database migration, update, or rollback acceptance; those remain later tasks.

The installed package came from reviewed source commit `15d17a3e344a91b57536dad672f1698c3854c166` in immutable builder image `sha256:d713a2f2e504763ed9d109d7567a723d3c454e0f80203a31d936b716067754bf` (`linux/amd64`). The image ran Python 3.12.14, NumPy 2.5.3, package 0.1.0, and UID/GID 1000:1000. The installed `release.py`, `release_build.py`, `build.py`, `export.py`, `bundle.py`, and GloVe registry bytes matched the reviewed commit. Dockerfile SHA-256: `57575eb42fe0845feaeed057cabd2d5524d4a15b96f1391cfa95b0abbbc55001`.

Accepted upstream identities were base dataset `sha256:78bb29bfe18a00519b5dcd06e9682e1c4c84528166f0c51a80d24ea28c7a62b6` and alias enrichment `sha256:ffa80087fc9d8bace4fe5517d7aa0fa7ba98bfed365988396b32828ec75a2403`. The GloVe text was 1,037,962,819 bytes, SHA-256 `91125602f730fea7ca768736c6f442e668b49db095682bf2aad375db061c21ed`. Base, alias, Nafiul, and GloVe input bytes matched their pre-run hashes after all operations. No input was regenerated.

The installed CLI ran `recsys release compose` twice with the same explicit base, alias, and Nafiul inputs; `recsys release build` with the resulting release identity and GloVe text at block sizes 256 and 128; and `recsys release export` once per matching build report and neighbors file. Each process ran in a fresh container with no network, read-only inputs/root filesystem, dropped capabilities, and a private output mount. The installed `recsys validate` and Rust `check_bundle` each accepted both actual bundle directories and reported 28,955 records.

| Operation | Wall seconds | Peak RSS KiB | Result |
| --- | ---: | ---: | --- |
| Compose A | 295.003 | 275,204 | pass |
| Compose repeat | 284.467 | 275,008 | pass; both output files byte-identical to A |
| Build A, block 256 | 443.668 | 429,108 | pass |
| Build B, block 128 | 459.015 | 313,284 | pass |
| Export A | 299.993 | 335,544 | pass |
| Export B | 303.062 | 335,396 | pass |

The release identity is `sha256:8da879636b476ffc9d72bf3a08447b778ef047232fc02cd69b1ed0ca02168b5c`. Its catalog has 28,955 items, 12,754 enriched items, 12,691 items with changed aliases and first alias, and 28,003 aliases before versus 40,851 after overlay. `catalog.json` is 14,722,754 bytes, SHA-256 `70abba18facb58bf77865c4dadb07e86ebde0191f5b4fe94cd70714a1d2e156c`.

Both builds produced byte-identical `neighbors.json` (7,279,574 bytes, SHA-256 `19f05e5a083a675675dbf734ba841ea1193aaa538839f74759e4d38f1e341be7`). Their build reports agree after excluding only `execution`: A SHA-256 `1aa472f9764b82033d9a31a7d472814c5c0df06cd3833837e68894a3057d741e`, B SHA-256 `cf10d521425c1775deeb219b06a298eff0baffc2c22a3c299f31f2510da56dfe`. Requested and effective rows are 256 for A and 128 for B; both use the 67,108,864-byte similarity block budget. Bundle A identity is `sha256:77d171f9ee8c30f606957afb7037276a66a543582491344661638d486d94f728`; B is `sha256:d8b773b50791a68ae30ec14f3bf38805df79350ec71edc4322e83113fbc8f5e8`. A is the reference for the later locked preparation task.

Measured vocabulary was 68,289 tokens, with 38,131 matched; 1,667,225 of 1,762,460 token occurrences matched. Vector status counts were `ready=28,872`, `oov_only=83`, `no_tokens=0`, and `zero_vector=0`. The neighbor-count histogram for 0 through 5 was `83, 0, 0, 0, 0, 28,872`. The 83 nonready IDs exactly match the report's zero-vector ID list and empty neighbor rows. MAL ID 1 has ordered neighbors `37991, 24439, 26443, 8425, 23133`; the first empty seed is MAL ID 29740.

The small synthetic fixture tests and focused checks passed before the reviewed code freeze. This full run establishes deterministic composition, block-size equivalence, provenance, and Python/Rust bundle readability for the pinned offline inputs. Private receipts, complete ID lists, and generated JSON remain outside Git.
