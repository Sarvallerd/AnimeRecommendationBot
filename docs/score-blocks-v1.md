# Validation score blocks v1

Package 0.6.0 writes `scores.bin` with prediction manifest schema 2 by default for standalone validation prediction and for the eight arm `ml02-native-v3` campaign. The unchanged `ml02-cpu-v1` campaign explicitly writes schema 1 `scores.jsonl.gz`. Standalone prediction can select it with `--score-format jsonl-v1`. The old `ml02-native-v2` campaign and its plan/report remain historical and require the pinned `a06e369b21f96c5d8ee37765e3116448e1d12b6e` checkout. Existing artifacts, manifests, resource bytes, and hashes are not converted. New preparation and a new campaign are required for the new producer.

## Container

All integers are little endian. The 64 byte file header uses `<8sHHIQQ32s`: exact magic `ARBSCR1` followed by NUL, version 1, flags 0, catalog count C, query count Q, dense row count D, and SHA-256 of canonical prediction manifest bytes with `scores` omitted. C is at least one. The manifest stores `score_format: arb-score-blocks-v1` and a descriptor with stored bytes/SHA-256, rows, canonical decoded bytes/SHA-256, C, D, and query block count. The manifest is at most 16 KiB.

Each block has a 64 byte `<BBHIIIQQ32s` header: kind, codec, reserved, count, raw length, stored length, first ordinal, dense count, raw SHA-256. Blocks appear in order: one ascending unique uint32 MAL ID catalog; one train popularity vector if Q is positive; then contiguous query blocks. There is no footer. Codec 1 is zlib level 1 when shorter than raw; otherwise codec 0 stores raw bytes. Every raw block is at most 4 MiB.

A vector is an LSB first availability bitmap of `ceil(C/8)` bytes followed by C signed int64 scores. Bitmap padding is zero. Unavailable items have zero payload; an available zero is distinct from null. Popularity contains exact nonnegative `positive_users` counts from train statistics. It is stored once per artifact and broadcast for the popularity arm and every zero seed query. A query record is 32 bytes of its `q1_` hexadecimal suffix, one mode byte (0 broadcast, 1 dense), followed by a vector only for dense mode. Query blocks contain exactly 128 rows except the last, or fewer when C requires a smaller deterministic block size: `min(128, floor(4194304/(33+8*C+ceil(C/8))))`. Popularity uses 128.

The canonical decoded hash covers the file header, each block header normalized to codec 0 and raw stored length, and each raw payload. It does not reconstruct JSON. Both stored and canonical decoded container limits are 64 GiB. The reader checks dimensions before allocation, bounded decompression, catalog, popularity, masks, null payload, every block and file digest, binding, query order and mode, EOF, and trailing bytes. Verification and evaluation consume bounded blocks; staged publication occurs only after the entire prediction stream is checked.

## Conditional storage admission

Let `V=8*C+ceil(C/8)`, `B=128` for popularity and the block size above for other arms, `K=ceil(Q/B)`, and `D=0` for popularity or `Q-Q0` for other arms. A maximum uncompressed score container is

`U=64+64+4*C+int(Q>0)*(64+V)+64*K+33*Q+D*V`.

The stored container is at most U because compression falls back to raw. Add 16 KiB for the manifest per arm. Fresh admission requires the sum of one popularity and seven other arm bounds, 8 GiB estimated other artifacts, 20 GiB reserve, and 4 GiB margin. Resume uses only variants without a verified successful prediction, plus 8 GiB and 24 GiB. The plan records trusted C, Q, Q0, query descriptor, every arm bound, and the fresh total. Both output and physical volumes are checked before work and every spawn; active writes require 24 GiB plus the next emit, and running work stops below 24 GiB. The 8 GiB other artifact estimate is conditional. Space is not reserved atomically, and this calculation does not guarantee campaign completion.

For an illustrative synthetic C=28,955, Q=45,938, Q0=0, the per arm bounds including manifest are 1,906,586 bytes for popularity and 10,809,430,482 bytes for each other arm. Fresh admission is 110,027,658,328 bytes (about 102.47 GiB) on each volume. Real Q0 is read from the trusted prepared query stream.

## Scope of evidence

Small fixtures test exact decoded scores and private metric bytes for all seven CPU variants, native common writer, corruption, replay, and storage checks. `scripts/benchmark_score_blocks.py` generates only synthetic C=28,955 cases with Q=41 and 1024, four score distributions, and five alternating repeats. It reports actual bytes, write time, verification plus normalization of both formats into Python score pairs, and a local unmasked top-ten ranking time. Its process peak RSS covers both formats in one case, so it is not per-format RSS. `scripts/benchmark_rank_adapters.py` separately times production legacy and binary ranking adapters for the same synthetic C and Q, with the same exclusions and ties in five alternating repeats. Neither script times full metric aggregation or establishes full ML-02 completion, model quality, or a selected N. Stored zlib bytes can vary with zlib implementation while canonical decoded bytes and scientific results remain fixed.
