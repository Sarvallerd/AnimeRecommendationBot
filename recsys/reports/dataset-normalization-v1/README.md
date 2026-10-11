# Neelagiri 2025 dataset normalization

The reviewed-fix producer completed two fresh full-data runs from the pinned Neelagiri Kaggle version 1 CSVs. Both published datasets passed a separate full-stream `dataset validate`, and all five output files matched byte for byte. Their identity is `sha256:78bb29bfe18a00519b5dcd06e9682e1c4c84528166f0c51a80d24ea28c7a62b6`. This is a separate research snapshot; it does not replace the bot's runtime catalog. Source terms are CC BY-NC-SA 4.0. Rating collection dates and event times are unknown, so chronological evaluation is unsupported.

The aggregate QA matched every frozen source-audit expectation: 28,955 catalog items; 124,298,357 raw rating rows; 69,606,111 distinct explicit pairs before mapping; 146,144 explicit pairs without metadata; one mixed explicit/unrated pair; 69,459,966 accepted pairs across 286,866 users and 28,514 MAL_IDs; 241,412 users with at least 15 distinct scores of 8–10; and 1,871/740 recent rated titles from 2024/2025 onward. The accepted completed-only subset has 65,372,644 pairs and 278,656 users. All pinned score, status, and five-item profile-history distributions matched; bin `0` means 0–4 items and bin `100` means 100+. The full aggregate QA is stored in `dataset-qa.json` in each private output, without source usernames or interaction rows.

| Output | Bytes | SHA-256 in both runs |
| --- | ---: | --- |
| `catalog.json` | 14,359,145 | `c4b54c4a3f0fcb9253ff627ce4a15db9fd9c3c3da75f3dee37184ce101a35eab` |
| `item-facts.json` | 2,852,007 | `3834154b3abba9fbb08191027741a46340aa8de0bedf63b13e20ea8f2c7af526` |
| `interactions.tsv.gz` | 332,401,043 | `19996cd5204fba814241337d6053bc4b2c4521e7c2609bc7b98975a3fa082aeb` |
| `dataset-qa.json` | 2,763 | `11bb5a6b0f8d16a969af726f50f412f68f7a08fc420d1e237ca792ea70fe994b` |
| `dataset-manifest.json` | 2,845 | `78bb29bfe18a00519b5dcd06e9682e1c4c84528166f0c51a80d24ea28c7a62b6` |

| Fresh run | Normalize wall / peak RSS | Published-byte validate wall / peak RSS |
| --- | --- | --- |
| E | 14:29.02 / 2,919,848 KiB | 3:45.65 / 198,068 KiB |
| F | 14:54.86 / 3,011,976 KiB | 3:45.97 / 198,064 KiB |

The locked no-editable package used Python 3.12.3, DuckDB 1.5.6, and zlib runtime 1.3. Each run independently copied and rehashed both source CSVs, built in its own private scratch, checked pinned QA, streamed validation over staged content, then atomically published a new output directory. A separate CLI validation checked the published bytes. DuckDB used a 2,048 MiB query memory limit, two threads, and a 32,768 MiB spill limit; the memory limit is not a process RSS cap. The commands, with private paths represented by placeholders, were:

```sh
recsys dataset normalize --snapshot neelagiri-2025-v1 \
  --data-dir CACHE --output-dir OUTPUT_E --scratch-dir SCRATCH_E \
  --memory-limit-mib 2048 --threads 2 --spill-limit-mib 32768
recsys dataset validate OUTPUT_E
recsys dataset normalize --snapshot neelagiri-2025-v1 \
  --data-dir CACHE --output-dir OUTPUT_F --scratch-dir SCRATCH_F \
  --memory-limit-mib 2048 --threads 2 --spill-limit-mib 32768
recsys dataset validate OUTPUT_F
```

Source SHA-256 before and after both runs was unchanged: details `023b36604b475c07f0952caa1954168ba86d5ebd94b65458c4de522f78e6d1ee`, ratings `b93f8f1a540510a8640243ec6bba771212ae33e1fba50f257621f9d1077439e1`, cache manifest `ee82bf5e5ad75e8911f32661169258a061e2c662949c292f8aa8a6a6007e6fec`, and packaged source registry `e55b9aa6d1c49e82c198af8e58bb2cb5dcbd4439615c31890e77e301f299b9ab`. Installed bytes matched checkout bytes throughout the runs. Producer SHA-256: `dataset.py` `b61da6809c45ac8099e78fceb64515bdcc0952d3b59697bf1d0a851114cce79f`; `dataset_metadata.py` `09a47627fce76e6ad72aeff5656ff011eeb90ad0d0ef97f4e96f022d3124a97d`; `dataset_interactions.py` `fcb6a274ff4c619f20c888021b0a6f56e7ee55ffad7ccd161754108b77004c75`; expectations `9e246049ac968ea5a95c11d90935f482f5d383b54166fab92b06952cd08cdbbd`.

Earlier diagnostic outputs are excluded from this identity. One run loaded an expectations resource changed during execution and failed its subsequent integrity check; another exposed an arbitrary gzip expansion guard, now replaced with bounded decompression. A later deterministic pair preceded the exact CSV-header check and independent review. That review found that tab-only usernames were accepted and `1e-999` underflowed to zero. The current producer checks 29 Unicode whitespace codepoints for blank usernames and classifies scores by exact string grammar, without floating-point conversion. The accepted QA and interaction bytes stayed identical; the manifest identity changed with the corrected producer fingerprint.

The locked installed-package Python suite passed 99 tests after these fixes. Contract tests, evidence tests, agent setup, Rust format/build/lint/tests, and the cross-language bundle checker passed locally. The coordinator ran synthetic Compose acceptance on the preceding code commit through a working WSL Docker configuration: exit 0 and `Compose acceptance passed`; private evidence is `compose-3fca290.log`. No Compose code changed in these fixes. Required CI and independent exact-SHA review remain pending. Source and cohort selection are documented in the [dataset audit](../dataset-audit-v1/README.md); format and validation rules are in the [dataset contract](../../DATASET.md).
