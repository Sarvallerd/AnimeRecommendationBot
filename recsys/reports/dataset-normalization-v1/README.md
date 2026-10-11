# Neelagiri 2025 dataset normalization

The dataset is the two pinned Neelagiri Kaggle version 1 CSVs, processed as a separate research snapshot. Source and cohort selection are documented in the [dataset audit](../dataset-audit-v1/README.md); the [dataset contract](../../DATASET.md) defines the five outputs and validation rules. Source terms are CC BY-NC-SA 4.0. Exact rating collection dates and rating event times are unknown, so chronological evaluation is unsupported. This report contains only aggregates and artifact hashes; private CSVs, pseudonymous user keys, and generated outputs remain outside Git.

The full run uses the locked package with DuckDB 1.5.6. Each attempt independently copies and rehashes both input CSVs before parsing. DuckDB uses a 2,048 MiB query memory limit, two threads, and a 32,768 MiB spill limit. The limit is not a process RSS cap. After aggregate checks against the frozen audit, normalization validates the published bytes, including the full compressed interaction stream. Reproducibility requires two successive fresh runs into different directories, independent scratch, identical hashes for all five output files, and the same manifest identity. The commands use private paths represented here by placeholders:

```sh
recsys dataset normalize --snapshot neelagiri-2025-v1 \
  --data-dir CACHE --output-dir OUTPUT_A --scratch-dir SCRATCH_A \
  --memory-limit-mib 2048 --threads 2 --spill-limit-mib 32768
recsys dataset validate OUTPUT_A
recsys dataset normalize --snapshot neelagiri-2025-v1 \
  --data-dir CACHE --output-dir OUTPUT_B --scratch-dir SCRATCH_B \
  --memory-limit-mib 2048 --threads 2 --spill-limit-mib 32768
recsys dataset validate OUTPUT_B
```

An initial diagnostic run is excluded from acceptance. Its full scan matched every pinned audit count and distribution and published the five files, but a subsequent CLI validation returned `dataset_integrity_error`: the packaged expectations resource was edited during that running process to remove an unmeasured `status_conflict_pairs=0` assumption. The manifest recorded the pre-edit resource SHA-256 while the second validation loaded the revised bytes. No accepted audit number was changed. Diagnostic wall time was 13:27.06, and peak resident set size was 2,707,872 KiB. This run is not an A/A member and its output is not a trusted dataset release.

A second attempted run, before the final streaming-validator fix, also returned `dataset_integrity_error` during staged output validation and published no output. Its wall time was 9:12.51 and peak resident set size was 3,126,880 KiB. Investigation found that valid highly compressible gzip chunks could expand beyond the validator's arbitrary 16 MiB per-chunk guard. The validator now bounds each decompression call's output and separately limits incomplete line length; a synthetic file that exceeds the old guard passes full interaction validation. This failed attempt is excluded from A/A acceptance.

Two fresh runs of the pre-header-check producer each completed normalization and a separate post-publication `dataset validate` with exit code 0. All five outputs matched byte for byte. Their shared identity was `sha256:a24281bf442efd4974a7a848d97c5439088d17c88ca50fefe457547cf59e7c89`. The complete aggregate QA matched the frozen source audit: 124,298,357 raw rows; 69,606,111 distinct explicit pairs before mapping; 146,144 pairs without catalog metadata; one explicit/unrated mixed pair; 69,459,966 accepted pairs across 286,866 users and 28,514 MAL_IDs; 241,412 users with at least 15 distinct scores of 8–10; and 1,871/740 recent rated titles from 2024/2025 onward. The accepted completed-only subset had 65,372,644 pairs and 278,656 users. The five-item user-history bins include 0–4 in bin `0` and 100+ in bin `100`.

| File | Bytes | SHA-256, identical in both runs |
| --- | ---: | --- |
| `catalog.json` | 14,359,145 | `c4b54c4a3f0fcb9253ff627ce4a15db9fd9c3c3da75f3dee37184ce101a35eab` |
| `item-facts.json` | 2,852,007 | `3834154b3abba9fbb08191027741a46340aa8de0bedf63b13e20ea8f2c7af526` |
| `interactions.tsv.gz` | 332,401,043 | `19996cd5204fba814241337d6053bc4b2c4521e7c2609bc7b98975a3fa082aeb` |
| `dataset-qa.json` | 2,763 | `11bb5a6b0f8d16a969af726f50f412f68f7a08fc420d1e237ca792ea70fe994b` |
| `dataset-manifest.json` | 2,845 | `a24281bf442efd4974a7a848d97c5439088d17c88ca50fefe457547cf59e7c89` |

| Diagnostic run | Normalize wall / peak RSS | Published-byte validate wall / peak RSS |
| --- | --- | --- |
| A | 13:31.05 / 2,766,456 KiB | 3:46.98 / 198,236 KiB |
| A-repeat | 12:50.24 / 2,597,216 KiB | 3:44.32 / 197,796 KiB |

The source SHA-256 values before and after the two runs were unchanged: details `023b36604b475c07f0952caa1954168ba86d5ebd94b65458c4de522f78e6d1ee`, ratings `b93f8f1a540510a8640243ec6bba771212ae33e1fba50f257621f9d1077439e1`. The installed producer fingerprints for these runs were `dataset.py` `b61da6809c45ac8099e78fceb64515bdcc0952d3b59697bf1d0a851114cce79f`, `dataset_metadata.py` `09a47627fce76e6ad72aeff5656ff011eeb90ad0d0ef97f4e96f022d3124a97d`, and `dataset_interactions.py` `a3d42898773ea64857212b575cfd1ef584499f1e7de5b3a1355c6cfa4e03ab19`; expectations resource `9e246049ac968ea5a95c11d90935f482f5d383b54166fab92b06952cd08cdbbd`.

An exact ratings-CSV header check was added after this diagnostic A/A, so these byte hashes are evidence for the previous producer, not final acceptance of the updated code. Final frozen A/A for the reviewed producer remains pending. No code, source CSV, or acceptance artifact is published as a runtime catalog by this task.

The post-diagnostic `dataset_interactions.py` SHA-256 is `75d92e3043be642471d017749c364eb7a340cdf7f274edc7b98fa2fb5d96d379`. It rejects renamed, reordered, or duplicate CSV headers before DuckDB parsing with a fixed `ratings_csv_invalid` code. The locked installed-package Python suite passed 97 tests; contract tests, evidence tests, Rust format/build/lint/tests, and the cross-language bundle checker passed locally. Synthetic Compose acceptance could not start because this host's Docker installation lacks the Compose CLI plugin (`docker compose version --short` returned `unknown flag: --short`); CI remains the required check for that job.
