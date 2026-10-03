# ARB-018: fixed 20-query comparison

This report compares the selected GloVe ranker, positive genre Jaccard, and a repaired original notebook method on a purposive MAL 2020 query set. Five queries came from each of four popularity bands. One declared LLM assessor cohort graded the blinded union of 268 query–candidate pairs from catalog evidence. There was no human cross-check. These results describe this sample and assessor; they do not establish general recommendation quality.

## Quality results

Each method has 100 possible top-five slots. Missing results contribute zero. A returned null grade remains unknown. Mean grade ranges from 0 to 2; precision counts grades of at least 1. A point estimate exists only when every returned result has a grade. Bounds are arithmetic limits for unknown grades, not confidence intervals.

| Method | Universe | Returned | Graded / returned | Mean grade@5 | Precision@5 | Assessor-judged same-franchise@5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Selected GloVe (`arb009-v1`) | 17,562 | 100/100 | 100/100 | 1.29 | 0.87 | unknown; [0.20, 0.21] |
| Genre Jaccard (`1`) | 17,562 | 100/100 | 86/100 | unknown; [1.16, 1.44] | unknown; [0.80, 0.94] | unknown; [0.22, 0.23] |
| Repaired legacy (`arb018-legacy-v1`) | 10,882 | 95/100 | 91/95 | unknown; [1.09, 1.17] | unknown; [0.77, 0.81] | unknown; [0.10, 0.12] |

Franchise values describe the assessor's judgment of a candidate's relationship to its query, not verified franchise IDs. Annotation coverage was 99/100, 99/100, and 93/95 returned results, respectively. Selected GloVe and Jaccard grade ranges overlap. The legacy universe differs, and MAL ID 35102 is excluded because its Aired field is missing. No general improvement threshold was set. [summary.md](summary.md) has all per-query results, grade histograms, statuses, source hashes, and individual timing samples; [summary.json](summary.json) holds the computed metrics.

## Measured workload

The comparison was rerun on 2026-10-03 with three fresh-process repetitions per method. Each worker prepared its method and ranked the same 20 queries. This does not time construction of the full production neighbor file. All nine workers succeeded, per-method outputs were identical across repetitions, and the selected method's ordered IDs and scores matched the published bundle for all 20 queries. The table gives medians.

| Method | Launch to exit | Preparation | Rank 20 | Whole-process peak RSS |
| --- | ---: | ---: | ---: | ---: |
| Selected GloVe | 10.687 s | 10.535 s | 0.058 s | 226.2 MiB |
| Genre Jaccard | 2.196 s | 1.625 s | 0.244 s | 178.3 MiB |
| Repaired legacy | 13.646 s | 13.273 s | 0.159 s | 411.8 MiB |

Preparation includes repeated source and report verification. Both GloVe methods stream and hash the entire embedding file; Jaccard checks its size only. Legacy also parses the CSVs again. Peak RSS covers the entire worker. OS page cache, background host activity, and scheduling were uncontrolled on the shared WSL2 host. [measurements.json](measurements.json) contains exact samples, method order, thread settings, runtime, and machine details.

## Data and assessment

The normalized catalog has 17,562 titles. Repaired notebook preprocessing retains 12,173 after metadata exclusions and 10,882 after the synopsis join. The method uses fixed 20-cluster KMeans, notebook-style metadata tokens, GloVe vector sums, 99 cosine candidates, same-cluster filtering, and Score/year sorting. Historical fitted clusters and vectors were unavailable; the comparator repairs row alignment, self exclusion, ties, and nonready vectors. [legacy-diagnostics.json](legacy-diagnostics.json) records counts and deterministic digests. MAL ID 39619 has boilerplate synopsis text in the legacy source while its normalized synopsis is null.

The assessor is **`arb018-gpt56sol-high-cohort-v1`, an LLM cohort** using `gpt-5.6-sol` at high reasoning effort. Seven fresh sequential sessions each judged a disjoint query partition with the same blinded prompt and its catalog subset. This is one complete cohort assessment, not seven independent full assessments. The 18 null pair grades remain unknown; some pairs appear in more than one method's results. The assessor did not receive method, rank, algorithm score, MAL rating, popularity, or previous batch grades. Catalog-only evidence, missing synopses, and separate contexts limit interpretation. [assessor-method.md](assessor-method.md) documents partitions, dates, recovery, and schema corrections. [judgments.json](judgments.json) preserves accepted rationales and verbatim citations. No human grades were inferred.

## Provenance and reproduction

The original comparison and first five completed batches date to 2026-09-27. After temporary files were lost, the 200 accepted rows in batches 01, 02, 03, 04, and 06 were recovered byte-for-byte from saved responses. Normalize, build, export, and comparison were rerun on 2026-10-03; deterministic comparison and packet hashes matched. Batches 05 and 07 were completed on 2026-10-03. The committed measurements contain the new timings. The comparison's `code_revision` is the actual measured source revision; the later rebase and report commit did not rerun measurements. Every recorded code-file hash still matches the rebased source.

| Artifact | SHA-256 |
| --- | --- |
| [comparison.json](comparison.json) | `cd829aba0ed71465540b7926a073853b651775bf8260203dd5e44cfb2c4d9551` |
| [measurements.json](measurements.json) | `2288bed4e79fe443ef8176b27fc5c10c0ee658a2b1316913266dcbb87b2332b4` |
| [legacy-diagnostics.json](legacy-diagnostics.json) | `c5cad4f90b2618fc90eee3d7add7d7d4d89d21c55af2e90efd253cd259823a27` |
| [judgments.json](judgments.json) | `daa8f48af717bb7977da9d3ab59674eb915feee654a4f506cee0da2487f849ec` |
| Blinded packet (kept outside Git) | `dad77d65536c4983bee66881988f7631b3843acabe091f240539df15f795e63e` |
| Catalog (kept outside Git) | `2ae604168ba10be67cca14cd8cba0235fc6f0d0de19fdc25b9fc91175ffd06d2` |
| Measured source revision | `5a55469b5d6dafefcb2eff6adc0f3b681f51dd1a` |

From `recsys/`, with the pinned MAL CSVs, GloVe, generated bundle, and reports available, rerun comparison into a new directory:

```sh
uv sync --locked --no-editable --extra quality
uv run --locked --no-editable --extra quality recsys quality compare \
  --bundle-dir "$BUNDLE_DIR" \
  --normalization-report "$NORMALIZATION_REPORT" \
  --build-report "$BUILD_REPORT" \
  --anime-csv "$ANIME_CSV" \
  --synopsis-csv "$SYNOPSIS_CSV" \
  --glove "$GLOVE_FILE" \
  --lockfile uv.lock \
  --code-revision 5a55469b5d6dafefcb2eff6adc0f3b681f51dd1a \
  --output-dir "$NEW_COMPARISON_DIR" \
  --repetitions 3
```

To revalidate and regenerate the committed summary, use this report and the pinned bundle catalog:

```sh
uv run --locked --no-editable --extra quality recsys quality summarize \
  --comparison reports/arb018-v1/comparison.json \
  --catalog "$BUNDLE_DIR/catalog.json" \
  --measurements reports/arb018-v1/measurements.json \
  --assessment reports/arb018-v1/judgments.json \
  --output-dir "$NEW_SUMMARY_DIR"
```

Raw MAL data, GloVe, the full catalog and assessment packet, and generated matrices remain outside Git. Source descriptors, package versions, and code hashes are in [summary.md](summary.md) and [comparison.json](comparison.json).
