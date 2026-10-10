# ARB-027: fresh ratings dataset audit

**Decision:** use the pinned [Anime Dataset 2025, version 1](https://www.kaggle.com/datasets/neelagiriaditya/anime-dataset-jan-1917-to-oct-2025) for **noncommercial research intake**. Its `details.csv` and `ratings.csv` are measured and pinned in [source-audit.json](source-audit.json). This selects research inputs; the active MAL 2020 registry, runtime bundle, and bot have not been replaced. Commercial use is outside the declared CC BY-NC-SA 4.0 grant. The intended production scope and underlying MAL-derived data rights remain unassessed, and release is ineligible until intake, normalization, alias coverage, bundle checks, and acceptance are complete.

| Candidate | Full-data result | Decision |
| --- | --- | --- |
| Neelagiri version 1, 2025 | 69,606,111 distinct explicit source pairs; 69,459,967 uniquely join the MAL metadata (**99.7900%**). After quarantining one pair with both score 0 and an explicit score: 69,459,966 accepted pairs, 286,866 rated profiles, 241,412 profiles with at least 15 distinct ratings ≥8, and 28,514 rated `MAL_ID`s. Of these items, 1,871 first aired in 2024–October 2025, including 740 in 2025. | Passes all research-data gates. |
| Ramazan version 8, 2025 | 148,170,496 rows and 20,237 one-to-one internal-anime-ID/MAL-URL mappings. The 1,774,522 user IDs are opaque contiguous integers pooled from MAL, AniList, and Kitsu. Their original namespaces and rating conversions are unavailable. | Insufficient evidence for an accepted user cohort; no source synopsis. |
| Shikimori, February 2026 | 67,071 rows but only 726 profiles with explicit scores, 315 with at least 15 distinct positive native items, and 3,540 distinct native items with scores. No verified native-ID-to-`MAL_ID` map. | Fails cohort and mapping gates. |
| Nafiul version 1, 2026 | 253,949 distinct explicit pairs; 246,873 uniquely join metadata (**97.21%**), after excluding missing and duplicate metadata IDs. Only 632 accepted profiles have at least 15 positive items. | Fails the 99% join and 5,000-positive-user gates. Its metadata is separately screened as a possible alias source. |

The thresholds were fixed before inspecting the files: at least 10,000 rated users, 5,000 users with 15 distinct positive titles, 5,000 rated `MAL_ID`s, 100 rated titles first released from 2024, 20 from 2025, 99% raw distinct-pair MAL join, and 100% accepted-pair join. “Positive” means explicit score ≥8 on the observed 1–10 scale. A zero or missing score does not mean dislike. Exact repeated rating triples are removed; score conflicts without reliable event time are excluded. Source user profiles are counted, not people deduplicated across services. A metadata row alone does not satisfy a rated-title gate. The JSON records each gate as `PASS`, `FAIL`, or `UNKNOWN` with its denominator; unknown evidence is never treated as zero or a pass.

The selected dataset was published on **2025-11-05**. Its name describes catalog coverage through October 2025, not a verified rating crawl date. The 740 accepted scored titles with an aired/airing status and first-release date from 2025 through October 31 support a **bounded inference** that the snapshot contains 2025 activity. Exact crawl dates and interaction event times remain unknown; this dataset cannot support a temporal train/test split or next-event claims. We excluded 98 scored 2025 titles and six scored 2026 titles marked “Not yet aired” from the recent-title gates, plus accepted items without a usable first-release date. Watch status was kept separate from score; the completed-status diagnostic alone has 65,372,644 accepted pairs, 278,656 users, and 238,279 users with at least 15 positive titles.

The selected metadata contains 28,955 unique MAL IDs, 23,828 nonempty synopses, and 22,972 nonempty parsed genre lists. A full scan of the **hash-verified current GloVe 6B 300d** input using the frozen `[a-z]+(?:'[a-z]+)?` tokenizer found matches in 28,855 projected catalog items and all 23,828 nonempty synopses. This diagnostic projects `title`, `title_japanese` as an alias, parsed genres, and synopsis into the existing text builder; it is not a normalized candidate bundle or a recommendation-quality result. ASCII words and GloVe matches do not prove that a synopsis is English. The selected CSV lacks explicit English-title and synonym fields, which may reduce search coverage. A separate metadata-only screen of the pinned [Nafiul version 1 file](https://www.kaggle.com/datasets/nafiulislam490/comprehensive-myanimelist-mal-dataset-2026) found a unique `MAL_ID` overlap for 28,558 of the 28,955 selected titles and nonempty English-title text for 12,689 of those matches; 26 overlapping IDs have ambiguous duplicate secondary rows. Its synonym field is free text and has not been parsed or quality-checked. No alias enrichment is adopted by this audit.

## Reproduction and interpretation

The JSON pins every consumed public artifact by version, URL, complete byte size, and SHA-256. Download those exact versions into a private local directory, verify hashes before parsing, and keep raw profiles, local scripts, and disk tables outside Git. `details.csv` is 20,178,926 bytes; `ratings.csv` is 4,504,586,658 bytes. Full inspection used Python 3.12.3, DuckDB 1.5.6 with disk-backed grouping, and NumPy 2.5.3 for the GloVe diagnostic. The current pinned GloVe member is 1,037,962,819 bytes with SHA-256 `91125602f730fea7ca768736c6f442e668b49db095682bf2aad375db061c21ed`. These are substantial inputs; the full files and local analysis methods are deliberately absent from the repository. Method-file hashes and independent cross-checks are in the JSON.

For the two selected source files, set `AUDIT_DIR` to a private local directory and run:

```sh
mkdir -p "$AUDIT_DIR"
curl --fail --location 'https://www.kaggle.com/api/v1/datasets/download/neelagiriaditya/anime-dataset-jan-1917-to-oct-2025/datasets%2Fdetails.csv?datasetVersionNumber=1' -o "$AUDIT_DIR/details.csv"
curl --fail --location 'https://www.kaggle.com/api/v1/datasets/download/neelagiriaditya/anime-dataset-jan-1917-to-oct-2025/datasets%2Fratings.csv?datasetVersionNumber=1' -o "$AUDIT_DIR/ratings.csv"
test "$(wc -c < "$AUDIT_DIR/details.csv")" -eq 20178926
test "$(wc -c < "$AUDIT_DIR/ratings.csv")" -eq 4504586658
printf '%s  %s\n' '023b36604b475c07f0952caa1954168ba86d5ebd94b65458c4de522f78e6d1ee20' "$AUDIT_DIR/details.csv" | sha256sum --check -
printf '%s  %s\n' 'b93f8f1a540510a8640243ec6bba771212ae33e1fba50f257621f9d1077439e1' "$AUDIT_DIR/ratings.csv" | sha256sum --check -
```

Kaggle may require its normal access credentials. The command URLs and hashes are public pins, not signed download links. Use DuckDB 1.5.6 or equivalent disk-backed grouping for the full pair calculation below. With `METHOD_DIR` pointing to the local, uncommitted audit scripts and their expected scratch layout, the full selected-candidate audit used these commands after download:

```sh
python3 "$METHOD_DIR/audit_neelagiri_metadata.py"
python3 "$METHOD_DIR/audit_neelagiri_ratings.py"
python3 "$METHOD_DIR/group_neelagiri_pairs.py"
python3 "$METHOD_DIR/cohorts_neelagiri.py"
python3 "$METHOD_DIR/audit_neelagiri_glove.py"
python3 "$METHOD_DIR/validate_report.py"
```

The scripts and full data are intentionally **not shipped**. Their hashes are recorded in the JSON; the commands require the private method files and the same pinned GloVe input. The artifact pins and algorithm description support an independent implementation, but this repository report alone is not a self-contained rerun. No live bot or database was needed for the audit.

For the selected source, read `username`, `anime_id`, `status`, and `score` from all ratings rows. Reject blank usernames and scores outside 0–10. Build the **raw explicit-pair** set from distinct `(username, anime_id)` where score is 1–10 *before* mapping exclusions: 69,606,111 pairs. Build an order-independent group over **all** rows for each `(username, anime_id)`, inspecting score minima/maxima and row count. Three rows repeat an explicit rating triple; no pair has two different explicit scores, while one pair combines zero and a nonzero score and is quarantined. Inner-join raw explicit pairs to unique `details.mal_id`: 69,459,967 map; 146,144 do not. The one additional mixed-state quarantine leaves 69,459,966 accepted pairs. Count users, distinct rated MAL items, and per-user distinct items rated ≥8 from that accepted set. First-release cohorts extract the calendar date from `details.start_date` (equivalent to `CAST(start_date AS DATE) <= DATE '2025-10-31'`) and require `status IN ('Finished Airing', 'Currently Airing')`, excluding unreleased and missing-date rows. This is a static preference snapshot, so do not use rating-file order as time.

The other candidates use the same explicit-score, deduplication, and pre-mapping denominator rules where their source semantics allow. Ramazan `animeID` is an internal 1–20,237 index; `mal_url` in `animes.csv` supplies the syntactic MAL mapping. Its 2026-year titles already have nonzero scores in a file updated September 2025; the cause is unknown. The author model-loading code injects timestamp `978300760`, which is synthetic, not rating event time. Shikimori's `anime` rating field is an embedded Python-literal object containing a Shikimori ID, not a MAL ID. Nafiul's 145 extra metadata rows span 134 duplicate MAL IDs; all ambiguous rows were quarantined in the join calculation. Reproduction details, exact counts, source URLs, limitations, and local-method SHA-256 values are in the JSON.

The next dependent tasks are DATA02 to freeze and acquire selected inputs and decide whether to audit alias/synopsis enrichment, DATA03 to normalize and quarantine unmapped pairs under the current catalog contract, and DATA04 to build and validate a new immutable bundle before changing the default. Full-catalog and private update/rollback acceptance remain separate DATA05/06 stages. The existing pinned MAL 2020 bundle stays the rollback source until those stages pass.
