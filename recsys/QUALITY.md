# Recommendation quality protocol, version `arb008-v1`

This protocol fixes a small diagnostic query set and a reproducible genre baseline. It does not claim that the baseline makes good recommendations. Its Jaccard score measures overlap in catalog genre labels; only an assessor can judge whether a candidate serves the query's main appeal.

## Inputs and query set

Run the baseline against the canonical `catalog.json` produced by normalizer version 1 from the pinned MAL 2020 sources. The packaged [`quality_queries_v1.json`](src/recsys/quality_queries_v1.json) records the exact catalog SHA-256, catalog count (17,562), source registry SHA-256, MAL repository commit, and both MAL CSV hashes. Each query records its MAL ID, normalized title, source popularity rank and member count, popularity band, and why it was selected. The program verifies the raw catalog hash, count, and title guards before writing files. It never replaces a missing query with another title. A changed query set needs a new version.

The 20 queries were chosen deliberately: five each at popularity ranks 1–100, 101–1,000, 1,001–5,000, and 5,001 and above. They cover 29 genre labels, TV series, films and OVA, and years 1969–2019. This is a diagnostic set, not a random or representative sample. Its popularity bands use the source CSV's `Popularity` field. The member counts are descriptive; neither value affects ranking or relevance.

## Baseline and files

Run from the repository root with the root locked environment:

```sh
uv run --locked --no-editable recsys quality baseline \
  --catalog /path/to/catalog.json \
  --output-dir /path/to/quality-output
```

The command reads the packaged specification regardless of the current directory. For each query, it compares the query's exact genre set with every catalog entry using Jaccard similarity: intersection size divided by union size. It excludes the query's own MAL ID and zero-overlap candidates, then sorts by descending score and ascending numeric MAL ID. It returns up to five unique candidates. It does not use rating, year, popularity, title, franchise, or synopsis. An empty query genre set produces `empty_genres`; no positive candidate produces `no_positive_candidates`. Short lists remain short.

The command writes canonical UTF-8 JSON with sorted keys and a trailing newline, using temporary files and atomic replacement:

- `genre-baseline.json` contains the query set and catalog hashes, method version and parameters, query statuses, ordered candidate IDs, ranks, and Jaccard scores. It is a diagnostic artifact, separate from the bot's production cosine-neighbor contract.
- `relevance-template.json` references the baseline file by its raw SHA-256 and lists each unique query–candidate pair in numeric order. Every row starts with `relevance: null`, `assessor_kind: "unjudged"`, and null assessor ID, method version, and rationale. Algorithm scores are never copied into relevance grades.

## Assessment rubric

Ask: **Would this candidate be useful to someone seeking the query's main appeal?** Assess the concrete relationship in theme, tone, narrative, setting, or characters. A shared genre label alone is insufficient. Use the same rubric for candidates in the same franchise; measure franchise concentration separately. MAL score and popularity are not relevance judgments.

| Grade | Meaning |
| --- | --- |
| `0` | No convincing relationship in the query's central appeal. Explain the mismatch. |
| `1` | A meaningful partial relationship, with substantial differences. |
| `2` | A strong relationship in the main appeal, supported by concrete similarities. |
| `null` | Unassessed, or evidence is insufficient to judge. Never treat it as grade 0. |

A scored row must identify its assessor, assessment method version, and rationale. Future assessor kinds are `human` and `llm`; an LLM assessment must also identify its model and prompt version. The files generated here contain no scored rows and no fabricated quality labels.

## ARB-018 comparison protocol

The [ARB-018 comparison](EVALUATION.md) evaluates the chosen GloVe cosine method, genre Jaccard baseline, and a repaired version of the original method on these same 20 queries and pinned inputs. Record versions and hashes for the sources, configuration, code, query set, and every run. Keep explicit short or unavailable result lists. Pool the union of query–candidate pairs, hide method, rank, and score in the assessment view, sort pairs deterministically, and assess each pair once per declared assessor. Reuse that judgment across methods. Keep human and LLM reports separate, show assessment coverage, and report each query as well as a macro average over all 20 queries. Never reduce the denominator to the queries with available results.

For a fully judged five-slot result, `mean_grade@5` is the sum of grades divided by five, ranging from 0 to 2. `precision@5` is the number of candidates graded at least 1 divided by five. Missing result slots count as zero. A returned candidate with a null grade prevents a fully judged score. The ARB-018 summary omits nDCG because the candidate pool is incomplete.

The original method's available provenance is [`src/app/recsys/utils.py`](../src/app/recsys/utils.py) (SHA-256 `2faf2555c00dfc67626ea00a2d926324686b501707a05177047cc3881f6dddc6`), [`create embeddings.ipynb`](../dev/notebooks/create%20embeddings.ipynb) (`c032878aa33fd9a9db0fae69a01f25a919d0c001e6221cfb15c2e196c77a7a96`), and [`EDA + data preprocessing.ipynb`](../dev/notebooks/EDA%20%2B%20data%20preprocessing.ipynb) (`a08391a7161e23e65b9da87dbdefb25e2ccbdabb836b86e1f44ab864617ae9d0`). That path summed GloVe vectors from serialized metadata, used cosine candidates, restricted them through 20-cluster KMeans over 41 binary genre indicators with random state 22, then sorted by Score and year. ARB-018 documents repairs for stable MAL ID alignment, self exclusion, ties, null handling, explicit KMeans settings and dependencies, and the original preprocessing exclusions. It describes this as a repaired comparison method because those changes prevent a claim of exact pipeline reproduction.

No quality threshold or improvement claim is set by this protocol.
