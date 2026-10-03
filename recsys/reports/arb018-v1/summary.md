# ARB-018 comparison

The fixed, purposive set has 20 queries. Each method ranks against its stated universe.
Returned results fill up to five slots per query; missing slots contribute zero. Unknown returned judgments stay unknown.
Bounds are arithmetic uncertainty bounds, not confidence intervals. No general quality threshold was set.

## Methods and source provenance

| Method | Version | Universe | Universe ID SHA-256 | Parameter SHA-256 | Returned slots | Unavailable | Short | Status counts |
| --- | --- | ---: | --- | --- | ---: | ---: | ---: | --- |
| glove-selected | arb009-v1 | 17562 | `8eb8a645d1e36164b95605458f22116ff6d93c203c58057973f9c3b93e790711` | `c5f9043ef8f13f5b6b951b472d06758b1a340ef40ad6bca015c6e011bac83dfd` | 100/100 | 0 | 0 | `{"ok": 20}` |
| genre-jaccard | 1 | 17562 | `8eb8a645d1e36164b95605458f22116ff6d93c203c58057973f9c3b93e790711` | `1fd29b5c1aa35444d11603ce6fef60a72df38feea3ef974aed68a08257306f91` | 100/100 | 0 | 0 | `{"ok": 20}` |
| legacy-repaired | arb018-legacy-v1 | 10882 | `6a621eed1bf9a089ac57f1a4370a3efdcf4e042b56826c632d753e44c062c6cd` | `79cfcf3078326f4d88429ba3b9b4997c4886f9e2ad2412012cf341a7c072b7ac` | 95/100 | 1 | 0 | `{"legacy_exclusion": 1, "ok": 19}` |

- glove-selected parameters: `{"k":5,"positive_only":true,"reduction":"numpy.einsum_ik_jk_ij_optimize_false","tie_break":"numeric_MAL_ID_ascending","universe":"normalized_catalog"}`.
- genre-jaccard parameters: `{"k":5,"positive_only":true,"tie_break":"numeric_MAL_ID_ascending","universe":"normalized_catalog"}`.
- legacy-repaired parameters: `{"feature_order":["Action","Military","Adventure","Fantasy","Music","Romance","Shoujo","Dementia","Psychological","Drama","Shounen Ai","Comedy","Demons","Ecchi","School","Parody","Shounen","Historical","Seinen","Mystery","Sci-Fi","Space","Horror","Martial Arts","Samurai","Mecha","Supernatural","Thriller","Magic","Sports","Harem","Vampire","Game","Super Power","Shoujo Ai","Kids","Police","Slice of Life","Yaoi","Josei","Cars"],"k":5,"kmeans":{"algorithm":"lloyd","copy_x":true,"init":"k-means++","max_iter":300,"n_clusters":20,"n_init":10,"random_state":22,"tol":0.0001,"verbose":0},"same_cluster":true,"score_year_sort":true,"top_cosine":99,"universe":"notebook_eligible"}`.

Query set `mal2020-stratified20-v1` SHA-256: `6a05478fb76027dfb5169afca6913df35796ee8d7d25432313d8ee70d4860153`.
Catalog SHA-256: `2ae604168ba10be67cca14cd8cba0235fc6f0d0de19fdc25b9fc91175ffd06d2`; legacy diagnostics SHA-256: `c5cad4f90b2618fc90eee3d7add7d7d4d89d21c55af2e90efd253cd259823a27`.
Code revision: `5a55469b5d6dafefcb2eff6adc0f3b681f51dd1a`; lockfile SHA-256: `7a357fe8fecf26d8409c6c07091c364a7a78e9b0a33bcd4c4466d3524c76ea9e`.
Registry SHA-256: `49bb399935f0ac7ec649fe2f4bd7715444c09c8513704866fa90b9bea29d182f`; bundle identity: `sha256:4a862f27b76afe86c467af380481217b2aec9176e38038bf0445eb10c3a9a57c`.
Manifest SHA-256: `4a862f27b76afe86c467af380481217b2aec9176e38038bf0445eb10c3a9a57c`; neighbors SHA-256: `fa4b7471f580c9cdede60550ea35df5a360833a31e6046918901737d166b713f`.
Normalization report SHA-256: `59595041f45ecfb113c2b878de68010e1f16e981ef814987aa5de7951bd1df1a`; build report SHA-256: `06e82b9bbebbcddbf012263771dd85e57c3d579a247ee0aedc58ced552be094a`.
Comparison SHA-256: `cd829aba0ed71465540b7926a073853b651775bf8260203dd5e44cfb2c4d9551`; measurements SHA-256: `2288bed4e79fe443ef8176b27fc5c10c0ee658a2b1316913266dcbb87b2332b4`; packet SHA-256: `dad77d65536c4983bee66881988f7631b3843acabe091f240539df15f795e63e`.

Raw sources (size in bytes, SHA-256):

- mal_anime: 5662362; `6e6d723f3e021e084a281d1ec557a0a10dfb9721d256b87596e02c27eb1f77e9`
- mal_synopsis: 7221844; `fc5065338f1d66063cc6b6ed328701c3e23118110689daa852d5876911674e33`
- glove_300d: 1037962819; `91125602f730fea7ca768736c6f442e668b49db095682bf2aad375db061c21ed`

Code file SHA-256 hashes:

- assessment.py: `9b4d4fa037cbe373e92da21c2b1c7de2d7f9bcdbb3d055816e2a4db8886773d0`
- assessment_prompt_v1.md: `3612c73634b0d687210ed6030c4ca0846b37aebeb645af71076fe8d1741dbada`
- build.py: `51106b320099c557f464f3122960f2c0dfe95972f9f313cda80c12e3d3d51745`
- bundle.py: `641315298c96313e6f16732d5661204a7b50f9006fad4db8fc2803a6dbf51f0a`
- cli.py: `08e920497403790da8706542020bcc94ca2950f28120562db5c779827d253716`
- evaluation.py: `b57e5d0d57bb11954ebc2998be7b21c0e6f37f1376e27d5f86c2b37d7da4a34a`
- export.py: `3722d4c884dc289aa1e9b39cbf76bfa4acadf83dea9b6221693f1e7f545f41f6`
- legacy.py: `55a568c6339fd5ecb9e8745a247e795d6c6e6b9da07f05a281fea80006a93a88`
- normalize.py: `af4a4d82eba5155a412d3d2b0e6fbdc69a69bf7f832591408226cc91992e5d13`
- quality.py: `ba995f633b54463aff2da9c48bca8854e994d3ad7a0f2c925ee86a33fe9b0ea4`
- sources.py: `def119a032f8e5e431d9eeac982712f0963602197613e22dfbe4dc84412d279e`

## Measurement context

Scope: `fresh-process-prepare-and-rank20-v1`; workload: 20 queries, universe counts `{"genre-jaccard": 17562, "glove-selected": 17562, "legacy-repaired": 10882}`.
Machine: `{"cpu_count": 20, "machine": "x86_64", "platform": "Linux-6.6.87.1-microsoft-standard-WSL2-x86_64-with-glibc2.39", "processor": "x86_64"}`.
Runtime: `{"executable": "/home/wozata/projects/AnimeRecommendationBot/.worktrees/arb-018/recsys/.venv/bin/python3", "implementation": "CPython", "python": "3.12.3"}`.
Native thread environment: `{"MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"}`.
Each worker rechecks the bundle, catalog, manifest, build and normalization reports, and both MAL CSV hashes, headers and IDs during measured preparation. This shared validation cost is included even for genre Jaccard.
Selected GloVe and repaired legacy workers stream and hash the complete GloVe file. Genre Jaccard checks only its file size; it does not read GloVe contents.
Each sample starts a fresh process, prepares its method, and ranks the same 20 queries. The OS page cache, background host activity, and CPU scheduling were uncontrolled on this shared development machine; full production neighbor generation was outside scope.

## Performance

- glove-selected:
  - end_to_end_wall_ns (ns): samples [10686752321, 10659847127, 10717534253]; median 10686752321; min 10659847127; max 10717534253.
  - prepare_wall_ns (ns): samples [10535404908, 10508584798, 10565399162]; median 10535404908; min 10508584798; max 10565399162.
  - rank20_wall_ns (ns): samples [57684639, 58635693, 57159026]; median 57684639; min 57159026; max 58635693.
  - peak_rss_bytes (bytes): samples [237015040, 237355008, 237187072]; median 237187072; min 237015040; max 237355008.
  - Effective native threads per sample (each detected pool): [[1], [1], [1]].
- genre-jaccard:
  - end_to_end_wall_ns (ns): samples [2317169462, 2195884091, 2191903402]; median 2195884091; min 2191903402; max 2317169462.
  - prepare_wall_ns (ns): samples [1724903411, 1625289948, 1620619958]; median 1625289948; min 1620619958; max 1724903411.
  - rank20_wall_ns (ns): samples [252885644, 242879245, 244100860]; median 244100860; min 242879245; max 252885644.
  - peak_rss_bytes (bytes): samples [187002880, 186740736, 186998784]; median 186998784; min 186740736; max 187002880.
  - Effective native threads per sample (each detected pool): [[], [], []].
- legacy-repaired:
  - end_to_end_wall_ns (ns): samples [14129277517, 13596009638, 13646407909]; median 13646407909; min 13596009638; max 14129277517.
  - prepare_wall_ns (ns): samples [13745667737, 13228669506, 13272529328]; median 13272529328; min 13228669506; max 13745667737.
  - rank20_wall_ns (ns): samples [165309382, 159356485, 159136000]; median 159356485; min 159136000; max 165309382.
  - peak_rss_bytes (bytes): samples [431726592, 431767552, 432340992]; median 431767552; min 431726592; max 432340992.
  - Effective native threads per sample (each detected pool): [[1, 1, 1], [1, 1, 1], [1, 1, 1]].

## Provenance and limits

Catalog SHA-256: `2ae604168ba10be67cca14cd8cba0235fc6f0d0de19fdc25b9fc91175ffd06d2`; measurements SHA-256: `2288bed4e79fe443ef8176b27fc5c10c0ee658a2b1316913266dcbb87b2332b4`.
The raw MAL 2020 CSVs were verified by every worker; GloVe bytes were fully verified by selected and legacy workers. The original fitted clusters and vectors were unavailable, so the legacy method is a repaired comparator.
MAL ID 35102 is excluded from the legacy universe because Aired is missing. MAL ID 39619 retains raw boilerplate synopsis text there, while its normalized catalog synopsis is null.
Assessors use only supplied catalog evidence; missing synopsis text and LLM uncertainty can widen reported bounds. These data do not establish general recommendation quality.

## Reproduction

Run `recsys quality compare` with the pinned bundle, normalization and build reports, raw CSVs, GloVe file, lockfile and code revision, then `recsys quality summarize` with complete declared assessment files. Exact CLI forms are in `recsys/EVALUATION.md`.


## Assessor arb018-gpt56sol-high-cohort-v1 (llm)

Method version: `sequential-blinded-catalog-v1`. Model: `gpt-5.6-sol`. Prompt: `arb018-catalog-evidence-v1`.

### glove-selected

Returned 100/100; assessed 100; coverage 1.0; full-five queries 20.
Mean grade@5 1.29 bounds [1.29, 1.29]; precision@5 0.8699999999999999 bounds [0.8699999999999999, 0.8699999999999999].
Assessor-judged same-franchise@5 None bounds [0.2, 0.21000000000000002]; franchise coverage 0.99.

| Query MAL_ID | Status | Returned | Assessed | Unknown | Mean grade@5 | Grade bounds | Precision@5 | Precision bounds | Franchise@5 | Franchise bounds |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | --- | ---: | --- |
| 1 | ok | 5 | 5 | 0 | 1.0 | [1.0, 1.0] | 0.8 | [0.8, 0.8] | 0.0 | [0.0, 0.0] |
| 20 | ok | 5 | 5 | 0 | 1.6 | [1.6, 1.6] | 0.8 | [0.8, 0.8] | 0.6 | [0.6, 0.6] |
| 30 | ok | 5 | 5 | 0 | 1.2 | [1.2, 1.2] | 0.8 | [0.8, 0.8] | 0.4 | [0.4, 0.4] |
| 199 | ok | 5 | 5 | 0 | 1.6 | [1.6, 1.6] | 1.0 | [1.0, 1.0] | 0.0 | [0.0, 0.0] |
| 1535 | ok | 5 | 5 | 0 | 0.8 | [0.8, 0.8] | 0.6 | [0.6, 0.6] | 0.0 | [0.0, 0.0] |
| 457 | ok | 5 | 5 | 0 | 1.8 | [1.8, 1.8] | 1.0 | [1.0, 1.0] | 0.6 | [0.6, 0.6] |
| 440 | ok | 5 | 5 | 0 | 1.2 | [1.2, 1.2] | 1.0 | [1.0, 1.0] | 0.0 | [0.0, 0.0] |
| 5680 | ok | 5 | 5 | 0 | 1.8 | [1.8, 1.8] | 1.0 | [1.0, 1.0] | 0.6 | [0.6, 0.6] |
| 877 | ok | 5 | 5 | 0 | 1.0 | [1.0, 1.0] | 1.0 | [1.0, 1.0] | 0.0 | [0.0, 0.0] |
| 10162 | ok | 5 | 5 | 0 | 1.4 | [1.4, 1.4] | 1.0 | [1.0, 1.0] | 0.0 | [0.0, 0.0] |
| 1065 | ok | 5 | 5 | 0 | 0.8 | [0.8, 0.8] | 0.8 | [0.8, 0.8] | 0.0 | [0.0, 0.0] |
| 326 | ok | 5 | 5 | 0 | 1.6 | [1.6, 1.6] | 0.8 | [0.8, 0.8] | 0.0 | [0.0, 0.0] |
| 885 | ok | 5 | 5 | 0 | 1.0 | [1.0, 1.0] | 0.8 | [0.8, 0.8] | 0.0 | [0.0, 0.0] |
| 16664 | ok | 5 | 5 | 0 | 0.8 | [0.8, 0.8] | 0.8 | [0.8, 0.8] | 0.0 | [0.0, 0.0] |
| 111 | ok | 5 | 5 | 0 | 1.0 | [1.0, 1.0] | 0.8 | [0.8, 0.8] | 0.0 | [0.0, 0.0] |
| 35102 | ok | 5 | 5 | 0 | 1.4 | [1.4, 1.4] | 1.0 | [1.0, 1.0] | None | [0.0, 0.2] |
| 1391 | ok | 5 | 5 | 0 | 2.0 | [2.0, 2.0] | 1.0 | [1.0, 1.0] | 0.6 | [0.6, 0.6] |
| 3258 | ok | 5 | 5 | 0 | 0.4 | [0.4, 0.4] | 0.4 | [0.4, 0.4] | 0.0 | [0.0, 0.0] |
| 1550 | ok | 5 | 5 | 0 | 2.0 | [2.0, 2.0] | 1.0 | [1.0, 1.0] | 0.2 | [0.2, 0.2] |
| 39619 | ok | 5 | 5 | 0 | 1.4 | [1.4, 1.4] | 1.0 | [1.0, 1.0] | 1.0 | [1.0, 1.0] |

### genre-jaccard

Returned 100/100; assessed 86; coverage 0.86; full-five queries 20.
Mean grade@5 None bounds [1.16, 1.44]; precision@5 None bounds [0.8, 0.9400000000000001].
Assessor-judged same-franchise@5 None bounds [0.22000000000000003, 0.23000000000000004]; franchise coverage 0.99.

| Query MAL_ID | Status | Returned | Assessed | Unknown | Mean grade@5 | Grade bounds | Precision@5 | Precision bounds | Franchise@5 | Franchise bounds |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | --- | ---: | --- |
| 1 | ok | 5 | 5 | 0 | 1.6 | [1.6, 1.6] | 1.0 | [1.0, 1.0] | 0.2 | [0.2, 0.2] |
| 20 | ok | 5 | 4 | 1 | None | [1.4, 1.8] | None | [0.8, 1.0] | None | [0.2, 0.4] |
| 30 | ok | 5 | 5 | 0 | 2.0 | [2.0, 2.0] | 1.0 | [1.0, 1.0] | 0.8 | [0.8, 0.8] |
| 199 | ok | 5 | 2 | 3 | None | [0.8, 2.0] | None | [0.4, 1.0] | 0.0 | [0.0, 0.0] |
| 1535 | ok | 5 | 3 | 2 | None | [1.2, 2.0] | None | [0.6, 1.0] | 0.2 | [0.2, 0.2] |
| 457 | ok | 5 | 5 | 0 | 2.0 | [2.0, 2.0] | 1.0 | [1.0, 1.0] | 1.0 | [1.0, 1.0] |
| 440 | ok | 5 | 4 | 1 | None | [1.0, 1.4] | None | [0.8, 1.0] | 0.0 | [0.0, 0.0] |
| 5680 | ok | 5 | 5 | 0 | 1.6 | [1.6, 1.6] | 1.0 | [1.0, 1.0] | 0.6 | [0.6, 0.6] |
| 877 | ok | 5 | 5 | 0 | 1.4 | [1.4, 1.4] | 1.0 | [1.0, 1.0] | 0.2 | [0.2, 0.2] |
| 10162 | ok | 5 | 3 | 2 | None | [0.4, 1.2] | None | [0.4, 0.8] | 0.2 | [0.2, 0.2] |
| 1065 | ok | 5 | 5 | 0 | 1.2 | [1.2, 1.2] | 1.0 | [1.0, 1.0] | 0.2 | [0.2, 0.2] |
| 326 | ok | 5 | 4 | 1 | None | [1.0, 1.4] | None | [0.8, 1.0] | 0.0 | [0.0, 0.0] |
| 885 | ok | 5 | 5 | 0 | 0.8 | [0.8, 0.8] | 0.8 | [0.8, 0.8] | 0.0 | [0.0, 0.0] |
| 16664 | ok | 5 | 4 | 1 | None | [0.4, 0.8] | None | [0.4, 0.6] | 0.0 | [0.0, 0.0] |
| 111 | ok | 5 | 4 | 1 | None | [0.8, 1.2] | None | [0.8, 1.0] | 0.0 | [0.0, 0.0] |
| 35102 | ok | 5 | 5 | 0 | 0.8 | [0.8, 0.8] | 0.8 | [0.8, 0.8] | 0.0 | [0.0, 0.0] |
| 1391 | ok | 5 | 4 | 1 | None | [0.8, 1.2] | None | [0.8, 1.0] | 0.0 | [0.0, 0.0] |
| 3258 | ok | 5 | 4 | 1 | None | [1.0, 1.4] | None | [0.6, 0.8] | 0.0 | [0.0, 0.0] |
| 1550 | ok | 5 | 5 | 0 | 2.0 | [2.0, 2.0] | 1.0 | [1.0, 1.0] | 0.8 | [0.8, 0.8] |
| 39619 | ok | 5 | 5 | 0 | 1.0 | [1.0, 1.0] | 1.0 | [1.0, 1.0] | 0.0 | [0.0, 0.0] |

### legacy-repaired

Returned 95/100; assessed 91; coverage 0.9578947368421052; full-five queries 19.
Mean grade@5 None bounds [1.09, 1.17]; precision@5 None bounds [0.77, 0.8099999999999999].
Assessor-judged same-franchise@5 None bounds [0.1, 0.12]; franchise coverage 0.9789473684210527.

| Query MAL_ID | Status | Returned | Assessed | Unknown | Mean grade@5 | Grade bounds | Precision@5 | Precision bounds | Franchise@5 | Franchise bounds |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | --- | ---: | --- |
| 1 | ok | 5 | 5 | 0 | 1.6 | [1.6, 1.6] | 1.0 | [1.0, 1.0] | 0.0 | [0.0, 0.0] |
| 20 | ok | 5 | 5 | 0 | 1.6 | [1.6, 1.6] | 1.0 | [1.0, 1.0] | 0.4 | [0.4, 0.4] |
| 30 | ok | 5 | 5 | 0 | 1.6 | [1.6, 1.6] | 1.0 | [1.0, 1.0] | 0.6 | [0.6, 0.6] |
| 199 | ok | 5 | 5 | 0 | 1.0 | [1.0, 1.0] | 0.8 | [0.8, 0.8] | 0.0 | [0.0, 0.0] |
| 1535 | ok | 5 | 5 | 0 | 1.2 | [1.2, 1.2] | 1.0 | [1.0, 1.0] | 0.0 | [0.0, 0.0] |
| 457 | ok | 5 | 5 | 0 | 1.6 | [1.6, 1.6] | 1.0 | [1.0, 1.0] | 0.6 | [0.6, 0.6] |
| 440 | ok | 5 | 5 | 0 | 1.4 | [1.4, 1.4] | 1.0 | [1.0, 1.0] | 0.2 | [0.2, 0.2] |
| 5680 | ok | 5 | 5 | 0 | 1.2 | [1.2, 1.2] | 1.0 | [1.0, 1.0] | 0.2 | [0.2, 0.2] |
| 877 | ok | 5 | 5 | 0 | 1.0 | [1.0, 1.0] | 1.0 | [1.0, 1.0] | 0.0 | [0.0, 0.0] |
| 10162 | ok | 5 | 5 | 0 | 0.8 | [0.8, 0.8] | 0.6 | [0.6, 0.6] | 0.0 | [0.0, 0.0] |
| 1065 | ok | 5 | 5 | 0 | 1.0 | [1.0, 1.0] | 0.8 | [0.8, 0.8] | 0.0 | [0.0, 0.0] |
| 326 | ok | 5 | 5 | 0 | 1.4 | [1.4, 1.4] | 0.8 | [0.8, 0.8] | 0.0 | [0.0, 0.0] |
| 885 | ok | 5 | 5 | 0 | 1.0 | [1.0, 1.0] | 0.6 | [0.6, 0.6] | 0.0 | [0.0, 0.0] |
| 16664 | ok | 5 | 5 | 0 | 1.2 | [1.2, 1.2] | 0.8 | [0.8, 0.8] | 0.0 | [0.0, 0.0] |
| 111 | ok | 5 | 5 | 0 | 1.4 | [1.4, 1.4] | 1.0 | [1.0, 1.0] | 0.0 | [0.0, 0.0] |
| 35102 | legacy_exclusion | 0 | 0 | 0 | 0.0 | [0.0, 0.0] | 0.0 | [0.0, 0.0] | 0.0 | [0.0, 0.0] |
| 1391 | ok | 5 | 5 | 0 | 1.0 | [1.0, 1.0] | 0.8 | [0.8, 0.8] | 0.0 | [0.0, 0.0] |
| 3258 | ok | 5 | 5 | 0 | 0.2 | [0.2, 0.2] | 0.2 | [0.2, 0.2] | 0.0 | [0.0, 0.0] |
| 1550 | ok | 5 | 5 | 0 | 1.4 | [1.4, 1.4] | 0.8 | [0.8, 0.8] | 0.0 | [0.0, 0.0] |
| 39619 | ok | 5 | 1 | 4 | None | [0.2, 1.8] | None | [0.2, 1.0] | None | [0.0, 0.4] |
