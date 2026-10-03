# ARB-018 assessor method and custody

`judgments.json` is one complete **LLM cohort** assessment, declared as `arb018-gpt56sol-high-cohort-v1`. The method is `sequential-blinded-catalog-v1`: seven fresh sequential `gpt-5.6-sol` sessions at high reasoning effort, each judging a disjoint query partition. It is neither a human assessment nor seven independent complete ratings. Every session received the same packaged prompt (`arb018-catalog-evidence-v1`, SHA-256 `3612c73634b0d687210ed6030c4ca0846b37aebeb645af71076fe8d1741dbada`) and only its catalog subset from packet SHA-256 `dad77d65536c4983bee66881988f7631b3843acabe091f240539df15f795e63e`.

The blinded packet has 268 unique query–candidate pairs. It excludes method names, ranks, algorithm scores, MAL ratings, popularity, and result frequency. Each session used the supplied title, aliases, genres, type, episodes, year, and synopsis. The rubric asks for theme, tone, narrative, setting, or character evidence; permits relevance 0–2 or null; and treats franchise relationship as a separate uncertain annotation. Every scored pair cites exact catalog excerpts from both anime. The strict validator checks pair coverage, side/field citation shape, verbatim excerpts, the 240-character quote limit, and prompt/packet hashes.

| Batch | Query MAL IDs | Pairs | Date | Custody |
| --- | --- | ---: | --- | --- |
| 01 | 1, 20, 30 | 37 | 2026-09-27 | Recovered exact accepted bytes |
| 02 | 111, 199, 326 | 39 | 2026-09-27 | Recovered exact accepted bytes |
| 03 | 440, 457, 877 | 39 | 2026-09-27 | Recovered exact accepted bytes |
| 04 | 885, 1065, 1391 | 43 | 2026-09-27 | Recovered exact accepted bytes |
| 06 | 5680, 10162, 16664 | 42 | 2026-09-27 | Recovered exact accepted bytes |
| 05 | 1535, 1550, 3258 | 43 | 2026-10-03 | Resumed assessor output |
| 07 | 35102, 39619 | 25 | 2026-10-03 | Resumed assessor output |

Execution order was 01, 02, 03, 04, 06, 05, 07. Temporary files were cleared on an environment restart. The first five completed batch files were recovered from saved response payloads and checked against their accepted SHA-256 hashes before aggregation. Batch 01 first arrived as a full JSON message and was then saved as a file. Batch 02 was written by the assessor as `response.json`, with path, SHA-256, and pair count returned to the coordinator. The coordinator saved batch 03 after an assessor child-write was rejected by automatic approval review. Later batches used file output. The batch 04 assessor corrected one citation exceeding 240 characters before final validation. In batch 05, the same assessor converted plain-string evidence to required `side`/`field`/`quote` objects after a strict schema rejection; relevance, rationale, and franchise decisions were preserved. Batch 07 received the citation-object schema up front. The coordinator made no relevance edits. [batch-provenance.json](batch-provenance.json) records batch hashes, pair counts, null counts, and order.

The final file contains 268 unique pair judgments, including 18 null relevance grades, and has SHA-256 `daa8f48af717bb7977da9d3ab59674eb915feee654a4f506cee0da2487f849ec`. The comparison, catalog, measurements, packet regeneration, and all judgments passed `recsys quality summarize` validation. Session boundaries, catalog-only evidence, and the lack of a human cross-check limit interpretation. A franchise flag is an assessor opinion, not a verified identifier.
