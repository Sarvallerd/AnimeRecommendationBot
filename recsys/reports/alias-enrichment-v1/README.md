# ARB-030 — pinned title alias enrichment

The selected source is the version-1 `anime_info.csv` artifact in Nafiul's Kaggle dataset `nafiulislam490/comprehensive-myanimelist-mal-dataset-2026`: 22,314,208 bytes, SHA-256 `e833476c0da3eadd4c033889e1e8defb2808baf95b0ae38b55009f45d5d72c35`. It is used only to propose title aliases for the already accepted Neelagiri dataset, whose manifest identity is `sha256:78bb29bfe18a00519b5dcd06e9682e1c4c84528166f0c51a80d24ea28c7a62b6`. The exact metadata collection date is unknown. The source declares CC BY-NC-SA 4.0, so the workflow is scoped to noncommercial research.

The pinned QA covers 29,816 source rows and 29,671 valid distinct MAL_IDs. There are 134 globally duplicated ID groups (279 rows); 26 groups (53 rows) overlap the base and are entirely excluded. The unambiguous overlap contains 28,558 of 28,955 base titles. Of those, 12,754 receive 12,911 alias strings: 12,688 English and 223 source-main strings, with 157 titles receiving two. The source has 979 unique IDs outside the base, 371 base IDs absent from the source, and no malformed MAL_ID or URL rows. These are fixed acceptance expectations rather than estimates of recommendation accuracy.

On the eligible overlap, 1 English field and 3 source-main fields contain forbidden format controls. No other field category is quarantined there. English equals the base primary title for 1,948 items and a base alias for 63; those English strings are retained intentionally as the preferred display role. Seven emitted English strings fail the current runtime Latin display predicate. The 14,199 nonempty synonym fields on eligible rows are ignored. All counts, including the complete source and eligible field scopes, are pinned in the packaged expectations resource and independently checked by the validator.

The frozen full-data A/A acceptance passed on the reviewed producer SHA-256 `64b702f8d4a6c9ec86b24db16bc290adfb111dbe528e53c9e4566116994de6b6` (Python 3.12.3, Unicode 15.0.0). Both normalization runs produced the same release identity, `sha256:ffa80087fc9d8bace4fe5517d7aa0fa7ba98bfed365988396b32828ec75a2403`, and all three published files match byte for byte:

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `aliases.json` | 1,002,900 | `ac5fd1a84a4dee3e88496a49a1137c6f3af7bc763d54c4c23aa156f6974d1761` |
| `alias-qa.json` | 1,759 | `5aeaaf2c3c155875dee911915b38f23d66182e265cab173043701cb9a604ff00` |
| `alias-manifest.json` | 1,788 | `ffa80087fc9d8bace4fe5517d7aa0fa7ba98bfed365988396b32828ec75a2403` |

Each normalization includes full base validation. Run B took 229.53 seconds and 273,060 KiB peak RSS; its separate published-byte validation took 228.94 seconds and 273,100 KiB. Run C took 225.50 seconds and 273,220 KiB; its completed independent validation took 234.34 seconds and 273,120 KiB. The first validation attempt for run C was interrupted without a completion receipt, so its result is unknown; the recorded 234.34-second result is from a fresh, successful validation bound to the release identity. An earlier diagnostic run A used a different producer fingerprint and is excluded from the accepted A/A pair.

The pinned Nafiul file and source manifest, all five accepted base files, registry, expectations, and installed and checkout producer bytes matched their frozen SHA-256 and size after A/A. Published directories had mode 0700 and files mode 0600. The installed test suite passed 113 tests, including 12 focused alias tests; setup and diff checks passed. This stage does not select a production bundle or establish live Telegram behavior.
