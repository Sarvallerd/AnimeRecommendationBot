# Recommendation bundle contract (v1)

The offline Python exporter writes `catalog.json`, `neighbors.json`, and `manifest.json`. See [the export guide](../recsys/EXPORT.md) for the pinned report checks and immutable publication command. The Rust bot loads these three files as one bundle. `tests/fixtures/bundle/source.json` is a small synthetic input used only to demonstrate provenance; the runtime does not need it.

Validate a bundle with:

```sh
python3 contracts/check_bundle.py tests/fixtures/bundle
python3 -m unittest discover -s contracts -p 'test_*.py'
```

The checker prints the bundle identity and catalog record count. It uses only Python's standard library. The three Draft 2020-12 schemas document local structure; the checker also enforces cross-file rules, exact bytes, and hashes. Producers and runtime loaders must enforce the same rules without requiring a JSON Schema engine.

## Files and IDs

Every root has integer `schema_version: 1` and exactly the fields shown in its schema. Unknown fields are rejected except within `algorithm.parameters` and source `metadata`, which are free JSON objects. A `MAL_ID` is an integer from 1 through 2147483647. Catalog and neighbor map keys are its canonical decimal string: no sign, leading zero, or whitespace.

`catalog.json` contains a nonempty `anime` map. Each entry has all eight fields. `title` is nonblank. `aliases` and `genres` are ordered arrays of unique, nonblank strings. `score` is a finite number from 1 through 10 or null; `year` is an integer from 1 through 9999 or null; `episodes` is a positive integer or null. `type` and `synopsis` are nonblank strings or null. Use null or an empty array for unknown metadata, never a placeholder string. Titles need not be unique. Consumers preserve Unicode and the written array order; they do not normalize names.

`neighbors.json` has exactly the catalog's source IDs, including sources with no recommendations. Each source list has at most five entries. An entry has a target `mal_id` and a finite cosine similarity in `(0, 1]`. The target must exist in the catalog, differ from the source, and appear only once per list. Lists are ordered by descending numeric similarity, then ascending numeric `MAL_ID` for ties. A recommendation's rank is its zero-based list index plus one. The producer excludes zero-vector sources, zero-vector candidates, and nonpositive similarities; it clamps tiny floating-point overshoots to 1 before serialization. The loader rejects any out-of-range value. There is no personalization, genre filter, or score/year reranking.

`manifest.json` names the `glove-mean-cosine` algorithm, its producer version, five maximum neighbors, cosine similarity, and free parameters. Consumers accept an unfamiliar nonblank producer version when the v1 wire format validates. Its nonempty `sources` array lists every consumed input with a raw-byte SHA-256 digest, sorted by `(name, version)` with unique pairs. Source metadata is free JSON. `files` has exactly `catalog.json` and `neighbors.json`, each with its raw-byte SHA-256 digest. All hashes are lowercase 64-character hex strings. The manifest contains no timestamp, machine path, random ID, or self-hash. Its hashes detect corruption; they do not authenticate a producer.

## Canonical bytes and identity

All three bundle files use UTF-8 without a BOM and end with one LF. The exact Python reference serialization is:

```python
(json.dumps(value, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
```

This is a project convention, not RFC 8785. Object keys, including ID keys, are sorted lexicographically; arrays retain their defined order. Duplicate JSON keys, invalid UTF-8, NaN, and infinity are invalid. Required integers must be JSON integers, while scores and similarities may be any finite JSON number. Hash the raw bytes, including the final LF. A Rust loader hashes bytes as read and need not reserialize floating-point numbers.

A consumer computes the full bundle identity as `sha256:` followed by the SHA-256 hex digest of the raw `manifest.json` bytes. The identity is not stored inside the manifest. Persist the full identity with recommendation feedback events so a later analysis can identify the exact recommendation bundle. A valid content or provenance change changes the identity.

The synthetic fixture has IDs `1`, `2`, `3`, `10`, `11`, `12`, `20`, and `99`. Its source vectors exercise cosine ties and numeric ID tie-breaking, a duplicate title, multilingual aliases, full and unknown metadata, negative similarity exclusion, and a zero vector with no neighbors. `source.json` is itself canonical and its actual raw-byte hash appears in the manifest.
