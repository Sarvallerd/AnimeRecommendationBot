"""Build deterministic cosine neighbors from the pinned normalized MAL catalog and GloVe."""

import hashlib
import json
import math
import os
import platform
import re
import tempfile
from collections import Counter
from pathlib import Path

from . import __version__
from .sources import SourceError, load_registry, validate_registry

BUILDER_VERSION = "arb009-v1"
MAX_ID = 2_147_483_647
DIMENSION = 300
SCORE_BLOCK_BUDGET = 64 * 1024 * 1024
TOKEN_PATTERN = r"[a-z]+(?:'[a-z]+)?"
FIELDS = ("title", "aliases", "genres", "synopsis")
CATALOG_FIELDS = {"title", "aliases", "genres", "score", "year", "type", "episodes", "synopsis"}


class BuildError(ValueError):
    """The inputs cannot produce a valid recommendation artifact."""


def _canonical(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode("utf-8")


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _unique(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise BuildError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def _nonfinite(value):
    raise BuildError(f"nonfinite JSON constant {value}")


def _read_canonical(path):
    raw = Path(path).read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique,
                           parse_constant=_nonfinite)
        if raw != _canonical(value):
            raise BuildError(f"noncanonical JSON: {path}")
    except (UnicodeError, json.JSONDecodeError, OverflowError, ValueError) as exc:
        raise BuildError(f"invalid canonical JSON {path}: {exc}") from exc
    return value, raw


def _positive_id(key):
    return (isinstance(key, str) and re.fullmatch(r"[1-9][0-9]*", key) is not None
            and len(key) <= 10 and int(key) <= MAX_ID)


def _catalog(value):
    if not isinstance(value, dict) or set(value) != {"schema_version", "anime"} or type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise BuildError("catalog must have exactly schema_version 1 and anime")
    anime = value["anime"]
    if not isinstance(anime, dict) or not anime:
        raise BuildError("catalog anime must be a nonempty object")
    for key, item in anime.items():
        if not _positive_id(key) or not isinstance(item, dict) or set(item) != CATALOG_FIELDS:
            raise BuildError(f"invalid catalog MAL_ID or fields: {key!r}")
        if not isinstance(item["title"], str) or not item["title"].strip():
            raise BuildError(f"invalid title: {key}")
        for field in ("aliases", "genres"):
            entries = item[field]
            if (not isinstance(entries, list) or any(not isinstance(x, str) or not x.strip() for x in entries)
                    or len(entries) != len(set(entries))):
                raise BuildError(f"invalid {field}: {key}")
        score = item["score"]
        if score is not None and (type(score) not in (int, float) or not 1 <= score <= 10
                                  or (type(score) is float and not math.isfinite(score))):
            raise BuildError(f"invalid score: {key}")
        for field, maximum in (("year", 9999), ("episodes", None)):
            x = item[field]
            if x is not None and (type(x) is not int or x < 1 or (maximum is not None and x > maximum)):
                raise BuildError(f"invalid {field}: {key}")
        for field in ("type", "synopsis"):
            x = item[field]
            if x is not None and (not isinstance(x, str) or not x.strip()):
                raise BuildError(f"invalid {field}: {key}")
    return anime


def _report(value, registry, registry_sha, catalog_sha, count):
    descriptors = {entry["id"]: entry for entry in registry["sources"]}
    if (not isinstance(value, dict) or type(value.get("schema_version")) is not int
            or value["schema_version"] != 1 or value.get("normalizer_version") != "1"
            or value.get("registry_sha256") != registry_sha
            or value.get("provenance") != registry["provenance"]
            or value.get("sources") != [descriptors["mal_anime"], descriptors["mal_synopsis"]]
            or not isinstance(value.get("files"), dict)
            or value["files"].get("catalog.json") != {"sha256": catalog_sha}
            or not isinstance(value.get("counts"), dict)
            or type(value["counts"].get("retained")) is not int
            or value["counts"]["retained"] != count):
        raise BuildError("normalization report does not match pinned MAL sources, registry, or catalog")


def tokenize(text: str) -> list[str]:
    """Extract lowercase ASCII words with one optional internal apostrophe."""
    return re.findall(TOKEN_PATTERN, text.lower())


def _tokens(entry):
    for field in FIELDS:
        values = entry[field] if field in ("aliases", "genres") else [entry[field]]
        for value in values:
            if value is not None:
                yield from tokenize(value)


def load_embeddings(path: Path, vocabulary: set[str], descriptor: dict) -> tuple[dict, dict]:
    """Hash all GloVe bytes; parse only requested 300-dimensional vectors."""
    import numpy as np

    vectors = {}
    digest = hashlib.sha256()
    size = 0
    with Path(path).open("rb") as stream:
        for line_number, raw in enumerate(stream, 1):
            digest.update(raw)
            size += len(raw)
            try:
                line = raw.decode("utf-8")
            except UnicodeError as exc:
                raise BuildError(f"GloVe line {line_number}: invalid UTF-8") from exc
            parts = line.split()
            if len(parts) < 2:
                raise BuildError(f"GloVe line {line_number}: structurally empty line")
            word = parts[0]
            if word not in vocabulary:
                continue
            if word in vectors:
                raise BuildError(f"GloVe line {line_number}: duplicate requested token {word!r}")
            if len(parts) != DIMENSION + 1:
                raise BuildError(f"GloVe line {line_number}: expected {DIMENSION} components for {word!r}")
            try:
                vector = np.array([float(component) for component in parts[1:]], dtype=np.float64)
            except (ValueError, OverflowError) as exc:
                raise BuildError(f"GloVe line {line_number}: invalid numeric component for {word!r}") from exc
            if not np.isfinite(vector).all():
                raise BuildError(f"GloVe line {line_number}: nonfinite component for {word!r}")
            vectors[word] = vector
    observed = digest.hexdigest()
    if type(descriptor.get("size")) is not int or descriptor.get("sha256") != observed or descriptor["size"] != size:
        raise BuildError(f"GloVe size/SHA256 mismatch: observed {size} bytes, {observed}")
    return vectors, {"sha256": observed, "size": size}


def build_vectors(anime: dict[str, dict], embeddings: dict) -> tuple[list[int], object, dict]:
    """Mean known token occurrences and normalize each anime vector in float64."""
    import numpy as np

    if not anime or any(not _positive_id(key) for key in anime):
        raise BuildError("invalid or empty MAL_ID set")
    for word, vector in embeddings.items():
        if not isinstance(word, str) or not isinstance(vector, np.ndarray) or vector.shape != (DIMENSION,) or not np.isfinite(vector).all():
            raise BuildError(f"invalid embedding for {word!r}")
    ids = sorted(int(key) for key in anime)
    vectors = np.zeros((len(ids), DIMENSION), dtype=np.float64)
    details = {}
    vocabulary = set()
    occurrences = 0
    matched_occurrences = 0
    statuses = Counter()
    for index, mal_id in enumerate(ids):
        entry = anime[str(mal_id)]
        tokens = list(_tokens(entry))
        count = len(tokens)
        vocabulary.update(tokens)
        occurrences += count
        matched = 0
        for token in tokens:
            vector = embeddings.get(token)
            if vector is not None:
                vectors[index] += vector
                matched += 1
        matched_occurrences += matched
        if matched:
            vectors[index] /= matched
            norm = np.linalg.norm(vectors[index])
            if not np.isfinite(norm):
                raise BuildError(f"nonfinite vector norm for MAL_ID {mal_id}")
            if norm:
                vectors[index] /= norm
            if not np.isfinite(vectors[index]).all():
                raise BuildError(f"nonfinite normalized vector for MAL_ID {mal_id}")
        status = "no_tokens" if not count else "oov_only" if not matched else "zero_vector" if not np.any(vectors[index]) else "ready"
        statuses[status] += 1
        details[str(mal_id)] = {"token_count": count, "matched_token_count": matched,
                                "vector_status": status}
    coverage = {"catalog_count": len(ids), "vocabulary_count": len(vocabulary),
                "matched_vocabulary_count": len(vocabulary & embeddings.keys()),
                "token_occurrences": occurrences, "matched_token_occurrences": matched_occurrences,
                "vector_status_counts": {key: statuses[key] for key in ("no_tokens", "oov_only", "zero_vector", "ready")},
                "zero_vector_ids": [mal_id for mal_id in ids if details[str(mal_id)]["vector_status"] != "ready"]}
    return ids, vectors, {"coverage": coverage, "anime": details}


def _effective_rows(count, block_size):
    if type(block_size) is not int or not 1 <= block_size <= 256:
        raise BuildError("block_size must be an integer in 1..256")
    rows = SCORE_BLOCK_BUDGET // (8 * count)
    if not rows:
        raise BuildError("catalog exceeds one-row similarity block budget")
    return min(block_size, count, rows)


def cosine_neighbors(mal_ids: list[int], vectors, *, block_size: int = 256) -> dict[str, list[dict]]:
    """Compute exact deterministic top five positive cosine scores in bounded blocks."""
    import numpy as np

    count = len(mal_ids)
    if (not count or any(type(x) is not int or x < 1 or x > MAX_ID for x in mal_ids)
            or len(set(mal_ids)) != count or mal_ids != sorted(mal_ids)):
        raise BuildError("MAL_IDs must be unique, positive, bounded, and numerically sorted")
    if (not isinstance(vectors, np.ndarray) or vectors.shape != (count, DIMENSION)
            or vectors.dtype != np.float64 or not np.isfinite(vectors).all()):
        raise BuildError("vectors must be a finite float64 matrix with one 300D row per MAL_ID")
    rows = _effective_rows(count, block_size)
    nonzero = np.any(vectors != 0, axis=1)
    ids_array = np.asarray(mal_ids, dtype=np.int64)
    result = {}
    for start in range(0, count, rows):
        scores = np.einsum("ik,jk->ij", vectors[start:start + rows], vectors, optimize=False)
        if not np.isfinite(scores).all():
            raise BuildError("nonfinite cosine similarity")
        if np.any(scores > 1 + 1e-12):
            raise BuildError("cosine similarity exceeds overshoot tolerance")
        np.minimum(scores, 1.0, out=scores)
        for local_index, row in enumerate(scores):
            index = start + local_index
            source_id = mal_ids[index]
            if not nonzero[index]:
                result[str(source_id)] = []
                continue
            row[index] = 0.0
            row[~nonzero] = 0.0
            candidates = np.flatnonzero(row > 0)
            if len(candidates) > 5:
                threshold = np.partition(row[candidates], -5)[-5]
                candidates = candidates[row[candidates] >= threshold]
            order = np.lexsort((ids_array[candidates], -row[candidates]))
            result[str(source_id)] = [{"mal_id": int(ids_array[target]), "similarity": float(row[target])}
                                      for target in candidates[order[:5]]]
    return result


def _source(name, version, digest, metadata):
    return {"name": name, "version": version, "sha256": digest, "metadata": metadata}


def _sources(registry, registry_sha, catalog_sha, report_sha, glove_info):
    descriptors = {entry["id"]: entry for entry in registry["sources"]}
    mal = registry["provenance"]["mal"]
    glove = registry["provenance"]["glove"]
    sources = [
        _source("glove_300d", "6B-300d", glove_info["sha256"],
                {"filename": descriptors["glove_300d"]["filename"], "size": glove_info["size"],
                 "archive_sha256": descriptors["glove_zip"]["sha256"], "member": descriptors["glove_300d"]["member"],
                 "project": glove["project"], "license": glove["pretrained_vectors_license"]}),
        _source("mal_anime", mal["commit"], descriptors["mal_anime"]["sha256"],
                {"filename": descriptors["mal_anime"]["filename"], "size": descriptors["mal_anime"]["size"],
                 "url": descriptors["mal_anime"]["url"], "repository": mal["repository"]}),
        _source("mal_synopsis", mal["commit"], descriptors["mal_synopsis"]["sha256"],
                {"filename": descriptors["mal_synopsis"]["filename"], "size": descriptors["mal_synopsis"]["size"],
                 "url": descriptors["mal_synopsis"]["url"], "repository": mal["repository"]}),
        _source("normalized_catalog", "normalizer-1", catalog_sha, {"filename": "catalog.json"}),
        _source("normalization_report", "1", report_sha, {"filename": "normalization-report.json"}),
        _source("source_registry", "1", registry_sha, {"filename": "source_registry.json"}),
    ]
    return sorted(sources, key=lambda item: (item["name"], item["version"]))


def _atomic(path, content):
    temp = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temp = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        temp.replace(path)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def build(catalog_path, normalization_report_path, glove_path, output_dir, *,
          block_size=256, registry=None, registry_sha256=None) -> Path:
    """Build neighbors and deterministic report after validating every pinned input."""
    import numpy as np

    if registry is None:
        registry, registry_sha256 = load_registry()
    else:
        validate_registry(registry)
        if registry_sha256 is None:
            registry_sha256 = _sha(_canonical(registry))
    descriptors = {entry["id"]: entry for entry in registry["sources"]}
    if not {"mal_anime", "mal_synopsis", "glove_300d", "glove_zip"} <= descriptors.keys():
        raise BuildError("source registry lacks required descriptors")
    catalog, catalog_raw = _read_canonical(catalog_path)
    anime = _catalog(catalog)
    report, report_raw = _read_canonical(normalization_report_path)
    catalog_sha, report_sha = _sha(catalog_raw), _sha(report_raw)
    _report(report, registry, registry_sha256, catalog_sha, len(anime))
    vocabulary = {token for entry in anime.values() for token in _tokens(entry)}
    embeddings, glove_info = load_embeddings(Path(glove_path), vocabulary, descriptors["glove_300d"])
    ids, vectors, diagnostics = build_vectors(anime, embeddings)
    del embeddings, vocabulary
    neighbors = cosine_neighbors(ids, vectors, block_size=block_size)
    del vectors
    for key, entry in diagnostics["anime"].items():
        entry["neighbor_count"] = len(neighbors[key])
    coverage = diagnostics["coverage"]
    histogram = Counter(len(items) for items in neighbors.values())
    coverage["neighbor_count_histogram"] = {str(key): histogram[key] for key in range(6)}
    neighbors_raw = _canonical({"schema_version": 1, "neighbors": neighbors})
    effective = _effective_rows(len(ids), block_size)
    algorithm = {"name": "glove-mean-cosine", "version": BUILDER_VERSION, "max_neighbors": 5,
                 "similarity": "cosine", "parameters": {
                     "field_order": list(FIELDS), "tokenizer": TOKEN_PATTERN,
                     "tokenizer_version": "arb009-v1", "lowercasing": "unicode-str.lower",
                     "pooling": "arithmetic_mean_known_occurrences", "token_weight": 1,
                     "dimension": DIMENSION, "dtype": "float64", "normalization": "L2",
                     "reduction": "numpy.einsum_ik_jk_ij_optimize_false",
                     "positive_only": True, "overshoot_tolerance": 1e-12,
                     "tie_break": "numeric_MAL_ID_ascending"}}
    code_dir = Path(__file__).parent
    build_report = {
        "schema_version": 1, "builder_version": BUILDER_VERSION,
        "algorithm": algorithm,
        "sources": _sources(registry, registry_sha256, catalog_sha, report_sha, glove_info),
        "registry_sha256": registry_sha256,
        "inputs": {"catalog.json": {"sha256": catalog_sha},
                   "normalization-report.json": {"sha256": report_sha},
                   "glove.6B.300d.txt": glove_info},
        "files": {"catalog.json": {"sha256": catalog_sha},
                  "neighbors.json": {"sha256": _sha(neighbors_raw)}},
        "producer": {"package_version": __version__, "python_version": platform.python_version(),
                     "numpy_version": np.__version__,
                     "code_files": {name: _sha((code_dir / name).read_bytes()) for name in ("build.py", "sources.py")}},
        "execution": {"requested_block_rows": block_size, "effective_block_rows": effective,
                      "similarity_block_budget_bytes": SCORE_BLOCK_BUDGET},
        "coverage": coverage, "anime": diagnostics["anime"],
    }
    report_bytes = _canonical(build_report)
    output_dir = Path(output_dir)
    if output_dir.is_symlink() or (output_dir.exists() and not output_dir.is_dir()):
        raise BuildError(f"unsafe output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = (("neighbors.json", neighbors_raw), ("build-report.json", report_bytes))
    for name, _ in outputs:
        target = output_dir / name
        if target.is_symlink() or target.is_dir():
            raise BuildError(f"unsafe output target: {target}")
    for name, content in outputs:
        _atomic(output_dir / name, content)
    return output_dir / "build-report.json"
