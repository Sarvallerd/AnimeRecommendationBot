"""Reproducible diagnostic queries and a genre-only recommendation baseline."""

import hashlib
import json
import os
import tempfile
from importlib import resources
from pathlib import Path

from .sources import SourceError, load_registry


class QualityError(ValueError):
    """The pinned quality inputs or query specification are invalid."""


PROTOCOL_VERSION = "arb008-v1"
METHOD_ID = "genre-jaccard"
BANDS = {"1-100": (1, 100), "101-1000": (101, 1000),
         "1001-5000": (1001, 5000), "5001+": (5001, None)}


def _canonical(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode("utf-8")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _load_json(raw: bytes, name: str) -> dict:
    try:
        value = json.loads(raw)
    except (UnicodeError, ValueError) as exc:
        raise QualityError(f"invalid {name} JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise QualityError(f"{name} must be a JSON object")
    return value


def _validate_spec_metadata(spec: dict) -> None:
    """Keep the stated source pins tied to the installed source registry."""
    if (type(spec.get("schema_version")) is not int or spec["schema_version"] != 1
            or spec.get("protocol_version") != PROTOCOL_VERSION
            or spec.get("normalizer_version") != "1"
            or type(spec.get("catalog_schema_version")) is not int
            or spec["catalog_schema_version"] != 1
            or not isinstance(spec.get("query_set_version"), str)
            or not spec["query_set_version"].strip()):
        raise QualityError("invalid quality query specification version")
    selection = spec.get("selection")
    expected_bands = {name: [lower, upper] for name, (lower, upper) in BANDS.items()}
    if (not isinstance(selection, dict) or selection.get("bands") != expected_bands
            or type(selection.get("queries_per_band")) is not int
            or selection["queries_per_band"] != 5
            or not isinstance(selection.get("method"), str) or not selection["method"].strip()
            or not isinstance(selection.get("limitations"), str)
            or not selection["limitations"].strip()):
        raise QualityError("invalid quality query selection metadata")
    try:
        registry, registry_sha = load_registry()
    except SourceError as exc:
        raise QualityError(f"invalid packaged source registry: {exc}") from exc
    provenance = spec.get("source_provenance")
    sources = {source["id"]: source for source in registry["sources"]}
    if (spec.get("registry_sha256") != registry_sha
            or not isinstance(provenance, dict)
            or provenance.get("mal_repository_commit") != registry["provenance"]["mal"]["commit"]
            or provenance.get("anime_csv_sha256") != sources["mal_anime"]["sha256"]
            or provenance.get("synopsis_csv_sha256") != sources["mal_synopsis"]["sha256"]):
        raise QualityError("quality query source provenance does not match packaged registry")


def _validate_catalog(catalog: dict, expected_count: int) -> dict[str, dict]:
    anime = catalog.get("anime")
    if type(catalog.get("schema_version")) is not int or catalog["schema_version"] != 1 or not isinstance(anime, dict):
        raise QualityError("catalog needs schema_version 1 and an anime object")
    if len(anime) != expected_count:
        raise QualityError(f"catalog has {len(anime)} entries; expected {expected_count}")
    for key, entry in anime.items():
        if (not isinstance(key, str) or not key.isascii() or not key.isdecimal()
                or str(int(key)) != key or not isinstance(entry, dict)):
            raise QualityError(f"invalid catalog MAL_ID {key!r}")
        genres = entry.get("genres")
        if (not isinstance(entry.get("title"), str) or not entry["title"].strip()
                or not isinstance(genres, list)
                or any(not isinstance(genre, str) or not genre.strip() for genre in genres)
                or len(set(genres)) != len(genres)):
            raise QualityError(f"invalid title or genres for MAL_ID {key}")
    return anime


def _validate_query_set(spec: dict, anime: dict[str, dict]) -> list[dict]:
    queries = spec.get("queries")
    if not isinstance(queries, list) or len(queries) != 20:
        raise QualityError("quality query set must contain exactly 20 queries")
    seen: set[int] = set()
    counts = dict.fromkeys(BANDS, 0)
    for query in queries:
        if not isinstance(query, dict):
            raise QualityError("invalid quality query entry")
        mal_id, title = query.get("mal_id"), query.get("title")
        rank, members = query.get("popularity_rank"), query.get("members")
        band, rationale = query.get("popularity_band"), query.get("rationale")
        if type(mal_id) is not int or mal_id <= 0 or mal_id in seen:
            raise QualityError(f"duplicate or invalid query MAL_ID {mal_id!r}")
        if (not isinstance(title, str) or not title.strip()
                or not isinstance(rationale, str) or not rationale.strip()
                or type(rank) is not int or rank <= 0
                or type(members) is not int or members < 0
                or not isinstance(band, str) or band not in BANDS):
            raise QualityError(f"invalid query metadata for MAL_ID {mal_id}")
        lower, upper = BANDS[band]
        if rank < lower or (upper is not None and rank > upper):
            raise QualityError(f"popularity band does not match rank for MAL_ID {mal_id}")
        entry = anime.get(str(mal_id))
        if entry is None or entry["title"] != title:
            raise QualityError(f"missing MAL_ID or title guard mismatch for query {mal_id}")
        seen.add(mal_id)
        counts[band] += 1
    if any(count != 5 for count in counts.values()):
        raise QualityError("quality query set needs five queries per popularity band")
    return queries


def genre_neighbors(anime: dict[str, dict], query_id: int, k: int = 5) -> list[dict]:
    """Rank positive Jaccard neighbors by score, then numeric MAL_ID."""
    if type(query_id) is not int or str(query_id) not in anime:
        raise QualityError(f"unknown query MAL_ID {query_id!r}")
    if type(k) is not int or k < 0:
        raise QualityError("k must be a nonnegative integer")
    source = set(anime[str(query_id)]["genres"])
    if not source or k == 0:
        return []
    ranked = []
    for key, entry in anime.items():
        candidate_id = int(key)
        if candidate_id == query_id:
            continue
        candidate = set(entry["genres"])
        overlap = len(source & candidate)
        if overlap:
            ranked.append((overlap / len(source | candidate), candidate_id))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [{"mal_id": mal_id, "score": score} for score, mal_id in ranked[:k]]


def _write_atomic(path: Path, content: bytes) -> None:
    temp = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.",
                                         suffix=".tmp", delete=False) as file:
            temp = Path(file.name)
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
        temp.replace(path)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def run_baseline(catalog_path: Path, output_dir: Path, *, query_set: dict | None = None) -> Path:
    """Validate pinned inputs and write the baseline and blank assessment template."""
    if query_set is None:
        spec_raw = resources.files("recsys").joinpath("quality_queries_v1.json").read_bytes()
        spec = _load_json(spec_raw, "packaged query specification")
    else:
        spec = query_set
        if not isinstance(spec, dict):
            raise QualityError("query specification must be a JSON object")
        spec_raw = _canonical(spec)
    _validate_spec_metadata(spec)
    raw = Path(catalog_path).read_bytes()
    observed = _sha(raw)
    if observed != spec.get("catalog_sha256"):
        raise QualityError(f"catalog SHA256 mismatch: observed {observed}")
    catalog = _load_json(raw, "catalog")
    expected_count = spec.get("catalog_count")
    if type(expected_count) is not int or expected_count <= 0:
        raise QualityError("invalid catalog count in query specification")
    anime = _validate_catalog(catalog, expected_count)
    queries = _validate_query_set(spec, anime)
    spec_sha = _sha(spec_raw)
    baseline_queries = []
    judgments = []
    for query in queries:
        query_id = query["mal_id"]
        recommendations = genre_neighbors(anime, query_id)
        status = ("empty_genres" if not anime[str(query_id)]["genres"] else
                  "ok" if recommendations else "no_positive_candidates")
        baseline_queries.append({"query_mal_id": query_id, "status": status,
                                 "recommendations": [dict(rank=index, **neighbor)
                                                     for index, neighbor in enumerate(recommendations, 1)]})
        for neighbor in recommendations:
            candidate_id = neighbor["mal_id"]
            judgments.append({"query_mal_id": query_id, "candidate_mal_id": candidate_id,
                              "query_title": query["title"],
                              "candidate_title": anime[str(candidate_id)]["title"],
                              "relevance": None, "assessor_kind": "unjudged",
                              "assessor_id": None, "method_version": None, "rationale": None})
    common = {"schema_version": 1, "protocol_version": PROTOCOL_VERSION,
              "query_set_version": spec["query_set_version"], "query_set_sha256": spec_sha,
              "catalog_sha256": observed}
    baseline = {**common, "method": {"id": METHOD_ID, "version": "1", "score": "jaccard",
                                       "parameters": {"k": 5, "positive_only": True,
                                                      "tie_break": "mal_id_ascending"}},
                "queries": baseline_queries}
    baseline_raw = _canonical(baseline)
    template = {**common, "runs": [{"method_id": METHOD_ID, "sha256": _sha(baseline_raw)}],
                "judgments": sorted(judgments, key=lambda row: (row["query_mal_id"],
                                                           row["candidate_mal_id"]))}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "genre-baseline.json"
    _write_atomic(path, baseline_raw)
    _write_atomic(output_dir / "relevance-template.json", _canonical(template))
    return path
