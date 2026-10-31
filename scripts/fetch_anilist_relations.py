"""Fetch AniList anime relations for every catalog MAL ID into a reproducible snapshot."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

ENDPOINT = "https://graphql.anilist.co"
USER_AGENT = "AnimeRecommendationBot (offline franchise snapshot)"
BATCH = 50
PAUSE_SECONDS = 2.1
QUERY = """query ($ids: [Int], $page: Int) {
  Page(page: $page, perPage: 50) {
    pageInfo { hasNextPage }
    media(idMal_in: $ids, type: ANIME) {
      idMal format startDate { year month day }
      relations { edges { relationType(version: 2) node { idMal type } } }
    }
  }
}"""


class FetchError(Exception):
    pass


def obj(value: object, where: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise FetchError(f"{where}: expected object")
    return {str(key): item for key, item in value.items()}


def arr(value: object, where: str) -> list[object]:
    if not isinstance(value, list):
        raise FetchError(f"{where}: expected array")
    return value


def optional_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def catalog_ids(catalog: Path) -> list[int]:
    loaded: object = json.loads(catalog.read_bytes())
    return sorted(int(key) for key in obj(obj(loaded, "catalog")["anime"], "catalog.anime"))


def post(ids: list[int], page: int) -> dict[str, object]:
    body = json.dumps({"query": QUERY, "variables": {"ids": ids, "page": page}}).encode()
    request = urllib.request.Request(
        ENDPOINT,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": USER_AGENT,
        },
    )
    for attempt in range(8):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                loaded: object = json.load(response)
                return obj(loaded, "response")
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504):
                raise FetchError(f"HTTP {exc.code}") from exc
            retry = str(exc.headers.get("Retry-After") or "")
            wait = int(retry) if retry.isdigit() else 2**attempt * 5
        except (urllib.error.URLError, TimeoutError) as exc:
            wait = 2**attempt * 5
            if attempt == 7:
                raise FetchError(str(exc)) from exc
        time.sleep(wait + 1)
    raise FetchError("retries exhausted")


def fetch_batch(ids: list[int]) -> list[object]:
    media: list[object] = []
    page = 1
    while True:
        response = post(ids, page)
        if response.get("errors"):
            # A partial page must not be cached as a complete batch.
            raise FetchError(f"GraphQL errors: {json.dumps(response['errors'])[:300]}")
        data = obj(response["data"], "data")
        found = obj(data["Page"], "Page")
        media.extend(arr(found["media"], "media"))
        if not obj(found["pageInfo"], "pageInfo")["hasNextPage"]:
            return media
        page += 1
        time.sleep(PAUSE_SECONDS)


def group_media(items: list[object], wanted: set[int]) -> dict[int, list[dict[str, object]]]:
    """Collect the AniList entries of every wanted MAL ID; AniList splits a few MAL entries."""
    found: dict[int, list[dict[str, object]]] = {}
    for item in items:
        media = obj(item, "media")
        mal_id = optional_int(media.get("idMal"))
        if mal_id is not None and mal_id in wanted:
            found.setdefault(mal_id, []).append(media)
    return found


def normalize(mal_id: int, entries: list[dict[str, object]]) -> dict[str, object]:
    """Merge one MAL ID's entries: format and start of the earliest, every anime relation."""
    starts: list[tuple[list[int], int, list[int | None], object]] = []
    relations: set[tuple[str, int]] = set()
    for media in entries:
        start = obj(media.get("startDate") or {}, "startDate")
        date = [optional_int(start.get(key)) for key in ("year", "month", "day")]
        known = [9999 if part is None else part for part in date]
        starts.append((known, len(starts), date, media.get("format")))
        for edge in arr(obj(media["relations"], "relations")["edges"], "edges"):
            edge = obj(edge, "edge")
            node = obj(edge["node"], "node")
            target = optional_int(node.get("idMal"))
            if node.get("type") == "ANIME" and target is not None and target != mal_id:
                relations.add((str(edge["relationType"]), target))
    _, _, date, media_format = min(starts)
    return {
        "mal_id": mal_id,
        "format": media_format,
        "start_date": date,
        "relations": sorted([kind, target] for kind, target in relations),
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True, help="bundle catalog.json")
    parser.add_argument("--cache", type=Path, required=True, help="resumable per-batch cache")
    parser.add_argument("--output", type=Path, required=True, help="snapshot directory")
    args = parser.parse_args(argv)
    catalog: Path = args.catalog
    cache: Path = args.cache
    output: Path = args.output
    ids = catalog_ids(catalog)
    wanted = set(ids)
    catalog_sha256 = hashlib.sha256(catalog.read_bytes()).hexdigest()
    cache.mkdir(parents=True, exist_ok=True)
    # Batches are numbered by position in the catalog, so a cache serves one catalog only.
    marker = cache / "catalog.sha256"
    if marker.exists() and marker.read_text(encoding="utf-8").strip() != catalog_sha256:
        raise FetchError(f"{cache}: the cache was filled for another catalog")
    marker.write_text(catalog_sha256 + "\n", encoding="utf-8")
    batches = [ids[start : start + BATCH] for start in range(0, len(ids), BATCH)]
    for index, batch in enumerate(batches):
        path = cache / f"batch-{index:05d}.json"
        if path.exists():
            continue
        partial = path.with_suffix(".tmp")
        partial.write_text(json.dumps(fetch_batch(batch)), encoding="utf-8")
        partial.replace(path)
        if index % 50 == 0:
            print(f"{index + 1}/{len(batches)} batches", flush=True)
        time.sleep(PAUSE_SECONDS)

    media: list[object] = []
    newest = 0.0
    for index in range(len(batches)):
        path = cache / f"batch-{index:05d}.json"
        newest = max(newest, path.stat().st_mtime)
        cached: object = json.loads(path.read_bytes())
        media.extend(arr(cached, "batch"))
    # A rebuild from the cache keeps the time the data was actually fetched.
    fetched_at = datetime.fromtimestamp(newest, UTC)
    records = {
        mal_id: normalize(mal_id, entries) for mal_id, entries in group_media(media, wanted).items()
    }
    lines = "".join(
        json.dumps(records[key], ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for key in sorted(records)
    )
    raw = gzip.compress(lines.encode("utf-8"), mtime=0)
    output.mkdir(parents=True, exist_ok=True)
    (output / "relations.jsonl.gz").write_bytes(raw)
    manifest = {
        "source": ENDPOINT,
        "catalog_sha256": catalog_sha256,
        "query_sha256": hashlib.sha256(QUERY.encode()).hexdigest(),
        "fetched_at": fetched_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "requested_mal_ids": len(ids),
        "found_mal_ids": len(records),
        "relations": sum(len(arr(record["relations"], "relations")) for record in records.values()),
        "files": {
            "relations.jsonl.gz": {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        },
    }
    (output / "source-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"{output} found={len(records)}/{len(ids)} sha256={manifest['files']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
