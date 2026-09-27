"""Validate a v1 offline recommendation bundle with only Python's standard library."""

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

MAX_ID = 2_147_483_647
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
CATALOG_FIELDS = {"title", "aliases", "genres", "score", "year", "type", "episodes", "synopsis"}


class ContractError(ValueError):
    """A bundle violates its wire or semantic contract."""


def require(condition, location, message):
    if not condition:
        raise ContractError(f"{location}: {message}")


def exact_fields(value, required, location):
    require(isinstance(value, dict), location, "expected object")
    missing = required - value.keys()
    extra = value.keys() - required
    require(not missing and not extra, location,
            f"fields differ; missing={sorted(missing)}, unexpected={sorted(extra)}")


def integer(value, location, low, high=None):
    require(type(value) is int and value >= low and (high is None or value <= high),
            location, f"expected integer in {low}..{high if high is not None else 'unbounded'}")


def nonblank(value, location):
    require(isinstance(value, str) and bool(value.strip()), location, "expected nonblank string")


def digest(value, location):
    require(isinstance(value, str) and SHA256.fullmatch(value) is not None,
            location, "expected lowercase SHA-256 hex digest")


def mal_id(value, location):
    require(isinstance(value, str) and re.fullmatch(r"[1-9][0-9]*", value) is not None,
            location, "expected canonical positive decimal MAL_ID key")
    require(len(value) <= 10 and int(value) <= MAX_ID, location, "MAL_ID exceeds 2147483647")


def number(value, location, low, high, exclusive_low=False):
    require(type(value) in (int, float) and (type(value) is int or math.isfinite(value)),
            location, "expected finite JSON number")
    require((value > low if exclusive_low else value >= low) and value <= high,
            location, f"number outside {'(' if exclusive_low else '['}{low},{high}]")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ContractError(f"nonfinite JSON constant {value}")


def read_canonical(path):
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ContractError(f"{path}: {exc.strerror or exc}") from exc
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object,
                           parse_constant=_reject_constant)
        canonical = (json.dumps(value, ensure_ascii=False, sort_keys=True,
                                separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
    except (UnicodeError, json.JSONDecodeError, ValueError, OverflowError) as exc:
        raise ContractError(f"{path}: invalid JSON: {exc}") from exc
    require(raw == canonical, str(path), "noncanonical JSON encoding")
    return value, raw


def check_catalog(value):
    exact_fields(value, {"schema_version", "anime"}, "catalog.json")
    integer(value["schema_version"], "catalog.json.schema_version", 1, 1)
    anime = value["anime"]
    require(isinstance(anime, dict) and bool(anime), "catalog.json.anime", "expected nonempty object")
    for key, entry in anime.items():
        loc = f"catalog.json.anime.{key}"
        mal_id(key, loc)
        exact_fields(entry, CATALOG_FIELDS, loc)
        nonblank(entry["title"], f"{loc}.title")
        for field in ("aliases", "genres"):
            items = entry[field]
            require(isinstance(items, list), f"{loc}.{field}", "expected array")
            for index, item in enumerate(items):
                nonblank(item, f"{loc}.{field}[{index}]")
            require(len(items) == len(set(items)), f"{loc}.{field}", "duplicate string")
        if entry["score"] is not None:
            number(entry["score"], f"{loc}.score", 1, 10)
        if entry["year"] is not None:
            integer(entry["year"], f"{loc}.year", 1, 9999)
        if entry["type"] is not None:
            nonblank(entry["type"], f"{loc}.type")
        if entry["episodes"] is not None:
            integer(entry["episodes"], f"{loc}.episodes", 1)
        if entry["synopsis"] is not None:
            nonblank(entry["synopsis"], f"{loc}.synopsis")
    return set(anime)


def check_neighbors(value, catalog_keys):
    exact_fields(value, {"schema_version", "neighbors"}, "neighbors.json")
    integer(value["schema_version"], "neighbors.json.schema_version", 1, 1)
    neighbors = value["neighbors"]
    require(isinstance(neighbors, dict), "neighbors.json.neighbors", "expected object")
    for key in neighbors:
        mal_id(key, f"neighbors.json.neighbors.{key}")
    require(set(neighbors) == catalog_keys, "neighbors.json.neighbors",
            "source MAL_ID keys must exactly match catalog")
    for source, items in neighbors.items():
        loc = f"neighbors.json.neighbors.{source}"
        require(isinstance(items, list) and len(items) <= 5, loc, "expected array of at most 5")
        seen = set()
        previous = None
        for index, item in enumerate(items):
            item_loc = f"{loc}[{index}]"
            exact_fields(item, {"mal_id", "similarity"}, item_loc)
            target = item["mal_id"]
            integer(target, f"{item_loc}.mal_id", 1, MAX_ID)
            require(str(target) in catalog_keys, f"{item_loc}.mal_id", "target missing from catalog")
            require(str(target) != source, f"{item_loc}.mal_id", "self-neighbor")
            require(target not in seen, f"{item_loc}.mal_id", "duplicate neighbor")
            seen.add(target)
            score = item["similarity"]
            number(score, f"{item_loc}.similarity", 0, 1, exclusive_low=True)
            ordering = (-score, target)
            require(previous is None or previous <= ordering, item_loc,
                    "neighbors must be sorted by descending similarity, then ascending MAL_ID")
            previous = ordering


def check_manifest(value, raw, file_bytes):
    exact_fields(value, {"schema_version", "algorithm", "sources", "files"}, "manifest.json")
    integer(value["schema_version"], "manifest.json.schema_version", 1, 1)
    algorithm = value["algorithm"]
    exact_fields(algorithm, {"name", "version", "max_neighbors", "similarity", "parameters"},
                 "manifest.json.algorithm")
    require(algorithm["name"] == "glove-mean-cosine", "manifest.json.algorithm.name",
            "expected glove-mean-cosine")
    nonblank(algorithm["version"], "manifest.json.algorithm.version")
    integer(algorithm["max_neighbors"], "manifest.json.algorithm.max_neighbors", 5, 5)
    require(algorithm["similarity"] == "cosine", "manifest.json.algorithm.similarity", "expected cosine")
    require(isinstance(algorithm["parameters"], dict), "manifest.json.algorithm.parameters",
            "expected object")
    sources = value["sources"]
    require(isinstance(sources, list) and bool(sources), "manifest.json.sources", "expected nonempty array")
    previous = None
    for index, source in enumerate(sources):
        loc = f"manifest.json.sources[{index}]"
        exact_fields(source, {"name", "version", "sha256", "metadata"}, loc)
        nonblank(source["name"], f"{loc}.name")
        nonblank(source["version"], f"{loc}.version")
        digest(source["sha256"], f"{loc}.sha256")
        require(isinstance(source["metadata"], dict), f"{loc}.metadata", "expected object")
        pair = (source["name"], source["version"])
        require(previous is None or previous < pair, loc,
                "sources must be unique and sorted by name, version")
        previous = pair
    files = value["files"]
    exact_fields(files, {"catalog.json", "neighbors.json"}, "manifest.json.files")
    for name, entry in files.items():
        loc = f"manifest.json.files.{name}"
        exact_fields(entry, {"sha256"}, loc)
        digest(entry["sha256"], f"{loc}.sha256")
        require(hashlib.sha256(file_bytes[name]).hexdigest() == entry["sha256"], loc,
                "SHA-256 mismatch")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def check_bundle(bundle_dir):
    """Return (bundle identity, catalog record count) after complete validation."""
    bundle_dir = Path(bundle_dir)
    manifest, manifest_raw = read_canonical(bundle_dir / "manifest.json")
    catalog, catalog_raw = read_canonical(bundle_dir / "catalog.json")
    neighbors, neighbors_raw = read_canonical(bundle_dir / "neighbors.json")
    identity = check_manifest(manifest, manifest_raw,
                              {"catalog.json": catalog_raw, "neighbors.json": neighbors_raw})
    keys = check_catalog(catalog)
    check_neighbors(neighbors, keys)
    return identity, len(keys)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle_dir", type=Path)
    args = parser.parse_args(argv)
    try:
        identity, count = check_bundle(args.bundle_dir)
    except ContractError as exc:
        print(exc, file=sys.stderr)
        return 1
    print(f"{identity} records={count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
