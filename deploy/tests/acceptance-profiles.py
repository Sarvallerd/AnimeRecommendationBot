#!/usr/bin/env python3
"""Strict, stdlib-only acceptance profile and retained bundle validation."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import stat
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROFILE_DIR = Path(__file__).with_name("acceptance_profiles")
NEW_RELEASE = "neelagiri-2025-nafiul-2026-v1"
LEGACY_RELEASE = "mal-2020-glove-v1"
PROFILE_FILES = {LEGACY_RELEASE: "legacy-v1.json", NEW_RELEASE: NEW_RELEASE + ".json"}
LOCK_RESOURCE = "release_neelagiri_2025_nafiul_2026_v1.json"
TRUSTED_NEW_PROFILE_SHA = "7137a3d064b27e4f929c7ffde1e0a6746682276364553c3e8a1422efe8399e60"
TRUSTED_LEGACY_PROFILE_SHA = "d1e640debd8912193d45f2a6537389c2fbad67306493819c8adb9a67f1e5b35c"
TRUSTED_LOCK_SHA = "46f3ffd60239675b802026025016ee6719cb0525325e088fbfa248012db3de9b"
HEX = re.compile(r"[0-9a-f]{64}\Z")
IDENTITY = re.compile(r"sha256:[0-9a-f]{64}\Z")


class ProfileError(ValueError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def need(condition: bool, code: str) -> None:
    if not condition:
        raise ProfileError(code)


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _unique(pairs):
    value = {}
    for key, item in pairs:
        need(key not in value, "DUPLICATE_JSON_KEY")
        value[key] = item
    return value


def _constant(_):
    raise ProfileError("INVALID_JSON")


def canonical(raw: bytes) -> dict:
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique, parse_constant=_constant)
        encoded = (
            json.dumps(
                value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
            )
            + "\n"
        ).encode()
    except ProfileError:
        raise
    except (UnicodeError, ValueError, OverflowError) as exc:
        raise ProfileError("INVALID_JSON") from exc
    need(raw == encoded and type(value) is dict, "NONCANONICAL_JSON")
    return value


def regular_bytes(path: Path) -> bytes:
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, "rb") as stream:
            need(stat.S_ISREG(os.fstat(stream.fileno()).st_mode), "UNSAFE_INPUT")
            return stream.read()
    except OSError as exc:
        raise ProfileError("UNSAFE_INPUT") from exc


def fields(value, names, code="INVALID_PROFILE"):
    need(type(value) is dict and set(value) == set(names), code)


def _expectations(value: dict) -> None:
    fields(
        value,
        ("catalog_count", "neighbor_count_histogram", "smoke_seed", "empty_seed"),
        "INVALID_EXPECTATIONS",
    )
    count = value["catalog_count"]
    need(type(count) is int and count > 0, "INVALID_EXPECTATIONS")
    hist = value["neighbor_count_histogram"]
    fields(hist, (str(i) for i in range(6)), "INVALID_EXPECTATIONS")
    need(
        all(type(n) is int and n >= 0 for n in hist.values()) and sum(hist.values()) == count,
        "INVALID_EXPECTATIONS",
    )
    fields(value["smoke_seed"], ("mal_id", "query", "neighbor_ids"), "INVALID_EXPECTATIONS")
    smoke = value["smoke_seed"]
    need(
        type(smoke["mal_id"]) is int
        and smoke["mal_id"] > 0
        and type(smoke["query"]) is str
        and bool(smoke["query"])
        and type(smoke["neighbor_ids"]) is list
        and len(smoke["neighbor_ids"]) == 5
        and all(type(i) is int and i > 0 for i in smoke["neighbor_ids"]),
        "INVALID_EXPECTATIONS",
    )
    empty = value["empty_seed"]
    need(
        empty is None
        or (
            type(empty) is dict
            and set(empty) == {"mal_id", "query"}
            and type(empty["mal_id"]) is int
            and empty["mal_id"] > 0
            and type(empty["query"]) is str
            and bool(empty["query"])
        ),
        "INVALID_EXPECTATIONS",
    )


class AcceptanceProfile:
    def __init__(self, value: dict, raw: bytes, lock: dict | None, lock_raw: bytes | None):
        self.id = value["profile_id"]
        self.release_id = value["release_id"]
        self.policy = value["policy"]
        self.sha256 = digest(raw)
        self.lock = lock
        self.lock_sha256 = digest(lock_raw) if lock_raw is not None else None
        self.expectations = (
            lock["semantic_expectations"] if lock is not None else value["expectations"]
        )
        self.reference = lock["reference"] if lock is not None else None
        self.inputs = lock["inputs"] if lock is not None else None


def load_profile(
    release_id: str, *, _profile_bytes: bytes | None = None, _lock_bytes: bytes | None = None
) -> AcceptanceProfile:
    need(release_id in PROFILE_FILES, "UNKNOWN_RELEASE")
    raw = (
        _profile_bytes
        if _profile_bytes is not None
        else regular_bytes(PROFILE_DIR / PROFILE_FILES[release_id])
    )
    if _profile_bytes is None:
        need(
            digest(raw)
            == (
                TRUSTED_LEGACY_PROFILE_SHA
                if release_id == LEGACY_RELEASE
                else TRUSTED_NEW_PROFILE_SHA
            ),
            "PROFILE_SHA_MISMATCH",
        )
    value = canonical(raw)
    fields(value, ("schema_version", "kind", "profile_id", "release_id", "policy", "expectations"))
    need(
        type(value["schema_version"]) is int
        and value["schema_version"] == 1
        and value["kind"] == "arb-acceptance-profile"
        and value["release_id"] == release_id,
        "INVALID_PROFILE",
    )
    if release_id == LEGACY_RELEASE:
        need(
            value["profile_id"] == "legacy-v1" and value["policy"] == "legacy-arb020-v1",
            "INVALID_PROFILE",
        )
        _expectations(value["expectations"])
        need(_lock_bytes is None, "INVALID_PROFILE")
        return AcceptanceProfile(value, raw, None, None)
    need(
        value["profile_id"] == release_id and value["policy"] == "release-lock-v1",
        "INVALID_PROFILE",
    )
    fields(value["expectations"], ("release_lock_resource", "release_lock_sha256"))
    expected = value["expectations"]
    need(
        expected["release_lock_resource"] == LOCK_RESOURCE
        and type(expected["release_lock_sha256"]) is str
        and HEX.fullmatch(expected["release_lock_sha256"]) is not None,
        "INVALID_PROFILE",
    )
    lock_raw = (
        _lock_bytes
        if _lock_bytes is not None
        else regular_bytes(ROOT / "src/recsys" / LOCK_RESOURCE)
    )
    if _lock_bytes is None:
        need(digest(lock_raw) == TRUSTED_LOCK_SHA, "LOCK_SHA_MISMATCH")
    need(digest(lock_raw) == expected["release_lock_sha256"], "LOCK_SHA_MISMATCH")
    lock = canonical(lock_raw)
    fields(
        lock,
        ("schema_version", "kind", "release_id", "inputs", "reference", "semantic_expectations"),
        "INVALID_LOCK",
    )
    need(
        type(lock["schema_version"]) is int
        and lock["schema_version"] == 1
        and lock["kind"] == "arb-release-lock"
        and lock["release_id"] == release_id
        and type(lock["inputs"]) is dict
        and type(lock["reference"]) is dict
        and type(lock["semantic_expectations"]) is dict,
        "INVALID_LOCK",
    )
    _expectations(
        {
            k: lock["semantic_expectations"][k]
            for k in ("catalog_count", "neighbor_count_histogram", "smoke_seed", "empty_seed")
            if k in lock["semantic_expectations"]
        }
    )
    semantic = lock["semantic_expectations"]
    need(
        set(semantic)
        == {
            "catalog_count",
            "neighbor_count_histogram",
            "smoke_seed",
            "empty_seed",
            "vector_status_counts",
            "zero_vector_ids",
        }
        and type(semantic["vector_status_counts"]) is dict
        and set(semantic["vector_status_counts"])
        == {"ready", "oov_only", "no_tokens", "zero_vector"}
        and type(semantic["zero_vector_ids"]) is list,
        "INVALID_LOCK",
    )
    ref = lock["reference"]
    need(
        set(ref)
        == {
            "block_size",
            "build_report_sha256",
            "bundle_identity",
            "catalog_sha256",
            "neighbors_sha256",
            "producer",
        }
        and ref["block_size"] == 256
        and type(ref["block_size"]) is int
        and all(
            type(ref[key]) is str and HEX.fullmatch(ref[key])
            for key in ("build_report_sha256", "catalog_sha256", "neighbors_sha256")
        )
        and type(ref["bundle_identity"]) is str
        and IDENTITY.fullmatch(ref["bundle_identity"]),
        "INVALID_LOCK",
    )
    return AcceptanceProfile(value, raw, lock, lock_raw)


def _bundle_module():
    path = ROOT / "src/recsys/bundle.py"
    spec = importlib.util.spec_from_file_location("arb_acceptance_bundle_contract", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BundleSummary:
    def __init__(self, identity, count, hashes, neighbors, histogram):
        self.identity = identity
        self.count = count
        self.hashes = hashes
        self.neighbors = neighbors
        self.histogram = histogram


def check_profile_bundle(
    bundle_dir: Path, profile: AcceptanceProfile, *, expected_identity: str
) -> BundleSummary:
    need(
        type(expected_identity) is str and IDENTITY.fullmatch(expected_identity) is not None,
        "INVALID_EXPECTED_IDENTITY",
    )
    directory = Path(bundle_dir)
    need(
        directory.is_absolute()
        and ".." not in directory.parts
        and all(not item.is_symlink() for item in (directory, *directory.parents))
        and directory.is_dir(),
        "UNSAFE_BUNDLE",
    )
    need(
        {item.name for item in directory.iterdir()}
        == {"catalog.json", "neighbors.json", "manifest.json"},
        "UNSAFE_BUNDLE",
    )
    raw = {
        name: regular_bytes(directory / name)
        for name in ("catalog.json", "neighbors.json", "manifest.json")
    }
    contract = _bundle_module()
    try:
        identity, count = contract.check_bundle_bytes(
            raw["catalog.json"], raw["neighbors.json"], raw["manifest.json"]
        )
        catalog = contract.parse_canonical(raw["catalog.json"], "catalog.json")["anime"]
        neighbors = contract.parse_canonical(raw["neighbors.json"], "neighbors.json")["neighbors"]
    except (ValueError, KeyError, TypeError) as exc:
        raise ProfileError("INVALID_BUNDLE") from exc
    need(identity == expected_identity, "BUNDLE_IDENTITY_MISMATCH")
    expected = profile.expectations
    histogram = {str(i): sum(len(items) == i for items in neighbors.values()) for i in range(6)}
    need(
        count == expected["catalog_count"] and histogram == expected["neighbor_count_histogram"],
        "BUNDLE_SEMANTICS_MISMATCH",
    )
    smoke = expected["smoke_seed"]
    sid = str(smoke["mal_id"])
    need(
        catalog.get(sid, {}).get("title") == smoke["query"]
        and [item["mal_id"] for item in neighbors.get(sid, [])] == smoke["neighbor_ids"],
        "BUNDLE_SMOKE_MISMATCH",
    )
    empty = expected["empty_seed"]
    if empty is not None:
        eid = str(empty["mal_id"])
        need(
            catalog.get(eid, {}).get("title") == empty["query"] and neighbors.get(eid) == [],
            "BUNDLE_EMPTY_MISMATCH",
        )
    return BundleSummary(
        identity, count, {name: digest(data) for name, data in raw.items()}, neighbors, histogram
    )
