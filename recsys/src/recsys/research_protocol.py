"""Frozen ML-01 policy; overrides are restricted to trusted synthetic tests."""

import hashlib
import math
from importlib import resources
from pathlib import Path

from .bundle import canonical_bytes, parse_canonical

RESOURCE = "research_protocol_v1.json"


class ResearchError(ValueError):
    """Public, fixed-code failure without private input values."""


def fail(code="research_integrity_error"):
    raise ResearchError(code)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def exact(value, fields):
    if type(value) is not dict or set(value) != set(fields):
        fail()


def integer(value, low=0, high=None):
    if type(value) is not int or value < low or (high is not None and value > high):
        fail()


def finite(value):
    if type(value) is int:
        return
    if type(value) is not float or not math.isfinite(value):
        fail()


def load_protocol(selector="ml01-v1", *, _protocol_bytes=None):
    if selector != "ml01-v1":
        fail("research_protocol_invalid")
    try:
        trusted = resources.files("recsys").joinpath(RESOURCE).read_bytes()
        raw = trusted if _protocol_bytes is None else _protocol_bytes
        value = parse_canonical(raw, RESOURCE)
        expected = parse_canonical(trusted, RESOURCE)
        exact(value, expected)
        # Production is immutable. Test-only overrides may adjust quantitative values,
        # while retaining shape and type of the frozen contract.
        if _protocol_bytes is None and raw != trusted:
            fail()
        def shape(actual, reference):
            if type(actual) is not type(reference):
                fail()
            if type(reference) is dict:
                exact(actual, reference)
                for key in reference:
                    shape(actual[key], reference[key])
            elif type(reference) is list:
                if len(actual) != len(reference):
                    fail()
                for left, right in zip(actual, reference):
                    shape(left, right)
            elif type(reference) is float:
                finite(actual)
            elif type(reference) is int:
                integer(actual)
            elif actual != reference:
                fail()
        shape(value, expected)
        return value
    except (OSError, ValueError, TypeError, OverflowError) as exc:
        raise ResearchError("research_protocol_invalid") from exc


def protocol_identity(*, _protocol_bytes=None):
    value = load_protocol(_protocol_bytes=_protocol_bytes)
    return "sha256:" + sha(canonical_bytes(value))


def split_user(user_key: str, protocol=None) -> str:
    if type(user_key) is not str or len(user_key) != 67 or not user_key.startswith("u1_"):
        fail()
    if any(char not in "0123456789abcdef" for char in user_key[3:]):
        fail()
    seed = (protocol or load_protocol())["query"]["split_seed"]
    bucket = int.from_bytes(hashlib.sha256(
        b"arb-ml-user-split-v1\0" + str(seed).encode() + b"\0" + user_key.encode()
    ).digest()[:8], "big") % 100
    return "train" if bucket < 80 else "validation" if bucket < 90 else "test"


def order_hash(seed: int, user: str, mal_id: int) -> bytes:
    return hashlib.sha256(
        b"arb-ml-item-order-v1\0" + str(seed).encode() + b"\0" +
        user.encode() + b"\0" + str(mal_id).encode()
    ).digest()


def sample_hash(seed: int, split: str, cohort: str, user: str) -> bytes:
    return hashlib.sha256(
        b"arb-ml-user-sample-v1\0" + str(seed).encode() + b"\0" +
        split.encode() + b"\0" + cohort.encode() + b"\0" + user.encode()
    ).digest()
