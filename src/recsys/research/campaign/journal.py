"""Fixed, supervised and resumable validation-only ML-02 campaign.

The worker entry uses only the standard library until the generation gate has
been acquired and checked. All expensive scientific work runs in workers.
"""

from __future__ import annotations

import json
import math
import os
import stat
from datetime import datetime
from pathlib import Path
from typing import NotRequired, TypedDict, cast

from recsys.research.campaign.common import (
    CAMPAIGN_SECONDS,
    PHASES,
    VARIANTS,
    _canonical,
    _fail,
    _iso,
    _now,
    _parse_iso,
    _sync_dir,
)


class CampaignEvent(TypedDict):
    seq: int
    event: str
    utc: str
    variant: NotRequired[str]
    phase: NotRequired[str]
    attempt: NotRequired[int]
    charged_seconds: NotRequired[float]
    status: NotRequired[str]
    generation: NotRequired[str]
    output_identity: NotRequired[str | None]
    resources: NotRequired[dict[str, object] | None]
    code: NotRequired[str | None]
    verification_resources: NotRequired[dict[str, object] | None]
    verification_identity: NotRequired[str | None]
    execution_spawned: NotRequired[bool]
    verification_spawned: NotRequired[bool]


type PhaseKey = tuple[str, str]
_STORAGE_PREFLIGHT_CODES = frozenset(
    {
        "research_campaign_storage_low",
        "research_campaign_storage_probe_failed",
        "research_campaign_storage_binding_changed",
        "research_campaign_storage_root_invalid",
    }
)
type SuccessState = tuple[
    dict[PhaseKey, CampaignEvent],
    dict[PhaseKey, int],
    dict[PhaseKey, float],
    dict[PhaseKey, CampaignEvent],
    set[PhaseKey],
]


def _validated_event(value: object) -> CampaignEvent:
    if type(value) is not dict:
        _fail()
    event = cast("dict[str, object]", value)
    if (
        any(type(key) is not str for key in event)
        or type(event.get("seq")) is not int
        or type(event.get("event")) is not str
        or type(event.get("utc")) is not str
    ):
        _fail()
    for key in ("variant", "phase", "status", "generation", "plan_identity"):
        if key in event and type(event[key]) is not str:
            _fail()
    if "phase" in event and event["phase"] not in PHASES:
        _fail()
    if "attempt" in event and type(event["attempt"]) is not int:
        _fail()
    for key in ("charged_seconds", "remaining_seconds"):
        if key in event and type(event[key]) not in (int, float):
            _fail()
    for key in ("output_identity", "verification_identity", "code"):
        if key in event and event[key] is not None and type(event[key]) is not str:
            _fail()
    for resource_key in ("resources", "verification_resources"):
        if resource_key in event and event[resource_key] is not None:
            resources = event[resource_key]
            if type(resources) is not dict or any(type(key) is not str for key in resources):
                _fail()
    if "resumed" in event and type(event["resumed"]) is not bool:
        _fail()
    if "execution_spawned" in event or "verification_spawned" in event:
        if (
            event["event"] != "attempt_finished"
            or event.get("status") != "failed"
            or event.get("code") not in _STORAGE_PREFLIGHT_CODES
            or event.get("output_identity") is not None
            or "execution_spawned" in event
            and "verification_spawned" in event
        ):
            _fail()
        if "execution_spawned" in event and (
            event["execution_spawned"] is not False
            or event.get("resources") is not None
            or event.get("verification_resources") is not None
        ):
            _fail()
        execution_resources = event.get("resources")
        if "verification_spawned" in event and (
            event["verification_spawned"] is not False
            or type(execution_resources) is not dict
            or execution_resources.get("cleanup_complete") is not True
            or event.get("verification_resources") is not None
        ):
            _fail()
    if event["event"] in ("attempt_started", "attempt_finished", "attempt_interrupted"):
        required = {"variant", "phase", "attempt"}
        if event["event"] == "attempt_started" or (
            event["event"] == "attempt_finished" and event.get("status") == "success"
        ):
            required.add("generation")
        if not required <= event.keys():
            _fail()
        if event["event"] != "attempt_started" and "charged_seconds" not in event:
            _fail()
        if event["event"] == "attempt_finished" and "status" not in event:
            _fail()
        if (
            event["event"] == "attempt_finished"
            and event.get("status") == "success"
            and (
                type(event.get("output_identity")) is not str
                or type(event.get("code")) not in (str, type(None))
            )
        ):
            _fail()
    return cast("CampaignEvent", event)


class _Journal:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.events: list[CampaignEvent] = []
        if path.exists():
            if path.is_symlink():
                _fail()
            info = path.stat()
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_size > 16 * 1024 * 1024
                or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) != 0o600
            ):
                _fail()
            for raw in path.read_bytes().splitlines(keepends=True):
                if not raw.endswith(b"\n"):
                    _fail()
                try:
                    event = json.loads(raw)
                except (ValueError, UnicodeError):
                    _fail()
                if raw != _canonical(event):
                    _fail()
                checked = _validated_event(event)
                if checked["seq"] != len(self.events) + 1:
                    _fail()
                self.events.append(checked)

    def append(self, name: str, **fields: object) -> CampaignEvent:
        if any(key in fields for key in ("seq", "event", "utc")):
            _fail()
        event = _validated_event(
            {"seq": len(self.events) + 1, "event": name, "utc": _iso(_now()), **fields}
        )
        if self.events and _parse_iso(event["utc"]) < _parse_iso(self.events[-1]["utc"]):
            _fail("research_campaign_clock_invalid")
        raw = _canonical(event)
        creating = not self.path.exists()
        fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            with os.fdopen(fd, "wb") as out:
                out.write(raw)
                out.flush()
                os.fsync(out.fileno())
            if creating:
                _sync_dir(self.path.parent)
        except BaseException:
            raise
        self.events.append(event)
        return event


def _successes(
    events: list[CampaignEvent], *, variants: tuple[str, ...] = VARIANTS
) -> SuccessState:
    """Restore durable successes, charges, attempts and unresolved starts."""
    successes: dict[PhaseKey, CampaignEvent] = {}
    attempts: dict[PhaseKey, int] = {}
    charged: dict[PhaseKey, float] = {}
    starts: dict[PhaseKey, CampaignEvent] = {}
    failed: set[PhaseKey] = set()
    latest_utc: datetime | None = None
    for event in events:
        timestamp = _parse_iso(event["utc"])
        if latest_utc is not None and timestamp < latest_utc:
            _fail("research_campaign_clock_invalid")
        latest_utc = timestamp
        if event["event"] not in ("attempt_started", "attempt_finished", "attempt_interrupted"):
            continue
        key = (event["variant"], event["phase"])
        if key[0] not in variants or key[1] not in PHASES:
            _fail()
        number = event["attempt"]
        if type(number) is not int or number < 1:
            _fail()
        if event["event"] == "attempt_started":
            if key in starts or number != attempts.get(key, 0) + 1:
                _fail()
            attempts[key] = number
            starts[key] = event
            failed.discard(key)
        else:
            if key not in starts or starts[key]["attempt"] != number:
                _fail()
            starts.pop(key)
            seconds = event["charged_seconds"]
            if (
                type(seconds) not in (int, float)
                or not math.isfinite(seconds)
                or not 0 <= seconds < CAMPAIGN_SECONDS * 2
            ):
                _fail()
            charged[key] = charged.get(key, 0.0) + seconds
            if event["event"] == "attempt_finished" and event["status"] == "success":
                if len(variants) != len(VARIANTS) and (
                    type(event.get("verification_identity")) is not str
                    or type(event.get("verification_resources")) is not dict
                ):
                    _fail("research_campaign_verification_invalid")
                successes[key] = event
                failed.discard(key)
            elif event["event"] == "attempt_finished":
                failed.add(key)
    return successes, attempts, charged, starts, failed
