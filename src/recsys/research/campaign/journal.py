"""Fixed, supervised and resumable validation-only ML-02 campaign.

The worker entry uses only the standard library until the generation gate has
been acquired and checked. All expensive scientific work runs in workers.
"""

from __future__ import annotations

import json
import os
import stat

from recsys.research.campaign.common import (
    AUX_PHASE,
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


class _Journal:
    def __init__(self, path):
        self.path = path
        self.events = []
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
                if raw != _canonical(event) or event.get("seq") != len(self.events) + 1:
                    _fail()
                self.events.append(event)

    def append(self, name, **fields):
        event = {"seq": len(self.events) + 1, "event": name, "utc": _iso(_now()), **fields}
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


def _successes(events):
    """Restore durable successes, charges, attempts and unresolved starts."""
    successes = {}
    attempts = {}
    charged = {}
    starts = {}
    failed = set()
    latest_utc = None
    for event in events:
        timestamp = _parse_iso(event["utc"])
        if latest_utc is not None and timestamp < latest_utc:
            _fail("research_campaign_clock_invalid")
        latest_utc = timestamp
        if event["event"] not in ("attempt_started", "attempt_finished", "attempt_interrupted"):
            continue
        key = (event["variant"], event["phase"])
        if key[0] not in VARIANTS or key[1] not in (*PHASES, AUX_PHASE):
            _fail()
        number = event["attempt"]
        if type(number) is not int or number < 1:
            _fail()
        if event["event"] == "attempt_started":
            if key in starts or number != attempts.get(key, 0) + 1:
                _fail()
            attempts[key] = number
            starts[key] = event
        else:
            if key not in starts or starts[key]["attempt"] != number:
                _fail()
            starts.pop(key)
            seconds = event["charged_seconds"]
            if type(seconds) not in (int, float) or not 0 <= seconds < CAMPAIGN_SECONDS * 2:
                _fail()
            charged[key] = charged.get(key, 0.0) + seconds
            if event["event"] == "attempt_finished" and event["status"] == "success":
                successes[key] = event
            elif event["event"] == "attempt_finished":
                failed.add(key)
    return successes, attempts, charged, starts, failed
