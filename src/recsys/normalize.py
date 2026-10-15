"""Deterministic normalization of the two verified MAL CSV sources."""

import csv
import hashlib
import io
import json
import math
import os
import re
import tempfile
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

from .sources import SourceError, load_registry, validate_registry

NORMALIZER_VERSION = "1"
MAX_ID = 2_147_483_647
ANIME_COLUMNS = ("MAL_ID", "Name", "English name", "Japanese name", "Genres", "Score",
                 "Episodes", "Aired", "Premiered", "Type", "Duration")
SYNOPSIS_COLUMNS = ("MAL_ID", "sypnopsis")
BOILERPLATES = {
    "No synopsis information has been added to this title. Help improve our database by adding a synopsis here .",
    "No synopsis has been added for this series yet. Click here to update this information.",
    "No synopsis has been added for this series yet.",
}
MONTH = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
AIRED = re.compile(rf"(?:[0-9]{{4}}|{MONTH}, [0-9]{{4}}|{MONTH} [0-9]{{1,2}}, [0-9]{{4}})(?: to .+)?\Z")
PREMIERED = re.compile(r"(?:Winter|Spring|Summer|Fall) ([0-9]{4})\Z")
DURATION = re.compile(
    r"(?:(?P<hours>[0-9]+) hr\.(?: (?P<minutes>[0-9]+) min\.)?"
    r"(?: (?P<seconds>[0-9]+) sec\.)?|"
    r"(?P<minutes_only>[0-9]+) min\.(?: (?P<seconds_after_minutes>[0-9]+) sec\.)?|"
    r"(?P<seconds_only>[0-9]+) sec\.)(?: (?P<per_episode>per ep\.))?\Z"
)
ASCII_INTEGER = re.compile(r"[0-9]+\Z")


def _canonical(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode("utf-8")


def _missing(value):
    return value.strip() in ("", "Unknown")


def _id(raw):
    value = raw.strip()
    if not ASCII_INTEGER.fullmatch(value):
        return None
    canonical = value.lstrip("0")
    if not canonical or len(canonical) > 10 or (len(canonical) == 10 and canonical > str(MAX_ID)):
        return None
    return int(canonical)


def _read_csv(path, descriptor, columns):
    """Verify one raw read, then parse only those verified bytes."""
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if len(raw) != descriptor["size"] or digest != descriptor["sha256"]:
        raise SourceError(f"{path}: size/SHA256 mismatch: observed {len(raw)} bytes, {digest}")
    try:
        stream = io.StringIO(raw.decode("utf-8", errors="strict"), newline="")
        reader = csv.reader(stream, strict=True)
        header = next(reader)
        missing = [column for column in columns if column not in header]
        repeated = [column for column in columns if header.count(column) > 1]
        if missing or repeated:
            raise SourceError(f"{path}: invalid consumed headers: missing={missing}, duplicate={repeated}")
        indexes = {column: header.index(column) for column in columns}
        rows = []
        for row_number, row in enumerate(reader, 2):
            if len(row) != len(header):
                raise SourceError(f"{path}: CSV row {row_number} has {len(row)} columns; expected {len(header)}")
            rows.append((row_number, {column: row[index] for column, index in indexes.items()}))
    except (UnicodeError, csv.Error, StopIteration) as exc:
        raise SourceError(f"{path}: invalid UTF-8 or CSV: {exc}") from exc
    return rows


def _ordered_unique(values):
    result = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result


def _year(value, premiered=False):
    if _missing(value):
        return None
    value = value.strip()
    if premiered:
        match = PREMIERED.fullmatch(value)
        return int(match.group(1)) if match and int(match.group(1)) > 0 else None
    if not AIRED.fullmatch(value):
        return None
    start = value.split(" to ", 1)[0]
    year = int(start[-4:])
    if year == 0:
        return None
    if not start[0].isdigit():
        month = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec").index(start[:3]) + 1
        dated = re.fullmatch(rf"{MONTH} ([0-9]{{1,2}}), [0-9]{{4}}", start)
        day = int(dated.group(1)) if dated else 1
        try:
            date(year, month, day)
        except ValueError:
            return None
    return year


def _decimal(value):
    digits = value.lstrip("0") or "0"
    if len(digits) > 4000:
        raise ValueError("decimal integer exceeds supported length")
    return int(digits)


def _duration(value):
    if _missing(value):
        return None, None
    match = DURATION.fullmatch(value.strip())
    if not match:
        return None, None
    try:
        seconds = (_decimal(match.group("hours") or "0") * 3600 +
                   _decimal(match.group("minutes") or match.group("minutes_only") or "0") * 60 +
                   _decimal(match.group("seconds") or match.group("seconds_after_minutes") or
                            match.group("seconds_only") or "0"))
    except ValueError:
        return None, None
    if seconds <= 0:
        return None, None
    scope = "per_episode" if match.group("per_episode") else "unspecified"
    return seconds, scope


def _validate_artifacts(anime, durations):
    """Check v1 catalog and duration invariants before installing any file."""
    if not anime or set(anime) != set(durations):
        raise SourceError("catalog and durations need the same nonempty MAL_ID set")
    fields = {"title", "aliases", "genres", "score", "year", "type", "episodes", "synopsis"}
    for key, entry in anime.items():
        if _id(key) is None or str(_id(key)) != key or set(entry) != fields:
            raise SourceError(f"invalid catalog entry {key}")
        if not isinstance(entry["title"], str) or not entry["title"].strip():
            raise SourceError(f"invalid title for MAL_ID {key}")
        for field in ("aliases", "genres"):
            values = entry[field]
            if (not isinstance(values, list) or any(not isinstance(value, str) or not value.strip()
                                                    for value in values) or len(set(values)) != len(values)):
                raise SourceError(f"invalid {field} for MAL_ID {key}")
        score = entry["score"]
        if score is not None and (type(score) not in (int, float) or not math.isfinite(score)
                                  or not 1 <= score <= 10):
            raise SourceError(f"invalid score for MAL_ID {key}")
        for field, maximum in (("year", 9999), ("episodes", None)):
            value = entry[field]
            if value is not None and (type(value) is not int or value < 1 or
                                      (maximum is not None and value > maximum)):
                raise SourceError(f"invalid {field} for MAL_ID {key}")
        for field in ("type", "synopsis"):
            value = entry[field]
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise SourceError(f"invalid {field} for MAL_ID {key}")
        duration = durations[key]
        if set(duration) != {"duration_seconds", "scope"}:
            raise SourceError(f"invalid duration entry for MAL_ID {key}")
        seconds, scope = duration["duration_seconds"], duration["scope"]
        if (seconds is None and scope is not None or
                seconds is not None and (type(seconds) is not int or seconds <= 0 or
                                         scope not in ("per_episode", "unspecified"))):
            raise SourceError(f"invalid duration for MAL_ID {key}")


def _write_atomic(path, content):
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


def normalize(anime_csv: Path, synopsis_csv: Path, output_dir: Path, *,
              registry: dict | None = None, registry_sha256: str | None = None) -> Path:
    """Write catalog, durations and report from verified MAL inputs; return report path."""
    if registry is None:
        registry, registry_sha256 = load_registry()
    else:
        validate_registry(registry)
        if registry_sha256 is None:
            registry_sha256 = hashlib.sha256(_canonical(registry)).hexdigest()
    descriptors = {source["id"]: source for source in registry["sources"]}
    if "mal_anime" not in descriptors or "mal_synopsis" not in descriptors:
        raise SourceError("registry needs mal_anime and mal_synopsis sources")
    anime_rows = _read_csv(Path(anime_csv), descriptors["mal_anime"], ANIME_COLUMNS)
    synopsis_rows = _read_csv(Path(synopsis_csv), descriptors["mal_synopsis"], SYNOPSIS_COLUMNS)
    exclusions = []
    exceptions = []
    counts = Counter()

    def exception(source, row, mal_id, field, reason, raw, normalized=None):
        exceptions.append({"source": source, "row": row, "id": mal_id, "field": field,
                           "reason": reason, "raw": raw, "normalized": normalized})

    def indexed(rows, source):
        index = {}
        for row_number, row in rows:
            raw = row["MAL_ID"]
            number = _id(raw)
            if number is None:
                exclusions.append({"source": source, "row": row_number, "field": "MAL_ID",
                                   "raw_id": raw, "parsed_id": None, "reason": "invalid_id"})
                continue
            if number in index:
                raise SourceError(f"{source}: duplicate normalized MAL_ID {number} at CSV row {row_number}")
            if raw != str(number):
                exception(source, row_number, number, "MAL_ID", "noncanonical_id", raw, str(number))
            index[number] = row_number, row
        return index

    anime_index = indexed(anime_rows, "mal_anime")
    synopsis_index = indexed(synopsis_rows, "mal_synopsis")
    anime = {}
    durations = {}
    title_ids = defaultdict(list)
    matched = 0
    for mal_id in sorted(anime_index):
        row_number, row = anime_index[mal_id]
        title = row["Name"].strip()
        alternatives = [row["English name"].strip(), row["Japanese name"].strip()]
        if _missing(title):
            title = next((candidate for candidate in alternatives if not _missing(candidate)), "")
            if not title:
                exclusions.append({"source": "mal_anime", "row": row_number, "field": "Name",
                                   "raw_id": row["MAL_ID"], "parsed_id": mal_id,
                                   "reason": "no_usable_title"})
                continue
            exception("mal_anime", row_number, mal_id, "Name", "title_fallback", row["Name"], title)
        aliases = _ordered_unique(candidate for candidate in alternatives
                                  if not _missing(candidate) and candidate != title)
        genres = _ordered_unique(part.strip() for part in row["Genres"].split(",")
                                 if not _missing(part))
        score = None
        raw_score = row["Score"]
        if not _missing(raw_score):
            try:
                score = float(raw_score.strip())
                if not math.isfinite(score) or not 1 <= score <= 10:
                    raise ValueError("score outside 1..10 or nonfinite")
            except (ValueError, OverflowError):
                exception("mal_anime", row_number, mal_id, "Score", "invalid_score", raw_score)
                score = None
        episodes = None
        raw_episodes = row["Episodes"]
        if not _missing(raw_episodes):
            stripped = raw_episodes.strip()
            try:
                if not ASCII_INTEGER.fullmatch(stripped):
                    raise ValueError("episodes must be decimal")
                episodes = _decimal(stripped)
                if episodes <= 0:
                    raise ValueError("episodes must be positive")
            except ValueError:
                episodes = None
                exception("mal_anime", row_number, mal_id, "Episodes", "invalid_episodes", raw_episodes)
        aired_year = _year(row["Aired"])
        premiered_year = _year(row["Premiered"], premiered=True)
        for field, value, parsed in (("Aired", row["Aired"], aired_year),
                                     ("Premiered", row["Premiered"], premiered_year)):
            if not _missing(value) and parsed is None:
                exception("mal_anime", row_number, mal_id, field, "invalid_year", value)
        if aired_year is not None and premiered_year is not None and aired_year != premiered_year:
            exception("mal_anime", row_number, mal_id, "year", "aired_premiered_disagreement",
                      {"Aired": row["Aired"], "Premiered": row["Premiered"]}, aired_year)
        year = aired_year if aired_year is not None else premiered_year
        type_value = None if _missing(row["Type"]) else row["Type"].strip()
        duration_seconds, scope = _duration(row["Duration"])
        if duration_seconds is None and not _missing(row["Duration"]):
            exception("mal_anime", row_number, mal_id, "Duration", "invalid_duration", row["Duration"])
        synopsis = None
        if mal_id in synopsis_index:
            matched += 1
            synopsis_row_number, synopsis_row = synopsis_index[mal_id]
            raw_synopsis = synopsis_row["sypnopsis"]
            stripped = raw_synopsis.strip()
            if stripped in BOILERPLATES:
                exception("mal_synopsis", synopsis_row_number, mal_id, "sypnopsis",
                          "boilerplate_synopsis", raw_synopsis)
                counts["boilerplate_synopsis"] += 1
            elif not _missing(stripped):
                synopsis = stripped
            elif stripped == "Unknown":
                counts["unknown_synopsis"] += 1
            else:
                counts["blank_synopsis"] += 1
        anime[str(mal_id)] = {"title": title, "aliases": aliases, "genres": genres,
                              "score": score, "year": year, "type": type_value,
                              "episodes": episodes, "synopsis": synopsis}
        durations[str(mal_id)] = {"duration_seconds": duration_seconds, "scope": scope}
        title_ids[title].append(mal_id)
        for field, value in (("score", score), ("episodes", episodes), ("year", year),
                             ("type", type_value), ("duration", duration_seconds),
                             ("synopsis", synopsis)):
            if value is None:
                counts[f"missing_{field}"] += 1
        if not genres:
            counts["missing_genres"] += 1
    if not anime:
        raise SourceError("normalization produced an empty catalog")
    orphan = sorted(set(synopsis_index) - {int(key) for key in anime})
    for mal_id in orphan:
        row_number, row = synopsis_index[mal_id]
        exclusions.append({"source": "mal_synopsis", "row": row_number, "field": "MAL_ID",
                           "raw_id": row["MAL_ID"], "parsed_id": mal_id,
                           "reason": "orphan_synopsis_id"})
    unmatched = sorted(int(key) for key in anime if int(key) not in synopsis_index)
    exclusions.sort(key=lambda item: (item["source"], item["row"], item["field"], item["reason"]))
    exceptions.sort(key=lambda item: (item["source"], item["row"], item["field"], item["reason"]))
    _validate_artifacts(anime, durations)
    catalog_bytes = _canonical({"schema_version": 1, "anime": anime})
    durations_bytes = _canonical({"schema_version": 1, "anime": durations})
    report = {
        "schema_version": 1, "normalizer_version": NORMALIZER_VERSION,
        "registry_sha256": registry_sha256,
        "provenance": registry["provenance"],
        "sources": [descriptors[source_id] for source_id in ("mal_anime", "mal_synopsis")],
        "files": {"catalog.json": {"sha256": hashlib.sha256(catalog_bytes).hexdigest()},
                  "durations.json": {"sha256": hashlib.sha256(durations_bytes).hexdigest()}},
        "counts": {
            "anime_input_rows": len(anime_rows), "synopsis_input_rows": len(synopsis_rows),
            "retained": len(anime), "excluded": len(exclusions),
            "anime_excluded": sum(item["source"] == "mal_anime" for item in exclusions),
            "synopsis_excluded": sum(item["source"] == "mal_synopsis" for item in exclusions),
            "matched_synopsis": matched, "unmatched_synopsis": len(unmatched),
            "orphan_synopsis": len(orphan),
            "missing": {field: counts[f"missing_{field}"] for field in
                        ("score", "episodes", "year", "type", "genres", "duration", "synopsis")},
            "reasons": dict(sorted(Counter(item["reason"] for item in exclusions + exceptions).items())),
            "blank_synopsis": counts["blank_synopsis"],
            "unknown_synopsis": counts["unknown_synopsis"],
        },
        "exclusions": exclusions, "exceptions": exceptions,
        "unmatched_catalog_ids": unmatched, "orphan_synopsis_ids": orphan,
        "duplicate_titles": [{"title": title, "ids": ids} for title, ids in sorted(title_ids.items())
                             if len(ids) > 1],
    }
    report_bytes = _canonical(report)
    output_dir = Path(output_dir)
    if output_dir.is_symlink() or (output_dir.exists() and not output_dir.is_dir()):
        raise SourceError(f"unsafe output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = (("catalog.json", catalog_bytes), ("durations.json", durations_bytes),
               ("normalization-report.json", report_bytes))
    for name, _ in outputs:
        target = output_dir / name
        if target.is_symlink() or target.is_dir():
            raise SourceError(f"unsafe output target: {target}")
    for name, content in outputs:
        _write_atomic(output_dir / name, content)
    return output_dir / "normalization-report.json"
