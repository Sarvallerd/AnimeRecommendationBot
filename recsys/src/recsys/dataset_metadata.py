"""Parse pinned Neelagiri details into the existing catalog wire format."""

import ast
import csv
import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path

from .bundle import MAX_ID, check_catalog

DETAIL_COLUMNS = ("mal_id", "title", "title_japanese", "url", "image_url", "type", "status",
                  "score", "scored_by", "start_date", "end_date", "synopsis", "rank",
                  "popularity", "members", "favorites", "genres", "studios", "themes",
                  "demographics", "source", "rating", "episodes", "season", "year",
                  "producers", "explicit_genres", "licensors", "streaming")
BOILERPLATES = {
    "No synopsis has been added for this series yet.",
    "No synopsis has been added for this series yet. Click here to update this information.",
}
ID_PATTERN = re.compile(r"[0-9]+\Z", re.ASCII)
DATE_PATTERN = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", re.ASCII)


@dataclass(frozen=True)
class MetadataResult:
    catalog: dict
    facts: dict
    qa: dict

    @property
    def ids(self) -> set[int]:
        return {int(key) for key in self.catalog["anime"]}


def _id(value: str | None) -> int | None:
    value = (value or "").strip()
    if not ID_PATTERN.fullmatch(value):
        return None
    number = int(value)
    return number if 1 <= number <= MAX_ID else None


def _decimal(value: str | None) -> Decimal | None:
    value = (value or "").strip()
    if not value:
        return None
    try:
        number = Decimal(value)
    except InvalidOperation:
        return None
    return number if number.is_finite() else None


def _score(value: str | None) -> float | None:
    number = _decimal(value)
    if number is None or not 1 <= number <= 10:
        return None
    result = float(number)
    return result if math.isfinite(result) else None


def _episodes(value: str | None) -> int | None:
    number = _decimal(value)
    if number is None or number <= 0 or number != number.to_integral_value():
        return None
    return int(number)


def _seasonal_year(value: str | None) -> int | None:
    number = _decimal(value)
    if (number is None or number != number.to_integral_value()
            or not 1 <= number <= 9999):
        return None
    return int(number)


def _date(value: str | None) -> str | None:
    prefix = (value or "").strip()[:10]
    if not DATE_PATTERN.fullmatch(prefix):
        return None
    try:
        return date.fromisoformat(prefix).isoformat()
    except ValueError:
        return None


def _genres(value: str | None) -> tuple[list[str], bool]:
    try:
        parsed = ast.literal_eval(value or "")
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return [], False
    if not isinstance(parsed, list) or any(not isinstance(item, str) for item in parsed):
        return [], False
    result = []
    for item in parsed:
        cleaned = item.strip()
        if cleaned and cleaned not in result:
            result.append(cleaned)
    return result, True


def _status(value: str | None) -> str:
    normalized = re.sub(r"[\s-]+", "_", (value or "").strip().lower())
    return {
        "finished_airing": "finished_airing", "currently_airing": "currently_airing",
        "not_yet_aired": "not_yet_aired",
    }.get(normalized, "unknown")


def _synopsis(value: str | None) -> str | None:
    cleaned = (value or "").strip()
    if not cleaned or " ".join(cleaned.split()) in BOILERPLATES:
        return None
    return cleaned


def normalize_metadata(details_path: Path) -> MetadataResult:
    """Read a verified details copy; no raw metadata values appear in errors."""
    catalog: dict[str, dict] = {}
    facts: dict[str, dict] = {}
    counts = Counter()
    duplicate_ids: set[str] = set()
    seen_ids: set[str] = set()
    try:
        with Path(details_path).open("r", encoding="utf-8", newline="") as source:
            reader = csv.DictReader(source, strict=True)
            if tuple(reader.fieldnames or ()) != DETAIL_COLUMNS:
                raise ValueError("metadata_schema_invalid")
            for row in reader:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError("metadata_schema_invalid")
                counts["metadata_rows"] += 1
                mal = _id(row["mal_id"])
                if mal is None:
                    counts["invalid_mal_id"] += 1
                    continue
                key = str(mal)
                if key in seen_ids:
                    duplicate_ids.add(key)
                    catalog.pop(key, None)
                    facts.pop(key, None)
                    counts["duplicate_mal_id_rows"] += 1
                    continue
                seen_ids.add(key)
                primary_title = row["title"].strip()
                title = primary_title or row["title_japanese"].strip()
                if not title:
                    counts["missing_title"] += 1
                    continue
                if not primary_title:
                    counts["japanese_title_fallback"] += 1
                japanese = row["title_japanese"].strip()
                if not japanese:
                    counts["missing_japanese_alias"] += 1
                aliases = [japanese] if japanese and japanese != title else []
                genres, genres_valid = _genres(row["genres"])
                if not row["genres"].strip():
                    counts["missing_genres"] += 1
                elif not genres_valid:
                    counts["invalid_genres"] += 1
                elif not genres:
                    counts["empty_genres"] += 1
                score = _score(row["score"])
                if not row["score"].strip():
                    counts["missing_score"] += 1
                elif score is None:
                    counts["invalid_score"] += 1
                episodes = _episodes(row["episodes"])
                if not row["episodes"].strip():
                    counts["missing_episodes"] += 1
                elif episodes is None:
                    counts["invalid_episodes"] += 1
                release_date = _date(row["start_date"])
                if not row["start_date"].strip():
                    counts["missing_start_date"] += 1
                elif release_date is None:
                    counts["invalid_start_date"] += 1
                seasonal = _seasonal_year(row["year"])
                if not row["year"].strip():
                    counts["missing_seasonal_year"] += 1
                elif seasonal is None:
                    counts["invalid_seasonal_year"] += 1
                year = int(release_date[:4]) if release_date else None
                if year is not None and seasonal is not None and year != seasonal:
                    counts["first_release_seasonal_year_disagreement"] += 1
                synopsis = _synopsis(row["synopsis"])
                if row["synopsis"].strip() and synopsis is None:
                    counts["synopsis_boilerplate"] += 1
                if not row["synopsis"].strip():
                    counts["missing_synopsis"] += 1
                if not row["type"].strip():
                    counts["missing_type"] += 1
                airing_status = _status(row["status"])
                if airing_status == "unknown":
                    counts["unknown_airing_status"] += 1
                catalog[key] = {
                    "title": title, "aliases": aliases, "genres": genres, "score": score,
                    "year": year, "type": row["type"].strip() or None,
                    "episodes": episodes, "synopsis": synopsis,
                }
                facts[key] = {
                    "first_release_date": release_date,
                    "airing_status": airing_status,
                    "seasonal_year": seasonal,
                }
    except (OSError, UnicodeError, csv.Error):
        raise ValueError("metadata_schema_invalid") from None
    counts["catalog_items"] = len(catalog)
    counts["duplicate_mal_ids"] = len(duplicate_ids)
    payload = {"schema_version": 1, "anime": catalog}
    try:
        check_catalog(payload)
    except ValueError:
        raise ValueError("metadata_schema_invalid") from None
    return MetadataResult(payload, {"schema_version": 1, "anime": facts}, dict(sorted(counts.items())))
