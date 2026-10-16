"""Small verified CSV fixtures for deterministic MAL normalization."""

import csv
import hashlib
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Sequence, TypedDict, cast

from recsys.contracts.bundle import AnimeRecord, check_catalog
from recsys.data.mal2020 import (
    BOILERPLATES,
    DurationRecord,
    normalize,
)
from recsys.data.sources import SourceError, SourceRegistry

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from contracts.check_bundle import read_canonical  # noqa: E402


class _ReportCounts(TypedDict):
    reasons: dict[str, int]
    missing: dict[str, int]
    unknown_synopsis: int
    blank_synopsis: int
    excluded: int
    matched_synopsis: int


class _Reason(TypedDict):
    reason: str


class _Report(TypedDict):
    files: dict[str, dict[str, str]]
    duplicate_titles: list[dict[str, object]]
    orphan_synopsis_ids: list[int]
    unmatched_catalog_ids: list[int]
    exceptions: list[_Reason]
    exclusions: list[_Reason]
    counts: _ReportCounts


def _object(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise AssertionError("expected JSON object")
    return value


def _items(value: object) -> list[object]:
    if not isinstance(value, list):
        raise AssertionError("expected JSON array")
    return value


def _integer(value: object) -> int:
    if type(value) is not int:
        raise AssertionError("expected JSON integer")
    return value


def _text(value: object) -> str:
    if not isinstance(value, str):
        raise AssertionError("expected JSON string")
    return value


def _int_map(value: object) -> dict[str, int]:
    return {key: _integer(item) for key, item in _object(value).items()}


def _durations(value: object) -> dict[str, DurationRecord]:
    anime = _object(_object(value)["anime"])
    result: dict[str, DurationRecord] = {}
    for key, raw in anime.items():
        record = _object(raw)
        seconds = record["duration_seconds"]
        scope = record["scope"]
        result[key] = {
            "duration_seconds": None if seconds is None else _integer(seconds),
            "scope": None if scope is None else _text(scope),
        }
    return result


def _report(value: object) -> _Report:
    document = _object(value)
    counts = _object(document["counts"])
    return {
        "files": {
            key: {"sha256": _text(_object(entry)["sha256"])}
            for key, entry in _object(document["files"]).items()
        },
        "duplicate_titles": [_object(item) for item in _items(document["duplicate_titles"])],
        "orphan_synopsis_ids": [_integer(item) for item in _items(document["orphan_synopsis_ids"])],
        "unmatched_catalog_ids": [
            _integer(item) for item in _items(document["unmatched_catalog_ids"])
        ],
        "exceptions": [
            {"reason": _text(_object(item)["reason"])} for item in _items(document["exceptions"])
        ],
        "exclusions": [
            {"reason": _text(_object(item)["reason"])} for item in _items(document["exclusions"])
        ],
        "counts": {
            "reasons": _int_map(counts["reasons"]),
            "missing": _int_map(counts["missing"]),
            "unknown_synopsis": _integer(counts["unknown_synopsis"]),
            "blank_synopsis": _integer(counts["blank_synopsis"]),
            "excluded": _integer(counts["excluded"]),
            "matched_synopsis": _integer(counts["matched_synopsis"]),
        },
    }


ANIME_HEADER = [
    "MAL_ID",
    "Name",
    "English name",
    "Japanese name",
    "Genres",
    "Score",
    "Episodes",
    "Aired",
    "Premiered",
    "Type",
    "Duration",
]
SYNOPSIS_HEADER = ["MAL_ID", "sypnopsis"]


def csv_bytes(header: Sequence[str], rows: Sequence[Sequence[str]]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def anime(
    mal_id: str = "1",
    name: str = "Título, один",
    english: str = "English, Title",
    japanese: str = "日本語",
    genres: str = "Action, Drama",
    score: str = "8.5",
    episodes: str = "12",
    aired: str = "Apr 3, 1998 to Apr 4, 1999",
    premiered: str = "Spring 1998",
    type_: str = "TV",
    duration: str = "1 hr. 2 min. 3 sec. per ep.",
) -> list[str]:
    return [
        mal_id,
        name,
        english,
        japanese,
        genres,
        score,
        episodes,
        aired,
        premiered,
        type_,
        duration,
    ]


class NormalizeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def run_fixture(
        self,
        anime_rows: Sequence[Sequence[str]],
        synopsis_rows: Sequence[Sequence[str]],
        *,
        anime_header: Sequence[str] = ANIME_HEADER,
        synopsis_header: Sequence[str] = SYNOPSIS_HEADER,
        output: str = "out",
        raw_anime: bytes | None = None,
    ) -> tuple[
        dict[str, AnimeRecord],
        dict[str, DurationRecord],
        _Report,
        tuple[bytes, bytes, bytes],
        SourceRegistry,
    ]:
        a = csv_bytes(anime_header, anime_rows) if raw_anime is None else raw_anime
        s = csv_bytes(synopsis_header, synopsis_rows)
        anime_path = self.root / "anime.csv"
        synopsis_path = self.root / "synopsis.csv"
        anime_path.write_bytes(a)
        synopsis_path.write_bytes(s)
        registry: SourceRegistry = {
            "schema_version": 1,
            "provenance": {"fixture": "local"},
            "sources": [
                {
                    "id": "mal_anime",
                    "filename": "anime.csv",
                    "size": len(a),
                    "sha256": hashlib.sha256(a).hexdigest(),
                    "url": "https://example.test/anime.csv",
                },
                {
                    "id": "mal_synopsis",
                    "filename": "synopsis.csv",
                    "size": len(s),
                    "sha256": hashlib.sha256(s).hexdigest(),
                    "url": "https://example.test/synopsis.csv",
                },
            ],
        }
        path = normalize(anime_path, synopsis_path, self.root / output, registry=registry)
        catalog, catalog_raw = read_canonical(path.parent / "catalog.json")
        durations_document, durations_raw = read_canonical(path.parent / "durations.json")
        report_document, report_raw = read_canonical(path)
        check_catalog(catalog)
        durations = _durations(durations_document)
        report = _report(report_document)
        self.assertEqual(set(catalog["anime"]), set(durations))
        self.assertEqual(
            hashlib.sha256(catalog_raw).hexdigest(), report["files"]["catalog.json"]["sha256"]
        )
        self.assertEqual(
            hashlib.sha256(durations_raw).hexdigest(), report["files"]["durations.json"]["sha256"]
        )
        return (
            cast("dict[str, AnimeRecord]", catalog["anime"]),
            durations,
            report,
            (catalog_raw, durations_raw, report_raw),
            registry,
        )

    def test_join_titles_aliases_dates_duration_and_order(self) -> None:
        rows = [
            anime(),
            anime(
                "2",
                "Título, один",
                "Título, один",
                "日本語",
                "Unknown, Action, Action",
                "Unknown",
                "Unknown",
                "Unknown",
                "Winter 2020",
                "Unknown",
                "3 sec.",
            ),
            anime(
                "3",
                "Unknown",
                "Fallback",
                "Fallback",
                "Unknown",
                "5",
                "1",
                "2001",
                "Unknown",
                "OVA",
                "Unknown",
            ),
        ]
        synopses = [["2", "A genuine plot with synopsis words."], ["1", "Plot"], ["999", "Orphan"]]
        entries, durations, report, raw, registry = self.run_fixture(rows, synopses)
        self.assertEqual(entries["1"]["aliases"], ["English, Title", "日本語"])
        self.assertEqual(entries["1"]["genres"], ["Action", "Drama"])
        self.assertEqual(entries["2"]["synopsis"], synopses[0][1])
        self.assertEqual(entries["2"]["genres"], ["Action"])
        self.assertEqual(entries["3"]["title"], "Fallback")
        self.assertEqual(entries["3"]["aliases"], [])
        self.assertEqual(entries["2"]["year"], 2020)
        self.assertEqual(entries["3"]["year"], 2001)
        self.assertEqual(durations["1"], {"duration_seconds": 3723, "scope": "per_episode"})
        self.assertEqual(durations["2"], {"duration_seconds": 3, "scope": "unspecified"})
        self.assertEqual(durations["3"], {"duration_seconds": None, "scope": None})
        self.assertEqual(report["duplicate_titles"], [{"title": "Título, один", "ids": [1, 2]}])
        self.assertEqual(report["orphan_synopsis_ids"], [999])
        self.assertEqual(report["counts"]["reasons"]["orphan_synopsis_id"], 1)
        self.assertEqual(report["unmatched_catalog_ids"], [3])
        self.assertIn("title_fallback", [e["reason"] for e in report["exceptions"]])
        again = self.run_fixture(rows, synopses, output="again")[3]
        self.assertEqual(raw, again)
        reordered = self.run_fixture(
            list(reversed(rows)), list(reversed(synopses)), output="reordered"
        )
        self.assertEqual(entries, reordered[0])
        self.assertEqual(durations, reordered[1])

    def test_missing_invalid_and_disagreement(self) -> None:
        rows = [
            anime(
                " 01 ",
                score="NaN",
                episodes="0",
                aired="Dec 31, 2019",
                premiered="Winter 2020",
                duration="0 sec.",
            ),
            anime(
                "2",
                score="inf",
                episodes="no",
                aired="Unknown",
                premiered="Unknown",
                duration="garbage",
            ),
            anime(
                "3",
                score="11",
                episodes="Unknown",
                aired="Apr, 2021 to ?",
                premiered="Unknown",
                duration="2 min.",
            ),
            anime(
                "4",
                score="Unknown",
                episodes="1",
                aired="Unknown",
                premiered="Unknown",
                duration="Unknown",
            ),
        ]
        entries, durations, report, _, _ = self.run_fixture(rows, [])
        self.assertEqual(entries["1"]["year"], 2019)
        self.assertEqual(entries["3"]["year"], 2021)
        self.assertEqual(entries["1"]["score"], None)
        self.assertEqual(entries["1"]["episodes"], None)
        self.assertEqual(durations["3"]["duration_seconds"], 120)
        reasons = [e["reason"] for e in report["exceptions"]]
        self.assertEqual(reasons.count("invalid_score"), 3)
        self.assertEqual(reasons.count("invalid_episodes"), 2)
        self.assertEqual(reasons.count("invalid_duration"), 2)
        self.assertIn("noncanonical_id", reasons)
        self.assertIn("aired_premiered_disagreement", reasons)
        self.assertEqual(report["counts"]["missing"]["score"], 4)
        huge = self.run_fixture(
            [anime(episodes="9" * 5000, duration="9" * 5000 + " sec.")], [], output="huge"
        )
        self.assertIsNone(huge[0]["1"]["episodes"])
        self.assertIsNone(huge[1]["1"]["duration_seconds"])
        invalid_date = self.run_fixture(
            [anime(aired="Feb 30, 2020", premiered="Spring 2021")], [], output="invalid_date"
        )
        self.assertEqual(invalid_date[0]["1"]["year"], 2021)
        self.assertIn("invalid_year", [e["reason"] for e in invalid_date[2]["exceptions"]])

    def test_boilerplates_blank_and_similar_genuine_text(self) -> None:
        rows = [anime(str(i), name=f"Title {i}") for i in range(1, 7)]
        synopses = [[str(i), text] for i, text in enumerate(sorted(BOILERPLATES), 1)]
        synopses += [
            ["4", "  "],
            ["5", "A plot says No synopsis has been added for this series yet."],
        ]
        entries, _, report, _, _ = self.run_fixture(rows, synopses)
        self.assertEqual([entries[str(i)]["synopsis"] for i in range(1, 5)], [None] * 4)
        self.assertIsNotNone(entries["5"]["synopsis"])
        self.assertEqual(report["counts"]["reasons"]["boilerplate_synopsis"], 3)
        self.assertEqual(report["counts"]["blank_synopsis"], 1)
        self.assertEqual(report["unmatched_catalog_ids"], [6])

    def test_unknown_synopsis_is_missing(self) -> None:
        entries, _, report, _, _ = self.run_fixture([anime()], [["1", "Unknown"]])
        self.assertIsNone(entries["1"]["synopsis"])
        self.assertEqual(report["counts"]["unknown_synopsis"], 1)

    def test_invalid_ids_titles_and_bad_source_files(self) -> None:
        rows = [
            anime("0"),
            anime("oops"),
            anime("1", name="Unknown", english="Unknown", japanese="Unknown"),
            anime("2"),
        ]
        entries, _, report, _, _ = self.run_fixture(rows, [["bad", "Plot"], ["2", "Plot"]])
        self.assertEqual(set(entries), {"2"})
        self.assertEqual(report["counts"]["excluded"], 4)
        self.assertEqual(report["counts"]["matched_synopsis"], 1)
        with self.assertRaisesRegex(SourceError, "duplicate normalized MAL_ID"):
            self.run_fixture([anime("1"), anime("01")], [], output="dupe")
        entries, _, report, _, _ = self.run_fixture(
            [anime("0" * 5000), anime("2")], [], output="long_id"
        )
        self.assertEqual(set(entries), {"2"})
        self.assertEqual(report["exclusions"][0]["reason"], "invalid_id")
        with self.assertRaisesRegex(SourceError, "duplicate normalized MAL_ID"):
            self.run_fixture([anime()], [["2", "A"], ["02", "B"]], output="syn_dupe")
        with self.assertRaisesRegex(SourceError, "invalid consumed headers"):
            self.run_fixture(
                [anime()], [], anime_header=ANIME_HEADER[:-1] + ["MAL_ID"], output="headers"
            )
        with self.assertRaisesRegex(SourceError, "columns"):
            self.run_fixture([anime()[:-1]], [], output="width")
        with self.assertRaisesRegex(SourceError, "invalid UTF-8"):
            self.run_fixture([], [], raw_anime=b"\xff", output="utf8")
        with self.assertRaisesRegex(SourceError, "invalid UTF-8 or CSV"):
            self.run_fixture(
                [], [], raw_anime=csv_bytes(ANIME_HEADER, []) + b'"1,broken\n', output="csv"
            )

    def test_hash_mismatch_and_empty_result(self) -> None:
        a = csv_bytes(ANIME_HEADER, [anime()])
        s = csv_bytes(SYNOPSIS_HEADER, [])
        a_path, s_path = self.root / "a.csv", self.root / "s.csv"
        a_path.write_bytes(a)
        s_path.write_bytes(s)
        registry: SourceRegistry = {
            "schema_version": 1,
            "provenance": {},
            "sources": [
                {
                    "id": name,
                    "filename": filename,
                    "size": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                    "url": "https://example.test/x",
                }
                for name, filename, content in (
                    ("mal_anime", "a.csv", a),
                    ("mal_synopsis", "s.csv", s),
                )
            ],
        }
        a_path.write_bytes(a + b"x")
        with self.assertRaisesRegex(SourceError, "size/SHA256 mismatch"):
            normalize(a_path, s_path, self.root / "mismatch", registry=registry)
        self.assertFalse((self.root / "mismatch").exists())
        cli = subprocess.run(
            [
                sys.executable,
                "-m",
                "recsys",
                "normalize",
                "--anime-csv",
                str(a_path),
                "--synopsis-csv",
                str(s_path),
                "--output-dir",
                str(self.root / "cli"),
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(cli.returncode, 1)
        self.assertIn("size/SHA256 mismatch", cli.stderr)
        self.assertNotIn("Traceback", cli.stderr)
        self.assertFalse((self.root / "cli").exists())
        a_path.write_bytes(csv_bytes(ANIME_HEADER, []))
        registry["sources"][0]["size"] = a_path.stat().st_size
        registry["sources"][0]["sha256"] = hashlib.sha256(a_path.read_bytes()).hexdigest()
        with self.assertRaisesRegex(SourceError, "empty catalog"):
            normalize(a_path, s_path, self.root / "empty", registry=registry)


if __name__ == "__main__":
    unittest.main()
