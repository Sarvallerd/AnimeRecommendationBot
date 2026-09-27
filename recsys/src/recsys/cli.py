"""Command-line entry point for offline recommendation preparation."""

import argparse
import sys
from collections.abc import Sequence

from . import __version__


def build_parser() -> argparse.ArgumentParser:
    """Construct the public CLI parser."""
    parser = argparse.ArgumentParser(prog="recsys", allow_abbrev=False)
    parser.add_argument(
        "--version",
        action="version",
        version=f"recsys {__version__}",
    )
    commands = parser.add_subparsers(dest="command")
    obtain_parser = commands.add_parser("obtain", help="download and verify pinned source data")
    obtain_parser.add_argument("--data-dir", default="data/raw", help="source cache directory")
    obtain_parser.add_argument("--offline", action="store_true", help="use verified cached files only")
    normalize_parser = commands.add_parser("normalize", help="normalize verified MAL CSV files")
    normalize_parser.add_argument("--anime-csv", required=True, help="path to anime.csv")
    normalize_parser.add_argument("--synopsis-csv", required=True, help="path to anime_with_synopsis.csv")
    normalize_parser.add_argument("--output-dir", required=True, help="directory for normalization artifacts")
    build_parser = commands.add_parser("build", help="build pinned GloVe cosine neighbors")
    build_parser.add_argument("--catalog", required=True, help="path to normalized catalog.json")
    build_parser.add_argument("--normalization-report", required=True, help="path to normalization-report.json")
    build_parser.add_argument("--glove", required=True, help="path to glove.6B.300d.txt")
    build_parser.add_argument("--output-dir", required=True, help="directory for build artifacts")
    build_parser.add_argument("--block-size", type=int, default=256, help="similarity rows per block (1..256)")
    export_parser = commands.add_parser("export", help="publish an immutable validated bundle")
    export_parser.add_argument("--catalog", required=True, help="path to catalog.json")
    export_parser.add_argument("--neighbors", required=True, help="path to neighbors.json")
    export_parser.add_argument("--build-report", required=True, help="path to build-report.json")
    export_parser.add_argument("--normalization-report", required=True,
                               help="path to normalization-report.json")
    export_parser.add_argument("--output-dir", required=True, help="bundle store directory")
    validate_parser = commands.add_parser("validate", help="validate a published bundle")
    validate_parser.add_argument("bundle_dir", help="bundle directory")
    quality_parser = commands.add_parser("quality", help="run offline recommendation quality diagnostics")
    quality_commands = quality_parser.add_subparsers(dest="quality_command", required=True)
    baseline_parser = quality_commands.add_parser("baseline", help="run the pinned genre baseline")
    baseline_parser.add_argument("--catalog", required=True, help="path to pinned catalog.json")
    baseline_parser.add_argument("--output-dir", required=True, help="directory for diagnostic artifacts")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return its status code."""
    args = sys.argv[1:] if argv is None else argv
    parser = build_parser()
    parsed = parser.parse_args(args)
    if not args:
        parser.print_help()
    elif parsed.command == "obtain":
        from pathlib import Path

        from .obtain import obtain
        from .sources import SourceError

        try:
            obtain(Path(parsed.data_dir), offline=parsed.offline)
        except (SourceError, OSError) as exc:
            print(f"recsys obtain: {exc}", file=sys.stderr)
            return 1
    elif parsed.command == "normalize":
        from pathlib import Path

        from .normalize import normalize
        from .sources import SourceError

        try:
            normalize(Path(parsed.anime_csv), Path(parsed.synopsis_csv), Path(parsed.output_dir))
        except (SourceError, OSError, ValueError) as exc:
            print(f"recsys normalize: {exc}", file=sys.stderr)
            return 1
    elif parsed.command == "build":
        from pathlib import Path

        from .build import BuildError, build
        from .sources import SourceError

        try:
            build(Path(parsed.catalog), Path(parsed.normalization_report), Path(parsed.glove),
                  Path(parsed.output_dir), block_size=parsed.block_size)
        except (BuildError, SourceError, OSError, ValueError) as exc:
            print(f"recsys build: {exc}", file=sys.stderr)
            return 1
    elif parsed.command == "export":
        from pathlib import Path

        from .bundle import ContractError, check_bundle
        from .export import export_bundle
        from .sources import SourceError

        try:
            path = export_bundle(Path(parsed.catalog), Path(parsed.neighbors),
                                 Path(parsed.build_report), Path(parsed.normalization_report),
                                 Path(parsed.output_dir))
            identity, count = check_bundle(path)
        except (ContractError, SourceError, OSError, ValueError) as exc:
            print(f"recsys export: {exc}", file=sys.stderr)
            return 1
        print(f"{path} {identity} records={count}")
    elif parsed.command == "validate":
        from pathlib import Path

        from .bundle import ContractError, check_bundle

        try:
            identity, count = check_bundle(Path(parsed.bundle_dir))
        except (ContractError, OSError) as exc:
            print(f"recsys validate: {exc}", file=sys.stderr)
            return 1
        print(f"{identity} records={count}")
    elif parsed.command == "quality" and parsed.quality_command == "baseline":
        from pathlib import Path

        from .quality import QualityError, run_baseline

        try:
            run_baseline(Path(parsed.catalog), Path(parsed.output_dir))
        except (QualityError, OSError) as exc:
            print(f"recsys quality baseline: {exc}", file=sys.stderr)
            return 1
    return 0
