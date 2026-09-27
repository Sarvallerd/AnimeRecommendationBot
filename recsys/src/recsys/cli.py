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
    return 0
