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
    from .sources import DEFAULT_SNAPSHOT, SNAPSHOT_IDS

    obtain_parser.add_argument("--snapshot", choices=SNAPSHOT_IDS, required=True,
                               help="pinned source snapshot")
    obtain_parser.add_argument("--data-dir", help="source cache directory (overrides snapshot default)")
    obtain_parser.add_argument("--offline", action="store_true", help="use verified cached files only")
    normalize_parser = commands.add_parser("normalize", help="normalize verified MAL CSV files")
    normalize_parser.add_argument("--anime-csv", required=True, help="path to anime.csv")
    normalize_parser.add_argument("--synopsis-csv", required=True, help="path to anime_with_synopsis.csv")
    normalize_parser.add_argument("--output-dir", required=True, help="directory for normalization artifacts")
    dataset_parser = commands.add_parser("dataset", help="normalize or validate a research dataset")
    dataset_commands = dataset_parser.add_subparsers(dest="dataset_command", required=True)
    dataset_normalize = dataset_commands.add_parser("normalize", help="normalize a pinned research snapshot")
    dataset_normalize.add_argument("--snapshot", required=True, choices=("neelagiri-2025-v1",))
    dataset_normalize.add_argument("--data-dir", required=True)
    dataset_normalize.add_argument("--output-dir", required=True)
    dataset_normalize.add_argument("--scratch-dir", required=True)
    dataset_normalize.add_argument("--memory-limit-mib", type=int, default=2048)
    dataset_normalize.add_argument("--threads", type=int, default=2)
    dataset_normalize.add_argument("--spill-limit-mib", type=int, default=32768)
    dataset_validate = dataset_commands.add_parser("validate", help="validate an immutable research dataset")
    dataset_validate.add_argument("dataset_dir")
    aliases_parser = commands.add_parser("aliases", help="normalize or validate pinned title aliases")
    aliases_commands = aliases_parser.add_subparsers(dest="aliases_command", required=True)
    aliases_normalize = aliases_commands.add_parser("normalize", help="create an immutable alias overlay")
    aliases_normalize.add_argument("--snapshot", required=True, choices=("nafiul-2026-v1",))
    aliases_normalize.add_argument("--data-dir", required=True)
    aliases_normalize.add_argument("--base-dir", required=True)
    aliases_normalize.add_argument("--base-identity", required=True)
    aliases_normalize.add_argument("--output-dir", required=True)
    aliases_validate = aliases_commands.add_parser("validate", help="revalidate pinned alias inputs and output")
    aliases_validate.add_argument("aliases_dir")
    aliases_validate.add_argument("--data-dir", required=True)
    aliases_validate.add_argument("--base-dir", required=True)
    aliases_validate.add_argument("--base-identity", required=True)
    aliases_validate.add_argument("--expected-identity")
    release_parser = commands.add_parser("release", help="compose, build, and export a catalog release")
    release_commands = release_parser.add_subparsers(dest="release_command", required=True)
    release_info = release_commands.add_parser("info", help="show the pinned current release")
    release_info.add_argument("--release", default="current", choices=("current", "neelagiri-2025-nafiul-2026-v1"))
    release_prepare = release_commands.add_parser("prepare", help="prepare a pinned release offline")
    release_prepare.add_argument("--release", default="current", choices=("current", "neelagiri-2025-nafiul-2026-v1", "mal-2020-glove-v1"))
    release_prepare.add_argument("--data-dir", required=True)
    release_compose = release_commands.add_parser("compose")
    for flag in ("base-dir", "base-identity", "aliases-dir", "alias-identity",
                 "alias-data-dir", "output-dir"):
        release_compose.add_argument(f"--{flag}", required=True)
    release_validate = release_commands.add_parser("validate")
    release_validate.add_argument("release_dir")
    for flag in ("release-identity", "base-dir", "aliases-dir", "alias-data-dir"):
        release_validate.add_argument(f"--{flag}", required=True)
    release_build = release_commands.add_parser("build")
    release_export = release_commands.add_parser("export")
    for command in (release_build, release_export):
        for flag in ("release-dir", "release-identity", "base-dir", "aliases-dir",
                     "alias-data-dir", "output-dir"):
            command.add_argument(f"--{flag}", required=True)
    release_build.add_argument("--glove", required=True)
    release_build.add_argument("--block-size", type=int, default=256)
    release_export.add_argument("--neighbors", required=True)
    release_export.add_argument("--build-report", required=True)
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
    compare_parser = quality_commands.add_parser("compare", help="compare three pinned recommendation methods")
    for option in ("bundle-dir", "normalization-report", "build-report", "anime-csv",
                   "synopsis-csv", "glove", "lockfile", "code-revision", "output-dir"):
        compare_parser.add_argument(f"--{option}", required=True)
    compare_parser.add_argument("--repetitions", type=int, default=3)
    summary_parser = quality_commands.add_parser("summarize", help="validate judgments and summarize comparison")
    for option in ("comparison", "catalog", "measurements", "output-dir"):
        summary_parser.add_argument(f"--{option}", required=True)
    summary_parser.add_argument("--assessment", required=True, action="append")
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
        from .sources import DEFAULT_SNAPSHOT, SourceError

        try:
            data_dir = (Path(parsed.data_dir) if parsed.data_dir is not None else
                        Path("data/raw") if parsed.snapshot == DEFAULT_SNAPSHOT else
                        Path("data/raw") / parsed.snapshot)
            obtain(data_dir, offline=parsed.offline, snapshot=parsed.snapshot)
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
    elif parsed.command == "dataset":
        from pathlib import Path
        from .dataset import DatasetError, normalize_dataset, validate_dataset

        try:
            if parsed.dataset_command == "normalize":
                import hashlib
                import json

                manifest = normalize_dataset(
                    parsed.snapshot, Path(parsed.data_dir), Path(parsed.output_dir),
                    scratch_dir=Path(parsed.scratch_dir),
                    memory_limit_mib=parsed.memory_limit_mib, threads=parsed.threads,
                    spill_limit_mib=parsed.spill_limit_mib)
                manifest_bytes = manifest.read_bytes()
                qa = json.loads((manifest.parent / "dataset-qa.json").read_bytes())
                counts = qa["counts"]
                identity = "sha256:" + hashlib.sha256(manifest_bytes).hexdigest()
                catalog_items = counts["catalog_items"]
                interactions = counts["accepted_pairs"]
                users = counts["accepted_users"]
                positive_users = counts["positive_users_ge_15"]
            else:
                summary = validate_dataset(Path(parsed.dataset_dir))
                identity = summary.identity
                catalog_items = summary.catalog_items
                interactions = summary.interactions
                users = summary.users
                positive_users = summary.positive_users_ge_15
        except (DatasetError, OSError) as exc:
            code = str(exc) if isinstance(exc, DatasetError) else "dataset_integrity_error"
            print(f"recsys dataset: {code}", file=sys.stderr)
            return 1
        print(f"{identity} catalog={catalog_items} "
              f"interactions={interactions} users={users} "
              f"positive_users_ge_15={positive_users}")
    elif parsed.command == "aliases":
        from pathlib import Path
        import hashlib

        from .aliases import AliasError, normalize_aliases, validate_aliases

        try:
            if parsed.aliases_command == "normalize":
                manifest = normalize_aliases(
                    Path(parsed.data_dir), Path(parsed.base_dir), Path(parsed.output_dir),
                    snapshot=parsed.snapshot, expected_base_identity=parsed.base_identity)
                identity = "sha256:" + hashlib.sha256(manifest.read_bytes()).hexdigest()
                import json
                qa = json.loads((manifest.parent / "alias-qa.json").read_bytes())["counts"]
                items, aliases, english, main = (qa[name] for name in
                    ("emitted_items", "emitted_strings", "emitted_english", "emitted_main"))
            else:
                summary = validate_aliases(
                    Path(parsed.aliases_dir), data_dir=Path(parsed.data_dir),
                    base_dir=Path(parsed.base_dir), expected_base_identity=parsed.base_identity,
                    expected_identity=parsed.expected_identity)
                identity, items, aliases, english, main = (
                    summary.identity, summary.items, summary.aliases, summary.english, summary.main)
        except (AliasError, OSError) as exc:
            code = str(exc) if isinstance(exc, AliasError) else "alias_integrity_error"
            print(f"recsys aliases: {code}", file=sys.stderr)
            return 1
        print(f"{identity} items={items} aliases={aliases} english={english} main={main}")
    elif parsed.command == "release":
        from pathlib import Path
        import hashlib
        from .release import ReleaseError, compose_release, validate_release
        from .release_build import ReleaseBuildError, build_release, export_release_bundle
        from .release_prepare import PrepareError
        try:
            if parsed.release_command == "info":
                from .release_prepare import release_info
                info = release_info(parsed.release)
                print(f"{info['release_id']} catalog={info['semantic_expectations']['catalog_count']} "
                      f"bundle={info['reference']['bundle_identity']}")
            elif parsed.release_command == "prepare":
                from .release_prepare import prepare_release
                path, identity, count = prepare_release(Path(parsed.data_dir), parsed.release)
                print(f"{path} {identity} records={count}")
                print(f"ARB_BUNDLE_DIR={path}")
            elif parsed.release_command == "compose":
                manifest = compose_release(
                    Path(parsed.base_dir), Path(parsed.aliases_dir), Path(parsed.alias_data_dir),
                    Path(parsed.output_dir), expected_base_identity=parsed.base_identity,
                    expected_alias_identity=parsed.alias_identity)
                identity = "sha256:" + hashlib.sha256(manifest.read_bytes()).hexdigest()
                import json
                counts = json.loads(manifest.read_bytes())["counts"]
                print(f"{identity} catalog={counts['catalog_count']} "
                      f"enrichment={counts['enrichment_items']} "
                      f"changed={counts['catalog_aliases_changed_items']}")
            elif parsed.release_command == "validate":
                summary = validate_release(
                    Path(parsed.release_dir), expected_identity=parsed.release_identity,
                    base_dir=Path(parsed.base_dir), aliases_dir=Path(parsed.aliases_dir),
                    alias_data_dir=Path(parsed.alias_data_dir))
                print(f"{summary.identity} catalog={summary.catalog_count} "
                      f"enrichment={summary.enrichment_items} changed={summary.catalog_aliases_changed_items}")
            elif parsed.release_command == "build":
                report = build_release(
                    Path(parsed.release_dir), Path(parsed.glove), Path(parsed.output_dir),
                    expected_release_identity=parsed.release_identity,
                    base_dir=Path(parsed.base_dir), aliases_dir=Path(parsed.aliases_dir),
                    alias_data_dir=Path(parsed.alias_data_dir), block_size=parsed.block_size)
                print(f"{report} release={parsed.release_identity}")
            else:
                path = export_release_bundle(
                    Path(parsed.release_dir), Path(parsed.neighbors), Path(parsed.build_report),
                    Path(parsed.output_dir), expected_release_identity=parsed.release_identity,
                    base_dir=Path(parsed.base_dir), aliases_dir=Path(parsed.aliases_dir),
                    alias_data_dir=Path(parsed.alias_data_dir))
                from .bundle import check_bundle
                identity, count = check_bundle(path)
                print(f"{path} {identity} records={count}")
        except (ReleaseError, ReleaseBuildError, PrepareError, OSError, ValueError) as exc:
            code = str(exc) if isinstance(exc, (ReleaseError, ReleaseBuildError, PrepareError)) else "release_integrity_error"
            print(f"recsys release: {code}", file=sys.stderr)
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
    elif parsed.command == "quality" and parsed.quality_command == "compare":
        from pathlib import Path
        from .evaluation import EvaluationError, compare
        try:
            compare(Path(parsed.bundle_dir), Path(parsed.normalization_report),
                    Path(parsed.build_report), Path(parsed.anime_csv), Path(parsed.synopsis_csv),
                    Path(parsed.glove), Path(parsed.lockfile), parsed.code_revision,
                    Path(parsed.output_dir), repetitions=parsed.repetitions)
        except (EvaluationError, OSError, ValueError) as exc:
            print(f"recsys quality compare: {exc}", file=sys.stderr)
            return 1
    elif parsed.command == "quality" and parsed.quality_command == "summarize":
        from pathlib import Path
        from .evaluation import EvaluationError, summarize
        try:
            summarize(Path(parsed.comparison), Path(parsed.catalog), Path(parsed.measurements),
                      [Path(path) for path in parsed.assessment], Path(parsed.output_dir))
        except (EvaluationError, OSError, ValueError) as exc:
            print(f"recsys quality summarize: {exc}", file=sys.stderr)
            return 1
    return 0
