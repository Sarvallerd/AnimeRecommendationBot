"""Command-line entry point for offline recommendation preparation."""

import argparse

from recsys import __version__


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
    from recsys.data.sources import SNAPSHOT_IDS

    obtain_parser.add_argument(
        "--snapshot", choices=SNAPSHOT_IDS, required=True, help="pinned source snapshot"
    )
    obtain_parser.add_argument(
        "--data-dir", help="source cache directory (overrides snapshot default)"
    )
    obtain_parser.add_argument(
        "--offline", action="store_true", help="use verified cached files only"
    )
    normalize_parser = commands.add_parser("normalize", help="normalize verified MAL CSV files")
    normalize_parser.add_argument("--anime-csv", required=True, help="path to anime.csv")
    normalize_parser.add_argument(
        "--synopsis-csv", required=True, help="path to anime_with_synopsis.csv"
    )
    normalize_parser.add_argument(
        "--output-dir", required=True, help="directory for normalization artifacts"
    )
    dataset_parser = commands.add_parser("dataset", help="normalize or validate a research dataset")
    dataset_commands = dataset_parser.add_subparsers(dest="dataset_command", required=True)
    dataset_normalize = dataset_commands.add_parser(
        "normalize", help="normalize a pinned research snapshot"
    )
    dataset_normalize.add_argument("--snapshot", required=True, choices=("neelagiri-2025-v1",))
    dataset_normalize.add_argument("--data-dir", required=True)
    dataset_normalize.add_argument("--output-dir", required=True)
    dataset_normalize.add_argument("--scratch-dir", required=True)
    dataset_normalize.add_argument("--memory-limit-mib", type=int, default=2048)
    dataset_normalize.add_argument("--threads", type=int, default=2)
    dataset_normalize.add_argument("--spill-limit-mib", type=int, default=32768)
    dataset_validate = dataset_commands.add_parser(
        "validate", help="validate an immutable research dataset"
    )
    dataset_validate.add_argument("dataset_dir")
    aliases_parser = commands.add_parser(
        "aliases", help="normalize or validate pinned title aliases"
    )
    aliases_commands = aliases_parser.add_subparsers(dest="aliases_command", required=True)
    aliases_normalize = aliases_commands.add_parser(
        "normalize", help="create an immutable alias overlay"
    )
    aliases_normalize.add_argument("--snapshot", required=True, choices=("nafiul-2026-v1",))
    aliases_normalize.add_argument("--data-dir", required=True)
    aliases_normalize.add_argument("--base-dir", required=True)
    aliases_normalize.add_argument("--base-identity", required=True)
    aliases_normalize.add_argument("--output-dir", required=True)
    aliases_validate = aliases_commands.add_parser(
        "validate", help="revalidate pinned alias inputs and output"
    )
    aliases_validate.add_argument("aliases_dir")
    aliases_validate.add_argument("--data-dir", required=True)
    aliases_validate.add_argument("--base-dir", required=True)
    aliases_validate.add_argument("--base-identity", required=True)
    aliases_validate.add_argument("--expected-identity")
    release_parser = commands.add_parser(
        "release", help="compose, build, and export a catalog release"
    )
    release_commands = release_parser.add_subparsers(dest="release_command", required=True)
    release_info = release_commands.add_parser("info", help="show the pinned current release")
    release_info.add_argument(
        "--release", default="current", choices=("current", "neelagiri-2025-nafiul-2026-v1")
    )
    release_prepare = release_commands.add_parser(
        "prepare", help="prepare a pinned release offline"
    )
    release_prepare.add_argument(
        "--release",
        default="current",
        choices=("current", "neelagiri-2025-nafiul-2026-v1", "mal-2020-glove-v1"),
    )
    release_prepare.add_argument("--data-dir", required=True)
    release_compose = release_commands.add_parser("compose")
    for flag in (
        "base-dir",
        "base-identity",
        "aliases-dir",
        "alias-identity",
        "alias-data-dir",
        "output-dir",
    ):
        release_compose.add_argument(f"--{flag}", required=True)
    release_validate = release_commands.add_parser("validate")
    release_validate.add_argument("release_dir")
    for flag in ("release-identity", "base-dir", "aliases-dir", "alias-data-dir"):
        release_validate.add_argument(f"--{flag}", required=True)
    release_build = release_commands.add_parser("build")
    release_export = release_commands.add_parser("export")
    for command in (release_build, release_export):
        for flag in (
            "release-dir",
            "release-identity",
            "base-dir",
            "aliases-dir",
            "alias-data-dir",
            "output-dir",
        ):
            command.add_argument(f"--{flag}", required=True)
    release_build.add_argument("--glove", required=True)
    release_build.add_argument("--block-size", type=int, default=256)
    release_export.add_argument("--neighbors", required=True)
    release_export.add_argument("--build-report", required=True)
    build_parser = commands.add_parser("build", help="build pinned GloVe cosine neighbors")
    build_parser.add_argument("--catalog", required=True, help="path to normalized catalog.json")
    build_parser.add_argument(
        "--normalization-report", required=True, help="path to normalization-report.json"
    )
    build_parser.add_argument("--glove", required=True, help="path to glove.6B.300d.txt")
    build_parser.add_argument("--output-dir", required=True, help="directory for build artifacts")
    build_parser.add_argument(
        "--block-size", type=int, default=256, help="similarity rows per block (1..256)"
    )
    export_parser = commands.add_parser("export", help="publish an immutable validated bundle")
    export_parser.add_argument("--catalog", required=True, help="path to catalog.json")
    export_parser.add_argument("--neighbors", required=True, help="path to neighbors.json")
    export_parser.add_argument("--build-report", required=True, help="path to build-report.json")
    export_parser.add_argument(
        "--normalization-report", required=True, help="path to normalization-report.json"
    )
    export_parser.add_argument("--output-dir", required=True, help="bundle store directory")
    validate_parser = commands.add_parser("validate", help="validate a published bundle")
    validate_parser.add_argument("bundle_dir", help="bundle directory")
    quality_parser = commands.add_parser(
        "quality", help="run offline recommendation quality diagnostics"
    )
    quality_commands = quality_parser.add_subparsers(dest="quality_command", required=True)
    baseline_parser = quality_commands.add_parser("baseline", help="run the pinned genre baseline")
    baseline_parser.add_argument("--catalog", required=True, help="path to pinned catalog.json")
    baseline_parser.add_argument(
        "--output-dir", required=True, help="directory for diagnostic artifacts"
    )
    compare_parser = quality_commands.add_parser(
        "compare", help="compare three pinned recommendation methods"
    )
    for option in (
        "bundle-dir",
        "normalization-report",
        "build-report",
        "anime-csv",
        "synopsis-csv",
        "glove",
        "lockfile",
        "code-revision",
        "output-dir",
    ):
        compare_parser.add_argument(f"--{option}", required=True)
    compare_parser.add_argument(
        "--originals-root", help="pinned historical checkout containing original notebooks"
    )
    compare_parser.add_argument("--repetitions", type=int, default=3)
    summary_parser = quality_commands.add_parser(
        "summarize", help="validate judgments and summarize comparison"
    )
    for option in ("comparison", "catalog", "measurements", "output-dir"):
        summary_parser.add_argument(f"--{option}", required=True)
    summary_parser.add_argument("--assessment", required=True, action="append")
    research = commands.add_parser("research", help="offline ML-01 and ML-02 research")
    research_commands = research.add_subparsers(dest="research_command", required=True)
    research_prepare = research_commands.add_parser("prepare")
    for flag in ("dataset-dir", "release-dir", "code-revision", "output-dir"):
        research_prepare.add_argument(f"--{flag}", required=True)
    research_prepare.add_argument("--protocol", required=True, choices=("ml01-v1",))
    research_validate = research_commands.add_parser("validate")
    research_validate.add_argument("prepared_dir")
    research_validate.add_argument("--expected-identity", required=True)
    training_verify = research_commands.add_parser(
        "training-verify", help="verify integrity of a versioned training artifact"
    )
    training_verify.add_argument("model_dir")
    training_verify.add_argument("--expected-identity", required=True)
    training_fit = research_commands.add_parser(
        "training-fit", help="fit the predeclared native seed-embedding candidate"
    )
    for flag in (
        "candidate",
        "prepared-dir",
        "expected-prepared-identity",
        "output-dir",
        "code-revision",
    ):
        training_fit.add_argument(f"--{flag}", required=True)
    training_fit.add_argument("--train-device", default="cuda:0")
    training_fit.add_argument("--inference-device", default="cpu")
    training_predict = research_commands.add_parser(
        "training-predict", help="make one bounded prediction from native weights"
    )
    for flag in (
        "prepared-dir",
        "expected-prepared-identity",
        "model-dir",
        "expected-model-identity",
        "output-dir",
        "code-revision",
    ):
        training_predict.add_argument(f"--{flag}", required=True)
    training_predict.add_argument("--seed-mal-id", type=int, action="append", default=[])
    training_predict.add_argument("--limit", type=int, default=5)
    training_predict.add_argument("--inference-device")
    model_fit = research_commands.add_parser("model-fit", help="fit a frozen ML-02 variant")
    for flag in (
        "prepared-dir",
        "expected-prepared-identity",
        "variant",
        "code-revision",
        "output-dir",
    ):
        model_fit.add_argument(f"--{flag}", required=True)
    model_fit.add_argument("--glove")
    model_fit.add_argument("--content-model")
    model_fit.add_argument("--puresvd-model")
    model_fit.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    predict_validation = research_commands.add_parser(
        "predict-validation", help="score validation queries"
    )
    for flag in ("prepared-dir", "expected-prepared-identity", "model-dir", "output-dir"):
        predict_validation.add_argument(f"--{flag}", required=True)
    predict_validation.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    predict_validation.add_argument("--measurement-output")
    gpu_preflight = research_commands.add_parser(
        "gpu-preflight", help="retired GPU scoring route and historical recovery guidance"
    )
    for flag in ("prepared-dir", "expected-prepared-identity", "model-dir", "output-dir"):
        gpu_preflight.add_argument(f"--{flag}")
    campaign_run = research_commands.add_parser(
        "campaign-run", help="run fixed ML-02 validation campaign"
    )
    for flag in (
        "prepared-dir",
        "expected-prepared-identity",
        "code-revision",
        "glove",
        "output-dir",
    ):
        campaign_run.add_argument(f"--{flag}", required=True)
    campaign_run.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    campaign_run.add_argument(
        "--campaign", choices=("ml02-cpu-v1", "ml02-native-v2"), default="ml02-cpu-v1"
    )
    campaign_run.add_argument("--native-train-device", choices=("cpu", "cuda:0"))
    campaign_run.add_argument("--physical-storage-root")
    campaign_run.add_argument("--cpu-workers", type=int, choices=(1, 2), default=2)
    campaign_run.add_argument("--resume", action="store_true")
    campaign_run.add_argument("--expected-campaign-identity")
    campaign_validate = research_commands.add_parser(
        "campaign-validate", help="verify retained ML-02 validation evidence"
    )
    for flag in (
        "prepared-dir",
        "expected-prepared-identity",
        "campaign-dir",
        "expected-campaign-identity",
    ):
        campaign_validate.add_argument(f"--{flag}", required=True)
    research_evaluate = research_commands.add_parser("evaluate")
    for flag in ("prepared-dir", "expected-prepared-identity", "predictions", "output-dir"):
        research_evaluate.add_argument(f"--{flag}", required=True)
    research_evaluate.add_argument("--split", required=True, choices=("validation", "test"))
    research_evaluate.add_argument("--selection")
    research_evaluate.add_argument("--expected-selection-identity")
    research_compare = research_commands.add_parser("compare")
    research_compare.add_argument("--evaluation", action="append", required=True)
    research_compare.add_argument("--expected-prepared-identity", required=True)
    research_compare.add_argument("--output-dir", required=True)
    research_finalize = research_commands.add_parser("finalize")
    for flag in ("prepared-dir", "decision", "output-dir"):
        research_finalize.add_argument(f"--{flag}", required=True)
    research_finalize.add_argument("--validation-report", action="append", required=True)
    research_test = research_commands.add_parser("test-queries")
    for flag in ("prepared-dir", "selection", "expected-selection-identity", "output-dir"):
        research_test.add_argument(f"--{flag}", required=True)
    return parser
