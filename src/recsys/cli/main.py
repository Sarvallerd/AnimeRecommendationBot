"""CLI command dispatch with lazy domain imports."""

import sys
from collections.abc import Sequence

from recsys.cli.parser import build_parser

REFERENCE_RECOVERY = (
    "Recover accepted v1 with Python 3.12.14 from pinned Git checkout "
    "1441573fbbbffc4693c7e1100aaae4458d9e88d4; follow root README.md "
    "and https://app.notion.com/p/3fb3908d6b1f8136a1dfff3c4a12108e."
)
GPU_SCORING_RECOVERY = (
    "GPU scoring is retired. Recover the historical implementation from pinned Git checkout "
    "{commit}; see docs/migrations/arb059-cupy-scoring-retirement.json."
)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return its status code."""
    args = sys.argv[1:] if argv is None else argv
    parser = build_parser()
    parsed = parser.parse_args(args)
    if not args:
        parser.print_help()
    elif parsed.command == "obtain":
        from pathlib import Path

        from recsys.data.obtain import obtain
        from recsys.data.sources import DEFAULT_SNAPSHOT, SourceError

        try:
            data_dir = (
                Path(parsed.data_dir)
                if parsed.data_dir is not None
                else Path("data/raw")
                if parsed.snapshot == DEFAULT_SNAPSHOT
                else Path("data/raw") / parsed.snapshot
            )
            obtain(data_dir, offline=parsed.offline, snapshot=parsed.snapshot)
        except (SourceError, OSError) as exc:
            print(f"recsys obtain: {exc}", file=sys.stderr)
            return 1
    elif parsed.command == "normalize":
        from pathlib import Path

        from recsys.data.mal2020 import normalize
        from recsys.data.sources import SourceError

        try:
            normalize(Path(parsed.anime_csv), Path(parsed.synopsis_csv), Path(parsed.output_dir))
        except (SourceError, OSError, ValueError) as exc:
            print(f"recsys normalize: {exc}", file=sys.stderr)
            return 1
    elif parsed.command == "dataset":
        from pathlib import Path

        from recsys.data.dataset import DatasetError, normalize_dataset, validate_dataset

        try:
            if parsed.dataset_command == "normalize":
                import hashlib
                import json

                manifest = normalize_dataset(
                    parsed.snapshot,
                    Path(parsed.data_dir),
                    Path(parsed.output_dir),
                    scratch_dir=Path(parsed.scratch_dir),
                    memory_limit_mib=parsed.memory_limit_mib,
                    threads=parsed.threads,
                    spill_limit_mib=parsed.spill_limit_mib,
                )
                manifest_bytes = manifest.read_bytes()
                qa = json.loads((manifest.parent / "dataset-qa.json").read_bytes())
                counts = qa["counts"]
                identity = "sha256:" + hashlib.sha256(manifest_bytes).hexdigest()
                catalog_items = counts["catalog_items"]
                interactions = counts["accepted_pairs"]
                users = counts["accepted_users"]
                positive_users = counts["positive_users_ge_15"]
            else:
                dataset_summary = validate_dataset(Path(parsed.dataset_dir))
                identity = dataset_summary.identity
                catalog_items = dataset_summary.catalog_items
                interactions = dataset_summary.interactions
                users = dataset_summary.users
                positive_users = dataset_summary.positive_users_ge_15
        except (DatasetError, OSError) as exc:
            code = str(exc) if isinstance(exc, DatasetError) else "dataset_integrity_error"
            print(f"recsys dataset: {code}", file=sys.stderr)
            return 1
        print(
            f"{identity} catalog={catalog_items} "
            f"interactions={interactions} users={users} "
            f"positive_users_ge_15={positive_users}"
        )
    elif parsed.command == "aliases":
        import hashlib
        from pathlib import Path

        from recsys.data.aliases import AliasError, normalize_aliases, validate_aliases

        try:
            if parsed.aliases_command == "normalize":
                manifest = normalize_aliases(
                    Path(parsed.data_dir),
                    Path(parsed.base_dir),
                    Path(parsed.output_dir),
                    snapshot=parsed.snapshot,
                    expected_base_identity=parsed.base_identity,
                )
                identity = "sha256:" + hashlib.sha256(manifest.read_bytes()).hexdigest()
                import json

                qa = json.loads((manifest.parent / "alias-qa.json").read_bytes())["counts"]
                items, aliases, english, main = (
                    qa[name]
                    for name in (
                        "emitted_items",
                        "emitted_strings",
                        "emitted_english",
                        "emitted_main",
                    )
                )
            else:
                alias_summary = validate_aliases(
                    Path(parsed.aliases_dir),
                    data_dir=Path(parsed.data_dir),
                    base_dir=Path(parsed.base_dir),
                    expected_base_identity=parsed.base_identity,
                    expected_identity=parsed.expected_identity,
                )
                identity, items, aliases, english, main = (
                    alias_summary.identity,
                    alias_summary.items,
                    alias_summary.aliases,
                    alias_summary.english,
                    alias_summary.main,
                )
        except (AliasError, OSError) as exc:
            code = str(exc) if isinstance(exc, AliasError) else "alias_integrity_error"
            print(f"recsys aliases: {code}", file=sys.stderr)
            return 1
        print(f"{identity} items={items} aliases={aliases} english={english} main={main}")
    elif parsed.command == "release":
        import hashlib
        from pathlib import Path

        from recsys.release.build import ReleaseBuildError, build_release, export_release_bundle
        from recsys.release.catalog import ReleaseError, compose_release, validate_release
        from recsys.release.prepare import PrepareError

        try:
            if parsed.release_command == "info":
                from recsys.release.lock import release_info

                info = release_info(parsed.release)
                print(
                    f"{info['release_id']} catalog={info['semantic_expectations']['catalog_count']} "
                    f"bundle={info['reference']['bundle_identity']}"
                )
            elif parsed.release_command == "prepare":
                from recsys.release.prepare import prepare_release

                path, identity, count = prepare_release(Path(parsed.data_dir), parsed.release)
                print(f"{path} {identity} records={count}")
                print(f"ARB_BUNDLE_DIR={path}")
            elif parsed.release_command == "compose":
                manifest = compose_release(
                    Path(parsed.base_dir),
                    Path(parsed.aliases_dir),
                    Path(parsed.alias_data_dir),
                    Path(parsed.output_dir),
                    expected_base_identity=parsed.base_identity,
                    expected_alias_identity=parsed.alias_identity,
                )
                identity = "sha256:" + hashlib.sha256(manifest.read_bytes()).hexdigest()
                import json

                counts = json.loads(manifest.read_bytes())["counts"]
                print(
                    f"{identity} catalog={counts['catalog_count']} "
                    f"enrichment={counts['enrichment_items']} "
                    f"changed={counts['catalog_aliases_changed_items']}"
                )
            elif parsed.release_command == "validate":
                release_summary = validate_release(
                    Path(parsed.release_dir),
                    expected_identity=parsed.release_identity,
                    base_dir=Path(parsed.base_dir),
                    aliases_dir=Path(parsed.aliases_dir),
                    alias_data_dir=Path(parsed.alias_data_dir),
                )
                print(
                    f"{release_summary.identity} catalog={release_summary.catalog_count} "
                    f"enrichment={release_summary.enrichment_items} changed={release_summary.catalog_aliases_changed_items}"
                )
            elif parsed.release_command == "build":
                report = build_release(
                    Path(parsed.release_dir),
                    Path(parsed.glove),
                    Path(parsed.output_dir),
                    expected_release_identity=parsed.release_identity,
                    base_dir=Path(parsed.base_dir),
                    aliases_dir=Path(parsed.aliases_dir),
                    alias_data_dir=Path(parsed.alias_data_dir),
                    block_size=parsed.block_size,
                )
                print(f"{report} release={parsed.release_identity}")
            else:
                path = export_release_bundle(
                    Path(parsed.release_dir),
                    Path(parsed.neighbors),
                    Path(parsed.build_report),
                    Path(parsed.output_dir),
                    expected_release_identity=parsed.release_identity,
                    base_dir=Path(parsed.base_dir),
                    aliases_dir=Path(parsed.aliases_dir),
                    alias_data_dir=Path(parsed.alias_data_dir),
                )
                from recsys.contracts.bundle import check_bundle

                identity, count = check_bundle(path)
                print(f"{path} {identity} records={count}")
        except (ReleaseError, ReleaseBuildError, PrepareError, OSError, ValueError) as exc:
            code = (
                str(exc)
                if isinstance(exc, (ReleaseError, ReleaseBuildError, PrepareError))
                else "release_integrity_error"
            )
            print(f"recsys release: {code}", file=sys.stderr)
            if code == "release_reference_producer_required":
                print(REFERENCE_RECOVERY, file=sys.stderr)
            return 1
    elif parsed.command == "research":
        from pathlib import Path

        from recsys.research.models.spec import (
            GPU_SCORING_RECOVERY_COMMIT,
            require_cpu_backend,
        )
        from recsys.research.protocol import ResearchError

        try:
            result: Path | str
            if parsed.research_command == "prepare":
                from recsys.research.prepare import prepare_research
                from recsys.research.protocol import load_protocol

                load_protocol(parsed.protocol)
                result = prepare_research(
                    Path(parsed.dataset_dir),
                    Path(parsed.release_dir),
                    Path(parsed.output_dir),
                    code_revision=parsed.code_revision,
                )
            elif parsed.research_command == "validate":
                from recsys.research.prepare import validate_preparation

                validate_preparation(
                    Path(parsed.prepared_dir), expected_identity=parsed.expected_identity
                )
                result = Path(parsed.prepared_dir) / "prepare-manifest.json"
            elif parsed.research_command == "training-verify":
                from recsys.research.training.artifacts import verify_training_artifact

                verified = verify_training_artifact(
                    Path(parsed.model_dir), expected_identity=parsed.expected_identity
                )
                result = (
                    f"{verified.identity} integrity_verified family={verified.spec.family} "
                    f"framework={verified.spec.framework} "
                    f"weights_format={verified.spec.weights_format}"
                )
            elif parsed.research_command == "model-fit":
                require_cpu_backend(parsed.backend)
                from recsys.research.models.package import fit_model

                assets = (
                    {"glove": Path(parsed.glove)}
                    if parsed.glove
                    else {
                        "content_model": Path(parsed.content_model),
                        "puresvd_model": Path(parsed.puresvd_model),
                    }
                    if parsed.content_model is not None and parsed.puresvd_model is not None
                    else None
                )
                if bool(parsed.content_model) != bool(parsed.puresvd_model) or (
                    parsed.glove and parsed.content_model
                ):
                    raise ValueError("research_model_invalid")
                result = fit_model(
                    Path(parsed.prepared_dir),
                    Path(parsed.output_dir),
                    expected_prepared_identity=parsed.expected_prepared_identity,
                    variant_id=parsed.variant,
                    code_revision=parsed.code_revision,
                    assets=assets,
                    backend=parsed.backend,
                )
            elif parsed.research_command == "predict-validation":
                require_cpu_backend(parsed.backend)
                from recsys.research.predict import predict_validation

                measured: dict[str, object] | None = {} if parsed.measurement_output else None
                result = predict_validation(
                    Path(parsed.prepared_dir),
                    Path(parsed.model_dir),
                    Path(parsed.output_dir),
                    expected_prepared_identity=parsed.expected_prepared_identity,
                    backend=parsed.backend,
                    measurements=measured,
                    measurement_output=Path(parsed.measurement_output)
                    if parsed.measurement_output
                    else None,
                )
            elif parsed.research_command == "gpu-preflight":
                require_cpu_backend("gpu")
                result = ""  # The retired route always raises above.
            elif parsed.research_command == "campaign-run":
                require_cpu_backend(parsed.backend)
                from recsys.research.campaign.runner import run_campaign

                result = run_campaign(
                    Path(parsed.prepared_dir),
                    Path(parsed.output_dir),
                    expected_prepared_identity=parsed.expected_prepared_identity,
                    code_revision=parsed.code_revision,
                    glove=Path(parsed.glove),
                    backend=parsed.backend,
                    cpu_workers=parsed.cpu_workers,
                    resume=parsed.resume,
                    expected_campaign_identity=parsed.expected_campaign_identity,
                )
            elif parsed.research_command == "campaign-validate":
                from recsys.research.campaign.report import validate_campaign

                validated = validate_campaign(
                    Path(parsed.prepared_dir),
                    Path(parsed.campaign_dir),
                    expected_prepared_identity=parsed.expected_prepared_identity,
                    expected_campaign_identity=parsed.expected_campaign_identity,
                )
                result = f"{parsed.expected_campaign_identity} {validated['status']}"
            elif parsed.research_command == "evaluate":
                from recsys.research.evaluation.evaluate import evaluate_predictions

                result = evaluate_predictions(
                    Path(parsed.prepared_dir),
                    Path(parsed.predictions),
                    Path(parsed.output_dir),
                    split=parsed.split,
                    expected_prepared_identity=parsed.expected_prepared_identity,
                    selection_path=Path(parsed.selection) if parsed.selection else None,
                    expected_selection_identity=parsed.expected_selection_identity,
                )
            elif parsed.research_command == "compare":
                from recsys.research.evaluation.compare import compare_evaluations

                result = compare_evaluations(
                    [Path(path) for path in parsed.evaluation],
                    Path(parsed.output_dir),
                    expected_prepared_identity=parsed.expected_prepared_identity,
                )
            elif parsed.research_command == "finalize":
                from recsys.research.evaluation.finalize import finalize_selection

                result = finalize_selection(
                    Path(parsed.prepared_dir),
                    Path(parsed.decision),
                    [Path(path) for path in parsed.validation_report],
                    Path(parsed.output_dir),
                )
            else:
                from recsys.research.evaluation.finalize import export_test_queries

                result = export_test_queries(
                    Path(parsed.prepared_dir),
                    Path(parsed.selection),
                    Path(parsed.output_dir),
                    expected_selection_identity=parsed.expected_selection_identity,
                )
            print(result)
        except (ResearchError, OSError, ValueError, KeyError, TypeError) as exc:
            code = str(exc) if isinstance(exc, ResearchError) else "research_integrity_error"
            print(f"recsys research: {code}", file=sys.stderr)
            if code == "research_gpu_scoring_retired":
                print(
                    GPU_SCORING_RECOVERY.format(commit=GPU_SCORING_RECOVERY_COMMIT),
                    file=sys.stderr,
                )
            return 1
    elif parsed.command == "build":
        from pathlib import Path

        from recsys.content.glove import BuildError, build
        from recsys.data.sources import SourceError

        try:
            build(
                Path(parsed.catalog),
                Path(parsed.normalization_report),
                Path(parsed.glove),
                Path(parsed.output_dir),
                block_size=parsed.block_size,
            )
        except (BuildError, SourceError, OSError, ValueError) as exc:
            print(f"recsys build: {exc}", file=sys.stderr)
            return 1
    elif parsed.command == "export":
        from pathlib import Path

        from recsys.contracts.bundle import ContractError, check_bundle
        from recsys.data.sources import SourceError
        from recsys.release.export import export_bundle

        try:
            path = export_bundle(
                Path(parsed.catalog),
                Path(parsed.neighbors),
                Path(parsed.build_report),
                Path(parsed.normalization_report),
                Path(parsed.output_dir),
            )
            identity, count = check_bundle(path)
        except (ContractError, SourceError, OSError, ValueError) as exc:
            print(f"recsys export: {exc}", file=sys.stderr)
            return 1
        print(f"{path} {identity} records={count}")
    elif parsed.command == "validate":
        from pathlib import Path

        from recsys.contracts.bundle import ContractError, check_bundle

        try:
            identity, count = check_bundle(Path(parsed.bundle_dir))
        except (ContractError, OSError) as exc:
            print(f"recsys validate: {exc}", file=sys.stderr)
            return 1
        print(f"{identity} records={count}")
    elif parsed.command == "quality" and parsed.quality_command == "baseline":
        from pathlib import Path

        from recsys.historical.genre_baseline import QualityError, run_baseline

        try:
            run_baseline(Path(parsed.catalog), Path(parsed.output_dir))
        except (QualityError, OSError) as exc:
            print(f"recsys quality baseline: {exc}", file=sys.stderr)
            return 1
    elif parsed.command == "quality" and parsed.quality_command == "compare":
        from pathlib import Path

        from recsys.historical.comparison import EvaluationError, compare

        try:
            compare(
                Path(parsed.bundle_dir),
                Path(parsed.normalization_report),
                Path(parsed.build_report),
                Path(parsed.anime_csv),
                Path(parsed.synopsis_csv),
                Path(parsed.glove),
                Path(parsed.lockfile),
                parsed.code_revision,
                Path(parsed.output_dir),
                repetitions=parsed.repetitions,
                originals_root=parsed.originals_root,
            )
        except (EvaluationError, OSError, ValueError) as exc:
            print(f"recsys quality compare: {exc}", file=sys.stderr)
            return 1
    elif parsed.command == "quality" and parsed.quality_command == "summarize":
        from pathlib import Path

        from recsys.historical.comparison import EvaluationError, summarize

        try:
            summarize(
                Path(parsed.comparison),
                Path(parsed.catalog),
                Path(parsed.measurements),
                [Path(path) for path in parsed.assessment],
                Path(parsed.output_dir),
            )
        except (EvaluationError, OSError, ValueError) as exc:
            print(f"recsys quality summarize: {exc}", file=sys.stderr)
            return 1
    return 0
