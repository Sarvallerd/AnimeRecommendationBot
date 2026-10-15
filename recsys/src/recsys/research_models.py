"""Private ML-02 model inputs, fixed variants, and immutable model packages."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import os
import platform
import re
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

from . import __version__
from .bundle import canonical_bytes, check_catalog
from .dataset import HEADER
from .release_prepare import release_info
from .research_artifacts import (MAX_STREAM, check_descriptor, check_identity, cleanup,
                                 descriptor, module_fingerprints, open_stage, publish,
                                 read_json, safe_path, separate, stream_tsv, write_json)
from .research_protocol import exact, fail, load_protocol, sha, split_user

CAMPAIGN = "research_campaign_v1.json"
COMMON_CODE = ("research_models.py", "research_predict.py", CAMPAIGN,
               "bundle.py", "dataset.py", "dataset_interactions.py",
               "release_prepare.py", "release_neelagiri_2025_nafiul_2026_v1.json",
               "research_protocol.py", "research_protocol_v1.json", "research_artifacts.py",
               "research_prepare.py")
BACKENDS = {"popularity": "research_model_popularity",
            "content-mean": "research_model_content",
            "content-max": "research_model_content",
            "content-union": "research_model_content",
            "itemknn": "research_model_cf", "puresvd": "research_model_cf",
            "hybrid": "research_model_hybrid"}
EXPECTED_VARIANTS = {
    "popularity": {},
    "content-mean": {"aggregation": "mean", "dimension": 300, "positive_only": True},
    "content-max": {"aggregation": "max", "dimension": 300, "positive_only": True},
    "content-union": {"dimension": 300, "positive_only": True, "rrf_k": 60, "top_k": 100},
    "itemknn": {"positive_min_rating": 8, "shrinkage": 10, "source_block": 128, "top_k": 200},
    "puresvd": {"flip_sign": True, "n_iter": 4, "n_oversamples": 16,
                "power_iteration_normalizer": "LU", "rank": 64, "random_state": 3737,
                "svd_lapack_driver": "gesdd", "transpose": False},
    "hybrid": {"content_weight": .5, "rrf_k": 60, "svd_weight": .5},
}
PREPARED_FILES = {"protocol.json", "model-catalog.json", "fit/manifest.json",
                  "fit/train-interactions.tsv.gz", "fit/train-stats.json",
                  "development/queries.jsonl.gz", "evaluator/splits.tsv.gz",
                  "evaluator/validation-labels.jsonl.gz", "evaluator/test-labels.jsonl.gz"}
MODEL_FIELDS = {"schema_version", "kind", "variant_id", "parameters",
                "campaign_spec_sha256", "prepared_identity", "protocol_identity",
                "fit_identity", "catalog_descriptor", "train_interactions", "train_stats",
                "external_inputs", "producer", "files", "metadata"}
HEX40 = re.compile(r"[0-9a-f]{40}\Z")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
NAME = re.compile(r"[a-z][a-z0-9_-]*\.(?:json|npy)\Z")


@dataclass(frozen=True)
class ScoringInputs:
    prepared_identity: str
    protocol_identity: str
    catalog_descriptor: dict
    fit_identity: str
    train_interactions_descriptor: dict
    train_stats_descriptor: dict
    mal_ids: tuple[int, ...]
    catalog: dict
    train_stats: dict


@dataclass(frozen=True)
class TrainingInputs(ScoringInputs):
    train_interactions_path: Path


@dataclass(frozen=True)
class ScoreVector:
    values: np.ndarray
    available: np.ndarray
    supported_seeds: int


def campaign():
    package = resources.files("recsys")
    raw = package.joinpath(CAMPAIGN).read_bytes()
    from .bundle import parse_canonical
    spec = parse_canonical(raw, CAMPAIGN)
    exact(spec, {"schema_version", "kind", "score_encoding", "variants"})
    if (type(spec["schema_version"]) is not int or spec["schema_version"] != 1 or
            spec["kind"] != "arb-ml02-campaign" or
            spec["score_encoding"] != {"nonpop_n_gt_0": "int-rint-half-even-times-1e9",
                                       "popularity_all_n": "raw-train-positive-user-counts",
                                       "zero_seed_all_variants": "raw-train-positive-user-counts"} or
            canonical_bytes(spec["variants"]) != canonical_bytes(EXPECTED_VARIANTS)):
        fail("research_model_invalid")
    return spec, sha(raw)


def _backend(variant_id):
    if type(variant_id) is not str or variant_id not in BACKENDS:
        fail("research_model_variant_invalid")
    try:
        return importlib.import_module("." + BACKENDS[variant_id], __package__)
    except ModuleNotFoundError as exc:
        if exc.name == f"{__package__}.{BACKENDS[variant_id]}":
            fail("research_model_unavailable")
        raise


def _stats(stats, ids):
    exact(stats, {"schema_version", "train_users", "train_completed_rows", "items"})
    if (type(stats["schema_version"]) is not int or stats["schema_version"] != 1 or
            type(stats["train_users"]) is not int or stats["train_users"] < 1 or
            type(stats["train_completed_rows"]) is not int or stats["train_completed_rows"] < 0 or
            type(stats["items"]) is not dict or set(stats["items"]) != set(ids)):
        fail("research_model_invalid")
    total = 0
    positive = []
    for key, value in stats["items"].items():
        exact(value, {"completed_users", "positive_users", "rating_sum", "rating_count", "band"})
        c, p, s, r = (value[name] for name in ("completed_users", "positive_users", "rating_sum", "rating_count"))
        if (any(type(n) is not int for n in (c, p, s, r)) or
                not 0 <= p <= c <= stats["train_users"] or r != c or
                not c <= s <= 10 * c):
            fail("research_model_invalid")
        total += c
        if p:
            positive.append(int(key))
    if total != stats["train_completed_rows"]:
        fail("research_model_invalid")
    ranked = sorted(positive, key=lambda item: (-stats["items"][str(item)]["positive_users"], item))
    head = set(ranked[:(len(ranked) + 4) // 5])
    for key, value in stats["items"].items():
        expected = "cold" if not value["positive_users"] else "head" if int(key) in head else "tail"
        if value["band"] != expected:
            fail("research_model_invalid")


def read_scoring_inputs(prepared_dir, *, expected_prepared_identity):
    """Read only development-visible preparation metadata, catalog, and train stats."""
    prepared_dir = safe_path(prepared_dir, directory=True)
    check_identity(expected_prepared_identity)
    prep, raw = read_json(prepared_dir / "prepare-manifest.json")
    exact(prep, {"schema_version", "kind", "protocol_identity", "release_lock_sha256",
                 "release_id", "dataset_identity", "catalog_release_identity", "catalog_sha256",
                 "producer", "files", "counts"})
    lock = release_info()
    if ("sha256:" + sha(raw) != expected_prepared_identity or
            type(prep["schema_version"]) is not int or prep["schema_version"] != 1 or
            prep["kind"] != "arb-ml-preparation" or
            prep["release_id"] != lock["release_id"] or
            prep["release_lock_sha256"] != sha(canonical_bytes(lock)) or
            prep["dataset_identity"] != lock["inputs"]["base"]["identity"] or
            prep["catalog_release_identity"] != lock["inputs"]["catalog_release"]["identity"] or
            prep["catalog_sha256"] != lock["inputs"]["catalog_release"]["catalog_sha256"]):
        fail("research_model_invalid")
    exact(prep["producer"], {"code_revision", "code_files_sha256", "python_version",
                             "package_version", "numpy_version"})
    if (prep["producer"]["code_files_sha256"] != module_fingerprints() or
            type(prep["producer"]["code_revision"]) is not str or
            HEX40.fullmatch(prep["producer"]["code_revision"]) is None):
        fail("research_model_invalid")
    files = prep["files"]
    if type(files) is not dict or set(files) != PREPARED_FILES:
        fail("research_model_invalid")
    for name, entry in files.items():
        check_descriptor(entry, stream=name.endswith(".gz"))
    protocol, _ = read_json(prepared_dir / "protocol.json", files["protocol.json"])
    if (protocol != load_protocol() or
            prep["protocol_identity"] != "sha256:" + sha(canonical_bytes(protocol))):
        fail("research_model_invalid")
    catalog, _ = read_json(prepared_dir / "model-catalog.json", files["model-catalog.json"])
    ids = check_catalog(catalog)
    if (len(ids) != lock["inputs"]["catalog_release"]["catalog_count"] or
            any(item["score"] is not None for item in catalog["anime"].values())):
        fail("research_model_invalid")
    fit, fit_raw = read_json(prepared_dir / "fit/manifest.json", files["fit/manifest.json"])
    if fit != {"schema_version": 1, "kind": "arb-ml-fit",
               "dataset_identity": prep["dataset_identity"],
               "protocol_identity": prep["protocol_identity"],
               "train_interactions": files["fit/train-interactions.tsv.gz"],
               "train_stats": files["fit/train-stats.json"]}:
        fail("research_model_invalid")
    stats, _ = read_json(prepared_dir / "fit/train-stats.json", files["fit/train-stats.json"])
    _stats(stats, ids)
    return ScoringInputs(expected_prepared_identity, prep["protocol_identity"],
                         files["model-catalog.json"], "sha256:" + sha(fit_raw),
                         files["fit/train-interactions.tsv.gz"], files["fit/train-stats.json"],
                         tuple(sorted(map(int, ids))), catalog, stats)


def read_training_inputs(prepared_dir, *, expected_prepared_identity):
    scores = read_scoring_inputs(prepared_dir, expected_prepared_identity=expected_prepared_identity)
    path = safe_path(Path(prepared_dir) / "fit/train-interactions.tsv.gz", directory=False)
    return TrainingInputs(**vars(scores), train_interactions_path=path)


def iter_training_rows(inputs: TrainingInputs):
    """Validate every train row and compressed/plain stream descriptor through EOF."""
    lines = iter(stream_tsv(inputs.train_interactions_path, inputs.train_interactions_descriptor))
    if next(lines, None) != HEADER:
        fail("research_model_invalid")
    ids = set(inputs.mal_ids)
    previous = None
    count = 0
    protocol = load_protocol()
    for raw in lines:
        fields = raw.removesuffix(b"\n").split(b"\t")
        if len(fields) != 4:
            fail("research_model_invalid")
        try:
            user = fields[0].decode("ascii")
            item = int(fields[1]); rating = int(fields[2])
            if (str(item).encode() != fields[1] or str(rating).encode() != fields[2] or
                    split_user(user, protocol) != "train" or fields[3] != b"completed" or
                    item not in ids or not 1 <= rating <= 10 or
                    (previous is not None and (user, item) <= previous)):
                fail("research_model_invalid")
        except (ValueError, UnicodeError):
            fail("research_model_invalid")
        previous = user, item
        count += 1
        yield user, item, rating
    if count != inputs.train_stats["train_completed_rows"]:
        fail("research_model_invalid")


def _file_descriptor(path):
    path = safe_path(path, directory=False)
    if path.stat().st_size > MAX_STREAM:
        fail("research_resource_limit")
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            size += len(block)
            if size > MAX_STREAM:
                fail("research_resource_limit")
            digest.update(block)
    return {"bytes": size, "sha256": digest.hexdigest()}


def _external(variant_id, assets, prepared_dir, output_dir):
    assets = {} if assets is None else assets
    if type(assets) is not dict:
        fail("research_model_invalid")
    required = {"glove"} if variant_id.startswith("content-") else set()
    if set(assets) != required or any(not isinstance(path, Path) for path in assets.values()):
        fail("research_model_invalid")
    entries = {}
    for name, path in assets.items():
        separate(path, prepared_dir, output_dir)
        desc = _file_descriptor(path)
        locked = release_info()["inputs"]["embeddings"]
        if name == "glove" and (desc != {"bytes": locked["text_size"],
                                        "sha256": locked["text_sha256"]}):
            fail("research_model_invalid")
        entries[name] = desc
    return entries


def _producer(code_revision, backend):
    import numpy as np
    if type(code_revision) is not str or HEX40.fullmatch(code_revision) is None:
        fail("research_model_invalid")
    names = (*COMMON_CODE, *backend.CODE_FILES)
    if len(set(names)) != len(names) or any(type(name) is not str or "/" in name or "\\" in name for name in names):
        fail("research_model_invalid")
    package = resources.files("recsys")
    dependencies = {"numpy": np.__version__}
    for name in getattr(backend, "DEPENDENCIES", ()):
        if name in dependencies or type(name) is not str:
            fail("research_model_invalid")
        dependencies[name] = importlib.metadata.version(name)
    return {"code_revision": code_revision,
            "code_files_sha256": {name: sha(package.joinpath(name).read_bytes()) for name in names},
            "python_version": platform.python_version(), "package_version": __version__,
            "dependency_versions": dependencies,
            "execution": {"backend": "cpu", "dtype": "float64", "blas_threads": 1}}


def _payloads(stage):
    import numpy as np
    if any(entry.is_symlink() or not entry.is_file() or NAME.fullmatch(entry.name) is None
           for entry in stage.iterdir()):
        fail("research_model_invalid")
    answer = {}
    for entry in sorted(stage.iterdir()):
        base = _file_descriptor(entry)
        if entry.suffix == ".json":
            read_json(entry, base)
        else:
            try:
                array = np.load(entry, allow_pickle=False, mmap_mode="r")
            except (OSError, ValueError, TypeError) as exc:
                raise ValueError("research_model_invalid") from exc
            if not isinstance(array, np.ndarray) or array.dtype.hasobject or array.dtype.fields:
                fail("research_model_invalid")
            base = {**base, "dtype": array.dtype.str, "shape": list(array.shape)}
        answer[entry.name] = base
    return answer


def _check_payloads(model_dir, files):
    import numpy as np
    if type(files) is not dict or any(type(name) is not str or NAME.fullmatch(name) is None
                                      for name in files):
        fail("research_model_invalid")
    observed = {entry.name for entry in model_dir.iterdir()}
    if observed != {"model-manifest.json", *files}:
        fail("research_model_invalid")
    for name, desc in files.items():
        entry = safe_path(model_dir / name, directory=False)
        if entry.suffix == ".json":
            exact(desc, {"bytes", "sha256"})
            check_descriptor(desc)
            read_json(entry, desc)
        else:
            exact(desc, {"bytes", "sha256", "dtype", "shape"})
            check_descriptor({key: desc[key] for key in ("bytes", "sha256")})
            if (type(desc["dtype"]) is not str or type(desc["shape"]) is not list or
                    any(type(n) is not int or n < 0 for n in desc["shape"])):
                fail("research_model_invalid")
        if _file_descriptor(entry) != {key: desc[key] for key in ("bytes", "sha256")}:
            fail("research_model_invalid")
        if entry.suffix == ".npy":
            try:
                array = np.load(entry, allow_pickle=False, mmap_mode="r")
            except (OSError, ValueError, TypeError) as exc:
                raise ValueError("research_model_invalid") from exc
            if (not isinstance(array, np.ndarray) or array.dtype.hasobject or array.dtype.fields or
                    array.dtype.str != desc["dtype"] or list(array.shape) != desc["shape"]):
                fail("research_model_invalid")


def fit_model(prepared_dir, output_dir, *, expected_prepared_identity, variant_id,
              code_revision, assets: dict[str, Path] | None = None):
    prepared_dir, output_dir = Path(prepared_dir), Path(output_dir)
    separate(prepared_dir, output_dir)
    inputs = read_training_inputs(prepared_dir, expected_prepared_identity=expected_prepared_identity)
    backend = _backend(variant_id)
    spec, campaign_hash = campaign()
    external = _external(variant_id, assets, prepared_dir, output_dir)
    producer = _producer(code_revision, backend)
    parameters = spec["variants"][variant_id]
    stage = open_stage(output_dir)
    try:
        metadata = backend.fit(inputs, parameters, {} if assets is None else assets, stage)
        if type(metadata) is not dict:
            fail("research_model_invalid")
        files = _payloads(stage)
        manifest = {"schema_version": 1, "kind": "arb-ml02-model",
                    "variant_id": variant_id, "parameters": parameters,
                    "campaign_spec_sha256": campaign_hash,
                    "prepared_identity": inputs.prepared_identity,
                    "protocol_identity": inputs.protocol_identity,
                    "fit_identity": inputs.fit_identity,
                    "catalog_descriptor": inputs.catalog_descriptor,
                    "train_interactions": inputs.train_interactions_descriptor,
                    "train_stats": inputs.train_stats_descriptor,
                    "external_inputs": external,
                    "producer": producer,
                    "files": files, "metadata": metadata}
        write_json(stage / "model-manifest.json", manifest)
        publish(stage, output_dir)
        return output_dir / "model-manifest.json"
    except Exception:
        cleanup(stage)
        raise


def load_model(model_dir, *, inputs: ScoringInputs):
    model_dir = safe_path(model_dir, directory=True)
    manifest, raw = read_json(model_dir / "model-manifest.json")
    exact(manifest, MODEL_FIELDS)
    variant = manifest["variant_id"]
    backend = _backend(variant)
    spec, campaign_hash = campaign()
    if type(manifest["producer"]) is not dict or type(manifest["producer"].get("code_revision")) is not str:
        fail("research_model_invalid")
    if (type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1 or
            manifest["kind"] != "arb-ml02-model" or
            canonical_bytes(manifest["parameters"]) != canonical_bytes(spec["variants"][variant]) or
            manifest["campaign_spec_sha256"] != campaign_hash or
            manifest["prepared_identity"] != inputs.prepared_identity or
            manifest["protocol_identity"] != inputs.protocol_identity or
            manifest["fit_identity"] != inputs.fit_identity or
            manifest["catalog_descriptor"] != inputs.catalog_descriptor or
            manifest["train_interactions"] != inputs.train_interactions_descriptor or
            manifest["train_stats"] != inputs.train_stats_descriptor or
            type(manifest["external_inputs"]) is not dict or
            manifest["producer"] != _producer(manifest["producer"]["code_revision"], backend)):
        fail("research_model_invalid")
    if variant in ("popularity", "itemknn", "puresvd") and manifest["external_inputs"]:
        fail("research_model_invalid")
    if variant.startswith("content-"):
        locked = release_info()["inputs"]["embeddings"]
        if manifest["external_inputs"] != {"glove": {"bytes": locked["text_size"],
                                                     "sha256": locked["text_sha256"]}}:
            fail("research_model_invalid")
    for desc in manifest["external_inputs"].values():
        check_descriptor(desc)
    _check_payloads(model_dir, manifest["files"])
    scorer = backend.load(model_dir, manifest, inputs)
    if not callable(getattr(scorer, "score", None)):
        fail("research_model_invalid")
    return manifest, scorer
