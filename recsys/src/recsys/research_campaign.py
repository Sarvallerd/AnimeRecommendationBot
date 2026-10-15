"""Fixed, supervised and resumable validation-only ML-02 campaign.

The worker entry uses only the standard library until the generation gate has
been acquired and checked. All expensive scientific work runs in workers.
"""

from __future__ import annotations

import fcntl
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import stat
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from importlib import resources
from pathlib import Path

VARIANTS = ("popularity", "content-mean", "content-max", "content-union",
            "itemknn", "puresvd", "hybrid")
PHASES = ("fit", "predict", "evaluate", "replay")
AUX_PHASE = "fit_cpu_component"
AUX_VARIANTS = ("content-mean", "puresvd")
FIT_SECONDS = 7200.0
EVALUATION_SECONDS = 3600.0
CAMPAIGN_SECONDS = 172800.0
RSS_LIMIT_BYTES = 8 * 1024**3
SOURCE_FILES = ("research_campaign.py", "research_resources.py", "research_models.py",
                "research_predict.py", "research_model_popularity.py",
                "research_model_content.py", "research_model_cf.py",
                "research_model_hybrid.py", "research_gpu.py", "research_evaluate.py",
                "research_compare.py", "research_artifacts.py", "research_protocol.py",
                "research_protocol_v1.json", "research_campaign_v1.json", "build.py",
                "sources.py", "bundle.py", "dataset.py", "dataset_interactions.py",
                "release_prepare.py", "release_neelagiri_2025_nafiul_2026_v1.json",
                "research_prepare.py")
_SAFE_CODE = re.compile(r"research_[a-z0-9_]+\Z")


def _fail(code="research_campaign_invalid"):
    from .research_protocol import fail
    fail(code)


def _canonical(value):
    from .bundle import canonical_bytes
    return canonical_bytes(value)


def _digest(raw):
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _now():
    return datetime.now(timezone.utc)


def _iso(value):
    return value.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _parse_iso(value):
    if type(value) is not str or not value.endswith("Z"):
        _fail()
    try:
        return datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        _fail()


def _private_file(path, raw):
    path = Path(path)
    with open(path, "xb", opener=lambda name, flags: os.open(name, flags, 0o600)) as out:
        out.write(raw)
        out.flush()
        os.fsync(out.fileno())
    _sync_dir(path.parent)


def _replace_private(path, raw):
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex)
    _private_file(temporary, raw)
    os.replace(temporary, path)
    _sync_dir(path.parent)


def _sync_dir(directory):
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _read_private(path, *, limit=16 * 1024 * 1024):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        _fail()
    info = path.stat()
    if (info.st_size > limit or info.st_uid != os.getuid() or
            stat.S_IMODE(info.st_mode) != 0o600):
        _fail()
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (ValueError, UnicodeError):
        _fail()
    if raw != _canonical(value):
        _fail()
    return value, raw


def _touch_lock(path):
    if path.is_symlink():
        _fail()
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    info = os.fstat(fd)
    if stat.S_IMODE(info.st_mode) != 0o600 or info.st_uid != os.getuid():
        os.close(fd)
        _fail()
    return fd


def _backend(variant, configured):
    return "cpu" if configured == "cpu" or variant in ("popularity", "itemknn", "hybrid") else "gpu"


def _expected_nodes(plan):
    nodes = {(variant, phase) for variant in VARIANTS for phase in PHASES}
    if plan["scheduling"]["gpu_workers"] == 1 and any(
            entry["backend"] == "gpu" for entry in plan["variants"]):
        nodes.update((variant, AUX_PHASE) for variant in AUX_VARIANTS)
    return nodes


def _component_phase(plan):
    return AUX_PHASE if plan["auxiliary_fits"] else "fit"


def _validate_plan_structure(plan):
    from .research_models import campaign
    spec, campaign_sha = campaign()
    if (type(plan) is not dict or plan.get("schema_version") != 1 or
            plan.get("kind") != "arb-ml02-campaign-plan" or
            plan.get("campaign_spec_sha256") != campaign_sha or
            type(plan.get("variants")) is not list or len(plan["variants"]) != 7 or
            plan.get("score_retention") != "keep"):
        _fail("research_campaign_plan_mismatch")
    configured = "gpu" if any(item.get("backend") == "gpu" for item in plan["variants"]) else "cpu"
    expected_aux = ([{"variant_id": variant, "phase": AUX_PHASE, "backend": "cpu",
                     "fit_budget_owner": variant, "consumer": "hybrid"}
                    for variant in AUX_VARIANTS] if configured == "gpu" else [])
    expected_components = [{"variant_id": variant,
                            "phase": AUX_PHASE if configured == "gpu" else "fit"}
                           for variant in AUX_VARIANTS]
    expected_variants = [{"variant_id": name, "parameters": spec["variants"][name],
                          "backend": _backend(name, configured),
                          **({"components": expected_components} if name == "hybrid" else {})}
                         for name in VARIANTS]
    expected_budgets = {"fit_seconds_per_variant": FIT_SECONDS,
                        "evaluation_seconds_per_variant": EVALUATION_SECONDS,
                        "campaign_wall_seconds": CAMPAIGN_SECONDS,
                        "sampled_rss_limit_bytes": RSS_LIMIT_BYTES,
                        "score_stream_limit_bytes": 64 * 1024**3,
                        "score_row_limit_bytes": 4 * 1024**2}
    schedule = plan.get("scheduling")
    if (plan["variants"] != expected_variants or
            plan.get("auxiliary_fits") != expected_aux or
            plan.get("execution_config_count") != (9 if configured == "gpu" else 7) or
            plan.get("budgets") != expected_budgets or
            type(schedule) is not dict or
            schedule.get("cpu_workers") not in (1, 2) or
            schedule != {"cpu_workers": schedule["cpu_workers"], "gpu_workers": 1,
                         "blas_threads": 1, "gpu_optional_benchmark": "not_scheduled"} or
            _parse_iso(plan["deadline_utc"]) - _parse_iso(plan["created_utc"]) !=
            timedelta(seconds=CAMPAIGN_SECONDS)):
        _fail("research_campaign_plan_mismatch")
    return configured


def _source_receipt(backend):
    package = resources.files("recsys")
    names = {name: hashlib.sha256(package.joinpath(name).read_bytes()).hexdigest()
             for name in SOURCE_FILES}
    dependencies = ("numpy", "scipy", "scikit-learn", "threadpoolctl")
    if backend == "gpu":
        from .research_models import GPU_DEPENDENCIES
        dependencies += GPU_DEPENDENCIES
    return {"code_files_sha256": names,
            "python_version": platform.python_version(),
            "package_versions": {name: importlib.metadata.version(name)
                                 for name in dependencies}}


def _plan(prepared_dir, expected_prepared_identity, code_revision, glove, backend,
          cpu_workers, created):
    from .research_models import campaign, read_scoring_inputs, _file_descriptor, release_info

    if (type(code_revision) is not str or re.fullmatch(r"[0-9a-f]{40}", code_revision) is None or
            backend not in ("cpu", "gpu") or type(backend) is not str or
            type(cpu_workers) is not int or cpu_workers not in (1, 2)):
        _fail()
    inputs = read_scoring_inputs(prepared_dir,
                                 expected_prepared_identity=expected_prepared_identity)
    spec, campaign_sha = campaign()
    locked = release_info()["inputs"]["embeddings"]
    glove_desc = _file_descriptor(glove)
    if glove_desc != {"bytes": locked["text_size"], "sha256": locked["text_sha256"]}:
        _fail()
    auxiliary = ([{"variant_id": variant, "phase": AUX_PHASE, "backend": "cpu",
                   "fit_budget_owner": variant, "consumer": "hybrid"}
                  for variant in AUX_VARIANTS] if backend == "gpu" else [])
    component_phase = AUX_PHASE if auxiliary else "fit"
    return {"schema_version": 1, "kind": "arb-ml02-campaign-plan",
            "prepared_identity": inputs.prepared_identity,
            "protocol_identity": inputs.protocol_identity,
            "fit_identity": inputs.fit_identity,
            "catalog_descriptor": inputs.catalog_descriptor,
            "train_interactions": inputs.train_interactions_descriptor,
            "train_stats": inputs.train_stats_descriptor,
            "campaign_spec_sha256": campaign_sha,
            "code_revision": code_revision,
            "producer": _source_receipt(backend),
            "variants": [{"variant_id": name, "parameters": spec["variants"][name],
                          "backend": _backend(name, backend),
                          **({"components": [{"variant_id": variant,
                                               "phase": component_phase}
                                              for variant in AUX_VARIANTS]}
                             if name == "hybrid" else {})} for name in VARIANTS],
            "auxiliary_fits": auxiliary,
            "execution_config_count": 9 if backend == "gpu" else 7,
            "assets": {"glove": glove_desc},
            "budgets": {"fit_seconds_per_variant": FIT_SECONDS,
                        "evaluation_seconds_per_variant": EVALUATION_SECONDS,
                        "campaign_wall_seconds": CAMPAIGN_SECONDS,
                        "sampled_rss_limit_bytes": RSS_LIMIT_BYTES,
                        "score_stream_limit_bytes": 64 * 1024**3,
                        "score_row_limit_bytes": 4 * 1024**2},
            "scheduling": {"cpu_workers": cpu_workers, "gpu_workers": 1,
                           "blas_threads": 1, "gpu_optional_benchmark": "not_scheduled"},
            "score_retention": "keep", "created_utc": _iso(created),
            "deadline_utc": _iso(created + timedelta(seconds=CAMPAIGN_SECONDS))}


class _Journal:
    def __init__(self, path):
        self.path = path
        self.events = []
        if path.exists():
            if path.is_symlink():
                _fail()
            info = path.stat()
            if (not stat.S_ISREG(info.st_mode) or info.st_size > 16 * 1024 * 1024 or
                    info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600):
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


def _worker_request(root, plan_identity, generation, variant, phase, attempt, prepared_dir,
                    glove, model_dir, prediction_dir, content_model, svd_model):
    artifact = root / "variants" / variant / "attempts" / phase / str(attempt) / "artifact"
    return {"schema_version": 1, "kind": "arb-ml02-campaign-worker",
            "root": str(root), "plan_identity": plan_identity,
            "generation": generation, "variant": variant, "phase": phase,
            "attempt": attempt, "prepared_dir": str(prepared_dir), "glove": str(glove),
            "model_dir": str(model_dir) if model_dir else None,
            "prediction_dir": str(prediction_dir) if prediction_dir else None,
            "content_model": str(content_model) if content_model else None,
            "svd_model": str(svd_model) if svd_model else None,
            "output_dir": str(artifact)}


def _worker_main(request_path):
    """A closed phase dispatcher; acquire the shared generation lock first."""
    request, _ = _read_private(request_path, limit=16 * 1024)
    wanted = {"schema_version", "kind", "root", "plan_identity", "generation",
              "variant", "phase", "attempt", "prepared_dir", "glove", "model_dir",
              "prediction_dir", "content_model", "svd_model", "output_dir"}
    if (type(request) is not dict or set(request) != wanted or
            request["schema_version"] != 1 or request["kind"] != "arb-ml02-campaign-worker" or
            request["variant"] not in VARIANTS or request["phase"] not in (*PHASES, AUX_PHASE) or
            type(request["attempt"]) is not int or request["attempt"] < 1):
        _fail()
    root = Path(request["root"])
    expected = root / "variants" / request["variant"] / "attempts" / request["phase"] / str(request["attempt"])
    if (not root.is_absolute() or Path(request_path) != expected / "request.json" or
            Path(request["output_dir"]) != expected / "artifact"):
        _fail()
    gate = _touch_lock(root / ".workers.lock")
    try:
        fcntl.flock(gate, fcntl.LOCK_SH)
        generation, _ = _read_private(root / "generation.json", limit=4096)
        plan, raw = _read_private(root / "campaign-plan.json")
        if (generation != {"generation": request["generation"]} or
                _digest(raw) != request["plan_identity"] or
                plan["variants"][VARIANTS.index(request["variant"])]["variant_id"] != request["variant"] or
                (request["variant"], request["phase"]) not in _expected_nodes(plan)):
            _fail("research_campaign_generation_stale")
        configured = _validate_plan_structure(plan)
        if plan.get("producer") != _source_receipt(configured):
            _fail("research_campaign_plan_mismatch")
        accepted, _, _, running, _ = _successes(_Journal(root / "journal.jsonl").events)
        start = running.get((request["variant"], request["phase"]))
        if (start is None or start["attempt"] != request["attempt"] or
                start["generation"] != request["generation"]):
            _fail("research_campaign_generation_stale")
        variant = request["variant"]
        phase = request["phase"]
        backend = ("cpu" if phase == AUX_PHASE else
                   plan["variants"][VARIANTS.index(variant)]["backend"])
        prepared = Path(request["prepared_dir"])
        output = Path(request["output_dir"])
        identity = plan["prepared_identity"]
        if not prepared.is_absolute() or not Path(request["glove"]).is_absolute():
            _fail()
        expected_model = _phase_path(root, accepted, variant, "fit")
        expected_prediction = _phase_path(root, accepted, variant, "predict")
        expected_content = _phase_path(root, accepted, "content-mean", _component_phase(plan))
        expected_svd = _phase_path(root, accepted, "puresvd", _component_phase(plan))
        required_paths = (("model_dir", expected_model),) if phase == "predict" else (
            (("prediction_dir", expected_prediction),) if phase in ("evaluate", "replay") else
            (("content_model", expected_content), ("svd_model", expected_svd))
            if phase == "fit" and variant == "hybrid" else ())
        for field, expected_path in required_paths:
            if request[field] != (str(expected_path) if expected_path is not None else None):
                _fail("research_campaign_request_invalid")
        # No scientific import or input read occurs before the gate checks.
        if phase in ("fit", AUX_PHASE):
            from .research_models import fit_model
            assets = ({"glove": Path(request["glove"])} if variant.startswith("content-") else
                      {"content_model": Path(request["content_model"]),
                       "puresvd_model": Path(request["svd_model"])} if variant == "hybrid" else None)
            manifest_path = fit_model(prepared, output, expected_prepared_identity=identity,
                                      variant_id=variant, code_revision=plan["code_revision"],
                                      assets=assets, backend=backend)
        elif phase == "predict":
            from .research_predict import predict_validation
            measured = {}
            manifest_path = predict_validation(
                prepared, Path(request["model_dir"]), output,
                expected_prepared_identity=identity, backend=backend,
                measurements=measured, measurement_output=expected / "resources" / "measurement.json")
        else:
            from .research_evaluate import evaluate_predictions
            manifest_path = evaluate_predictions(
                prepared, Path(request["prediction_dir"]), output, split="validation",
                expected_prepared_identity=identity)
        receipt = {"schema_version": 1, "kind": "arb-ml02-campaign-worker-receipt",
                   "status": "ok", "manifest_sha256": _digest(manifest_path.read_bytes()),
                   "phase": phase, "variant": variant, "attempt": request["attempt"],
                   "generation": request["generation"]}
        _private_file(expected / "result.json", _canonical(receipt))
        return 0
    except BaseException as error:
        code = str(error)
        if _SAFE_CODE.fullmatch(code) is None:
            code = "research_campaign_worker_failed"
        receipt = {"schema_version": 1, "kind": "arb-ml02-campaign-worker-receipt",
                   "status": "failed", "code": code, "phase": request["phase"],
                   "variant": request["variant"], "attempt": request["attempt"],
                   "generation": request["generation"]}
        try:
            _private_file(expected / "result.json", _canonical(receipt))
        except OSError:
            pass
        return 1
    finally:
        fcntl.flock(gate, fcntl.LOCK_UN)
        os.close(gate)


def _ensure_private(directory):
    directory = Path(directory)
    if directory.exists():
        if directory.is_symlink() or not directory.is_dir() or directory.stat().st_mode & 0o077:
            _fail()
        return
    _ensure_private(directory.parent)
    directory.mkdir(mode=0o700)
    _sync_dir(directory.parent)


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


def _phase_directory(root, variant, phase, attempt):
    return root / "variants" / variant / "attempts" / phase / str(attempt)


def _phase_path(root, successes, variant, phase):
    event = successes.get((variant, phase))
    if event is None:
        return None
    return _phase_directory(root, variant, phase, event["attempt"]) / "artifact"


def _read_receipt(attempt_dir, variant, phase, attempt, generation):
    result, _ = _read_private(attempt_dir / "result.json", limit=4096)
    common = {"schema_version": 1, "kind": "arb-ml02-campaign-worker-receipt",
              "phase": phase, "variant": variant, "attempt": attempt,
              "generation": generation}
    if (type(result) is not dict or any(result.get(k) != v for k, v in common.items()) or
            result.get("status") != "ok" or
            set(result) != {*common, "status", "manifest_sha256"} or
            type(result["manifest_sha256"]) is not str):
        _fail("research_campaign_receipt_invalid")
    return result


def _verify_phase(root, plan, successes, variant, phase, attempt, generation, prepared_dir):
    """Validate one artifact and its chain after every active process has stopped."""
    from .research_artifacts import read_json, safe_path
    from .research_models import load_model, read_scoring_inputs

    attempt_dir = _phase_directory(root, variant, phase, attempt)
    artifact = safe_path(attempt_dir / "artifact", directory=True)
    receipt = _read_receipt(attempt_dir, variant, phase, attempt, generation)
    manifest_name = {"fit": "model-manifest.json", AUX_PHASE: "model-manifest.json",
                     "predict": "prediction-manifest.json",
                     "evaluate": "evaluation-manifest.json",
                     "replay": "evaluation-manifest.json"}[phase]
    manifest, raw = read_json(artifact / manifest_name)
    manifest_identity = _digest(raw)
    if receipt["manifest_sha256"] != manifest_identity:
        _fail("research_campaign_receipt_invalid")
    backend = ("cpu" if phase == AUX_PHASE else
               plan["variants"][VARIANTS.index(variant)]["backend"])
    inputs = read_scoring_inputs(prepared_dir,
                                 expected_prepared_identity=plan["prepared_identity"])
    if phase in ("fit", AUX_PHASE):
        if (manifest["variant_id"] != variant or
                manifest["producer"]["code_revision"] != plan["code_revision"]):
            _fail()
        if backend == "gpu":
            from .research_gpu import gpu_execution
            with gpu_execution():
                _, scorer = load_model(artifact, inputs=inputs, backend=backend)
                scorer.close()
        else:
            _, scorer = load_model(artifact, inputs=inputs, backend=backend)
            close = getattr(scorer, "close", None)
            if callable(close):
                close()
    elif phase == "predict":
        from .research_evaluate import _predictions
        from .research_models import _file_descriptor
        prepared_manifest, _ = read_json(prepared_dir / "prepare-manifest.json")
        prediction, prediction_identity = _predictions(
            artifact, prepared_manifest, plan["prepared_identity"], "validation",
            inputs.fit_identity)
        model_path = _phase_path(root, successes, variant, "fit")
        if model_path is None:
            _fail()
        _, model_raw = read_json(model_path / "model-manifest.json")
        if (prediction["model_id"] != variant or
                prediction["config_sha256"] != hashlib.sha256(model_raw).hexdigest() or
                prediction["code_revision"] != plan["code_revision"] or
                _file_descriptor(artifact / "scores.jsonl.gz") !=
                {key: prediction["scores"][key] for key in ("bytes", "sha256")}):
            _fail("research_campaign_prediction_invalid")
        if backend == "gpu":
            measured, _ = _read_private(attempt_dir / "resources" / "measurement.json")
            if (measured.get("status") != "published" or measured.get("backend") != "gpu" or
                    measured.get("gpu_admission", {}).get("purpose") !=
                    "parity-only-gpu-admission"):
                _fail("research_campaign_admission_invalid")
        return manifest_identity, prediction_identity
    else:
        from .research_compare import _evaluation
        eval_manifest, eval_identity, rows = _evaluation(
            artifact, plan["prepared_identity"])
        prediction_path = _phase_path(root, successes, variant, "predict")
        if prediction_path is None:
            _fail()
        _, prediction_raw = read_json(prediction_path / "prediction-manifest.json")
        if (eval_manifest["model_id"] != variant or
                eval_manifest["prediction_identity"] != _digest(prediction_raw) or
                eval_manifest["prediction_code_revision"] != plan["code_revision"] or
                eval_manifest["fit_identity"] != plan["fit_identity"] or
                len(rows) == 0):
            _fail("research_campaign_evaluation_invalid")
        if phase == "replay":
            original = _phase_path(root, successes, variant, "evaluate")
            if original is None:
                _fail()
            for name in ("report.json", "user-metrics.jsonl.gz", "evaluation-manifest.json"):
                if (artifact / name).read_bytes() != (original / name).read_bytes():
                    _fail("research_campaign_replay_mismatch")
        return manifest_identity, eval_identity
    return manifest_identity, manifest_identity


def _worker_env():
    environment = dict(os.environ)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        environment[name] = "1"
    return environment


def _process_fields(result):
    return {"outcome": result.outcome, "returncode": result.returncode,
            "elapsed_seconds": result.elapsed_seconds,
            "sampled_peak_rss_bytes": result.sampled_peak_rss_bytes,
            "wait4_peak_rss_bytes": result.wait4_peak_rss_bytes,
            "observed_peak_rss_bytes": result.observed_peak_rss_bytes,
            "rss_samples": result.rss_samples,
            "max_sample_gap_seconds": result.max_sample_gap_seconds,
            "cleanup_complete": result.cleanup_complete}


def _bytes_equal(left, right):
    with Path(left).open("rb") as one, Path(right).open("rb") as two:
        while True:
            a, b = one.read(1024 * 1024), two.read(1024 * 1024)
            if a != b:
                return False
            if not a:
                return True


def _next_ready(successes, failed, active_keys, plan, cpu_used, gpu_used,
                cpu_workers):
    for variant in VARIANTS:
        for phase in (*PHASES, AUX_PHASE):
            key = (variant, phase)
            if key not in _expected_nodes(plan):
                continue
            if key in successes or key in failed or key in active_keys:
                continue
            index = PHASES.index(phase) if phase in PHASES else 0
            if phase == AUX_PHASE and (variant, "fit") not in successes:
                continue
            if index and (variant, PHASES[index - 1]) not in successes:
                continue
            if phase == "fit" and variant == "hybrid" and any(
                    (dependency, _component_phase(plan)) not in successes for dependency in
                    ("content-mean", "puresvd")):
                continue
            lane = ("cpu" if phase == AUX_PHASE else
                    plan["variants"][VARIANTS.index(variant)]["backend"]
                    if phase in ("fit", "predict") else "cpu")
            if lane == "gpu" and gpu_used or lane == "cpu" and cpu_used >= cpu_workers:
                continue
            return variant, phase, lane
    return None


def _phase_budget_remaining(charged, variant, phase):
    budget = FIT_SECONDS if phase in ("fit", AUX_PHASE) else EVALUATION_SECONDS
    used = sum(charged.get((variant, step), 0.0)
               for step in (("fit", AUX_PHASE) if phase in ("fit", AUX_PHASE) else PHASES[1:]))
    return budget - used


def _remaining(plan, charged, variant, phase):
    until = (_parse_iso(plan["deadline_utc"]) - _now()).total_seconds()
    return min(_phase_budget_remaining(charged, variant, phase), until)


def _prepare_attempt(root, plan, plan_identity, generation, successes, attempts,
                     variant, phase, prepared_dir, glove):
    number = attempts.get((variant, phase), 0) + 1
    directory = _phase_directory(root, variant, phase, number)
    _ensure_private(directory)
    _ensure_private(directory / "resources")
    model = _phase_path(root, successes, variant, "fit")
    predictions = _phase_path(root, successes, variant, "predict")
    content = _phase_path(root, successes, "content-mean", _component_phase(plan))
    svd = _phase_path(root, successes, "puresvd", _component_phase(plan))
    request = _worker_request(root, plan_identity, generation, variant, phase, number,
                              prepared_dir, glove, model, predictions, content, svd)
    _private_file(directory / "request.json", _canonical(request))
    return directory, number


def run_campaign(prepared_dir, output_dir, *, expected_prepared_identity,
                 code_revision, glove: Path, backend="cpu", cpu_workers=2,
                 resume=False, expected_campaign_identity=None):
    """Run exactly seven fixed validation variants with durable, bounded workers."""
    from .research_artifacts import safe_path, separate
    from .research_resources import ResourceSupervisor

    prepared_dir = Path(prepared_dir).absolute()
    glove = Path(glove).absolute()
    root = Path(output_dir).absolute()
    separate(prepared_dir, glove, root)
    safe_path(prepared_dir, directory=True)
    safe_path(glove, directory=False)
    if type(resume) is not bool or (resume and expected_campaign_identity is None) or (
            not resume and expected_campaign_identity is not None):
        _fail()
    if resume:
        if root.is_symlink() or not root.is_dir() or root.stat().st_mode & 0o077:
            _fail()
    else:
        if root.exists() or root.is_symlink():
            _fail()
        root.mkdir(mode=0o700)
        _sync_dir(root.parent)
    lock = _touch_lock(root / ".campaign.lock")
    gate = None
    try:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            _fail("research_campaign_busy")
        if resume:
            prior, raw = _read_private(root / "campaign-plan.json")
            created = _parse_iso(prior["created_utc"])
            planned = _plan(prepared_dir, expected_prepared_identity,
                            code_revision, glove, backend, cpu_workers, created)
            if prior != planned or _digest(raw) != expected_campaign_identity:
                _fail("research_campaign_plan_mismatch")
            plan, plan_identity = prior, _digest(raw)
        else:
            created = _now()
            plan = _plan(prepared_dir, expected_prepared_identity,
                         code_revision, glove, backend, cpu_workers, created)
            raw = _canonical(plan)
            _private_file(root / "campaign-plan.json", raw)
            _sync_dir(root)
            plan_identity = _digest(raw)
        _validate_plan_structure(plan)
        if _now() > _parse_iso(plan["deadline_utc"]):
            _fail("research_campaign_wall_limit")
        gate = _touch_lock(root / ".workers.lock")
        try:
            fcntl.flock(gate, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            _fail("research_campaign_workers_active")
        journal = _Journal(root / "journal.jsonl")
        successes, attempts, charged, starts, failed = _successes(journal.events)
        if starts:
            # A crashed coordinator cannot certify that a native descendant was
            # reaped. Preserve every orphan attempt and fail closed.
            _fail("research_campaign_interrupted_uncertain")
        for (variant, phase), event in successes.items():
            # A completed stage is reusable only after all current source/input
            # receipts and every payload/cross-link have been checked.
            identity, _ = _verify_phase(root, plan, successes, variant, phase,
                                        event["attempt"], event["generation"], prepared_dir)
            if identity != event.get("output_identity"):
                _fail("research_campaign_chain_invalid")
        generation = uuid.uuid4().hex
        _replace_private(root / "generation.json", _canonical({"generation": generation}))
        fcntl.flock(gate, fcntl.LOCK_UN)
        journal.append("run_started", generation=generation, plan_identity=plan_identity,
                       resumed=resume)
        active = {}
        pending = []
        frozen = bool(failed)
        failure_code = "research_campaign_previous_failure" if failed else None
        supervisor = ResourceSupervisor(rss_limit_bytes=RSS_LIMIT_BYTES)
        original_error = None
        try:
            supervisor.__enter__()
            while True:
                now = _now()
                if now >= _parse_iso(plan["deadline_utc"]) and not frozen:
                    frozen, failure_code = True, "research_campaign_wall_limit"
                    journal.append("campaign_failure", code=failure_code)
                    for handle in active:
                        supervisor.cancel(handle)
                if not frozen:
                    # Start independent ready phases while slots are free. Heavy
                    # artifact validation is deferred until no child is active.
                    while True:
                        cpu_used = sum(item["lane"] == "cpu" for item in active.values())
                        gpu_used = sum(item["lane"] == "gpu" for item in active.values())
                        choice = _next_ready(successes, failed,
                                             {(item["variant"], item["phase"]) for item in active.values()} |
                                             {(item["variant"], item["phase"]) for item in pending},
                                             plan, cpu_used, gpu_used, cpu_workers)
                        if choice is None:
                            break
                        variant, phase, lane = choice
                        remaining = _remaining(plan, charged, variant, phase)
                        if remaining <= 0:
                            frozen, failure_code = True, "research_campaign_budget_limit"
                            journal.append("campaign_failure", code=failure_code)
                            break
                        directory, number = _prepare_attempt(
                            root, plan, plan_identity, generation, successes, attempts,
                            variant, phase, prepared_dir, glove)
                        attempts[(variant, phase)] = number
                        journal.append("attempt_started", generation=generation,
                                       variant=variant, phase=phase, attempt=number,
                                       remaining_seconds=remaining)
                        # Preserve the virtualenv executable symlink: resolving it
                        # switches -I workers to the system interpreter.
                        argv = [str(Path(sys.executable).absolute()), "-I", "-m",
                                "recsys.research_campaign", "--worker-request",
                                str(directory / "request.json")]
                        started = time.monotonic()
                        try:
                            handle = supervisor.start(argv, cwd=root,
                                                      log_dir=directory / "logs",
                                                      remaining_seconds=remaining,
                                                      env=_worker_env())
                        except BaseException:
                            charge = time.monotonic() - started
                            charged[(variant, phase)] = charged.get((variant, phase), 0) + charge
                            failed.add((variant, phase))
                            frozen, failure_code = True, "research_campaign_spawn_failed"
                            journal.append("attempt_finished", generation=generation,
                                           variant=variant, phase=phase, attempt=number,
                                           status="failed", code=failure_code,
                                           charged_seconds=charge, resources=None)
                            break
                        active[handle] = {"variant": variant, "phase": phase, "lane": lane,
                                          "attempt": number, "generation": generation,
                                          "observed_elapsed": 0.0, "warned": False}
                if not active:
                    if pending:
                        for item in pending:
                            variant, phase = item["variant"], item["phase"]
                            result = item["result"]
                            charge = max(item["observed_elapsed"], result.elapsed_seconds)
                            charged[(variant, phase)] = charged.get((variant, phase), 0) + charge
                            success = result.outcome == "ok" and result.cleanup_complete
                            artifact_identity = None
                            if success:
                                try:
                                    artifact_identity, _ = _verify_phase(
                                        root, plan, successes, variant, phase,
                                        item["attempt"], generation, prepared_dir)
                                except BaseException:
                                    success = False
                                    failure_code = "research_campaign_artifact_invalid"
                            if success and (_phase_budget_remaining(charged, variant, phase) < 0 or
                                            _now() > _parse_iso(plan["deadline_utc"])):
                                success = False
                                failure_code = "research_campaign_budget_limit"
                            event = journal.append(
                                "attempt_finished", generation=generation, variant=variant,
                                phase=phase, attempt=item["attempt"],
                                status="success" if success else "failed",
                                code=None if success else failure_code or "research_campaign_phase_failed",
                                charged_seconds=charge, resources=_process_fields(result),
                                output_identity=artifact_identity if success else None)
                            if success:
                                successes[(variant, phase)] = event
                            else:
                                failed.add((variant, phase))
                                frozen = True
                                failure_code = event["code"]
                        pending.clear()
                    if frozen or set(successes) == _expected_nodes(plan):
                        break
                    if _next_ready(successes, failed, set(), plan, 0, 0,
                                   cpu_workers) is None:
                        frozen, failure_code = True, "research_campaign_dependency_blocked"
                        journal.append("campaign_failure", code=failure_code)
                        break
                    continue
                snapshot = supervisor.poll(wait_seconds=.1)
                # An interim cleanup failure has no completed event. Inspect every
                # unfinished handle after every poll, keep slots, and freeze work.
                for handle, item in tuple(active.items()):
                    result = supervisor.result(handle)
                    if result is None:
                        continue
                    item["observed_elapsed"] = max(item["observed_elapsed"],
                                                    result.elapsed_seconds)
                    if not result.cleanup_complete:
                        if not item["warned"]:
                            item["warned"] = True
                            journal.append("cleanup_warning", generation=generation,
                                           variant=item["variant"], phase=item["phase"],
                                           attempt=item["attempt"],
                                           code="research_campaign_cleanup_failed")
                        frozen, failure_code = True, "research_campaign_cleanup_failed"
                        for other in active:
                            if other != handle:
                                supervisor.cancel(other)
                        continue
                    if handle not in snapshot.completed:
                        _fail("research_campaign_supervisor_invalid")
                    item["result"] = result
                    pending.append(item)
                    del active[handle]
                if frozen and active:
                    for handle in active:
                        supervisor.cancel(handle)
        except BaseException as error:
            original_error = error
            try:
                journal.append("campaign_failure", code="research_campaign_controller_failed")
            except BaseException:
                # A backward UTC clock can reject this second event too. Keep
                # the first failure and leave started attempts unconfirmed.
                pass
            raise
        finally:
            try:
                supervisor.close()
            except BaseException:
                if original_error is None:
                    raise
            if original_error is not None:
                # Only a confirmed clean supervisor result permits a future
                # retry; orphan or failed cleanup keeps its started record open.
                for handle, item in active.items():
                    result = supervisor.result(handle)
                    if result is None or not result.cleanup_complete:
                        continue
                    try:
                        journal.append("attempt_interrupted", generation=generation,
                                       variant=item["variant"], phase=item["phase"],
                                       attempt=item["attempt"],
                                       charged_seconds=max(item["observed_elapsed"],
                                                           result.elapsed_seconds),
                                       resources=_process_fields(result))
                    except BaseException:
                        pass
                for item in pending:
                    result = item["result"]
                    if result.cleanup_complete:
                        try:
                            journal.append("attempt_interrupted", generation=generation,
                                           variant=item["variant"], phase=item["phase"],
                                           attempt=item["attempt"],
                                           charged_seconds=max(item["observed_elapsed"],
                                                               result.elapsed_seconds),
                                           resources=_process_fields(result))
                        except BaseException:
                            pass
        complete = not frozen and set(successes) == _expected_nodes(plan)
        if complete:
            report = _public_projection(root, plan, successes, prepared_dir)
            if _now() > _parse_iso(plan["deadline_utc"]):
                complete, failure_code = False, "research_campaign_wall_limit"
        else:
            report = {"variants": [], "primary_ndcg5": None, "paired_vs_popularity": None}
        report.update({"schema_version": 1, "kind": "arb-ml02-campaign-report",
                       "plan_identity": plan_identity, "prepared_identity": plan["prepared_identity"],
                       "status": "complete" if complete else "incomplete",
                       "failure_code": None if complete else failure_code,
                       "diagnostics": {"single_query_latency": "not_measured",
                                       "batch_throughput": "not_measured",
                                       "reason": "standalone_benchmark_not_scheduled"}})
        journal.append("campaign_finished", generation=generation,
                       status=report["status"], code=report["failure_code"])
        _replace_private(root / "campaign-report.json", _canonical(report))
        manifest = {"schema_version": 1, "kind": "arb-ml02-campaign-report-manifest",
                    "plan_identity": plan_identity, "status": report["status"],
                    "report": {"bytes": len(_canonical(report)),
                               "sha256": hashlib.sha256(_canonical(report)).hexdigest()}}
        _replace_private(root / "campaign-report-manifest.json", _canonical(manifest))
        return root / "campaign-report-manifest.json"
    finally:
        if gate is not None:
            os.close(gate)
        fcntl.flock(lock, fcntl.LOCK_UN)
        os.close(lock)


def _public_projection(root, plan, successes, prepared_dir):
    """Build only anonymous validation aggregates from strictly paired rows."""
    import numpy as np
    from .research_artifacts import read_json
    from .research_compare import _bootstrap, _evaluation, _primary, _quantile
    from .research_evaluate import _aggregate, _prepared
    from .research_models import _file_descriptor
    from .research_protocol import load_protocol

    prepared, protocol, catalog, fit, fit_identity, stats = _prepared(
        prepared_dir, plan["prepared_identity"])
    if fit_identity != plan["fit_identity"] or protocol != load_protocol():
        _fail("research_campaign_input_invalid")
    all_rows = {}
    variants = []
    auxiliaries = []
    journal_events = _Journal(root / "journal.jsonl").events
    _, attempts, charged, _, _ = _successes(journal_events)
    reference_metadata = None
    reference_n0 = None
    for variant in VARIANTS:
        paths = {phase: _phase_path(root, successes, variant, phase) for phase in PHASES}
        if any(value is None for value in paths.values()):
            _fail("research_campaign_incomplete")
        model, model_raw = read_json(paths["fit"] / "model-manifest.json")
        prediction, prediction_raw = read_json(paths["predict"] / "prediction-manifest.json")
        evaluation, evaluation_identity, rows = _evaluation(
            paths["evaluate"], plan["prepared_identity"])
        initial_report, _ = read_json(paths["evaluate"] / "report.json",
                                      evaluation["files"]["report.json"])
        recomputed = _aggregate(rows.values(), catalog, stats)
        expected_report = {"schema_version": 1, "kind": "arb-ml-evaluation-report",
                           "split": "validation", "prediction_identity": _digest(prediction_raw),
                           "prepared_identity": plan["prepared_identity"],
                           "queries": len(rows) // 2, "groups": recomputed}
        if _canonical(initial_report) != _canonical(expected_report):
            _fail("research_campaign_aggregate_mismatch")
        if (model["variant_id"] != variant or prediction["model_id"] != variant or
                prediction["config_sha256"] != hashlib.sha256(model_raw).hexdigest() or
                evaluation["model_id"] != variant or
                evaluation["prediction_identity"] != _digest(prediction_raw) or
                evaluation["config_sha256"] != prediction["config_sha256"] or
                evaluation["prediction_code_revision"] != plan["code_revision"] or
                evaluation["selection_identity"] is not None or
                prediction["split"] != "validation" or
                _file_descriptor(paths["predict"] / "scores.jsonl.gz") !=
                {key: prediction["scores"][key] for key in ("bytes", "sha256")}):
            _fail("research_campaign_chain_invalid")
        for name in ("report.json", "user-metrics.jsonl.gz", "evaluation-manifest.json"):
            if (paths["evaluate"] / name).read_bytes() != (paths["replay"] / name).read_bytes():
                _fail("research_campaign_replay_mismatch")
        metadata = {key: (row["user_key"], row["split"], row["cohort"], row["mask"],
                          row["order"], row["n"], row["history_bin"], row["seed_band"],
                          row["recent_2024_seed_count"])
                    for key, row in rows.items()}
        n0 = {key: (row["ranking"], row["metrics"])
              for key, row in rows.items() if row["n"] == 0}
        if reference_metadata is None:
            reference_metadata, reference_n0 = metadata, n0
        elif metadata != reference_metadata or n0 != reference_n0:
            _fail("research_campaign_unpaired")
        all_rows[variant] = rows
        phase_resources = {}
        for phase in PHASES:
            event = successes[(variant, phase)]
            evidence = event["resources"]
            if (type(evidence) is not dict or evidence.get("outcome") != "ok" or
                    evidence.get("cleanup_complete") is not True):
                _fail("research_campaign_resource_invalid")
            phase_resources[phase] = {
                key: evidence[key] for key in ("elapsed_seconds", "sampled_peak_rss_bytes",
                                                  "wait4_peak_rss_bytes", "observed_peak_rss_bytes",
                                                  "rss_samples", "max_sample_gap_seconds")}
        measurement = None
        if plan["variants"][VARIANTS.index(variant)]["backend"] == "gpu":
            measured, _ = _read_private(paths["predict"].parent / "resources" / "measurement.json")
            pool = measured.get("dedicated_pool_session_including_admission")
            if (measured.get("status") != "published" or type(pool) is not dict or
                    set(pool) != {"pool_used_bytes", "pool_reserved_bytes", "pool_limit_bytes",
                                  "pool_peak_used_bytes", "pool_peak_reserved_bytes"} or
                    any(type(value) is not int or value < 0 for value in pool.values())):
                _fail("research_campaign_measurement_invalid")
            measurement = {"gpu_admission_seconds": measured["gpu_admission_seconds"],
                           "queries": measured["queries"],
                           "score_pairs": measured["score_pairs"],
                           "synchronized_scoring_seconds": measured["synchronized_scoring_seconds"],
                           "encoding_gzip_write_seconds": measured["encoding_gzip_write_seconds"],
                           "dedicated_pool_session_including_admission": dict(pool),
                           "memory_scope": "dedicated-pool-only-excludes-context-and-cublas"}
        variants.append({"variant_id": variant,
                         "backend": plan["variants"][VARIANTS.index(variant)]["backend"],
                         "model_identity": _digest(model_raw),
                         "prediction_identity": _digest(prediction_raw),
                         "evaluation_identity": evaluation_identity,
                         "replay_identity": _digest((paths["replay"] / "evaluation-manifest.json").read_bytes()),
                         "evaluation_replay_bytes_equal": True,
                         "queries": initial_report["queries"],
                         "scores_compressed_bytes": prediction["scores"]["bytes"],
                         "scores_uncompressed_bytes": prediction["scores"]["uncompressed_bytes"],
                         "resources": phase_resources, "predict_measurement": measurement,
                         "fit_budget": {
                             "primary_charged_seconds": charged.get((variant, "fit"), 0.0),
                             "auxiliary_charged_seconds": charged.get((variant, AUX_PHASE), 0.0),
                             "all_attempts_charged_seconds": charged.get((variant, "fit"), 0.0) +
                                                              charged.get((variant, AUX_PHASE), 0.0),
                             "remaining_seconds": FIT_SECONDS - charged.get((variant, "fit"), 0.0) -
                                                  charged.get((variant, AUX_PHASE), 0.0)},
                         "checked_groups": recomputed})
    for variant in AUX_VARIANTS if plan["auxiliary_fits"] else ():
        path = _phase_path(root, successes, variant, AUX_PHASE)
        if path is None:
            _fail("research_campaign_incomplete")
        manifest, raw = read_json(path / "model-manifest.json")
        if (manifest["variant_id"] != variant or
                manifest["producer"]["execution"]["backend"] != "cpu"):
            _fail("research_campaign_component_invalid")
        evidence = successes[(variant, AUX_PHASE)]["resources"]
        if evidence.get("outcome") != "ok" or evidence.get("cleanup_complete") is not True:
            _fail("research_campaign_resource_invalid")
        auxiliaries.append({"variant_id": variant, "backend": "cpu",
                            "phase": AUX_PHASE, "status": "complete",
                            "component_identity": _digest(raw),
                            "consumer": "hybrid", "fit_budget_owner": variant,
                            "resources": {key: evidence[key] for key in
                                          ("elapsed_seconds", "sampled_peak_rss_bytes",
                                           "wait4_peak_rss_bytes", "observed_peak_rss_bytes",
                                           "rss_samples", "max_sample_gap_seconds")}})
    hybrid_path = _phase_path(root, successes, "hybrid", "fit")
    hybrid_manifest, _ = read_json(hybrid_path / "model-manifest.json")
    if hybrid_manifest["producer"]["execution"]["backend"] != "cpu":
        _fail("research_campaign_component_invalid")
    for part, variant in (("content", "content-mean"), ("svd", "puresvd")):
        origin = _phase_path(root, successes, variant, _component_phase(plan))
        source, raw = read_json(origin / "model-manifest.json")
        if hybrid_manifest["metadata"]["components"][part]["identity"] != _digest(raw):
            _fail("research_campaign_component_invalid")
        if not _bytes_equal(origin / "model-manifest.json", hybrid_path / (part + "_manifest.json")):
            _fail("research_campaign_component_invalid")
        prefix = "content_" if part == "content" else "svd_"
        for logical in source["files"]:
            if not _bytes_equal(origin / logical, hybrid_path / (prefix + logical)):
                _fail("research_campaign_component_invalid")
    reference_users, reference_cells = _primary(all_rows["popularity"])
    main_users = len(reference_users)
    for variant in VARIANTS[1:]:
        users, cells = _primary(all_rows[variant])
        if users != reference_users or any(
                cells[user][arm][0] != reference_cells[user][arm][0]
                for user in users for arm in reference_cells[user]):
            _fail("research_campaign_unpaired")
    primary = paired = None
    if main_users >= 20:
        matrices = {}
        for variant in VARIANTS:
            _, cells = _primary(all_rows[variant])
            matrices[variant] = np.asarray([
                [sum(cells[user][(f"random-liked-{order}", n)][1]
                     for order in (1, 2, 3)) / 3 for n in range(1, 11)]
                for user in reference_users], dtype=np.float64)
        primary = {variant: {"users": main_users,
                             "mean_ndcg5_by_n": [float(x) for x in matrix.mean(axis=0)]}
                   for variant, matrix in matrices.items()}
        paired = {}
        comparison = protocol["comparison"]
        for variant in VARIANTS[1:]:
            delta = matrices[variant] - matrices["popularity"]
            sample = _bootstrap(delta, comparison["seed"],
                                comparison["bootstrap_replicates"])
            paired[variant] = {"users": main_users,
                               "mean_difference_by_n": [float(x) for x in delta.mean(axis=0)],
                               "pointwise_ci95_by_n": [
                                   [_quantile(sample[:, index], .025, "lower"),
                                    _quantile(sample[:, index], .975, "higher")]
                                   for index in range(10)],
                               "bootstrap_replicates": comparison["bootstrap_replicates"],
                               "bootstrap_seed": comparison["seed"]}
    return {"variants": variants, "auxiliary_fits": auxiliaries,
            "execution_config_count": plan["execution_config_count"],
            "planned_nodes": len(_expected_nodes(plan)),
            "actual_completed_model_packages": sum(
                phase in ("fit", AUX_PHASE) for _variant, phase in successes),
            "actual_attempts_started": sum(attempts.values()),
            "actual_attempts_completed": sum(
                event["event"] == "attempt_finished" for event in journal_events),
            "hybrid_admission": {"status": "not_applicable",
                                 "reason": "declared_cpu_execution"},
            "primary_users": main_users,
            "primary_ndcg5": primary, "paired_vs_popularity": paired,
            "small_cohort_suppressed": main_users < 20}


def validate_campaign(prepared_dir, campaign_dir, *, expected_prepared_identity,
                      expected_campaign_identity):
    """Revalidate retained artifacts and published anonymous aggregates."""
    from .research_artifacts import safe_path, separate

    prepared_dir = Path(prepared_dir).absolute()
    root = Path(campaign_dir).absolute()
    separate(prepared_dir, root)
    safe_path(prepared_dir, directory=True)
    safe_path(root, directory=True)
    plan, raw = _read_private(root / "campaign-plan.json")
    configured = _validate_plan_structure(plan)
    if (_digest(raw) != expected_campaign_identity or
            plan["prepared_identity"] != expected_prepared_identity or
            plan["producer"] != _source_receipt(configured) or
            tuple(entry["variant_id"] for entry in plan["variants"]) != VARIANTS):
        _fail("research_campaign_plan_mismatch")
    journal = _Journal(root / "journal.jsonl")
    successes, attempts, charged, starts, failed = _successes(journal.events)
    if starts or failed or set(successes) != _expected_nodes(plan):
        _fail("research_campaign_incomplete")
    for (variant, phase), event in successes.items():
        identity, _ = _verify_phase(root, plan, successes, variant, phase,
                                    event["attempt"], event["generation"], prepared_dir)
        if identity != event["output_identity"]:
            _fail("research_campaign_chain_invalid")
    projection = _public_projection(root, plan, successes, prepared_dir)
    report, report_raw = _read_private(root / "campaign-report.json")
    manifest, _ = _read_private(root / "campaign-report-manifest.json")
    expected_fields = {"schema_version": 1, "kind": "arb-ml02-campaign-report",
                       "plan_identity": expected_campaign_identity,
                       "prepared_identity": expected_prepared_identity,
                       "status": "complete", "failure_code": None,
                       "diagnostics": {"single_query_latency": "not_measured",
                                       "batch_throughput": "not_measured",
                                       "reason": "standalone_benchmark_not_scheduled"}}
    if (any(report.get(key) != value for key, value in expected_fields.items()) or
            any(report.get(key) != value for key, value in projection.items()) or
            set(report) != {*expected_fields, *projection} or
            manifest != {"schema_version": 1, "kind": "arb-ml02-campaign-report-manifest",
                         "plan_identity": expected_campaign_identity, "status": "complete",
                         "report": {"bytes": len(report_raw),
                                    "sha256": hashlib.sha256(report_raw).hexdigest()}}):
        _fail("research_campaign_report_invalid")
    return report


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker-request":
        try:
            worker_status = _worker_main(Path(sys.argv[2]))
        except BaseException:
            worker_status = 2
        sys.exit(worker_status)
    else:
        sys.exit(2)
