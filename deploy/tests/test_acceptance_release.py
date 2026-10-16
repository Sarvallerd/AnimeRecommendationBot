"""Small saved-evidence tests for historical read-only verification."""

import contextlib
import copy
import datetime as dt
import importlib.util
import io
import json
import os
import pathlib
import shutil
import stat
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from test_acceptance_evidence import database
from test_acceptance_profiles import FIXTURE, encoded

SCRIPT = pathlib.Path(__file__).with_name("release-acceptance.py")
SPEC = importlib.util.spec_from_file_location("arb033_release_tests", SCRIPT)
r = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(r)


def variant(source, target, version):
    target.mkdir()
    for name in ("catalog.json", "neighbors.json", "manifest.json"):
        shutil.copyfile(source / name, target / name)
    manifest = json.loads((target / "manifest.json").read_bytes())
    manifest["sources"][0]["version"] = version
    (target / "manifest.json").write_bytes(encoded(manifest))
    return "sha256:" + r.sha((target / "manifest.json").read_bytes())


def saved(root, name, raw):
    return r.private_file(root / name, raw)


def make_fixture(root):
    root.chmod(0o700)
    bundles = root / "bundles"
    bundles.mkdir()
    a = bundles / "A"
    b = bundles / "B"
    old = bundles / "old"
    a.mkdir()
    for name in ("catalog.json", "neighbors.json", "manifest.json"):
        shutil.copyfile(FIXTURE / name, a / name)
    aid = "sha256:" + r.sha((a / "manifest.json").read_bytes())
    b.mkdir()
    for name in ("catalog.json", "neighbors.json", "manifest.json"):
        shutil.copyfile(FIXTURE / name, b / name)
    bid = aid
    oid = variant(FIXTURE, old, "3")
    semantics = {
        "catalog_count": 8,
        "neighbor_count_histogram": {"0": 1, "1": 0, "2": 1, "3": 0, "4": 0, "5": 6},
        "smoke_seed": {"mal_id": 1, "query": "Общее название", "neighbor_ids": [10, 11, 12, 2, 3]},
        "empty_seed": {"mal_id": 99, "query": "Тихий ноль"},
        "vector_status_counts": {"ready": 7, "oov_only": 1, "no_tokens": 0, "zero_vector": 0},
        "zero_vector_ids": [99],
    }
    lock = copy.deepcopy(r.p.load_profile(r.NEW).lock)
    layout = {}
    for key, filename in (
        ("base", "dataset-manifest.json"),
        ("aliases", "alias-manifest.json"),
        ("release", "release-manifest.json"),
    ):
        directory = root / key
        directory.mkdir()
        (directory / filename).write_bytes(encoded({"kind": key}))
        layout[key] = str(directory)
        lock["inputs"][{"base": "base", "aliases": "aliases", "release": "catalog_release"}[key]][
            "identity"
        ] = "sha256:" + r.file_sha(directory / filename)
    shutil.copyfile(a / "catalog.json", pathlib.Path(layout["release"]) / "catalog.json")
    lock["inputs"]["catalog_release"]["catalog_sha256"] = r.file_sha(a / "catalog.json")
    alias_raw = root / "alias_raw"
    alias_raw.mkdir()
    (alias_raw / "aliases.csv").write_bytes(b"small")
    layout["alias_raw"] = str(alias_raw)
    glove_dir = root / "glove"
    glove_dir.mkdir()
    glove = glove_dir / "glove.6B.300d.txt"
    glove.write_bytes(b"glove")
    layout["glove"] = str(glove)
    lock["inputs"]["embeddings"]["text_sha256"] = r.file_sha(glove)
    lock["inputs"]["embeddings"]["text_size"] = glove.stat().st_size
    lock["semantic_expectations"] = semantics
    lock["reference"]["bundle_identity"] = aid
    lock["reference"]["catalog_sha256"] = r.file_sha(a / "catalog.json")
    lock["reference"]["neighbors_sha256"] = r.file_sha(a / "neighbors.json")
    ref = lock["reference"]["producer"]
    summaries = {
        key: r.p.BundleSummary(
            identity,
            8,
            {
                name: r.file_sha(path / name)
                for name in ("catalog.json", "neighbors.json", "manifest.json")
            },
            {},
            semantics["neighbor_count_histogram"],
        )
        for key, path, identity in (("A", a, aid), ("B", b, bid))
    }
    reports = {}
    for label, block in (("A", 256), ("B", 128)):
        details = {
            key: {
                "vector_status": "oov_only" if key == "99" else "ready",
                "neighbor_count": len(value),
                "token_count": 1,
                "matched_token_count": 0 if key == "99" else 1,
            }
            for key, value in json.loads((a / "neighbors.json").read_bytes())["neighbors"].items()
        }
        reports[label] = {
            "schema_version": 2,
            "kind": "arb-release-build",
            "catalog_release_identity": lock["inputs"]["catalog_release"]["identity"],
            "inputs": {
                "glove.6B.300d.txt": {"sha256": lock["inputs"]["embeddings"]["text_sha256"]}
            },
            "files": {
                "catalog.json": {"sha256": summaries[label].hashes["catalog.json"]},
                "neighbors.json": {"sha256": summaries[label].hashes["neighbors.json"]},
            },
            "producer": {
                "python_version": ref["python_version"],
                "numpy_version": ref["numpy_version"],
                "package_version": ref["package_version"],
                "code_files": {
                    name: ref["code_files_sha256"][name]
                    for name in ("build.py", "release_build.py")
                },
            },
            "execution": {"requested_block_rows": block, "effective_block_rows": block},
            "coverage": {
                "neighbor_count_histogram": semantics["neighbor_count_histogram"],
                "vector_status_counts": semantics["vector_status_counts"],
                "zero_vector_ids": [99],
            },
            "anime": details,
        }
    for label, path in (("A", a), ("B", b)):
        manifest = json.loads((path / "manifest.json").read_bytes())
        manifest["sources"].extend(
            [
                {
                    "name": "build_report",
                    "version": "arb031-v1",
                    "sha256": r.sha(encoded(reports[label])),
                    "metadata": {
                        "filename": "build-report.json",
                        "producer": reports[label]["producer"],
                    },
                },
                {
                    "name": "bundle_exporter",
                    "version": "arb031-v1",
                    "sha256": ref["code_files_sha256"]["release_build.py"],
                    "metadata": {
                        "filename": "release_build.py",
                        "package_version": ref["package_version"],
                        "export_sha256": ref["code_files_sha256"]["export.py"],
                        "bundle_sha256": ref["code_files_sha256"]["bundle.py"],
                    },
                },
            ]
        )
        manifest["sources"].sort(key=lambda item: (item["name"], item["version"]))
        (path / "manifest.json").write_bytes(encoded(manifest))
        summaries[label].identity = "sha256:" + r.file_sha(path / "manifest.json")
        summaries[label].hashes["manifest.json"] = r.file_sha(path / "manifest.json")
    aid = summaries["A"].identity
    bid = summaries["B"].identity
    lock["reference"]["bundle_identity"] = aid
    lock["reference"]["build_report_sha256"] = r.sha(encoded(reports["A"]))
    lock_raw = encoded(lock)
    profile_data = json.loads((r.p.PROFILE_DIR / r.p.PROFILE_FILES[r.NEW]).read_bytes())
    profile_data["expectations"]["release_lock_sha256"] = r.sha(lock_raw)
    profile_raw = encoded(profile_data)
    old_data = json.loads((r.p.PROFILE_DIR / r.p.PROFILE_FILES[r.p.LEGACY_RELEASE]).read_bytes())
    old_data["expectations"] = {
        k: semantics[k]
        for k in ("catalog_count", "neighbor_count_histogram", "smoke_seed", "empty_seed")
    }
    old_raw = encoded(old_data)
    evidence = root / "evidence"
    evidence.mkdir(mode=0o700)
    records = {
        "profile": saved(evidence, "profile.json", profile_raw),
        "old_profile": saved(evidence, "old-profile.json", old_raw),
        "lock": saved(evidence, "release-lock.json", lock_raw),
    }
    original = (
        "ARB_DATA_DIR="
        + str(root)
        + "\nARB_BUNDLE_DIR="
        + str(old)
        + "\nARB_UID=1000\nARB_GID=1000\nARB_LOG=info\nARB_BOT_ENV=/tmp/bot.env\n"
        "ARB_POSTGRES_ENV=/tmp/postgres.env\n"
    ).encode()
    records["original_settings"] = saved(evidence, "original-settings.env", original)
    for key in ("tool", "compose", "builder_dockerfile", "bot_dockerfile", "snapshot_sql"):
        records[key] = saved(evidence, key + ".source", key.encode())
    probe = {
        "python_version": ref["python_version"],
        "numpy_version": ref["numpy_version"],
        "package_version": ref["package_version"],
        "code_files_sha256": ref["code_files_sha256"],
        "lock_sha256": r.sha(lock_raw),
        "registry_sha256": lock["inputs"]["embeddings"]["registry_sha256"],
    }
    records["probe"] = r.save_json(evidence / "producer-probe.json", probe)
    records["input_files"] = r.save_json(
        evidence / "input-files.json",
        r.files_under(
            [pathlib.Path(layout[key]) for key in ("base", "aliases", "release", "alias_raw")]
            + [glove_dir]
        ),
    )
    records["input_layout"] = r.save_json(evidence / "input-layout.json", layout)
    records["bundle_files"] = r.save_json(
        evidence / "bundle-files.json",
        {key: r.bundle_files(path) for key, path in (("old", old), ("A", a), ("B", b))},
    )
    pg = {
        "id": "pg-1",
        "image_id": "pg-image",
        "database_volume": "isolated-volume",
        "started_at": "2026-10-01T00:00:00Z",
        "running": True,
    }
    db = database()

    def shot(settings_hash, prepare=None, minute=1):
        return {
            "schema_version": 1,
            "kind": "full",
            "snapshot_sql_sha256": r.sha(b"snapshot_sql"),
            "source_commit": "f" * 40,
            "captured_at": (
                dt.datetime(2026, 10, 1, tzinfo=dt.timezone.utc) + dt.timedelta(minutes=minute)
            ).isoformat(),
            "deployment": {
                "project": "arb033-test",
                "compose_sha256": r.sha(b"compose"),
                "settings_sha256": settings_hash,
            },
            "runtime": {"bot": None, "prepare": prepare, "postgres": pg},
            "database": copy.deepcopy(db),
        }

    records["baseline"] = r.save_json(evidence / "snapshot-baseline.json", shot(r.sha(original)))
    builds = {}
    for label, path, identity, block in (("A", a, aid, 256), ("B", b, bid, 128)):
        report_raw = encoded(reports[label])
        builds[label] = {
            "identity": identity,
            "path": str(path),
            "count": 8,
            "report_sha256": r.sha(report_raw),
            "file_sha256": summaries[label].hashes,
            "producer": reports[label]["producer"],
            "block_size": block,
            "report": saved(evidence, "build-report-" + label + ".json", report_raw),
        }
    sequence = []
    for index, label in enumerate(r.LABELS):
        path, identity = (
            (old, oid) if label.startswith("old") else (b, bid) if label == "new-B" else (a, aid)
        )
        settings_raw = r.change_selection(original, path)
        settings_desc = saved(evidence, "settings-" + label + ".env", settings_raw)
        prepare = {
            "id": "prepare-" + str(index),
            "image_id": "bot-image",
            "status": "exited",
            "exit_code": 0,
            "user": "10001:10001",
            "read_only_root": True,
            "restart_count": 0,
            "artifact_mount": {"source": str(path), "rw": False},
            "bundle_id": identity,
            "bundle_count": 8,
            "started_at": (
                dt.datetime(2026, 10, 1, tzinfo=dt.timezone.utc)
                + dt.timedelta(minutes=index + 1, seconds=30)
            ).isoformat(),
        }
        snap_desc = r.save_json(
            evidence / ("snapshot-" + label + ".json"),
            shot(r.sha(settings_raw), prepare, index + 2),
        )
        sequence.append(
            {
                "label": label,
                "bundle_identity": identity,
                "settings": settings_desc,
                "snapshot": snap_desc,
            }
        )
    report = {
        "schema_version": 2,
        "kind": "arb-offline-release-acceptance",
        "scope": "full-inputs-offline-compatibility",
        "status": "PASS",
        "reason": "PASS",
        "profile": {
            "id": r.NEW,
            "sha256": r.sha(profile_raw),
            "release_id": r.NEW,
            "release_lock_sha256": r.sha(lock_raw),
        },
        "source": {
            "commit": "f" * 40,
            "tool_sha256": r.sha(b"tool"),
            "compose_sha256": r.sha(b"compose"),
            "dockerfile_sha256": {
                "builder": r.sha(b"builder_dockerfile"),
                "bot": r.sha(b"bot_dockerfile"),
            },
        },
        "original": {
            "settings_sha256": r.sha(original),
            "settings_uid": os.getuid(),
            "settings_gid": os.getgid(),
            "settings_mode": 0o600,
            "bundle_identity": oid,
            "bundle_path": str(old),
            "bundle_file_sha256": {
                name: r.file_sha(old / name)
                for name in ("catalog.json", "neighbors.json", "manifest.json")
            },
        },
        "inputs": {
            "base_identity": lock["inputs"]["base"]["identity"],
            "alias_identity": lock["inputs"]["aliases"]["identity"],
            "catalog_release_identity": lock["inputs"]["catalog_release"]["identity"],
            "embeddings_sha256": lock["inputs"]["embeddings"]["text_sha256"],
        },
        "builds": builds,
        "runtime": {
            "project": "arb033-test",
            "builder_image_id": "builder-image",
            "bot_image_id": "bot-image",
            "postgres_image_id": "pg-image",
            "database_volume": "isolated-volume",
            "baseline_snapshot_sha256": records["baseline"]["sha256"],
            "prepare_sequence": sequence,
        },
        "checks": {key: "PASS" for key in r.CHECKS},
        "live_acceptance": "PENDING",
        "production_history_preservation": "PENDING",
        "live_verifier": "REQUIRES_DATA06_V2",
        "saved_evidence": records,
    }
    r.save_json(evidence / "preparation-report.json", report)
    return evidence, report


class ReleaseEvidenceTests(unittest.TestCase):
    def test_linked_and_primary_checkout_share_private_roots_and_project_lock(self):
        primary = r.primary_root()
        self.assertEqual(r.primary_root(primary), primary)
        self.assertEqual(r.private_roots(), r.private_roots(primary))
        self.assertEqual(r.project_lock_dir(), r.project_lock_dir(primary))
        self.assertEqual(
            r.project_lock_dir() / "arb033-test.lock",
            primary / ".arb/acceptance-locks/arb033-test.lock",
        )
        self.assertIn(primary / ".tools/arb033-acceptance", r.private_roots())

    def test_probe_binds_installed_and_checkout_glove_registry(self):
        profile = r.p.load_profile(r.NEW)
        ref = profile.reference["producer"]
        probe = {key: ref[key] for key in ("python_version", "numpy_version", "package_version")}
        probe.update(
            code_files_sha256=ref["code_files_sha256"],
            lock_sha256=profile.lock_sha256,
            registry_sha256=profile.inputs["embeddings"]["registry_sha256"],
        )
        r.check_probe(probe, profile)
        self.assertIn("source_registry_glove_6b_300d_v1.json", r.probe_code())
        with self.assertRaisesRegex(r.ReleaseError, "BUILD_MISMATCH"):
            r.check_probe({**probe, "registry_sha256": "0" * 64}, profile)
        actual_file_sha = r.file_sha
        with mock.patch.object(
            r,
            "file_sha",
            side_effect=lambda path: (
                "0" * 64
                if pathlib.Path(path).name == "source_registry_glove_6b_300d_v1.json"
                else actual_file_sha(path)
            ),
        ):
            with self.assertRaisesRegex(r.ReleaseError, "BUILD_MISMATCH"):
                r.check_probe(probe, profile)

    def test_retained_input_mutation_stops_next_prepare(self):
        for changed in ("input", "old_bundle"):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as scratch:
                root = pathlib.Path(scratch)
                root.chmod(0o700)
                source = root / "source"
                source.mkdir()
                item = source / "input.bin"
                item.write_bytes(b"original")
                old = root / "old"
                old.mkdir()
                for name in ("catalog.json", "neighbors.json", "manifest.json"):
                    (old / name).write_bytes(b"old")
                retained = {
                    "input_paths": [source],
                    "input_files": r.files_under([source]),
                    "old_files": r.bundle_files(old),
                }
                settings = root / "settings.env"
                settings.write_bytes(b"original")
                state = (
                    r.sha(settings.read_bytes()),
                    settings.stat().st_uid,
                    settings.stat().st_gid,
                    stat.S_IMODE(settings.stat().st_mode),
                )
                t = mock.Mock(project="arb033-test", settings=settings)
                selected = {
                    "old": {"path": old},
                    "new-A": {"path": root / "A"},
                    "new-B": {"path": root / "B"},
                }
                phase_settings = {label: settings for label in r.LABELS}
                baseline = {"captured_at": "2026-10-01T00:01:00Z"}
                state_flag = {"begun": False}

                def first_phase(*args):
                    (item if changed == "input" else old / "manifest.json").write_bytes(b"changed")
                    return ({"label": "old"}, "prepare-1", "2026-10-01T00:02:00Z")

                with (
                    mock.patch.object(r, "clean"),
                    mock.patch.object(r.e, "container", return_value=None),
                    mock.patch.object(r, "snapshot_phase", side_effect=first_phase) as prepare,
                ):
                    with self.assertRaisesRegex(r.ReleaseError, "INPUT_CHANGED"):
                        r.run_prepare_sequence(
                            t,
                            root,
                            settings,
                            state,
                            "f" * 40,
                            baseline,
                            selected,
                            phase_settings,
                            "bot-image",
                            {},
                            retained,
                            old,
                            state_flag,
                        )
                    self.assertEqual(prepare.call_count, 1)
                    self.assertTrue(state_flag["begun"])

    def test_restore_is_bounded_and_refuses_changed_settings_or_running_bot(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = pathlib.Path(scratch)
            settings = root / "original.env"
            settings.write_bytes(b"original")
            old_settings = root / "old.env"
            old_settings.write_bytes(b"old")
            old_path = root / "old-bundle"
            old_path.mkdir()
            identity = "sha256:" + "a" * 64
            old = SimpleNamespace(identity=identity)
            pg0 = {
                "id": "pg-1",
                "image_id": "pg-image",
                "database_volume": "volume-1",
                "started_at": "2026-10-01T00:00:00Z",
            }
            state = (
                r.sha(settings.read_bytes()),
                settings.stat().st_uid,
                settings.stat().st_gid,
                stat.S_IMODE(settings.stat().st_mode),
            )
            t = mock.Mock(project="arb033-test", settings=settings)
            t.compose.return_value = ""
            restored = {
                "status": "exited",
                "exit_code": 0,
                "bundle_id": identity,
                "artifact_mount": {"source": str(old_path), "rw": False},
            }
            with (
                mock.patch.object(r, "quiescent", return_value=pg0),
                mock.patch.object(r, "check_retained"),
                mock.patch.object(r.e, "container", return_value=restored),
            ):
                self.assertEqual(
                    r.restore_old(t, settings, state, old_settings, old_path, old, pg0, {}), "PASS"
                )
            t.compose.reset_mock()
            settings.write_bytes(b"changed")
            self.assertEqual(
                r.restore_old(t, settings, state, old_settings, old_path, old, pg0, {}), "PENDING"
            )
            t.compose.assert_not_called()
            settings.write_bytes(b"original")
            with mock.patch.object(r, "quiescent", side_effect=r.ReleaseError("BOT_RUNNING")):
                self.assertEqual(
                    r.restore_old(t, settings, state, old_settings, old_path, old, pg0, {}),
                    "PENDING",
                )
            t.compose.assert_not_called()
            for field in ("id", "image_id", "database_volume", "started_at"):
                with (
                    self.subTest(field=field),
                    mock.patch.object(r, "quiescent", return_value={**pg0, field: "replaced"}),
                ):
                    self.assertEqual(
                        r.restore_old(t, settings, state, old_settings, old_path, old, pg0, {}),
                        "PENDING",
                    )
                    t.compose.assert_not_called()
            with (
                mock.patch.object(r, "quiescent", return_value=pg0),
                mock.patch.object(r, "check_retained", side_effect=r.ReleaseError("INPUT_CHANGED")),
            ):
                self.assertEqual(
                    r.restore_old(t, settings, state, old_settings, old_path, old, pg0, {}),
                    "PENDING",
                )
                t.compose.assert_not_called()
            with (
                mock.patch.object(r, "quiescent", return_value=pg0),
                mock.patch.object(r, "check_retained"),
            ):
                t.compose.side_effect = r.ReleaseError("TRANSPORT_TIMEOUT")
                self.assertEqual(
                    r.restore_old(t, settings, state, old_settings, old_path, old, pg0, {}), "FAIL"
                )

    def test_cli_failure_redacts_private_canary(self):
        with tempfile.TemporaryDirectory() as scratch:
            evidence = pathlib.Path(scratch)
            evidence.chmod(0o700)
            (evidence / "preparation-report.json").write_bytes(b"PRIVATE_CANARY")
            output = io.StringIO()
            with contextlib.redirect_stderr(output):
                self.assertEqual(r.main(["verify", "--evidence-dir", str(evidence)]), 1)
            self.assertEqual(output.getvalue(), "FAIL EVIDENCE_INVALID\n")

    def test_phase_requires_fresh_prepare_and_identical_database(self):
        with tempfile.TemporaryDirectory() as scratch:
            evidence = pathlib.Path(scratch)
            evidence.chmod(0o700)
            settings = evidence / "settings.env"
            settings.write_text("ARB_BUNDLE_DIR=/tmp/bundle\n")
            bundle = evidence / "bundle"
            bundle.mkdir()
            expected = {"path": bundle, "identity": "sha256:" + "a" * 64, "count": 8}
            pg = {
                "id": "pg-1",
                "image_id": "pg-image",
                "database_volume": "volume-1",
                "started_at": "2026-10-01T00:00:00Z",
                "running": True,
            }
            prepare = {
                "id": "prepare-2",
                "image_id": "bot-image",
                "status": "exited",
                "exit_code": 0,
                "user": "10001:10001",
                "read_only_root": True,
                "restart_count": 0,
                "artifact_mount": {"source": str(bundle), "rw": False},
                "bundle_id": expected["identity"],
                "bundle_count": 8,
                "started_at": "2026-10-01T00:01:30Z",
            }
            baseline = {"database": database(), "captured_at": "2026-10-01T00:01:00Z"}
            shot = {
                "runtime": {"bot": None, "prepare": prepare, "postgres": pg},
                "database": copy.deepcopy(baseline["database"]),
                "captured_at": "2026-10-01T00:02:00Z",
            }
            transport = mock.Mock(project="arb033-test", settings=settings)
            transport.compose.return_value = ""
            with (
                mock.patch.object(r, "quiescent", return_value=pg),
                mock.patch.object(r, "postgres_health", return_value=pg),
                mock.patch.object(r.e, "container", return_value=prepare),
                mock.patch.object(r.e, "snapshot", return_value=shot),
            ):
                phase, identifier, captured = r.snapshot_phase(
                    transport,
                    evidence,
                    "old",
                    settings,
                    baseline,
                    expected,
                    "prepare-1",
                    baseline["captured_at"],
                    "bot-image",
                    pg,
                )
                self.assertEqual(identifier, "prepare-2")
                self.assertEqual(captured, shot["captured_at"])
                self.assertEqual(phase["bundle_identity"], expected["identity"])
                changed = copy.deepcopy(shot)
                changed["database"]["migrations"][0]["applied_at"] = "2026-10-02T00:00:00Z"
                with (
                    mock.patch.object(r.e, "snapshot", return_value=changed),
                    self.assertRaisesRegex(r.ReleaseError, "DATABASE_HISTORY"),
                ):
                    r.snapshot_phase(
                        transport,
                        evidence,
                        "changed",
                        settings,
                        baseline,
                        expected,
                        "prepare-1",
                        baseline["captured_at"],
                        "bot-image",
                        pg,
                    )
                with self.assertRaisesRegex(r.ReleaseError, "PHASE_MISMATCH"):
                    r.snapshot_phase(
                        transport,
                        evidence,
                        "stale",
                        settings,
                        baseline,
                        expected,
                        "prepare-2",
                        baseline["captured_at"],
                        "bot-image",
                        pg,
                    )
                with (
                    mock.patch.object(
                        r, "postgres_health", return_value={**pg, "database_volume": "other"}
                    ),
                    self.assertRaisesRegex(r.ReleaseError, "DATABASE_HISTORY"),
                ):
                    r.snapshot_phase(
                        transport,
                        evidence,
                        "volume",
                        settings,
                        baseline,
                        expected,
                        "prepare-1",
                        baseline["captured_at"],
                        "bot-image",
                        pg,
                    )

    def test_recompute_and_historical_read_only(self):
        with tempfile.TemporaryDirectory() as scratch:
            evidence, report = make_fixture(pathlib.Path(scratch))
            with (
                mock.patch.object(r.p, "TRUSTED_NEW_PROFILE_SHA", report["profile"]["sha256"]),
                mock.patch.object(
                    r.p, "TRUSTED_LOCK_SHA", report["profile"]["release_lock_sha256"]
                ),
                mock.patch.object(
                    r.p,
                    "TRUSTED_LEGACY_PROFILE_SHA",
                    r.sha((evidence / "old-profile.json").read_bytes()),
                ),
            ):
                self.assertEqual(r.verify_record(evidence)["status"], "PASS")
                baseline = evidence / "snapshot-baseline.json"
                altered = json.loads(baseline.read_bytes())
                altered["database"]["migrations"][0]["applied_at"] = "2026-10-02T00:00:00Z"
                baseline.write_bytes(encoded(altered))
                with self.assertRaises(r.ReleaseError):
                    r.verify_record(evidence)

    def test_historical_phase_times_reject_equal_reversed_and_malformed(self):
        cases = (
            ("equal_capture", lambda shot: shot.update(captured_at="2026-10-01T00:01:00+00:00")),
            ("reversed_capture", lambda shot: shot.update(captured_at="2026-09-30T23:59:00+00:00")),
            (
                "early_prepare",
                lambda shot: shot["runtime"]["prepare"].update(
                    started_at="2026-10-01T00:00:59+00:00"
                ),
            ),
            (
                "late_prepare",
                lambda shot: shot["runtime"]["prepare"].update(
                    started_at="2026-10-01T00:03:00+00:00"
                ),
            ),
            ("naive_capture", lambda shot: shot.update(captured_at="2026-10-01T00:02:00")),
            (
                "malformed_prepare",
                lambda shot: shot["runtime"]["prepare"].update(started_at="not-a-time"),
            ),
        )
        for name, change in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as scratch:
                evidence, report = make_fixture(pathlib.Path(scratch))
                trusted = {
                    "TRUSTED_NEW_PROFILE_SHA": report["profile"]["sha256"],
                    "TRUSTED_LOCK_SHA": report["profile"]["release_lock_sha256"],
                    "TRUSTED_LEGACY_PROFILE_SHA": r.sha(
                        (evidence / "old-profile.json").read_bytes()
                    ),
                }
                first = evidence / "snapshot-old.json"
                shot = json.loads(first.read_bytes())
                change(shot)
                first.write_bytes(encoded(shot))
                report["runtime"]["prepare_sequence"][0]["snapshot"]["sha256"] = r.sha(
                    first.read_bytes()
                )
                (evidence / "preparation-report.json").write_bytes(encoded(report))
                with (
                    mock.patch.multiple(r.p, **trusted),
                    self.assertRaises((r.ReleaseError, r.e.EvidenceError)),
                ):
                    r.verify_record(evidence)

    def test_forged_pass_and_settings_phase_binding(self):
        with tempfile.TemporaryDirectory() as scratch:
            evidence, report = make_fixture(pathlib.Path(scratch))
            trusted = {
                "TRUSTED_NEW_PROFILE_SHA": report["profile"]["sha256"],
                "TRUSTED_LOCK_SHA": report["profile"]["release_lock_sha256"],
                "TRUSTED_LEGACY_PROFILE_SHA": r.sha((evidence / "old-profile.json").read_bytes()),
            }
            with mock.patch.multiple(r.p, **trusted):
                altered = copy.deepcopy(report)
                altered["checks"]["database_history"] = "FAIL"
                (evidence / "preparation-report.json").write_bytes(encoded(altered))
                with self.assertRaises(r.ReleaseError):
                    r.verify_record(evidence)
                (evidence / "preparation-report.json").write_bytes(encoded(report))
                phase = evidence / "settings-new-B.env"
                content = phase.read_bytes().replace(b"ARB_LOG=info", b"ARB_LOG=debug")
                phase.write_bytes(content)
                altered = copy.deepcopy(report)
                altered["runtime"]["prepare_sequence"][2]["settings"]["sha256"] = r.sha(content)
                snap_path = evidence / "snapshot-new-B.json"
                shot = json.loads(snap_path.read_bytes())
                shot["deployment"]["settings_sha256"] = r.sha(content)
                snap_path.write_bytes(encoded(shot))
                altered["runtime"]["prepare_sequence"][2]["snapshot"]["sha256"] = r.sha(
                    snap_path.read_bytes()
                )
                (evidence / "preparation-report.json").write_bytes(encoded(altered))
                with self.assertRaises(r.ReleaseError):
                    r.verify_record(evidence)

    def test_bounded_transport_and_guarded_dispatch(self):
        with mock.patch.object(
            r.subprocess, "run", side_effect=r.subprocess.TimeoutExpired("docker", 1)
        ):
            with self.assertRaisesRegex(r.ReleaseError, "TRANSPORT_TIMEOUT"):
                r.command(["docker", "version"], timeout=1)
        result = r.subprocess.run(
            [
                "bash",
                str(r.ROOT / "deploy/tests/full-catalog-acceptance.sh"),
                "--release",
                "current",
                "--project",
                "arb033-test",
                "--settings",
                "/tmp/x",
                "--data-dir",
                "/tmp/y",
                "--evidence-dir",
                "/tmp/z",
            ],
            cwd=r.ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertNotIn("docker", result.stderr.lower())


if __name__ == "__main__":
    unittest.main()
