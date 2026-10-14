#!/usr/bin/env python3
"""Run isolated, non-Telegram acceptance against FakeTelegram and synthetic Compose."""

import argparse
import datetime as dt
import json
import os
import re
import secrets
import signal
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "target" / "level1"
PG_TESTS = {
    "db::tests::migration_preserves_legacy_and_rejects_unknown_versions",
    "db::tests::records_replays_and_provenance",
    "db::tests::boundaries_concurrency_and_reconnect",
    "postgres_public_router_records_profile_query_and_stable_resolution",
    "postgres_records_real_coordinates_and_provenance",
    "arb016_postgres_public_router_persists_scores_with_provenance",
    "postgres_public_router_preserves_history_and_current_rating",
    "arb017_postgres_exact_row_and_confirmation_retry",
    "arb017_postgres_interrupted_backend_reconnects_on_explicit_retry",
}
LEVEL1_TESTS = {
    "level1_recommendations_alias",
    "level1_empty",
    "level1_recommendation_scores",
    "level1_anime_scores",
    "level1_navigation_all_stages",
    "level1_description_pages",
    "level1_feedback_unicode",
    "level1_feedback_failure_navigation_retry",
    "level1_context_reset_bundle_change",
    "level1_transport_failure",
}
COMPOSE_MARKERS = {
    "LEVEL1_COMPOSE_BUILD_PASS",
    "LEVEL1_COMPOSE_LEGACY_A_PASS",
    "LEVEL1_COMPOSE_CURRENT_B_PASS",
    "LEVEL1_COMPOSE_ROLLBACK_A_PASS",
    "LEVEL1_COMPOSE_PERSISTENCE_PASS",
    "LEVEL1_COMPOSE_GUARDS_PASS",
    "LEVEL1_COMPOSE_CLEANUP_PASS",
}
CASE_RE = re.compile(r"^test (\S+) \.\.\. (ok|FAILED|ignored)$", re.MULTILINE)
PORT_RE = re.compile(r"^127\.0\.0\.1:(\d+)$")
EXCLUDED = ("ARB_", "COMPOSE_", "POSTGRES_")
SECRET_NAMES = {"TELOXIDE_TOKEN", "DATABASE_URL", "ARTIFACTS_DIR", "BOT_TOKEN"}


def require_cases(text: str, required: set[str]) -> None:
    seen = {}
    for name, status in CASE_RE.findall(text):
        if name in required:
            seen.setdefault(name, []).append(status)
    bad = {name: seen.get(name) for name in required if seen.get(name) != ["ok"]}
    if bad:
        raise RuntimeError(f"Required tests did not pass exactly once: {bad}")


def require_markers(text: str) -> None:
    lines = text.splitlines()
    bad = {marker: lines.count(marker) for marker in COMPOSE_MARKERS if lines.count(marker) != 1}
    if bad or not lines or lines[-1] != "LEVEL1_COMPOSE_CLEANUP_PASS":
        raise RuntimeError(f"Synthetic Compose markers missing or repeated: {bad}")


class Interrupted(RuntimeError):
    pass


class Runner:
    def __init__(self, scope: str):
        self.scope = scope
        self.run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + secrets.token_hex(5)
        self.directory = OUTPUT / self.run_id
        self.directory.mkdir(parents=True, mode=0o700)
        self.directory.chmod(0o700)
        self.env = {key: value for key, value in os.environ.items()
                    if key not in SECRET_NAMES and not key.startswith(EXCLUDED)}
        self.env["CARGO_TERM_COLOR"] = "never"
        self.project = "arb035-l1-" + secrets.token_hex(5)
        self.empty_env = self.directory / "empty.env"
        self.empty_env.touch(mode=0o600)
        self.compose_file = ROOT / "deploy/tests/compose.level1.yaml"
        self.active = None
        self.cases = {}
        self.cleanup_ok = True
        self.started_pg = False

    def command(self, name, argv, timeout, extra_env=None, check=True):
        env = self.env.copy()
        if extra_env:
            env.update(extra_env)
        log = self.directory / f"{name}.log"
        with log.open("w", encoding="utf-8") as output:
            self.active = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=output,
                                           stderr=subprocess.STDOUT, start_new_session=True,
                                           text=True)
            try:
                code = self.active.wait(timeout=timeout)
            except BaseException:
                try:
                    os.killpg(self.active.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    # The shell phase traps TERM and needs time for scoped Docker cleanup.
                    self.active.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(self.active.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    self.active.wait(timeout=5)
                raise
            finally:
                self.active = None
        content = log.read_text(encoding="utf-8")
        if check and code != 0:
            raise RuntimeError(f"{name} exited {code}; see {log}")
        return content

    def compose(self, name, args, timeout=120):
        cmd = ["docker", "compose", "--project-directory", str(ROOT),
               "--env-file", str(self.empty_env), "-f", str(self.compose_file),
               "-p", self.project, *args]
        return self.command(name, cmd, timeout)

    def dialogue(self):
        self.command("rust-ordinary", ["cargo", "+1.95.0", "test", "--locked"], 1200)
        self.started_pg = True
        self.compose("pg-up", ["up", "--wait", "-d", "postgres"], 180)
        port_text = self.compose("pg-port", ["port", "postgres", "5432"], 30).strip()
        match = PORT_RE.fullmatch(port_text)
        if not match:
            raise RuntimeError("Disposable database did not publish an isolated loopback port")
        url = f"postgresql://arb_ci:arb_ci@127.0.0.1:{match.group(1)}/arb_ci_test?sslmode=disable"
        result = self.command("rust-postgres", ["cargo", "+1.95.0", "test", "--locked", "--", "--ignored", "--test-threads=1"],
                              1200, {"ARB_TEST_DATABASE_URL": url})
        require_cases(result, PG_TESTS | LEVEL1_TESTS)
        for name in sorted(PG_TESTS | LEVEL1_TESTS):
            self.cases[name] = {"status": "PASS", "source": "Rust router or database integration test on disposable PostgreSQL"}
        self.cleanup()
        if not self.cleanup_ok:
            raise RuntimeError("disposable dialogue PostgreSQL cleanup failed")
        self.cases["dialogue"] = {"status": "PASS", "source": "FakeTelegram and disposable PostgreSQL",
                                   "required_tests": len(PG_TESTS | LEVEL1_TESTS)}

    def synthetic_compose(self):
        # The shell harness owns its separate project and removes it before printing the final marker.
        result = self.command("compose", ["bash", "deploy/tests/compose-acceptance.sh"], 1800,
                              {"ARB_LEVEL1_OUTPUT": str(self.directory / "compose")})
        require_markers(result)
        for marker in sorted(COMPOSE_MARKERS):
            self.cases[marker] = {"status": "PASS", "source": "synthetic Compose fixture and database snapshot"}
        self.cases["compose"] = {"status": "PASS", "source": "synthetic A-B-A Compose and full database snapshots"}

    def cleanup(self):
        if not self.started_pg:
            return
        try:
            self.compose("pg-down", ["down", "--volumes", "--remove-orphans", "--timeout", "10"], 120)
            listed = self.compose("pg-verify-cleanup", ["ps", "-a", "-q"], 30).strip()
            if listed:
                raise RuntimeError("owned PostgreSQL project still has containers")
            self.started_pg = False
            self.cases["dialogue-postgres-cleanup"] = {"status": "PASS", "source": "scoped Docker Compose down and empty ps"}
        except Exception as error:
            self.cleanup_ok = False
            self.cases["cleanup"] = {"status": "FAIL", "reason": str(error)}

    def report(self, status):
        source = self.command("source-head", ["git", "rev-parse", "HEAD"], 10).strip()
        dirty = bool(self.command("source-status", ["git", "status", "--porcelain"], 10).strip())
        report = {"kind": "arb-level1", "schema_version": 1, "scope": self.scope,
                  "status": status, "source_sha": source, "source_dirty": dirty,
                  "cases": self.cases,
                  "limits": {"real_telegram": "NOT_RUN", "human_ui": "NOT_RUN",
                             "full_pinned_catalog": "NOT_RUN", "poller_process_restart": "NOT_RUN"}}
        (self.directory / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"{status}: {self.directory / 'report.json'}")


def main():
    def interrupted(signum, _frame):
        raise Interrupted(f"interrupted by signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=["all", "dialogue", "compose"], default="all")
    args = parser.parse_args()
    runner = Runner(args.scope)
    failure = None
    try:
        if args.scope in ("all", "dialogue"):
            runner.dialogue()
        if args.scope in ("all", "compose"):
            runner.synthetic_compose()
    except BaseException as error:
        failure = error
    finally:
        runner.cleanup()
    status = "PASS" if failure is None and runner.cleanup_ok else "FAIL"
    if failure:
        runner.cases["runner"] = {"status": "FAIL", "reason": str(failure)}
    runner.report(status)
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
