"""Offline behavior tests for the master merge policy readback."""

from __future__ import annotations

import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, cast
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import github_merge_policy as merge

SHA = "a" * 40
NOW = datetime(2026, 10, 15, 12, tzinfo=timezone.utc)


def _object(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise AssertionError("expected fixture object with string keys")
    return cast("dict[str, object]", value)


def _items(value: object) -> list[object]:
    if not isinstance(value, list):
        raise AssertionError("expected fixture list")
    return cast("list[object]", value)


def _policy_rule(policy: dict[str, object], index: int) -> dict[str, object]:
    return _object(_items(policy["rules"])[index])


def _ruleset(snapshot: dict[str, object]) -> dict[str, object]:
    return _object(_items(snapshot["rulesets"])[0])


def _rules(snapshot: dict[str, object], *, effective: bool = False) -> list[object]:
    return _items(snapshot["effective_rules"] if effective else _ruleset(snapshot)["rules"])


def _rule(snapshot: dict[str, object], index: int, *, effective: bool = False) -> dict[str, object]:
    return _object(_rules(snapshot, effective=effective)[index])


def _params(rule: dict[str, object]) -> dict[str, object]:
    return _object(rule["parameters"])


def _checks(snapshot: dict[str, object]) -> list[object]:
    return _items(snapshot["check_runs"])


def _check(snapshot: dict[str, object], index: int = 0) -> dict[str, object]:
    return _object(_checks(snapshot)[index])


def _repo(snapshot: dict[str, object]) -> dict[str, object]:
    return _object(snapshot["repository"])


def _branch_commit(snapshot: dict[str, object]) -> dict[str, object]:
    return _object(_object(snapshot["final_branch"])["commit"])


def fixture(*, applied: bool = False) -> dict[str, object]:
    policy = merge.load_policy()
    ruleset = dict(deepcopy(policy), id=77, source_type="Repository", source=merge.REPOSITORY)
    effective = [
        dict(
            deepcopy(_object(rule)),
            ruleset_id=77,
            ruleset_source_type="Repository",
            ruleset_source=merge.REPOSITORY,
        )
        for rule in _items(policy["rules"])
    ]
    checks = [
        dict(
            name=name,
            head_sha=SHA,
            status="completed",
            conclusion="success",
            app={"id": merge.APP_ID, "slug": "github-actions"},
            completed_at="2026-10-15T11:00:00Z",
        )
        for name in sorted(merge.CHECKS)
    ]
    return {
        "expected_master_sha": SHA,
        "repository": {
            "id": merge.REPOSITORY_ID,
            "full_name": merge.REPOSITORY,
            "default_branch": "master",
            "private": False,
            "visibility": "public",
            "archived": False,
            "disabled": False,
            "allow_squash_merge": True,
            "permissions": {"admin": True},
        },
        "branch": {"name": "master", "commit": {"sha": SHA}},
        "final_branch": {"name": "master", "commit": {"sha": SHA}},
        "protection": None,
        "ruleset_summaries": [{"id": 77}] if applied else [],
        "rulesets": [ruleset] if applied else [],
        "effective_rules": effective if applied else [],
        "check_runs": checks,
    }


def evaluate(snapshot: dict[str, object], mode: str = "plan") -> dict[str, str | int | None]:
    return merge.validate_snapshot(merge.load_policy(), snapshot, mode, now=NOW)


class DeclarationTests(unittest.TestCase):
    def test_offline_policy_is_exact_and_safely_parseable(self) -> None:
        policy = merge.load_policy()
        self.assertEqual(
            [_object(r)["type"] for r in _items(policy["rules"])],
            [
                "deletion",
                "non_fast_forward",
                "required_linear_history",
                "pull_request",
                "required_status_checks",
            ],
        )
        with tempfile.TemporaryDirectory() as path:
            bad = Path(path) / "policy.json"
            bad.write_text('{"name":"x","name":"y"}')
            with self.assertRaises(merge.PolicyError):
                merge.load_policy(bad)
            bad.write_text('{"name":NaN}')
            with self.assertRaises(merge.PolicyError):
                merge.load_policy(bad)

    def test_declaration_rejects_looser_values_and_unknown_fields(self) -> None:
        changes: tuple[Callable[[dict[str, object]], object], ...] = (
            lambda p: p.update(bypass_actors=[{"actor_id": 1}]),
            lambda p: _params(_policy_rule(p, 3)).update(required_approving_review_count=1),
            lambda p: _params(_policy_rule(p, 3)).update(allowed_merge_methods=["squash", "merge"]),
            lambda p: _params(_policy_rule(p, 3)).update(
                require_extra_approval_for_unattributed_changes=True
            ),
            lambda p: _params(_policy_rule(p, 4)).update(extra=True),
        )
        for change in changes:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as path:
                policy = merge.load_policy()
                change(policy)
                file = Path(path) / "policy.json"
                file.write_text(json.dumps(policy))
                with self.assertRaises(merge.PolicyError):
                    merge.load_policy(file)


class SnapshotTests(unittest.TestCase):
    def test_clean_plan_and_unapplied_check(self) -> None:
        self.assertEqual(evaluate(fixture()), {"action": "create", "ruleset_id": None})
        with self.assertRaisesRegex(merge.PolicyMismatch, "not applied"):
            evaluate(fixture(), "check")

    def test_applied_noop_check_and_known_server_defaults(self) -> None:
        snap = fixture(applied=True)
        _ruleset(snap).update(
            node_id="RRS_fixture",
            _links={"self": {"href": "fixture"}},
            created_at="2026-10-15T00:00:00Z",
            updated_at="2026-10-15T01:00:00Z",
        )
        _rules(snap).reverse()
        _rules(snap, effective=True).reverse()
        for rules in (_rules(snap), _rules(snap, effective=True)):
            pr = _object(next(x for x in rules if _object(x)["type"] == "pull_request"))
            _params(pr)["required_reviewers"] = []
            _params(pr)["dismissal_restriction"] = {"enabled": False, "allowed_actors": []}
            pr["id"] = 99
            checks = _object(
                next(x for x in rules if _object(x)["type"] == "required_status_checks")
            )
            _items(_params(checks)["required_status_checks"]).reverse()
        self.assertEqual(evaluate(snap), {"action": "noop", "ruleset_id": 77})
        self.assertEqual(evaluate(snap, "check")["ruleset_id"], 77)

    def test_unknown_and_missing_ruleset_detail_fields_fail_closed(self) -> None:
        snap = fixture(applied=True)
        _ruleset(snap)["mystery_enforcement_setting"] = True
        with self.assertRaisesRegex(merge.PolicyMismatch, "detail fields"):
            evaluate(snap, "check")
        snap = fixture(applied=True)
        del _ruleset(snap)["conditions"]
        with self.assertRaisesRegex(merge.PolicyMismatch, "detail fields"):
            evaluate(snap, "check")

    def test_current_user_bypass_metadata_is_only_absent_or_never(self) -> None:
        self.assertEqual(evaluate(fixture(applied=True), "check")["action"], "noop")
        snap = fixture(applied=True)
        _ruleset(snap)["current_user_can_bypass"] = "never"
        self.assertEqual(evaluate(snap, "check")["action"], "noop")
        invalid_values: tuple[object, ...] = (
            "always",
            "pull_requests_only",
            "exempt",
            "sometimes",
            "",
            None,
            True,
            0,
            [],
        )
        for value in invalid_values:
            with self.subTest(value=value):
                snap = fixture(applied=True)
                _ruleset(snap)["current_user_can_bypass"] = value
                with self.assertRaisesRegex(merge.PolicyMismatch, "current user bypass"):
                    evaluate(snap, "check")
        snap = fixture(applied=True)
        _ruleset(snap).update(current_user_can_bypass="never", mystery_enforcement_setting=True)
        with self.assertRaisesRegex(merge.PolicyMismatch, "detail fields"):
            evaluate(snap, "check")

    def test_preview_extra_approval_response_only_allows_absent_or_true(self) -> None:
        field = "require_extra_approval_for_unattributed_changes"
        self.assertEqual(evaluate(fixture(applied=True), "check")["action"], "noop")
        for location in ("detail", "effective"):
            with self.subTest(location=location, value=True):
                snap = fixture(applied=True)
                rules = _rules(snap, effective=location != "detail")
                pr = _object(
                    next(rule for rule in rules if _object(rule)["type"] == "pull_request")
                )
                _params(pr)[field] = True
                self.assertEqual(evaluate(snap, "check")["action"], "noop")
            invalid_values: tuple[object, ...] = (False, 1, 0, None, "true", "false", [], {})
            for value in invalid_values:
                with self.subTest(location=location, value=value):
                    snap = fixture(applied=True)
                    rules = _rules(snap, effective=location != "detail")
                    pr = _object(
                        next(rule for rule in rules if _object(rule)["type"] == "pull_request")
                    )
                    _params(pr)[field] = value
                    with self.assertRaisesRegex(merge.PolicyMismatch, "extra approval"):
                        evaluate(snap, "check")
            snap = fixture(applied=True)
            rules = _rules(snap, effective=location != "detail")
            pr = _object(next(rule for rule in rules if _object(rule)["type"] == "pull_request"))
            _params(pr).update({field: True, "unknown_neighbor": True})
            with self.subTest(location=location, value="unknown neighbor"):
                with self.assertRaises(merge.PolicyMismatch):
                    evaluate(snap, "check")

    def test_repository_identity_permissions_and_master_race(self) -> None:
        changes: list[Callable[[dict[str, object]], object]] = [
            lambda s: _repo(s).update(id=4),
            lambda s: _repo(s).update(full_name="other/repo"),
            lambda s: _repo(s).update(default_branch="main"),
            lambda s: _repo(s).update(private=True),
            lambda s: _object(_repo(s)["permissions"]).update(admin=False),
            lambda s: _repo(s).update(allow_squash_merge=False),
            lambda s: _branch_commit(s).update(sha="b" * 40),
        ]
        for change in changes:
            with self.subTest(change=change):
                snap = fixture()
                change(snap)
                with self.assertRaises(merge.PolicyMismatch):
                    evaluate(snap)

    def test_foreign_inactive_duplicate_and_classic_protection(self) -> None:
        changes: list[Callable[[dict[str, object]], object]] = [
            lambda s: s.update(protection={"enabled": True}),
            lambda s: _items(s["ruleset_summaries"]).append({"id": 9}),
            lambda s: _rules(s, effective=True).append({"type": "deletion"}),
        ]
        for change in changes:
            with self.subTest(change=change):
                snap = fixture()
                change(snap)
                with self.assertRaises((merge.PolicyMismatch, merge.PolicyError)):
                    evaluate(snap)
        for key, value in (
            ("enforcement", "evaluate"),
            ("target", "tag"),
            ("conditions", {"ref_name": {"include": ["~ALL"], "exclude": []}}),
            ("bypass_actors", [{"actor_id": 1}]),
        ):
            snap = fixture(applied=True)
            _ruleset(snap)[key] = value
            with self.subTest(key=key), self.assertRaises(merge.PolicyMismatch):
                evaluate(snap)

    def test_effective_rules_and_policy_drift(self) -> None:
        changes: list[Callable[[dict[str, object]], object]] = [
            lambda s: _rules(s, effective=True).pop(),
            lambda s: _rule(s, 0, effective=True).update(ruleset_id=8),
            lambda s: _rule(s, 0, effective=True).update(ruleset_source="other/repo"),
            lambda s: _rule(s, 0, effective=True).update(parameters={"x": 1}),
            lambda s: _params(_rule(s, 3)).update(unknown=True),
            lambda s: _params(_rule(s, 3)).update(allowed_merge_methods=["squash", "merge"]),
            lambda s: _params(_rule(s, 4)).update(strict_required_status_checks_policy=False),
            lambda s: _object(_items(_params(_rule(s, 4))["required_status_checks"])[0]).update(
                integration_id=9
            ),
        ]
        for change in changes:
            with self.subTest(change=change):
                snap = fixture(applied=True)
                change(snap)
                with self.assertRaises(merge.PolicyMismatch):
                    evaluate(snap)

    def test_check_runs_require_success_unique_recent_correct_app_and_sha(self) -> None:
        changes: list[Callable[[dict[str, object]], object]] = [
            lambda s: _checks(s).pop(),
            lambda s: _checks(s).append(deepcopy(_check(s))),
            lambda s: _check(s).update(status="queued"),
            lambda s: _check(s).update(conclusion="neutral"),
            lambda s: _check(s).update(conclusion="skipped"),
            lambda s: _check(s).update(conclusion="failure"),
            lambda s: _check(s).update(head_sha="b" * 40),
            lambda s: _object(_check(s)["app"]).update(id=123),
            lambda s: _check(s).update(completed_at="2026-10-01T00:00:00Z"),
        ]
        for change in changes:
            with self.subTest(change=change):
                snap = fixture()
                change(snap)
                with self.assertRaises(merge.PolicyMismatch):
                    evaluate(snap)
        snap = fixture()
        _checks(snap).append(dict(name="extra-check"))
        self.assertEqual(evaluate(snap)["action"], "create")
        snap = fixture(applied=True)
        _check(snap)["completed_at"] = "2025-01-01T00:00:00Z"
        self.assertEqual(evaluate(snap)["action"], "noop")


class ApiTests(unittest.TestCase):
    def test_plan_cli_exposes_create_and_noop_actions(self) -> None:
        for applied, action in ((False, "create"), (True, "noop")):
            with self.subTest(action=action):
                snap = fixture(applied=applied)
                completed = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
                for check in _checks(snap):
                    _object(check)["completed_at"] = completed

                class FakeApi:
                    def get_json(self, endpoint: str) -> object:
                        if endpoint == merge.API:
                            return snap["repository"]
                        if endpoint == f"{merge.API}/branches/master":
                            return snap["branch"]
                        if endpoint == f"{merge.API}/branches/master/protection":
                            raise merge.ApiStatus(404, "Branch not protected")
                        if endpoint == f"{merge.API}/rulesets/77" and applied:
                            return _ruleset(snap)
                        raise AssertionError(f"unexpected GET {endpoint}")

                    def get_pages(
                        self, endpoint: str, *, item_key: str | None = None
                    ) -> list[object]:
                        if endpoint.startswith(f"{merge.API}/rulesets?"):
                            return _items(snap["ruleset_summaries"])
                        if endpoint == f"{merge.API}/rules/branches/master":
                            return _rules(snap, effective=True)
                        if endpoint.startswith(f"{merge.API}/commits/{SHA}/check-runs?"):
                            return _checks(snap)
                        raise AssertionError(f"unexpected GET pages {endpoint}")

                output = io.StringIO()
                with patch.object(merge, "GhApi", return_value=FakeApi()), redirect_stdout(output):
                    self.assertEqual(merge.main(["plan", "--expected-master-sha", SHA]), 0)
                result = json.loads(output.getvalue())
                self.assertEqual(result["schema_version"], 1)
                self.assertEqual(result["mode"], "plan")
                self.assertEqual(result["repository"], merge.REPOSITORY)
                self.assertEqual(result["master_sha"], SHA)
                self.assertEqual(result["action"], action)
                self.assertEqual(result["result"], action)
                self.assertEqual("ruleset_payload" in result, not applied)
                self.assertEqual(result.get("ruleset_id"), 77 if applied else None)

    def test_only_get_and_fixed_host_headers(self) -> None:
        body = '{"id":631205863}'

        class Proc:
            returncode = 0
            stdout = "HTTP/2.0 200 OK\r\ncontent-type: application/json\r\n\r\n" + body

        with patch.object(merge.subprocess, "run", return_value=Proc()) as run:
            self.assertEqual(merge.GhApi().get_json(merge.API), {"id": merge.REPOSITORY_ID})
        argv = run.call_args.args[0]
        self.assertIn("GET", argv)
        self.assertIn("github.com", argv)
        self.assertIn(merge.API_VERSION, " ".join(argv))
        self.assertFalse(any(x in argv for x in ("POST", "PATCH", "PUT", "DELETE")))
        with self.assertRaises(merge.PolicyError):
            merge.GhApi().get_json("repos/other/repo")

    def test_only_specific_protection_404_is_absent(self) -> None:
        class FakeApi:
            def __init__(self, message: str) -> None:
                self.message = message

            def get_json(self, endpoint: str) -> object:
                if endpoint.endswith("/protection"):
                    raise merge.ApiStatus(404, self.message)
                if endpoint.endswith("/branches/master"):
                    return {"name": "master", "commit": {"sha": SHA}}
                return {}

            def get_pages(self, endpoint: str, *, item_key: str | None = None) -> list[object]:
                return []

        self.assertIsNone(
            merge.collect_snapshot(FakeApi("Branch not protected"), SHA)["protection"]
        )
        with self.assertRaises(merge.ApiStatus):
            merge.collect_snapshot(FakeApi("Not Found"), SHA)

    def test_pagination_requires_total_count_and_bounded_completion(self) -> None:
        api = merge.GhApi()
        with patch.object(api, "get_json", return_value={"total_count": 2, "check_runs": [{}]}):
            with self.assertRaisesRegex(merge.PolicyError, "incomplete"):
                api.get_pages(f"{merge.API}/commits/{SHA}/check-runs", item_key="check_runs")
        with patch.object(api, "get_json", return_value=[{}] * 100):
            with self.assertRaisesRegex(merge.PolicyError, "pagination limit"):
                api.get_pages(f"{merge.API}/rulesets")

    def test_transport_timeout_malformed_json_and_wrong_404_fail_closed(self) -> None:
        api = merge.GhApi()
        with patch.object(
            merge.subprocess, "run", side_effect=subprocess.TimeoutExpired(["gh"], 25)
        ):
            with self.assertRaisesRegex(merge.PolicyError, "unavailable"):
                api.get_json(merge.API)

        class Proc:
            returncode = 0
            stdout = "HTTP/2.0 200 OK\r\ncontent-type: application/json\r\n\r\n{invalid"

        with patch.object(merge.subprocess, "run", return_value=Proc()):
            with self.assertRaisesRegex(merge.PolicyError, "malformed JSON"):
                api.get_json(merge.API)
        Proc.returncode = 1
        Proc.stdout = 'HTTP/2.0 404 Not Found\r\n\r\n{"message":"Not Found"}'
        with patch.object(merge.subprocess, "run", return_value=Proc()):
            with self.assertRaises(merge.ApiStatus) as caught:
                api.get_json(merge.API)
        self.assertEqual(caught.exception.status, 404)
        self.assertEqual(str(caught.exception), "GitHub API returned HTTP 404")


if __name__ == "__main__":
    unittest.main()
