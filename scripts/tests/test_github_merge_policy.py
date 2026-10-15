"""Offline behavior tests for the master merge policy readback."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import github_merge_policy as merge

SHA = "a" * 40
NOW = datetime(2026, 10, 15, 12, tzinfo=timezone.utc)


def fixture(*, applied: bool = False) -> dict:
    policy = merge.load_policy()
    ruleset = dict(deepcopy(policy), id=77, source_type="Repository", source=merge.REPOSITORY)
    effective = [dict(deepcopy(rule), ruleset_id=77, ruleset_source_type="Repository",
                      ruleset_source=merge.REPOSITORY) for rule in policy["rules"]]
    checks = [dict(name=name, head_sha=SHA, status="completed", conclusion="success",
                   app={"id": merge.APP_ID, "slug": "github-actions"}, completed_at="2026-10-15T11:00:00Z")
              for name in sorted(merge.CHECKS)]
    return {"expected_master_sha": SHA,
            "repository": {"id": merge.REPOSITORY_ID, "full_name": merge.REPOSITORY,
                           "default_branch": "master", "private": False, "visibility": "public",
                           "archived": False, "disabled": False, "allow_squash_merge": True,
                           "permissions": {"admin": True}},
            "branch": {"name": "master", "commit": {"sha": SHA}},
            "final_branch": {"name": "master", "commit": {"sha": SHA}},
            "protection": None, "ruleset_summaries": [{"id": 77}] if applied else [],
            "rulesets": [ruleset] if applied else [], "effective_rules": effective if applied else [],
            "check_runs": checks}


def evaluate(snapshot: dict, mode: str = "plan") -> dict:
    return merge.validate_snapshot(merge.load_policy(), snapshot, mode, now=NOW)


class DeclarationTests(unittest.TestCase):
    def test_offline_policy_is_exact_and_safely_parseable(self) -> None:
        policy = merge.load_policy()
        self.assertEqual([r["type"] for r in policy["rules"]],
                         ["deletion", "non_fast_forward", "required_linear_history",
                          "pull_request", "required_status_checks"])
        with tempfile.TemporaryDirectory() as path:
            bad = Path(path) / "policy.json"
            bad.write_text('{"name":"x","name":"y"}')
            with self.assertRaises(merge.PolicyError):
                merge.load_policy(bad)
            bad.write_text('{"name":NaN}')
            with self.assertRaises(merge.PolicyError):
                merge.load_policy(bad)

    def test_declaration_rejects_looser_values_and_unknown_fields(self) -> None:
        for change in (lambda p: p.update(bypass_actors=[{"actor_id": 1}]),
                       lambda p: p["rules"][3]["parameters"].update(required_approving_review_count=1),
                       lambda p: p["rules"][3]["parameters"].update(allowed_merge_methods=["squash", "merge"]),
                       lambda p: p["rules"][4]["parameters"].update(extra=True)):
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
        snap["rulesets"][0].update(node_id="RRS_fixture", _links={"self": {"href": "fixture"}},
                                   created_at="2026-10-15T00:00:00Z", updated_at="2026-10-15T01:00:00Z")
        snap["rulesets"][0]["rules"].reverse()
        snap["effective_rules"].reverse()
        for rules in (snap["rulesets"][0]["rules"], snap["effective_rules"]):
            pr = next(x for x in rules if x["type"] == "pull_request")
            pr["parameters"]["required_reviewers"] = []
            pr["parameters"]["dismissal_restriction"] = {"enabled": False, "allowed_actors": []}
            pr["id"] = 99
            checks = next(x for x in rules if x["type"] == "required_status_checks")
            checks["parameters"]["required_status_checks"].reverse()
        self.assertEqual(evaluate(snap), {"action": "noop", "ruleset_id": 77})
        self.assertEqual(evaluate(snap, "check")["ruleset_id"], 77)

    def test_unknown_and_missing_ruleset_detail_fields_fail_closed(self) -> None:
        snap = fixture(applied=True)
        snap["rulesets"][0]["mystery_enforcement_setting"] = True
        with self.assertRaisesRegex(merge.PolicyMismatch, "detail fields"):
            evaluate(snap, "check")
        snap = fixture(applied=True)
        del snap["rulesets"][0]["conditions"]
        with self.assertRaisesRegex(merge.PolicyMismatch, "detail fields"):
            evaluate(snap, "check")

    def test_repository_identity_permissions_and_master_race(self) -> None:
        changes = [lambda s: s["repository"].update(id=4),
                   lambda s: s["repository"].update(full_name="other/repo"),
                   lambda s: s["repository"].update(default_branch="main"),
                   lambda s: s["repository"].update(private=True),
                   lambda s: s["repository"]["permissions"].update(admin=False),
                   lambda s: s["repository"].update(allow_squash_merge=False),
                   lambda s: s["final_branch"]["commit"].update(sha="b" * 40)]
        for change in changes:
            with self.subTest(change=change):
                snap = fixture()
                change(snap)
                with self.assertRaises(merge.PolicyMismatch):
                    evaluate(snap)

    def test_foreign_inactive_duplicate_and_classic_protection(self) -> None:
        changes = [lambda s: s.update(protection={"enabled": True}),
                   lambda s: s["ruleset_summaries"].append({"id": 9}),
                   lambda s: s["effective_rules"].append({"type": "deletion"})]
        for change in changes:
            with self.subTest(change=change):
                snap = fixture()
                change(snap)
                with self.assertRaises((merge.PolicyMismatch, merge.PolicyError)):
                    evaluate(snap)
        for key, value in (("enforcement", "evaluate"), ("target", "tag"),
                           ("conditions", {"ref_name": {"include": ["~ALL"], "exclude": []}}),
                           ("bypass_actors", [{"actor_id": 1}])):
            snap = fixture(applied=True)
            snap["rulesets"][0][key] = value
            with self.subTest(key=key), self.assertRaises(merge.PolicyMismatch):
                evaluate(snap)

    def test_effective_rules_and_policy_drift(self) -> None:
        changes = [lambda s: s["effective_rules"].pop(),
                   lambda s: s["effective_rules"][0].update(ruleset_id=8),
                   lambda s: s["effective_rules"][0].update(ruleset_source="other/repo"),
                   lambda s: s["effective_rules"][0].update(parameters={"x": 1}),
                   lambda s: s["rulesets"][0]["rules"][3]["parameters"].update(unknown=True),
                   lambda s: s["rulesets"][0]["rules"][3]["parameters"].update(allowed_merge_methods=["squash", "merge"]),
                   lambda s: s["rulesets"][0]["rules"][4]["parameters"].update(strict_required_status_checks_policy=False),
                   lambda s: s["rulesets"][0]["rules"][4]["parameters"]["required_status_checks"][0].update(integration_id=9)]
        for change in changes:
            with self.subTest(change=change):
                snap = fixture(applied=True)
                change(snap)
                with self.assertRaises(merge.PolicyMismatch):
                    evaluate(snap)

    def test_check_runs_require_success_unique_recent_correct_app_and_sha(self) -> None:
        changes = [lambda s: s["check_runs"].pop(),
                   lambda s: s["check_runs"].append(deepcopy(s["check_runs"][0])),
                   lambda s: s["check_runs"][0].update(status="queued"),
                   lambda s: s["check_runs"][0].update(conclusion="neutral"),
                   lambda s: s["check_runs"][0].update(conclusion="skipped"),
                   lambda s: s["check_runs"][0].update(conclusion="failure"),
                   lambda s: s["check_runs"][0].update(head_sha="b" * 40),
                   lambda s: s["check_runs"][0]["app"].update(id=123),
                   lambda s: s["check_runs"][0].update(completed_at="2026-10-01T00:00:00Z")]
        for change in changes:
            with self.subTest(change=change):
                snap = fixture()
                change(snap)
                with self.assertRaises(merge.PolicyMismatch):
                    evaluate(snap)
        snap = fixture()
        snap["check_runs"].append(dict(name="extra-check"))
        self.assertEqual(evaluate(snap)["action"], "create")
        snap = fixture(applied=True)
        snap["check_runs"][0]["completed_at"] = "2025-01-01T00:00:00Z"
        self.assertEqual(evaluate(snap)["action"], "noop")


class ApiTests(unittest.TestCase):
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
            def __init__(self, message):
                self.message = message
            def get_json(self, endpoint):
                if endpoint.endswith("/protection"):
                    raise merge.ApiStatus(404, self.message)
                if endpoint.endswith("/branches/master"):
                    return {"name": "master", "commit": {"sha": SHA}}
                return {}
            def get_pages(self, endpoint, **kwargs):
                return []
        self.assertIsNone(merge.collect_snapshot(FakeApi("Branch not protected"), SHA)["protection"])
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
        with patch.object(merge.subprocess, "run", side_effect=subprocess.TimeoutExpired(["gh"], 25)):
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
