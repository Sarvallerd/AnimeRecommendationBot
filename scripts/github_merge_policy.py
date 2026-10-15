#!/usr/bin/env python3
"""Validate the declared master ruleset and inspect GitHub without changing settings."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import urlencode

REPOSITORY = "Sarvallerd/AnimeRecommendationBot"
REPOSITORY_ID = 631205863
MASTER = "master"
APP_ID = 15368
CHECKS = frozenset({"agent-setup", "rust", "python", "compose", "dev-environment"})
POLICY = Path(__file__).resolve().parents[1] / ".github" / "merge-policy.json"
API = f"repos/{REPOSITORY}"
API_VERSION = "2026-03-10"
SHA = re.compile(r"[0-9a-f]{40}\Z")


class PolicyError(Exception):
    """Invalid declaration, incomplete API evidence, or inaccessible API."""


class PolicyMismatch(Exception):
    """A complete snapshot differs from the approved policy."""


class ApiStatus(PolicyError):
    def __init__(self, status: int, message: str = "") -> None:
        self.status = status
        self.message = message
        super().__init__(f"GitHub API returned HTTP {status}")


def _pairs(items: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in items:
        if key in result:
            raise PolicyError("duplicate JSON key")
        result[key] = value
    return result


def _constant(value: str) -> None:
    raise PolicyError(f"invalid JSON constant: {value}")


def _json(raw: str) -> object:
    try:
        return json.loads(raw, object_pairs_hook=_pairs, parse_constant=_constant)
    except (ValueError, UnicodeError) as exc:
        raise PolicyError("malformed JSON") from exc


def _keys(obj: object, expected: set[str], label: str) -> dict:
    if type(obj) is not dict or set(obj) != expected:
        raise PolicyError(f"invalid {label} fields")
    return obj


def _same(actual: object, expected: object, label: str) -> None:
    if type(actual) is not type(expected) or actual != expected:
        raise PolicyError(f"invalid {label}")


def _rules(raw: object, *, server: bool = False) -> dict[str, dict]:
    if type(raw) is not list or len(raw) != 5:
        raise PolicyError("expected five rules")
    result = {}
    simple = {"deletion", "non_fast_forward", "required_linear_history"}
    for item in raw:
        if type(item) is not dict or type(item.get("type")) is not str:
            raise PolicyError("invalid rule")
        rule = dict(item)
        if server:
            for key in ("id", "ruleset_id", "ruleset_source", "ruleset_source_type", "_links",
                        "created_at", "updated_at", "node_id", "source", "source_type", "url"):
                rule.pop(key, None)
        kind = rule.get("type")
        if kind in result:
            raise PolicyError("duplicate rule")
        if kind in simple:
            _keys(rule, {"type"}, kind)
        elif kind == "pull_request":
            _keys(rule, {"type", "parameters"}, kind)
            params = dict(rule["parameters"]) if type(rule["parameters"]) is dict else {}
            if server:
                if params.get("required_reviewers") == []:
                    params.pop("required_reviewers")
                if params.get("dismissal_restriction") == {"enabled": False, "allowed_actors": []}:
                    params.pop("dismissal_restriction")
            _keys(params, {"allowed_merge_methods", "dismiss_stale_reviews_on_push",
                           "require_code_owner_review", "require_last_push_approval",
                           "required_approving_review_count", "required_review_thread_resolution"}, kind)
            methods = params["allowed_merge_methods"]
            if type(methods) is not list or any(type(x) is not str for x in methods) or len(methods) != len(set(methods)):
                raise PolicyError("invalid merge methods")
            params["allowed_merge_methods"] = sorted(methods)
            for name in ("dismiss_stale_reviews_on_push", "require_code_owner_review",
                         "require_last_push_approval", "required_review_thread_resolution"):
                if type(params[name]) is not bool:
                    raise PolicyError(f"invalid {name}")
            if type(params["required_approving_review_count"]) is not int:
                raise PolicyError("invalid approval count")
            rule["parameters"] = params
        elif kind == "required_status_checks":
            _keys(rule, {"type", "parameters"}, kind)
            params = _keys(rule["parameters"], {"strict_required_status_checks_policy",
                                                 "do_not_enforce_on_create", "required_status_checks"}, kind)
            if type(params["strict_required_status_checks_policy"]) is not bool or type(params["do_not_enforce_on_create"]) is not bool:
                raise PolicyError("invalid status check flags")
            checks = params["required_status_checks"]
            if type(checks) is not list or len(checks) != 5:
                raise PolicyError("expected five required checks")
            normalized = []
            for check in checks:
                _keys(check, {"context", "integration_id"}, "required check")
                if type(check["context"]) is not str or type(check["integration_id"]) is not int:
                    raise PolicyError("invalid required check")
                normalized.append(check)
            if len({x["context"] for x in normalized}) != len(normalized):
                raise PolicyError("duplicate required check")
            rule["parameters"] = dict(params, required_status_checks=sorted(normalized, key=lambda x: x["context"]))
        else:
            raise PolicyError("unknown rule type")
        result[kind] = rule
    if set(result) != simple | {"pull_request", "required_status_checks"}:
        raise PolicyError("wrong rule set")
    return result


def load_policy(path: Path = POLICY) -> dict:
    try:
        raw = path.read_bytes()
        policy = _json(raw.decode("utf-8"))
    except (OSError, UnicodeError) as exc:
        raise PolicyError("cannot read policy") from exc
    _keys(policy, {"name", "target", "enforcement", "bypass_actors", "conditions", "rules"}, "policy")
    for key, value in (("name", "arb-master-merge-policy-v1"), ("target", "branch"),
                       ("enforcement", "active"), ("bypass_actors", [])):
        _same(policy[key], value, key)
    _keys(policy["conditions"], {"ref_name"}, "conditions")
    _keys(policy["conditions"]["ref_name"], {"include", "exclude"}, "ref condition")
    _same(policy["conditions"]["ref_name"]["include"], ["refs/heads/master"], "included refs")
    _same(policy["conditions"]["ref_name"]["exclude"], [], "excluded refs")
    rules = _rules(policy["rules"])
    expected_pr = {"allowed_merge_methods": ["squash"], "dismiss_stale_reviews_on_push": True,
                   "require_code_owner_review": False, "require_last_push_approval": False,
                   "required_approving_review_count": 0, "required_review_thread_resolution": True}
    _same(rules["pull_request"]["parameters"], expected_pr, "pull request rule")
    status = rules["required_status_checks"]["parameters"]
    _same(status["strict_required_status_checks_policy"], True, "strict status checks")
    _same(status["do_not_enforce_on_create"], False, "status checks on create")
    _same({x["context"] for x in status["required_status_checks"]}, set(CHECKS), "required check names")
    if any(x["integration_id"] != APP_ID for x in status["required_status_checks"]):
        raise PolicyError("wrong required check app")
    return policy


class GhApi:
    """Fixed-host GET client. No token reads and no admin write methods."""

    def get_json(self, endpoint: str) -> object:
        if not endpoint.startswith(API + "/") and endpoint != API:
            raise PolicyError("unexpected API endpoint")
        command = ["gh", "api", "--hostname", "github.com", "--method", "GET", "--include",
                   "-H", "Accept: application/vnd.github+json",
                   "-H", f"X-GitHub-Api-Version: {API_VERSION}", endpoint]
        try:
            proc = subprocess.run(command, capture_output=True, text=True, timeout=25, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise PolicyError("GitHub API unavailable") from exc
        match = re.search(r"(?:\A|\r?\n\r?\n)HTTP/\S+ (\d{3})[^\r\n]*\r?\n", proc.stdout)
        if match is None:
            raise PolicyError("GitHub API response missing status")
        status = int(match.group(1))
        separator = re.search(r"\r?\n\r?\n", proc.stdout[match.start():])
        if separator is None:
            raise PolicyError("GitHub API response missing body")
        body = proc.stdout[match.start() + separator.end():]
        if status == 404:
            obj = _json(body)
            message = obj.get("message", "") if type(obj) is dict else ""
            raise ApiStatus(status, message)
        if proc.returncode != 0 or status != 200:
            raise ApiStatus(status)
        return _json(body)

    def get_pages(self, endpoint: str, *, item_key: str | None = None) -> list:
        all_items = []
        for page in range(1, 21):
            sep = "&" if "?" in endpoint else "?"
            obj = self.get_json(f"{endpoint}{sep}{urlencode({'per_page': 100, 'page': page})}")
            if item_key is None:
                items = obj
            else:
                if type(obj) is not dict or type(obj.get("total_count")) is not int:
                    raise PolicyError("incomplete paginated response")
                items = obj.get(item_key)
            if type(items) is not list or len(items) > 100:
                raise PolicyError("invalid paginated response")
            all_items.extend(items)
            if len(items) < 100:
                if item_key is not None and len(all_items) != obj["total_count"]:
                    raise PolicyError("incomplete paginated response")
                return all_items
        raise PolicyError("pagination limit exceeded")


def collect_snapshot(api: GhApi, expected_sha: str) -> dict:
    if SHA.fullmatch(expected_sha) is None:
        raise PolicyError("expected master SHA must be 40 lowercase hex characters")
    repo = api.get_json(API)
    branch = api.get_json(f"{API}/branches/{MASTER}")
    try:
        protection = api.get_json(f"{API}/branches/{MASTER}/protection")
    except ApiStatus as exc:
        if exc.status == 404 and exc.message == "Branch not protected":
            protection = None
        else:
            raise
    summaries = api.get_pages(f"{API}/rulesets?includes_parents=true")
    if any(type(x) is not dict or type(x.get("id")) is not int for x in summaries):
        raise PolicyError("invalid ruleset summary")
    ids = [x["id"] for x in summaries]
    if len(ids) != len(set(ids)):
        raise PolicyError("duplicate ruleset summary")
    details = [api.get_json(f"{API}/rulesets/{identifier}") for identifier in ids]
    effective = api.get_pages(f"{API}/rules/branches/{MASTER}")
    checks = api.get_pages(f"{API}/commits/{expected_sha}/check-runs?filter=latest", item_key="check_runs")
    final_branch = api.get_json(f"{API}/branches/{MASTER}")
    return {"expected_master_sha": expected_sha, "repository": repo, "branch": branch, "protection": protection,
            "ruleset_summaries": summaries, "rulesets": details,
            "effective_rules": effective, "check_runs": checks, "final_branch": final_branch}


def _sha_from_branch(branch: object) -> str:
    if type(branch) is not dict or branch.get("name") != MASTER or type(branch.get("commit")) is not dict:
        raise PolicyError("invalid branch response")
    sha = branch["commit"].get("sha")
    if type(sha) is not str or SHA.fullmatch(sha) is None:
        raise PolicyError("invalid branch SHA")
    return sha


def _validate_checks(checks: object, expected_sha: str, *, recent: bool, now: datetime) -> None:
    if type(checks) is not list:
        raise PolicyError("invalid check runs")
    seen = set()
    found = set()
    for check in checks:
        if type(check) is not dict or type(check.get("name")) is not str:
            raise PolicyError("invalid check run")
        name = check["name"]
        if name in seen:
            raise PolicyMismatch("ambiguous latest check run")
        seen.add(name)
        if name not in CHECKS:
            continue
        app = check.get("app")
        if (check.get("head_sha") != expected_sha or check.get("status") != "completed"
                or check.get("conclusion") != "success" or type(app) is not dict
                or type(app.get("id")) is not int or app["id"] != APP_ID
                or app.get("slug") != "github-actions"):
            raise PolicyMismatch(f"required check {name} is not a successful GitHub Actions run")
        if recent:
            value = check.get("completed_at")
            if type(value) is not str:
                raise PolicyMismatch(f"required check {name} has no completion time")
            try:
                completed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError as exc:
                raise PolicyError("invalid check completion time") from exc
            if completed.tzinfo is None or not now - timedelta(days=7) <= completed <= now + timedelta(minutes=5):
                raise PolicyMismatch(f"required check {name} is stale")
        found.add(name)
    if found != CHECKS:
        raise PolicyMismatch("missing required successful check runs")


def _validate_repo(repo: object) -> None:
    if type(repo) is not dict:
        raise PolicyError("invalid repository response")
    permissions = repo.get("permissions")
    required = {"id": REPOSITORY_ID, "full_name": REPOSITORY, "default_branch": MASTER,
                "private": False, "archived": False, "disabled": False,
                "visibility": "public", "allow_squash_merge": True}
    for key, expected in required.items():
        if type(repo.get(key)) is not type(expected) or repo[key] != expected:
            raise PolicyMismatch(f"repository {key} differs")
    if type(permissions) is not dict or permissions.get("admin") is not True:
        raise PolicyMismatch("repository admin permission is unavailable")


def _validate_applied(policy: dict, detail: object, effective: object) -> int:
    if type(detail) is not dict or type(detail.get("id")) is not int:
        raise PolicyError("invalid ruleset detail")
    required_keys = set(policy) | {"id", "source_type", "source"}
    metadata_keys = {"node_id", "_links", "created_at", "updated_at", "current_user_can_bypass"}
    if not required_keys <= set(detail) or set(detail) - required_keys - metadata_keys:
        raise PolicyMismatch("ruleset detail fields differ")
    if "current_user_can_bypass" in detail and detail["current_user_can_bypass"] != "never":
        raise PolicyMismatch("ruleset current user bypass differs")
    identifier = detail["id"]
    expected_top = {key: policy[key] for key in ("name", "target", "enforcement", "bypass_actors", "conditions")}
    for key, expected in expected_top.items():
        if detail.get(key) != expected or type(detail.get(key)) is not type(expected):
            raise PolicyMismatch(f"ruleset {key} differs")
    if detail.get("source_type") != "Repository" or detail.get("source") != REPOSITORY:
        raise PolicyMismatch("ruleset source differs")
    try:
        if _rules(detail.get("rules"), server=True) != _rules(policy["rules"]):
            raise PolicyMismatch("ruleset rules differ")
        if type(effective) is not list or len(effective) != 5:
            raise PolicyMismatch("effective branch rules differ")
        for rule in effective:
            if (type(rule) is not dict or rule.get("ruleset_id") != identifier
                    or rule.get("ruleset_source_type") != "Repository"
                    or rule.get("ruleset_source") != REPOSITORY):
                raise PolicyMismatch("effective rule source differs")
        if _rules(effective, server=True) != _rules(policy["rules"]):
            raise PolicyMismatch("effective branch rules differ")
    except PolicyError as exc:
        raise PolicyMismatch(str(exc)) from exc
    return identifier


def validate_snapshot(policy: dict, snapshot: dict, mode: str, *, now: datetime | None = None) -> dict:
    if type(snapshot) is not dict:
        raise PolicyError("invalid snapshot")
    expected_sha = snapshot.get("expected_master_sha")
    if mode not in {"plan", "check"} or type(expected_sha) is not str or SHA.fullmatch(expected_sha) is None:
        raise PolicyError("invalid mode or expected SHA")
    _validate_repo(snapshot.get("repository"))
    if _sha_from_branch(snapshot.get("branch")) != expected_sha or _sha_from_branch(snapshot.get("final_branch")) != expected_sha:
        raise PolicyMismatch("master changed during inspection")
    if snapshot.get("protection") is not None:
        raise PolicyMismatch("classic branch protection exists")
    summaries = snapshot.get("ruleset_summaries")
    details = snapshot.get("rulesets")
    effective = snapshot.get("effective_rules")
    if type(summaries) is not list or type(details) is not list or len(summaries) != len(details):
        raise PolicyError("incomplete ruleset snapshot")
    if type(effective) is not list:
        raise PolicyError("invalid effective rules")
    if len(summaries) == 0 and len(effective) == 0:
        if mode == "check":
            raise PolicyMismatch("merge rules are not applied")
        action = "create"
        identifier = None
    elif len(summaries) == 1 and len(details) == 1:
        summary = summaries[0]
        detail = details[0]
        if type(summary) is not dict or type(detail) is not dict or summary.get("id") != detail.get("id"):
            raise PolicyError("ruleset summary/detail mismatch")
        identifier = _validate_applied(policy, detail, effective)
        action = "noop"
    else:
        raise PolicyMismatch("foreign or duplicate rulesets exist")
    _validate_checks(snapshot.get("check_runs"), expected_sha, recent=action == "create",
                     now=now or datetime.now(timezone.utc))
    return {"action": action, "ruleset_id": identifier}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate")
    for name in ("plan", "check"):
        sub.add_parser(name).add_argument("--expected-master-sha", required=True)
    args = parser.parse_args(argv)
    try:
        policy = load_policy()
        digest = hashlib.sha256(POLICY.read_bytes()).hexdigest()
        result = {"schema_version": 1, "mode": args.command, "repository": REPOSITORY,
                  "repository_id": REPOSITORY_ID, "master_sha": None, "policy_sha256": digest}
        if args.command == "validate":
            result["result"] = "valid"
        else:
            expected = args.expected_master_sha
            if SHA.fullmatch(expected) is None:
                raise PolicyError("expected master SHA must be 40 lowercase hex characters")
            verdict = validate_snapshot(policy, collect_snapshot(GhApi(), expected), args.command)
            result.update(master_sha=expected, result=verdict["action"] if args.command == "plan" else "compliant")
            if args.command == "plan":
                result["action"] = verdict["action"]
            if verdict["ruleset_id"] is not None:
                result["ruleset_id"] = verdict["ruleset_id"]
            if verdict["action"] == "create":
                result["ruleset_payload"] = policy
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    except PolicyMismatch as exc:
        print(f"merge policy mismatch: {exc}", file=sys.stderr)
        return 1
    except PolicyError as exc:
        print(f"merge policy inspection failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
