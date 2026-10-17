"""Behavioral fixtures for the static nested-package boundary guard."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from typing import cast

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from check_module_boundaries import check_repository  # noqa: E402


def _object(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise AssertionError("fixture policy must be an object with string keys")
    return cast("dict[str, object]", value)


def _items(value: object) -> list[object]:
    if not isinstance(value, list):
        raise AssertionError("fixture policy must contain a list")
    return cast("list[object]", value)


class BoundariesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        package = self.root / "src/recsys"
        package.parent.mkdir(parents=True)
        shutil.copytree(REPO / "src/recsys", package)
        (self.root / "scripts").mkdir()
        shutil.copy2(
            REPO / "scripts/module_boundaries.json", self.root / "scripts/module_boundaries.json"
        )

    def source(self, module: str) -> Path:
        return self.root / "src/recsys" / f"{module}.py"

    def append(self, module: str, code: str) -> None:
        with self.source(module).open("a", encoding="utf-8") as stream:
            stream.write("\n" + code + "\n")

    def reset(self, module: str) -> None:
        self.source(module).write_bytes((REPO / "src/recsys" / f"{module}.py").read_bytes())

    def policy(self) -> dict[str, object]:
        return _object(json.loads((self.root / "scripts/module_boundaries.json").read_text()))

    def save_policy(self, policy: dict[str, object]) -> None:
        (self.root / "scripts/module_boundaries.json").write_text(json.dumps(policy))

    def assert_category(self, category: str) -> None:
        errors = check_repository(self.root)
        self.assertTrue(any(error.startswith(category + ":") for error in errors), errors)

    def test_actual_graph_and_no_source_execution(self) -> None:
        self.assertEqual(check_repository(REPO), [])
        self.append("contracts/bundle", "raise RuntimeError('source executed')")
        self.assertEqual(check_repository(self.root), [])

    def test_release_cannot_import_research(self) -> None:
        self.append("release/catalog", "from recsys.research.models.dispatch import _backend")
        self.assert_category("forbidden_dependency")

    def test_backend_cannot_import_package_or_dispatch(self) -> None:
        for target in ("package", "dispatch"):
            with self.subTest(target=target):
                self.append(
                    "research/models/content",
                    f"from recsys.research.models.{target} import fit_model",
                )
                self.assert_category("forbidden_dependency")
                self.reset("research/models/content")

    def test_evaluate_cannot_import_finalize(self) -> None:
        self.append(
            "research/evaluation/evaluate",
            "from recsys.research.evaluation.finalize import finalize",
        )
        self.assert_category("new_cycle")

    def test_campaign_report_cannot_import_runner(self) -> None:
        self.append(
            "research/campaign/report", "from recsys.research.campaign.runner import run_campaign"
        )
        self.assert_category("new_cycle")

    def test_lazy_type_checking_and_optional_imports_are_edges(self) -> None:
        for code in (
            "def late():\n    from recsys.research.models.dispatch import _backend",
            "if TYPE_CHECKING:\n    from recsys.research.models.dispatch import _backend",
            "try:\n    from recsys.research.models.dispatch import _backend\nexcept ImportError:\n    pass",
        ):
            with self.subTest(code=code.splitlines()[0]):
                self.append("release/catalog", code)
                self.assert_category("forbidden_dependency")
                self.reset("release/catalog")

    def test_new_cycle_and_self_import(self) -> None:
        self.append("data/metadata", "from recsys.data.dataset import HEADER")
        self.assert_category("new_cycle")
        self.reset("data/metadata")
        for code in ("import recsys.release.catalog", "from . import catalog"):
            self.append("release/catalog", code)
            self.assert_category("new_cycle")
            self.reset("release/catalog")

    def test_inventory_and_unknown_internal_import(self) -> None:
        self.source("brand_new").write_text("")
        self.assert_category("inventory_mismatch")
        self.source("brand_new").unlink()
        self.append("release/catalog", "import recsys.does_not_exist")
        self.assert_category("unknown_import")

    def test_research_prepare_cannot_import_release_builder(self) -> None:
        self.append("research/prepare", "from recsys.release.prepare import prepare_release")
        self.assert_category("forbidden_dependency")

    def test_literal_dynamic_and_alias_respect_boundaries(self) -> None:
        for code in (
            "import importlib\nimportlib.import_module('recsys.research.evaluation.evaluate')",
            "import importlib as loader\nloader.import_module('recsys.research.evaluation.evaluate')",
            "from importlib import import_module as loader\nloader('recsys.research.evaluation.evaluate')",
            "__import__('recsys.research.evaluation.evaluate')",
            "loader = __import__\nloader('recsys.research.evaluation.evaluate')",
        ):
            with self.subTest(code=code.splitlines()[0]):
                self.append("release/catalog", code)
                self.assert_category("forbidden_dependency")
                self.reset("release/catalog")

    def test_direct_submodule_and_builtin_loaders(self) -> None:
        for code in (
            "import importlib.util\nimportlib.import_module('recsys.research.evaluation.evaluate')",
            "from builtins import __import__ as loader\nloader('recsys.research.evaluation.evaluate')",
            "import builtins as bi\nbi.__import__('recsys.research.evaluation.evaluate')",
        ):
            with self.subTest(code=code.splitlines()[0]):
                self.append("release/catalog", code)
                self.assert_category("forbidden_dependency")
                self.reset("release/catalog")
        for code in (
            "import importlib.util\nimportlib.import_module(f'recsys.{name}')",
            "from builtins import __import__ as loader\nloader('recsys.research.evaluation.evaluate', globals())",
        ):
            self.append("release/catalog", code)
            self.assert_category("unsupported_dynamic_import")
            self.reset("release/catalog")

    def test_dynamic_dispatch_pin_and_finite_targets(self) -> None:
        source = self.source("research/models/dispatch")
        content = source.read_text().replace(
            '"content-mean": "content"', '"content-mean": "invalid"', 1
        )
        source.write_text(content)
        self.assert_category("dynamic_dispatch_changed")
        policy = self.policy()
        _object(policy["dynamic_dispatch"])["source_sha256"] = hashlib.sha256(
            source.read_bytes()
        ).hexdigest()
        self.save_policy(policy)
        self.assert_category("dynamic_dispatch_changed")

    def test_registry_extra_variant_reusing_backend_fails_after_repin(self) -> None:
        source = self.source("research/models/dispatch")
        content = source.read_text().replace(
            '    "popularity": "popularity",',
            '    "extra": "content",\n    "popularity": "popularity",',
            1,
        )
        source.write_text(content)
        policy = self.policy()
        _object(policy["dynamic_dispatch"])["source_sha256"] = hashlib.sha256(
            source.read_bytes()
        ).hexdigest()
        self.save_policy(policy)
        self.assert_category("dynamic_dispatch_changed")

    def test_unresolved_loader_wildcard_and_callable_fail(self) -> None:
        self.append(
            "release/catalog", "import importlib\nimportlib.import_module(f'recsys.{name}')"
        )
        self.assert_category("unsupported_dynamic_import")
        self.reset("release/catalog")
        self.append("release/catalog", "from recsys.data.dataset import *")
        self.assert_category("unknown_import")
        self.reset("release/catalog")
        self.append("release/catalog", "import importlib\nreceiver(importlib.import_module)")
        self.assert_category("unsupported_dynamic_import")

    def test_unknown_package_member(self) -> None:
        self.append("release/catalog", "from recsys import nonexistent_member")
        self.assert_category("unknown_import")

    def test_malformed_policy_and_stale_exception(self) -> None:
        policy_path = self.root / "scripts/module_boundaries.json"
        original = policy_path.read_bytes()
        policy_path.write_text('{"schema_version":1,"schema_version":1}')
        self.assert_category("config_invalid")
        policy_path.write_bytes(original)
        policy = self.policy()
        _items(_object(_object(policy["subsystems"])["data"])["depends_on"]).append("missing_group")
        self.save_policy(policy)
        self.assert_category("config_invalid")
        policy_path.write_bytes(original)
        policy = self.policy()
        _items(policy["exceptions"]).append(
            {
                "source": "recsys.release.catalog",
                "target": "recsys.research.models.dispatch",
                "names": ["_backend"],
                "reason": "negative fixture",
            }
        )
        self.save_policy(policy)
        self.assert_category("stale_exception")

    def test_module_package_collision(self) -> None:
        package = self.root / "src/recsys/collision"
        package.mkdir()
        (package / "__init__.py").write_text("")
        self.source("collision").write_text("")
        self.assert_category("config_invalid")

    def test_nested_package_relative_import_uses_package_anchor(self) -> None:
        package = self.root / "src/recsys/nested"
        package.mkdir()
        (package / "__init__.py").write_text("__version__ = '1'\nfrom . import child\n")
        (package / "child.py").write_text("from .. import __version__\n")
        policy = self.policy()
        _items(_object(_object(policy["subsystems"])["core"])["modules"]).extend(
            ["recsys.nested", "recsys.nested.child"]
        )
        self.save_policy(policy)
        self.assertEqual(check_repository(self.root), [])


if __name__ == "__main__":
    unittest.main()
