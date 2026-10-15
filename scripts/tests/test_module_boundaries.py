"""Behavioral fixtures for the static boundary guard; no scientific data is loaded."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from check_module_boundaries import check_repository  # noqa: E402


class BoundariesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        package = self.root / "recsys/src/recsys"
        package.parent.mkdir(parents=True)
        shutil.copytree(REPO / "recsys/src/recsys", package)
        (self.root / "scripts").mkdir()
        shutil.copy2(REPO / "scripts/module_boundaries.json", self.root / "scripts/module_boundaries.json")

    def source(self, module: str) -> Path:
        return self.root / "recsys/src/recsys" / f"{module}.py"

    def append(self, module: str, code: str) -> None:
        with self.source(module).open("a", encoding="utf-8") as stream:
            stream.write("\n" + code + "\n")

    def policy(self) -> dict:
        return json.loads((self.root / "scripts/module_boundaries.json").read_text(encoding="utf-8"))

    def save_policy(self, policy: dict) -> None:
        (self.root / "scripts/module_boundaries.json").write_text(json.dumps(policy), encoding="utf-8")

    def assert_category(self, category: str) -> None:
        errors = check_repository(self.root)
        self.assertTrue(any(error.startswith(category + ":") for error in errors), errors)

    def test_actual_graph_and_no_source_execution(self) -> None:
        self.assertEqual(check_repository(REPO), [])
        self.append("bundle", "raise RuntimeError('source was executed')")
        self.assertEqual(check_repository(self.root), [])

    def test_release_cannot_import_research(self) -> None:
        self.append("release", "from .research_models import _backend")
        self.assert_category("forbidden_dependency")

    def test_model_cannot_import_evaluation_or_prepare(self) -> None:
        self.append("research_predict", "from .research_evaluate import evaluate\nfrom .research_prepare import prepare")
        self.assert_category("forbidden_dependency")

    def test_lazy_type_checking_and_optional_imports_are_edges(self) -> None:
        for code in (
            "def late():\n    from .research_models import _backend",
            "if TYPE_CHECKING:\n    from .research_models import _backend",
            "try:\n    from .research_models import _backend\nexcept ImportError:\n    pass",
        ):
            with self.subTest(code=code.splitlines()[0]):
                path = self.source("release")
                original = path.read_bytes()
                self.append("release", code)
                self.assert_category("forbidden_dependency")
                path.write_bytes(original)

    def test_new_cycle_and_new_internal_edge_in_existing_cycle(self) -> None:
        self.append("dataset_metadata", "from .dataset import HEADER")
        self.assert_category("new_cycle")
        self.source("dataset_metadata").write_bytes((REPO / "recsys/src/recsys/dataset_metadata.py").read_bytes())
        self.append("research_model_cf", "from .research_model_content import fit")
        self.assert_category("new_cycle")

    def test_absolute_and_relative_self_imports_are_cycles(self) -> None:
        for code in ("import recsys.release", "from . import release"):
            with self.subTest(code=code):
                path = self.source("release")
                original = path.read_bytes()
                self.append("release", code)
                self.assert_category("new_cycle")
                path.write_bytes(original)

    def test_inventory_and_unknown_internal_import(self) -> None:
        self.source("brand_new").write_text("", encoding="utf-8")
        self.assert_category("inventory_mismatch")
        self.source("brand_new").unlink()
        self.append("release", "import recsys.does_not_exist")
        self.assert_category("unknown_import")

    def test_symbol_limited_exception(self) -> None:
        self.append("research_prepare", "from .release_prepare import prepare_release")
        self.assert_category("forbidden_dependency")

    def test_literal_dynamic_and_alias_both_respect_boundaries(self) -> None:
        for code in (
            "import importlib\nimportlib.import_module('recsys.research_evaluate')",
            "import importlib as loader\nloader.import_module('.research_evaluate', __package__)",
            "from importlib import import_module as loader\nloader('recsys.research_evaluate')",
            "__import__('recsys.research_evaluate')",
            "loader = __import__\nloader('recsys.research_evaluate')",
        ):
            with self.subTest(code=code.splitlines()[0]):
                path = self.source("release")
                original = path.read_bytes()
                self.append("release", code)
                self.assert_category("forbidden_dependency")
                path.write_bytes(original)

    def test_direct_submodule_and_builtin_loader_bindings(self) -> None:
        for code in (
            "import importlib.util\nimportlib.import_module('recsys.research_evaluate')",
            "from builtins import __import__ as loader\nloader('recsys.research_evaluate')",
            "import builtins as bi\nbi.__import__('recsys.research_evaluate')",
        ):
            with self.subTest(code=code.splitlines()[0]):
                path = self.source("release")
                original = path.read_bytes()
                self.append("release", code)
                self.assert_category("forbidden_dependency")
                path.write_bytes(original)
        for code in (
            "import importlib.util\nimportlib.import_module(f'recsys.{name}')",
            "from builtins import __import__ as loader\nloader('recsys.research_evaluate', globals())",
        ):
            with self.subTest(code=code.splitlines()[0]):
                path = self.source("release")
                original = path.read_bytes()
                self.append("release", code)
                self.assert_category("unsupported_dynamic_import")
                path.write_bytes(original)

    def test_dynamic_dispatch_pin_and_finite_targets(self) -> None:
        source = self.source("research_models")
        content = source.read_text(encoding="utf-8")
        self.assertIn('"research_model_content"', content)
        content = content.replace('"research_model_content"', '"research_evaluate"', 1)
        source.write_text(content, encoding="utf-8")
        self.assert_category("dynamic_dispatch_changed")
        policy = self.policy()
        policy["dynamic_dispatch"]["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
        self.save_policy(policy)
        self.assert_category("dynamic_dispatch_changed")

    def test_registry_extra_variant_reusing_backend_fails_after_repin(self) -> None:
        source = self.source("research_models")
        content = source.read_text(encoding="utf-8")
        self.assertIn('BACKENDS = {"popularity": "research_model_popularity",', content)
        content = content.replace('BACKENDS = {"popularity": "research_model_popularity",',
                                  'BACKENDS = {"extra": "research_model_content", "popularity": "research_model_popularity",', 1)
        source.write_text(content, encoding="utf-8")
        policy = self.policy()
        policy["dynamic_dispatch"]["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
        self.save_policy(policy)
        self.assert_category("dynamic_dispatch_changed")

    def test_unresolved_dynamic_loader_and_internal_wildcard(self) -> None:
        self.append("release", "import importlib\nimportlib.import_module(f'recsys.{name}')")
        self.assert_category("unsupported_dynamic_import")
        self.source("release").write_bytes((REPO / "recsys/src/recsys/release.py").read_bytes())
        self.append("release", "from .dataset import *")
        self.assert_category("unknown_import")

    def test_loader_callable_and_unknown_package_member_fail(self) -> None:
        self.append("release", "import importlib\nreceiver(importlib.import_module)")
        self.assert_category("unsupported_dynamic_import")
        self.source("release").write_bytes((REPO / "recsys/src/recsys/release.py").read_bytes())
        self.append("release", "from recsys import nonexistent_member")
        self.assert_category("unknown_import")

    def test_malformed_policy_and_stale_exceptions(self) -> None:
        policy_path = self.root / "scripts/module_boundaries.json"
        original = policy_path.read_bytes()
        policy_path.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
        self.assert_category("config_invalid")
        policy_path.write_bytes(original)
        policy = self.policy()
        policy["subsystems"]["data"]["depends_on"].append("missing_group")
        self.save_policy(policy)
        self.assert_category("config_invalid")
        policy_path.write_bytes(original)
        policy = self.policy()
        policy["exceptions"][0]["names"].append("unused_symbol")
        self.save_policy(policy)
        self.assert_category("stale_exception")
        policy_path.write_bytes(original)
        policy = self.policy()
        policy["existing_cycles"][0]["edges"].pop()
        self.save_policy(policy)
        self.assert_category("new_cycle")

    def test_module_package_collision_rejects_both_sources(self) -> None:
        package = self.root / "recsys/src/recsys/collision"
        package.mkdir()
        (package / "__init__.py").write_text("", encoding="utf-8")
        self.source("collision").write_text("", encoding="utf-8")
        policy = self.policy()
        policy["subsystems"]["core"]["modules"].append("recsys.collision")
        self.save_policy(policy)
        self.assert_category("config_invalid")
        self.assertTrue(any("module/package collision" in error for error in check_repository(self.root)))

    def test_nested_package_relative_import_uses_package_anchor(self) -> None:
        package = self.root / "recsys/src/recsys/nested"
        package.mkdir()
        (package / "__init__.py").write_text("__version__ = '1'\nfrom . import child\n", encoding="utf-8")
        (package / "child.py").write_text("from .. import __version__\n", encoding="utf-8")
        policy = self.policy()
        policy["subsystems"]["core"]["modules"].extend(["recsys.nested", "recsys.nested.child"])
        self.save_policy(policy)
        self.assertEqual(check_repository(self.root), [])


if __name__ == "__main__":
    unittest.main()
