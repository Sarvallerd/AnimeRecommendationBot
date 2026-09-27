"""Ensure importing the package does no command or preparation work."""

import subprocess
import sys
import tempfile
import unittest


class ImportTests(unittest.TestCase):
    def test_imports_have_no_side_effects(self) -> None:
        script = """
import importlib
import pathlib
import sys

sys.argv = ["recsys", "--invalid"]
before = set(pathlib.Path.cwd().iterdir())
for name in ("recsys", "recsys.cli", "recsys.__main__", "recsys.sources", "recsys.obtain", "recsys.normalize", "recsys.build", "recsys.bundle", "recsys.export"):
    importlib.import_module(name)
assert set(pathlib.Path.cwd().iterdir()) == before
assert not any(name == "numpy" or name.startswith("numpy.") for name in sys.modules)
assert not any(name == "pandas" or name.startswith("pandas.") for name in sys.modules)
assert not any(name == "app" or name.startswith("app.") for name in sys.modules)
assert not any(name == "config" or name.startswith("config.") for name in sys.modules)
"""
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, "-c", script],
                cwd=directory,
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "")
        self.assertEqual(result.stderr, "")


if __name__ == "__main__":
    unittest.main()
