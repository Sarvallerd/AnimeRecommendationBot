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
for name in ("recsys", "recsys.cli.main", "recsys.__main__", "recsys.data.sources", "recsys.data.obtain", "recsys.data.mal2020", "recsys.data.dataset", "recsys.data.metadata", "recsys.data.interactions", "recsys.data.aliases", "recsys.release.catalog", "recsys.release.build", "recsys.release.prepare", "recsys.content.glove", "recsys.contracts.bundle", "recsys.release.export", "recsys.historical.legacy_vectors", "recsys.historical.comparison", "recsys.historical.assessment", "recsys.research.protocol", "recsys.research.artifacts", "recsys.research.prepare", "recsys.research.evaluation.evaluate", "recsys.research.evaluation.compare", "recsys.research.evaluation.finalize", "recsys.research.models.package", "recsys.research.predict", "recsys.research.models.popularity", "recsys.research.training", "recsys.research.training.spec", "recsys.research.training.artifacts", "recsys.research.training.candidate", "recsys.research.training.examples", "recsys.research.training.execution", "recsys.research.training.predict", "recsys.research.training.receipts", "recsys.research.training.torch_model", "recsys.research.training.types", "recsys.research.training.worker"):
    importlib.import_module(name)
assert set(pathlib.Path.cwd().iterdir()) == before
assert not any(name == "numpy" or name.startswith("numpy.") for name in sys.modules)
assert not any(name == "sklearn" or name.startswith("sklearn.") for name in sys.modules)
assert not any(name == "cupy" or name.startswith("cupy.") for name in sys.modules)
assert not any(name == "torch" or name.startswith("torch.") for name in sys.modules)
assert not any(name == "catboost" or name.startswith("catboost.") for name in sys.modules)
assert not any(name == "safetensors" or name.startswith("safetensors.") for name in sys.modules)
assert not any(name == "threadpoolctl" for name in sys.modules)
assert not any(name == "pandas" or name.startswith("pandas.") for name in sys.modules)
assert not any(name == "duckdb" or name.startswith("duckdb.") for name in sys.modules)
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
