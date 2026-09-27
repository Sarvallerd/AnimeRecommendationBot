"""Standalone checkout-local wrapper for the packaged stdlib bundle validator."""

import importlib.util
import sys
from pathlib import Path

_module_path = Path(__file__).resolve().parents[1] / "recsys" / "src" / "recsys" / "bundle.py"
_spec = importlib.util.spec_from_file_location("_checkout_bundle", _module_path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
ContractError = _module.ContractError
read_canonical = _module.read_canonical
check_catalog = _module.check_catalog
check_neighbors = _module.check_neighbors
check_manifest = _module.check_manifest
check_bundle = _module.check_bundle
main = _module.main

if __name__ == "__main__":
    sys.exit(main())
