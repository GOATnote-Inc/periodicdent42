"""Pytest configuration for app package."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
APP_SRC = PROJECT_ROOT / "src"

for path in (str(PROJECT_ROOT), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

if "src" not in sys.modules:
    spec = importlib.util.spec_from_file_location(
        "src", APP_SRC / "__init__.py", submodule_search_locations=[str(APP_SRC)]
    )
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    module.__path__ = [str(APP_SRC)]  # type: ignore[attr-defined]
    sys.modules["src"] = module
