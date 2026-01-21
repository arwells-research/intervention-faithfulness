# tests/conftest.py

from __future__ import annotations
 
import sys
from pathlib import Path
 
import pytest

# Ensure repo root is on sys.path so `import intervention_faithfulness` works under `pytest`
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def _ensure_project_on_syspath() -> None:
    """
    Make the project importable in test runs without requiring editable installs.

    Supports both:
    - repo-root layout: intervention_faithfulness/...
    - src layout:       src/intervention_faithfulness/...
    """
    repo_root = Path(__file__).resolve().parents[1]
    src_root = repo_root / "src"

    # Prepend so local code wins over any installed package with same name.
    for p in (repo_root, src_root):
         ps = str(p)
         if p.exists() and ps not in sys.path:
             sys.path.insert(0, ps)
 
    _ensure_project_on_syspath()
 

def pytest_configure(config):
    # Ensure headless-friendly matplotlib backend for CI
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
    except Exception:
        pass    
    # Keep test output readable for first-time users.
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks integration-style tests")