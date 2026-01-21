"""
plugins — Built-in plugin packages.

Contract (v0.1)
---------------
Importing intervention_faithfulness.plugins must register all built-in plugins:
  - data plugins
  - feature plugins

Registration is performed by importing the subpackages, which in turn import
their concrete plugin modules for side-effect registration into the runtime registry.
"""

from __future__ import annotations

from . import data as _data  # noqa: F401
from . import features as _features  # noqa: F401

__all__ = []
