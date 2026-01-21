"""
plugins.features — Built-in feature plugins.

Contract (v0.1)
---------------
Importing this package must register all built-in feature plugins into the runtime registry.

WIP policy
----------
When you add a new built-in feature plugin module, add one import line here.
"""

from __future__ import annotations

# Import built-in plugins for side-effect registration.
from .ewma_dissipation import EWMADissipationFeature  # noqa: F401
from .integrated_current import IntegratedCurrentFeature  # noqa: F401
from .prev_switch_count import PrevSwitchCountFeature  # noqa: F401
from .time_since_last import TimeSinceLastFeature  # noqa: F401

__all__ = [
    "EWMADissipationFeature",
    "IntegratedCurrentFeature",
    "PrevSwitchCountFeature",
    "TimeSinceLastFeature",
]
