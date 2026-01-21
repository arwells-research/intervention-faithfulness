"""
plugins.data — Built-in data plugins.

Contract (v0.1)
---------------
Importing this package must register all built-in data plugins into the runtime registry.

WIP policy
----------
When you add a new built-in data plugin module, add one import line here.
"""

from __future__ import annotations

# Import built-in DATA plugins for side-effect registration.
from .nanowire_switching import NanowireSwitchingPlugin  # noqa: F401
from .faithful_synthetic import FaithfulSyntheticDataPlugin  # noqa: F401
from .unfaithful_cut_synthetic import UnfaithfulCutSyntheticDataPlugin  # noqa: F401
from .sigma2i_unfaithful_cut_linear import Sigma2IUnfaithfulCutLinear  # noqa: F401

__all__ = ["NanowireSwitchingPlugin"]
