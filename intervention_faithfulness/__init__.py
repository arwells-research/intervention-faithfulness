"""
intervention_faithfulness: intervention-faithfulness diagnostics and plugins.
"""

# Ensure built-in plugins are registered at import time (v0.1 registry contract).
# This keeps list_plugins() and FaithfulnessTest.from_plugin(...) working without
# requiring users to manually import plugin modules.
from . import plugins as _plugins  # noqa: F401

# Re-export the main entrypoint used by tests.
from .core.FaithfulnessTest import FaithfulnessTest  # noqa: F401
