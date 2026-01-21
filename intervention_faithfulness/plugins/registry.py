"""
plugins/registry.py — Plugin registry, discovery, and help text (v0.1)

This module provides a simple, self-documenting plugin registry.

v0.1 goals:
- Zero-config built-in registry (works out of the box)
- User discoverability:
    - list_plugins()
    - FaithfulnessTest.plugin_help(name)  (delegates here)
- Clear separation:
    - core imports registry
    - registry does NOT import core algorithms

Planned (v0.2+):
- Optional entry-point discovery for third-party plugins
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Type, Union

import textwrap


# ---------------------------------------------------------------------
# Metadata contract
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class PluginMetadata:
    """
    Minimal self-documenting metadata for plugins.

    Required fields:
        - name
        - description
        - expected_format
        - example_usage

    Optional fields:
        - tags
        - links
    """

    name: str
    description: str
    expected_format: str
    example_usage: str
    tags: Optional[List[str]] = None
    links: Optional[Dict[str, str]] = None


# ---------------------------------------------------------------------
# Base marker classes (runtime only; ABC usage is optional)
# ---------------------------------------------------------------------

class DataPlugin:
    """
    Marker base class for data plugins.

    Required:
        - metadata: PluginMetadata
        - load(source, **kwargs)
        - to_trials(raw, **kwargs) -> pandas.DataFrame

    Optional:
        - defaults() -> dict
        - validate(df) -> list[str]
    """

    metadata: PluginMetadata

    def defaults(self) -> Dict[str, Any]:
        return {
            "divergence": "js",
            "min_samples": 50,
            "tail_mode": False,
            "quantile_focus": 0.95,
        }

    def validate(self, df) -> List[str]:
        return []


class FeaturePlugin:
    """
    Marker base class for feature plugins.

    Required:
        - metadata: PluginMetadata
        - compute(trials_df, **params) -> pandas.DataFrame

    Optional:
        - requires() -> list[str]
        - parameters() -> dict
    """

    metadata: PluginMetadata

    def requires(self) -> List[str]:
        return []

    def parameters(self) -> Dict[str, Any]:
        return {}


# ---------------------------------------------------------------------
# Registry storage
# ---------------------------------------------------------------------

_DATA_PLUGINS: Dict[str, Type[DataPlugin]] = {}
_FEATURE_PLUGINS: Dict[str, Type[FeaturePlugin]] = {}


# ---------------------------------------------------------------------
# Registration API (used by built-in plugins at import time)
# ---------------------------------------------------------------------

def register_data_plugin(plugin_cls: Type[DataPlugin]) -> None:
    meta = getattr(plugin_cls, "metadata", None)
    if meta is None or not isinstance(meta, PluginMetadata):
        raise TypeError("Data plugin must define `metadata: PluginMetadata`.")
    name = meta.name
    if name in _DATA_PLUGINS:
        raise ValueError(f"Duplicate data plugin name: {name}")
    _DATA_PLUGINS[name] = plugin_cls


def register_feature_plugin(plugin_cls: Type[FeaturePlugin]) -> None:
    meta = getattr(plugin_cls, "metadata", None)
    if meta is None or not isinstance(meta, PluginMetadata):
        raise TypeError("Feature plugin must define `metadata: PluginMetadata`.")
    name = meta.name
    if name in _FEATURE_PLUGINS:
        raise ValueError(f"Duplicate feature plugin name: {name}")
    _FEATURE_PLUGINS[name] = plugin_cls


# ---------------------------------------------------------------------
# Lookup API (used by FaithfulnessTest)
# ---------------------------------------------------------------------

def get_data_plugin(name: str) -> DataPlugin:
    if name not in _DATA_PLUGINS:
        raise KeyError(f"Unknown data plugin: {name}. Use list_plugins() to see available plugins.")
    return _DATA_PLUGINS[name]()


def get_feature_plugin(name: str) -> FeaturePlugin:
    if name not in _FEATURE_PLUGINS:
        raise KeyError(f"Unknown feature plugin: {name}. Use list_plugins() to see available plugins.")
    return _FEATURE_PLUGINS[name]()

# ---------------------------------------------------------------------
# Programmatic discovery helpers (tests + power users)
# ---------------------------------------------------------------------

def list_data_plugins() -> List[str]:
    """
    Return available data plugin names (machine-readable).
 
    Note: list_plugins() returns a human-readable listing; this returns just names.
    """
    return sorted(_DATA_PLUGINS.keys())
 
 
def list_feature_plugins() -> List[str]:
    """
    Return available feature plugin names (machine-readable).
 
    Note: list_plugins() returns a human-readable listing; this returns just names.
    """
    return sorted(_FEATURE_PLUGINS.keys())
 
 

# ---------------------------------------------------------------------
# Discovery helpers (user-facing)
# ---------------------------------------------------------------------

def list_plugins() -> str:
    """
    Return a human-readable listing of available plugins.

    Intended usage:
        from intervention_faithfulness import list_plugins
        print(list_plugins())
    """
    lines: List[str] = []

    lines.append("Data Plugins:")
    if not _DATA_PLUGINS:
        lines.append("  (none)")
    else:
        for name in sorted(_DATA_PLUGINS.keys()):
            meta = _DATA_PLUGINS[name].metadata
            lines.append(f"  - {meta.name:<18} {meta.description}")

    lines.append("")
    lines.append("Feature Plugins:")
    if not _FEATURE_PLUGINS:
        lines.append("  (none)")
    else:
        for name in sorted(_FEATURE_PLUGINS.keys()):
            meta = _FEATURE_PLUGINS[name].metadata
            lines.append(f"  - {meta.name:<18} {meta.description}")

    return "\n".join(lines)


def plugin_help(name: str) -> str:
    """
    Return a self-contained help block for a plugin (data or feature).
    """
    plugin_cls = None
    kind = None

    if name in _DATA_PLUGINS:
        plugin_cls = _DATA_PLUGINS[name]
        kind = "DATA"
    elif name in _FEATURE_PLUGINS:
        plugin_cls = _FEATURE_PLUGINS[name]
        kind = "FEATURE"
    else:
        raise KeyError(f"Unknown plugin: {name}. Use list_plugins() to see available plugins.")

    meta = plugin_cls.metadata
    inst = plugin_cls()

    lines: List[str] = []
    title = f"{meta.name.upper()} PLUGIN ({kind})"
    lines.append(title)
    lines.append("─" * len(title))
    lines.append("")
    lines.append(meta.description.strip())
    lines.append("")

    lines.append("Expected data / requirements:")
    lines.append(f"  {meta.expected_format.strip()}")
    lines.append("")

    if kind == "DATA":
        d = inst.defaults() if hasattr(inst, "defaults") else {}
        if d:
            lines.append("Automatic defaults:")
            for k in sorted(d.keys()):
                lines.append(f"  - {k}: {d[k]}")
            lines.append("")
    else:
        req = inst.requires() if hasattr(inst, "requires") else []
        if req:
            lines.append("Requires columns:")
            for c in req:
                lines.append(f"  - {c}")
            lines.append("")
        params = inst.parameters() if hasattr(inst, "parameters") else {}
        if params:
            lines.append("Parameters:")
            for k in sorted(params.keys()):
                lines.append(f"  - {k}: {params[k]}")
            lines.append("")

    if meta.tags:
        lines.append("Tags:")
        lines.append("  " + ", ".join(meta.tags))
        lines.append("")

    if meta.links:
        lines.append("Links:")
        for k, v in meta.links.items():
            lines.append(f"  - {k}: {v}")
        lines.append("")

    lines.append("Example usage:")
    lines.append("")
    lines.append(textwrap.indent(meta.example_usage.strip("\n"), "  "))

    return "\n".join(lines)


# ---------------------------------------------------------------------
# v0.2+ placeholder: entry-point discovery
# ---------------------------------------------------------------------

def discover_entrypoint_plugins() -> None:
    """
    Placeholder for future third-party plugin discovery.
    Intentionally not implemented in v0.1 to avoid packaging complexity.
    """
    return


 
__all__ = [
    # Metadata / base classes
    "PluginMetadata",
    "DataPlugin",
    "FeaturePlugin",
    # Registration / lookup
    "register_data_plugin",
    "register_feature_plugin",
    "get_data_plugin",
    "get_feature_plugin",
    # Discovery
    "list_plugins",
    "plugin_help",
    "list_data_plugins",
    "list_feature_plugins",
    # Future
    "discover_entrypoint_plugins",
]