"""
plugins/features/ewma_dissipation.py — Feature plugin: EWMA power/energy dissipation proxy (v0.1)

Adds:
    history_ewma_dissipation

Motivation:
- In many superconducting and switching systems, the relevant "memory" variable is
  not raw current but recent *dissipation* (I*V) or energy deposited.
- Even with one-row-per-trial data, this can capture recovery / heating / poisoning effects.

Computation:
- Compute per-trial dissipation proxy:
      diss = (current_col * voltage_col)
  or user may supply a dissipation_col directly.
- Compute EWMA across trials:
      ewma(diss)

Grouping:
- Optional groupby columns (device, intervention, etc.). Default keeps device separate if present.

"""

from __future__ import annotations

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from intervention_faithfulness.plugins.registry import (
    FeaturePlugin,
    PluginMetadata,
    register_feature_plugin,
)


@register_feature_plugin
class EWMADissipationFeature(FeaturePlugin):
    metadata = PluginMetadata(
        name="ewma_dissipation",
        description="Exponentially-weighted recent dissipation proxy (I*V) across trials.",
        expected_format=(
            "Requires either:\n"
            "  - dissipation_col (explicit), or\n"
            "  - both a current column and a voltage column (state_current/state_I and state_voltage/state_V).\n"
            "Adds: history_ewma_dissipation"
        ),
        example_usage=(
            "test.add_feature('ewma_dissipation', tau=25)\n"
        ),
        tags=["history", "thermal", "dissipation", "ewma"],
    )

    def parameters(self) -> Dict[str, Any]:
        return {
            "tau": {
                "type": "float",
                "default": 25.0,
                "description": "EWMA time constant in trial steps (larger = longer memory).",
            },
            "alpha": {
                "type": "float | None",
                "default": None,
                "description": "Direct EWMA alpha override. If provided, tau is ignored.",
            },
            "dissipation_col": {
                "type": "str | None",
                "default": None,
                "description": "If provided, uses this column directly as dissipation proxy.",
            },
            "current_col": {
                "type": "str",
                "default": "auto",
                "description": "Current column to use if dissipation_col not provided. 'auto' prefers state_current then state_I.",
            },
            "voltage_col": {
                "type": "str",
                "default": "auto",
                "description": "Voltage column to use if dissipation_col not provided. 'auto' prefers state_voltage then state_V.",
            },
            "groupby": {
                "type": "list[str] | None",
                "default": None,
                "description": "Optional grouping keys for EWMA (e.g., ['regime_device']).",
            },
            "output_col": {
                "type": "str",
                "default": "history_ewma_dissipation",
                "description": "Output column name.",
            },
        }

    def compute(
        self,
        trials_df: pd.DataFrame,
        *,
        tau: float = 25.0,
        alpha: Optional[float] = None,
        dissipation_col: Optional[str] = None,
        current_col: str = "auto",
        voltage_col: str = "auto",
        groupby: Optional[List[str]] = None,
        output_col: str = "history_ewma_dissipation",
        **kwargs,
    ) -> pd.DataFrame:
        df = trials_df.copy()

        if output_col in df.columns:
            raise ValueError(f"Column '{output_col}' already exists. Refusing to overwrite.")

        # Determine dissipation proxy
        if dissipation_col is not None:
            if dissipation_col not in df.columns:
                raise ValueError(f"dissipation_col='{dissipation_col}' not found in trials_df.")
            diss = pd.to_numeric(df[dissipation_col], errors="coerce")
        else:
            i_col = self._select_current_col(df, current_col=current_col)
            v_col = self._select_voltage_col(df, voltage_col=voltage_col)
            i = pd.to_numeric(df[i_col], errors="coerce")
            v = pd.to_numeric(df[v_col], errors="coerce")
            diss = i * v

        # Resolve alpha
        a = self._resolve_alpha(alpha=alpha, tau=tau)

        # Choose grouping
        if groupby is None:
            groupby = []
            if "regime_device" in df.columns:
                groupby.append("regime_device")
            if not groupby:
                groupby = None
        else:
            for g in groupby:
                if g not in df.columns:
                    raise ValueError(f"groupby key '{g}' not found in trials_df")

        if groupby is None:
            df[output_col] = diss.ewm(alpha=a, adjust=False).mean()
            return df

        def _ewma_group(s: pd.Series) -> pd.Series:
            return s.ewm(alpha=a, adjust=False).mean()

        df[output_col] = diss.groupby([df[g] for g in groupby], sort=False).transform(_ewma_group)
        return df

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _resolve_alpha(self, *, alpha: Optional[float], tau: float) -> float:
        if alpha is not None:
            a = float(alpha)
            if not (0.0 < a <= 1.0):
                raise ValueError("alpha must be in (0, 1].")
            return a
        t = max(1.0, float(tau))
        # alpha from exponential time constant (trial steps)
        return 1.0 - np.exp(-1.0 / t)

    def _select_current_col(self, df: pd.DataFrame, *, current_col: str) -> str:
        if current_col != "auto":
            if current_col not in df.columns:
                raise ValueError(f"Requested current_col='{current_col}' not found.")
            return current_col
        for c in ["state_current", "state_I", "state_i"]:
            if c in df.columns:
                return c
        for c in df.columns:
            if c.startswith("state_") and "curr" in c.lower():
                return c
        raise ValueError(
            "Could not auto-detect a current column. Provide current_col explicitly."
        )

    def _select_voltage_col(self, df: pd.DataFrame, *, voltage_col: str) -> str:
        if voltage_col != "auto":
            if voltage_col not in df.columns:
                raise ValueError(f"Requested voltage_col='{voltage_col}' not found.")
            return voltage_col
        for c in ["state_voltage", "state_V", "state_v"]:
            if c in df.columns:
                return c
        for c in df.columns:
            if c.startswith("state_") and ("volt" in c.lower() or c.lower().endswith("_v")):
                return c
        raise ValueError(
            "Could not auto-detect a voltage column. Provide voltage_col explicitly."
        )