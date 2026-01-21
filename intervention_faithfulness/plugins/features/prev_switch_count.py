"""
plugins/features/prev_switch_count.py — Feature plugin: previous switching event count (v0.1)

Adds:
    history_prev_switch_count

Intended use:
- Switching systems where recovery / quasiparticle / hotspot dynamics create
  multi-event correlations.
- Works on one-row-per-trial datasets (across-trials history).

Assumption:
- Each row represents a trial, and trial ordering in the table corresponds
  to temporal ordering within each group (device/regime/intervention).
  If you have a timestamp column, you should sort upstream before running,
  or we can add time-aware sorting in a future version.

"""

from __future__ import annotations

from typing import Dict, Any, List, Optional
import pandas as pd

from intervention_faithfulness.plugins.registry import (
    FeaturePlugin,
    PluginMetadata,
    register_feature_plugin,
)


@register_feature_plugin
class PrevSwitchCountFeature(FeaturePlugin):
    metadata = PluginMetadata(
        name="prev_switch_count",
        description="Counts recent switching events across trials (multi-event correlation proxy).",
        expected_format=(
            "Requires either:\n"
            "  - an explicit binary switch indicator column (default: outcome_is_switch), or\n"
            "  - a categorical outcome where switched/not can be derived, or\n"
            "  - user supplies outcome_to_switch_fn.\n"
            "Adds: history_prev_switch_count"
        ),
        example_usage=(
            "test.add_feature('prev_switch_count', window=10)\n"
        ),
        tags=["history", "switching", "memory", "events"],
    )

    def parameters(self) -> Dict[str, Any]:
        return {
            "window": {
                "type": "int",
                "default": 10,
                "description": "How many previous trials to include in the count.",
            },
            "groupby": {
                "type": "list[str] | None",
                "default": None,
                "description": "Optional grouping keys (e.g., ['regime_device']). "
                               "Counts are computed within each group.",
            },
            "switch_indicator_col": {
                "type": "str",
                "default": "outcome_is_switch",
                "description": "Optional existing boolean column indicating a switching event.",
            },
            "derive_from_outcome": {
                "type": "bool",
                "default": True,
                "description": "If True, attempt to derive a switch indicator from outcome if switch_indicator_col missing.",
            },
            "output_col": {
                "type": "str",
                "default": "history_prev_switch_count",
                "description": "Output column name.",
            },
        }

    def compute(
        self,
        trials_df: pd.DataFrame,
        *,
        window: int = 10,
        groupby: Optional[List[str]] = None,
        switch_indicator_col: str = "outcome_is_switch",
        derive_from_outcome: bool = True,
        output_col: str = "history_prev_switch_count",
        **kwargs,
    ) -> pd.DataFrame:
        df = trials_df.copy()

        if output_col in df.columns:
            raise ValueError(f"Column '{output_col}' already exists. Refusing to overwrite.")

        if window < 1:
            raise ValueError("window must be >= 1")

        # Determine switch indicator series
        if switch_indicator_col in df.columns:
            sw = df[switch_indicator_col].astype(bool)
        else:
            if not derive_from_outcome:
                raise ValueError(
                    f"switch_indicator_col='{switch_indicator_col}' not found, "
                    "and derive_from_outcome=False."
                )
            sw = self._derive_switch_indicator(df)

        sw = sw.fillna(False).astype(int)

        # Choose grouping if not provided
        if groupby is None:
            groupby = []
            if "regime_device" in df.columns:
                groupby.append("regime_device")
            if not groupby:
                groupby = None

        # Rolling count of previous switches (exclude current trial)
        if groupby is None:
            df[output_col] = (
                sw.shift(1)
                .rolling(window=window, min_periods=1)
                .sum()
                .astype(float)
            )
            return df

        for g in groupby:
            if g not in df.columns:
                raise ValueError(f"groupby key '{g}' not found in trials_df")

        def _group_roll(s: pd.Series) -> pd.Series:
            return (
                s.shift(1)
                .rolling(window=window, min_periods=1)
                .sum()
            )

        df[output_col] = sw.groupby([df[g] for g in groupby], sort=False).transform(_group_roll).astype(float)
        return df

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _derive_switch_indicator(self, df: pd.DataFrame) -> pd.Series:
        """
        Heuristic derivation:
        - If outcome is boolean: use it.
        - If outcome is categorical string: look for tokens indicating switch.
        - Otherwise: cannot infer; returns all False with a warning-like behavior.

        Users with numeric outcomes (time_to_switch) typically need an explicit
        success/failure indicator if "switch" is not guaranteed per trial.
        """
        out = df["outcome"]

        # Boolean outcomes
        if out.dtype == bool:
            return out

        # String categorical outcomes
        if out.dtype == object:
            def _is_switch(x: Any) -> bool:
                if x is None:
                    return False
                s = str(x).strip().lower()
                if s in ("switch", "switched", "1", "true", "yes"):
                    return True
                if s in ("no_switch", "noswitch", "0", "false", "no"):
                    return False
                # Token contains switch?
                return ("switch" in s) and ("no" not in s)
            return out.map(_is_switch)

        # Numeric outcomes: cannot infer switch vs no-switch from time-to-switch alone
        # Return False and let users provide an explicit indicator column.
        return pd.Series([False] * len(df), index=df.index)