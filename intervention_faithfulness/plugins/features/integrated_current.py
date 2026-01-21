"""
plugins/features/integrated_current.py — Feature plugin: integrated current history proxy (v0.1)

This feature plugin adds a history variable intended to capture "thermal / winding / inertia"
effects that depend on recent drive history.

Key constraint:
- Many real datasets are "one row per trial" (no within-trial time series).
  In that case, true ∫I(t)dt over nanoseconds is not directly computable.

Therefore this plugin supports TWO modes:

1) sequence_mode="within_trial"
   - Requires a per-trial current time series stored as an array-like object.
   - Computes a true integral over the last window_ns.

2) sequence_mode="across_trials" (default)
   - Works on one-row-per-trial data.
   - Computes an exponentially-weighted moving average (EWMA) proxy across trials
     within each intervention/regime grouping (or globally).
   - This represents "memory of recent pulses / dissipated drive", which is often
     the practical quantity available.

Outputs:
- Adds exactly one column:
    history_integrated_current

Column naming is stable and intentionally generic.

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
class IntegratedCurrentFeature(FeaturePlugin):
    metadata = PluginMetadata(
        name="integrated_current",
        description="Integrated-current history proxy (within-trial integral if available; otherwise EWMA across trials).",
        expected_format=(
            "Requires at least one of:\n"
            "  - state_current (recommended) or state_I\n"
            "Optionally supports within-trial series mode if you provide:\n"
            "  - a column containing array-like current traces per trial (e.g., current_trace)\n"
            "and matching timebase info (dt_ns or time_trace)."
        ),
        example_usage=(
            "from intervention_faithfulness import FaithfulnessTest\n"
            "\n"
            "test = FaithfulnessTest.from_plugin('nanowire_switching', 'data.csv')\n"
            "test.add_feature('integrated_current', window_ns=50)\n"
            "results = test.diagnose()\n"
        ),
        tags=["history", "thermal", "switching", "proxy"],
    )

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    def requires(self) -> List[str]:
        # We support either state_current or state_I, but requires() must be explicit.
        # Core will check these; we implement a soft selection in compute().
        return []

    def parameters(self) -> Dict[str, Any]:
        return {
            "window_ns": {
                "type": "float",
                "default": 50.0,
                "description": "History window in ns (within-trial mode) or time constant proxy (across-trials mode).",
            },
            "sequence_mode": {
                "type": "str",
                "default": "across_trials",
                "choices": ["across_trials", "within_trial"],
                "description": "How to compute the feature depending on data availability.",
            },
            "current_col": {
                "type": "str",
                "default": "auto",
                "description": "State current column name: 'auto' prefers state_current then state_I.",
            },
            "groupby": {
                "type": "list[str] | None",
                "default": None,
                "description": "Optional grouping keys for across-trials EWMA (e.g., ['regime_device', 'intervention_id']).",
            },
            "trace_col": {
                "type": "str",
                "default": "current_trace",
                "description": "Within-trial mode: column name holding array-like current traces per trial.",
            },
            "dt_ns": {
                "type": "float | None",
                "default": None,
                "description": "Within-trial mode: sampling interval in ns (if no explicit time axis provided).",
            },
            "time_col": {
                "type": "str | None",
                "default": None,
                "description": "Within-trial mode: optional column holding array-like time samples per trial (ns).",
            },
        }

    def compute(
        self,
        trials_df: pd.DataFrame,
        *,
        window_ns: float = 50.0,
        sequence_mode: str = "across_trials",
        current_col: str = "auto",
        groupby: Optional[List[str]] = None,
        trace_col: str = "current_trace",
        dt_ns: Optional[float] = None,
        time_col: Optional[str] = None,
        output_col: str = "history_integrated_current",
        **kwargs,
    ) -> pd.DataFrame:
        df = trials_df.copy()

        if output_col in df.columns:
            raise ValueError(
                f"Column '{output_col}' already exists. Refusing to overwrite."
            )

        if sequence_mode not in ("across_trials", "within_trial"):
            raise ValueError(f"Unsupported sequence_mode: {sequence_mode}")

        if sequence_mode == "within_trial":
            df[output_col] = self._compute_within_trial(
                df,
                window_ns=window_ns,
                trace_col=trace_col,
                dt_ns=dt_ns,
                time_col=time_col,
            )
            return df

        # Default: across-trials EWMA proxy
        col = self._select_current_col(df, current_col=current_col)

        # Choose grouping keys: if user doesn't specify, keep it simple and global
        if groupby is None:
            # Sensible default: keep regimes separate if present; do not over-segment.
            groupby = []
            if "regime_device" in df.columns:
                groupby.append("regime_device")
            if "intervention_id" in df.columns:
                groupby.append("intervention_id")
            if not groupby:
                groupby = None

        df[output_col] = self._compute_across_trials_ewma(
            df,
            current_state_col=col,
            window_ns=window_ns,
            groupby=groupby,
        )

        return df

    # ------------------------------------------------------------------
    # Mode 1: within-trial integral
    # ------------------------------------------------------------------

    def _compute_within_trial(
        self,
        df: pd.DataFrame,
        *,
        window_ns: float,
        trace_col: str,
        dt_ns: Optional[float],
        time_col: Optional[str],
    ) -> pd.Series:
        if trace_col not in df.columns:
            raise ValueError(
                f"within_trial mode requires column '{trace_col}' containing array-like current traces."
            )

        # Compute last-window integral for each trace
        out = []
        for idx, trace in enumerate(df[trace_col].tolist()):
            if trace is None:
                out.append(np.nan)
                continue

            x = np.asarray(trace, dtype=float)
            if x.ndim != 1 or x.size == 0:
                out.append(np.nan)
                continue

            if time_col is not None:
                if time_col not in df.columns:
                    raise ValueError(
                        f"time_col='{time_col}' requested but column is missing."
                    )
                t = df[time_col].iloc[idx]
                if t is None:
                    out.append(np.nan)
                    continue
                t_arr = np.asarray(t, dtype=float)
                if t_arr.shape != x.shape:
                    out.append(np.nan)
                    continue
                # Use the last window_ns segment
                t_end = float(t_arr[-1])
                mask = t_arr >= (t_end - float(window_ns))
                if mask.sum() < 2:
                    out.append(np.nan)
                    continue
                out.append(float(np.trapz(x[mask], t_arr[mask])))
            else:
                if dt_ns is None:
                    raise ValueError(
                        "within_trial mode requires either time_col (per-trial time axis) or dt_ns (sampling interval)."
                    )
                n = int(np.ceil(float(window_ns) / float(dt_ns)))
                n = max(1, min(n, x.size))
                segment = x[-n:]
                # Approx integral: sum * dt
                out.append(float(np.sum(segment) * float(dt_ns)))

        return pd.Series(out, index=df.index)

    # ------------------------------------------------------------------
    # Mode 2: across-trials EWMA proxy
    # ------------------------------------------------------------------

    def _compute_across_trials_ewma(
        self,
        df: pd.DataFrame,
        *,
        current_state_col: str,
        window_ns: float,
        groupby: Optional[List[str]],
    ) -> pd.Series:
        """
        Compute an EWMA proxy across trials.

        Interpretation:
        - This is not a literal nanosecond integral unless trials are time-ordered
          and window_ns is interpreted as a *memory time constant*.
        - In practice, it behaves as a low-dimensional "recent drive memory" feature.

        We treat window_ns as a time constant; translate to alpha via:
            alpha = 1 - exp(-1 / tau)
        where tau is window_ns in "trial steps".

        If you have real timestamps, a future version can use time-delta-aware EWMA.
        """
        x = pd.to_numeric(df[current_state_col], errors="coerce")

        # Convert tau to alpha in a stable way; interpret tau in trial steps
        tau = max(1.0, float(window_ns))
        alpha = 1.0 - np.exp(-1.0 / tau)

        if groupby is None:
            return x.ewm(alpha=alpha, adjust=False).mean()

        # Ensure groupby keys exist
        for g in groupby:
            if g not in df.columns:
                raise ValueError(f"groupby key '{g}' not found in trials_df")

        # Apply EWMA within each group
        def _ewma_group(s: pd.Series) -> pd.Series:
            return s.ewm(alpha=alpha, adjust=False).mean()

        return x.groupby([df[g] for g in groupby], sort=False).transform(_ewma_group)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _select_current_col(self, df: pd.DataFrame, *, current_col: str) -> str:
        if current_col != "auto":
            if current_col not in df.columns:
                raise ValueError(f"Requested current_col='{current_col}' not found in trials_df.")
            return current_col

        # Auto-detect common names
        preferred = ["state_current", "state_I", "state_i", "state_Current"]
        for c in preferred:
            if c in df.columns:
                return c

        # Fallback: any state_* that looks like current
        for c in df.columns:
            if c.startswith("state_") and "curr" in c.lower():
                return c

        raise ValueError(
            "Could not auto-detect a current state column. "
            "Provide current_col explicitly (e.g., current_col='state_I')."
        )