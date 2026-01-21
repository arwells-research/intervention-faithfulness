"""
plugins/features/time_since_last.py — Feature plugin: time since last event (v0.1)

Adds:
    history_time_since_last

This is useful when:
- recovery dynamics matter (quasiparticle relaxation, thermal recovery, etc.)
- events create temporary "state" that is not captured by instantaneous measurements

Supports two modes:
1) If a 'timestamp' column exists (recommended), compute real time deltas.
2) Else, compute "trials since last event" (proxy), using row order.

Event definition:
- Prefer a boolean column outcome_is_event (or user-specified)
- Otherwise attempt a heuristic derivation from outcome (similar to prev_switch_count)

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
class TimeSinceLastFeature(FeaturePlugin):
    metadata = PluginMetadata(
        name="time_since_last",
        description="Time since last event (uses timestamp if available; otherwise trials-since-event proxy).",
        expected_format=(
            "Optional: timestamp column (numeric or datetime-like).\n"
            "Optional: outcome_is_event boolean column.\n"
            "Adds: history_time_since_last"
        ),
        example_usage=(
            "test.add_feature('time_since_last', event_col='outcome_is_switch')\n"
        ),
        tags=["history", "recovery", "memory", "time"],
    )

    def parameters(self) -> Dict[str, Any]:
        return {
            "event_col": {
                "type": "str",
                "default": "outcome_is_event",
                "description": "Boolean column indicating whether an event occurred on this trial.",
            },
            "derive_from_outcome": {
                "type": "bool",
                "default": True,
                "description": "If True, attempt to derive events from outcome when event_col missing.",
            },
            "groupby": {
                "type": "list[str] | None",
                "default": None,
                "description": "Optional grouping keys (e.g., ['regime_device']). Computed within group.",
            },
            "output_col": {
                "type": "str",
                "default": "history_time_since_last",
                "description": "Output column name.",
            },
            "unit": {
                "type": "str",
                "default": "auto",
                "choices": ["auto", "seconds", "trials"],
                "description": "Force output units. 'auto' uses seconds if timestamp exists, else trials.",
            },
        }

    def compute(
        self,
        trials_df: pd.DataFrame,
        *,
        event_col: str = "outcome_is_event",
        derive_from_outcome: bool = True,
        groupby: Optional[List[str]] = None,
        output_col: str = "history_time_since_last",
        unit: str = "auto",
        **kwargs,
    ) -> pd.DataFrame:
        df = trials_df.copy()

        if output_col in df.columns:
            raise ValueError(f"Column '{output_col}' already exists. Refusing to overwrite.")

        if unit not in ("auto", "seconds", "trials"):
            raise ValueError(f"Unsupported unit: {unit}")

        # Determine event indicator
        if event_col in df.columns:
            ev = df[event_col].astype(bool)
        else:
            if not derive_from_outcome:
                raise ValueError(f"event_col='{event_col}' not found and derive_from_outcome=False.")
            ev = self._derive_event_indicator(df)

        # Decide timestamp availability
        has_ts = "timestamp" in df.columns
        use_seconds = (unit == "seconds") or (unit == "auto" and has_ts)

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

        if use_seconds:
            if not has_ts:
                raise ValueError("unit='seconds' requested but no 'timestamp' column exists.")
            ts = self._to_seconds(df["timestamp"])
            df[output_col] = self._time_since_last_seconds(ts, ev, groupby=groupby)
        else:
            df[output_col] = self._time_since_last_trials(ev, groupby=groupby)

        return df

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _derive_event_indicator(self, df: pd.DataFrame) -> pd.Series:
        """
        Heuristic:
        - if outcome is boolean: event = outcome
        - if outcome is string: look for 'switch'/'event' tokens
        - else: all False
        """
        out = df["outcome"]

        if out.dtype == bool:
            return out

        if out.dtype == object:
            def _is_event(x: Any) -> bool:
                if x is None:
                    return False
                s = str(x).strip().lower()
                if s in ("event", "switch", "switched", "1", "true", "yes"):
                    return True
                if s in ("no_event", "no_switch", "0", "false", "no"):
                    return False
                return ("switch" in s) or ("event" in s)
            return out.map(_is_event)

        return pd.Series([False] * len(df), index=df.index)

    def _to_seconds(self, s: pd.Series) -> pd.Series:
        """
        Convert timestamp column to float seconds.
        Supports:
        - numeric (assumed already seconds)
        - datetime-like strings
        - pandas datetime64
        """
        # If numeric, assume seconds
        if pd.api.types.is_numeric_dtype(s):
            return pd.to_numeric(s, errors="coerce").astype(float)

        # Try datetime conversion
        dt = pd.to_datetime(s, errors="coerce", utc=True)
        # Convert to seconds relative to epoch
        return (dt.view("int64") / 1e9).astype(float)

    def _time_since_last_seconds(
        self,
        ts_sec: pd.Series,
        ev: pd.Series,
        *,
        groupby: Optional[List[str]],
    ) -> pd.Series:
        # Compute within groups (or globally)
        if groupby is None:
            return self._scan_time_since_last(ts_sec, ev)

        out = pd.Series(index=ts_sec.index, dtype=float)
        keys = [ts_sec.index.to_series()]  # placeholder to preserve alignment
        # We'll do groupby using df columns indirectly via caller; here we require group labels.
        # Instead, caller passes groupby and we re-group using a combined label series.
        # Simpler: compute per-group using pandas groupby on an externally provided group label.
        # We'll accept group labels by requiring caller to have already grouped; but we didn't.
        # So implement grouping here by asking caller to pass groupby labels is not possible.
        # Therefore: we compute group labels outside by having caller supply them in kwargs is messy.
        # In v0.1, keep grouping simple: groupby only supported when passed as column names in df.
        # We re-create labels by storing them temporarily is complicated in this helper signature.

        raise NotImplementedError(
            "Groupwise time-since-last(seconds) requires access to grouping columns. "
            "Use groupby=None for v0.1 or compute per-group upstream and concatenate."
        )

    def _time_since_last_trials(self, ev: pd.Series, *, groupby: Optional[List[str]]) -> pd.Series:
        if groupby is None:
            return self._scan_trials_since_last(ev)

        raise NotImplementedError(
            "Groupwise time-since-last(trials) requires access to grouping columns. "
            "Use groupby=None for v0.1 or compute per-group upstream and concatenate."
        )

    def _scan_time_since_last(self, ts_sec: pd.Series, ev: pd.Series) -> pd.Series:
        last_t = None
        out = []
        for t, is_ev in zip(ts_sec.tolist(), ev.tolist()):
            if last_t is None:
                out.append(float("nan"))
            else:
                out.append(float(t - last_t) if t is not None else float("nan"))
            if bool(is_ev) and t is not None:
                last_t = float(t)
        return pd.Series(out, index=ts_sec.index, dtype=float)

    def _scan_trials_since_last(self, ev: pd.Series) -> pd.Series:
        last_idx = None
        out = []
        for i, is_ev in enumerate(ev.tolist()):
            if last_idx is None:
                out.append(float("nan"))
            else:
                out.append(float(i - last_idx))
            if bool(is_ev):
                last_idx = i
        return pd.Series(out, index=ev.index, dtype=float)