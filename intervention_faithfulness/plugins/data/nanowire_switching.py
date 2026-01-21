"""
plugins/data/nanowire_switching.py — Data plugin: superconducting switching trials (v0.1)

Purpose:
- Load common nanowire / Josephson switching trial datasets (CSV or HDF5)
- Convert to the canonical trials table:
    - trial_id
    - intervention_id
    - outcome
    - state_* columns

This plugin is intentionally minimal and conservative.
It does NOT assume microscopic mechanisms (vortices/hotspots/etc.).
It only standardizes data + defaults for the diagnostic.

Expected raw formats (typical):
- CSV with columns:
    trial_id, current, voltage, ramp_rate, time_to_switch
- HDF5 containing a tabular dataset with equivalent columns

You may override column names via kwargs.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, List
import pandas as pd

from intervention_faithfulness.plugins.registry import (
    DataPlugin,
    PluginMetadata,
    register_data_plugin,
)


@register_data_plugin
class NanowireSwitchingPlugin(DataPlugin):
    metadata = PluginMetadata(
        name="nanowire_switching",
        description="Superconducting nanowire / JJ switching experiments (trials under ramp/pulse interventions).",
        expected_format=(
            "CSV/HDF5 tabular data with repeated trials.\n"
            "Minimum columns (default names):\n"
            "  - trial_id\n"
            "  - ramp_rate (or pulse_id)  -> intervention\n"
            "  - time_to_switch (or I_sw) -> outcome\n"
            "Optional state columns:\n"
            "  - current, voltage, temperature, flux_bias, etc."
        ),
        example_usage=(
            "from intervention_faithfulness import FaithfulnessTest\n"
            "\n"
            "test = FaithfulnessTest.from_plugin(\n"
            "    'nanowire_switching',\n"
            "    'data/switching_runs.csv'\n"
            ")\n"
            "results = test.diagnose()\n"
            "results.export_certificate('model_validity.pdf')\n"
        ),
        tags=["superconducting", "nanowire", "josephson", "switching", "intervention"],
    )

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self, source: Any, **kwargs) -> pd.DataFrame:
        """
        Load raw data from CSV or HDF5.

        Parameters
        ----------
        source:
            Path to .csv or .h5/.hdf5, or an existing pandas DataFrame.
        kwargs:
            Passed through to pandas loaders where relevant.
            For HDF5, you may pass key=... or other read_hdf args.

        Returns
        -------
        pd.DataFrame
            Raw table.
        """
        if isinstance(source, pd.DataFrame):
            return source.copy()

        if not isinstance(source, str):
            raise ValueError(f"Unsupported source type: {type(source)} (expected path or DataFrame).")

        path = source.lower().strip()
        if path.endswith(".csv"):
            return pd.read_csv(source, **kwargs)

        if path.endswith(".h5") or path.endswith(".hdf5") or path.endswith(".hdf"):
            # kwargs may include key=...
            return pd.read_hdf(source, **kwargs)

        raise ValueError(f"Unsupported file format for source: {source}")

    # ------------------------------------------------------------------
    # Conversion to canonical trials table
    # ------------------------------------------------------------------

    def to_trials(
        self,
        raw: pd.DataFrame,
        *,
        trial_id_col: str = "trial_id",
        intervention_col: str = "ramp_rate",
        outcome_col: str = "time_to_switch",
        state_vars: Optional[List[str]] = None,
        state_prefix: str = "state_",
        coerce_numeric_outcome: bool = True,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Convert raw switching data into the canonical trials table.

        Parameters
        ----------
        raw:
            Raw DataFrame as loaded by load().
        trial_id_col:
            Column name for trial identifier.
        intervention_col:
            Column name for intervention identifier (ramp_rate, pulse_id, etc.).
        outcome_col:
            Column name for the measured outcome (time_to_switch, I_sw, etc.).
        state_vars:
            List of raw columns to include as state variables.
            If None, uses a conservative default intersection of known columns.
        state_prefix:
            Prefix for canonical state columns.
        coerce_numeric_outcome:
            If True, attempts to coerce outcome to numeric when possible.

        Returns
        -------
        pd.DataFrame
            Canonical trials table.
        """
        self._require_columns(raw, [trial_id_col, intervention_col, outcome_col])

        if state_vars is None:
            # Conservative default: include common measurement columns if present.
            candidates = ["current", "voltage", "temperature", "flux_bias", "field"]
            state_vars = [c for c in candidates if c in raw.columns]

        trials = pd.DataFrame()
        trials["trial_id"] = raw[trial_id_col]
        trials["intervention_id"] = raw[intervention_col]
        trials["outcome"] = raw[outcome_col]

        if coerce_numeric_outcome:
            trials["outcome"] = pd.to_numeric(trials["outcome"], errors="ignore")

        # Add state_* columns
        for col in state_vars:
            trials[f"{state_prefix}{col}"] = raw[col]

        # Optional timestamp passthrough if present and not already mapped
        if "timestamp" in raw.columns and "timestamp" not in trials.columns:
            trials["timestamp"] = raw["timestamp"]

        # Optional regime passthrough: device id, temperature setpoint, etc.
        # Users may override by pre-labeling regime_* columns in raw.
        for c in raw.columns:
            if c.startswith("regime_") and c not in trials.columns:
                trials[c] = raw[c]

        return trials

    # ------------------------------------------------------------------
    # Defaults tuned for switching / reliability use-cases
    # ------------------------------------------------------------------

    def defaults(self) -> Dict[str, Any]:
        """
        Domain defaults for superconducting switching data.

        Notes:
        - Switching reliability questions often depend on distribution tails.
        - Jensen–Shannon is a safe default divergence for distributions.
        """
        return {
            "divergence": "js",
            "min_samples": 50,
            "tail_mode": True,
            "quantile_focus": 0.95,
        }

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, df: pd.DataFrame) -> List[str]:
        """
        Domain-specific warnings (non-fatal). The core schema validator is separate.

        Returns list of warning strings.
        """
        warns: List[str] = []

        # Heuristic: if outcome looks categorical with few unique values, warn
        if "outcome" in df.columns:
            try:
                nunique = int(df["outcome"].nunique(dropna=True))
                if nunique <= 3:
                    warns.append(
                        "Outcome has very few unique values; if this is a binary event, "
                        "consider whether a time-to-event or continuous metric is available."
                    )
            except Exception:
                pass

        # If no state columns, warn (allowed, but often not intended)
        state_cols = [c for c in df.columns if c.startswith("state_")]
        if not state_cols:
            warns.append(
                "No state_* columns detected. This is allowed, but switching analyses typically "
                "include at least one measured state (e.g., current or voltage)."
            )

        return warns

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _require_columns(df: pd.DataFrame, cols: List[str]) -> None:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in raw data: {missing}")