"""
core/schema.py — Canonical trials table schema + validation (v0.1)

Defines the canonical table contract used by the core diagnostic.

Canonical trials table (minimum):
- trial_id (hashable)
- intervention_id (hashable; may be numeric or categorical)
- outcome (numeric recommended; categorical allowed for some divergences in future)

Optional:
- state_* columns (instantaneous reduced state variables)
- history_* columns (candidate completion / memory variables)
- timestamp (optional; numeric seconds or datetime-like)

Validation policy:
- Raise on missing required columns
- Return non-fatal warnings for common pitfalls (low samples, non-numeric outcomes, etc.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SchemaWarning:
    code: str
    message: str


REQUIRED_COLUMNS = ["trial_id", "intervention_id", "outcome"]


def validate_trials_table(df: pd.DataFrame) -> List[SchemaWarning]:
    """
    Validate a canonical trials table.

    Raises
    ------
    ValueError
        If required columns are missing or df is not a DataFrame.

    Returns
    -------
    List[SchemaWarning]
        Non-fatal warnings. Empty list means "looks good."
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"trials_df must be a pandas DataFrame, got {type(df)}")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    warns: List[SchemaWarning] = []

    # trial_id sanity
    if df["trial_id"].isna().any():
        warns.append(SchemaWarning("trial_id_nan", "trial_id contains NaN values; grouping may be unstable."))

    # intervention_id sanity
    if df["intervention_id"].isna().any():
        warns.append(
            SchemaWarning(
                "intervention_id_nan",
                "intervention_id contains NaN values; consider labeling interventions explicitly.",
            )
        )

    # outcome sanity
    out = df["outcome"]
    if out.isna().any():
        warns.append(SchemaWarning("outcome_nan", "outcome contains NaN values; these rows will be dropped in scoring."))

    # Numeric outcome recommendation
    if not pd.api.types.is_numeric_dtype(out):
        coerced = pd.to_numeric(out, errors="coerce")
        frac_numeric = float(np.isfinite(coerced).mean()) if len(coerced) else 0.0
        if frac_numeric < 0.9:
            warns.append(
                SchemaWarning(
                    "outcome_non_numeric",
                    "outcome is not numeric (or mostly non-numeric). v0.1 fracture scoring assumes numeric outcomes "
                    "for histogram/Wasserstein divergences. Consider providing a numeric outcome (e.g., time_to_switch).",
                )
            )

    # State/history columns presence
    state_cols = [c for c in df.columns if c.startswith("state_")]
    hist_cols = [c for c in df.columns if c.startswith("history_")]

    if not state_cols:
        warns.append(
            SchemaWarning(
                "no_state_cols",
                "No state_* columns found. This is allowed, but interpretation becomes intervention-only conditioning.",
            )
        )

    # Timestamp checks (optional)
    if "timestamp" in df.columns:
        ts = df["timestamp"]
        if pd.api.types.is_numeric_dtype(ts):
            if (pd.to_numeric(ts, errors="coerce").isna().mean() > 0.1):
                warns.append(
                    SchemaWarning(
                        "timestamp_parse",
                        "timestamp is numeric but many values fail numeric coercion; check dtype/content.",
                    )
                )
        else:
            dt = pd.to_datetime(ts, errors="coerce", utc=True)
            if (dt.isna().mean() > 0.1):
                warns.append(
                    SchemaWarning(
                        "timestamp_parse",
                        "timestamp is present but many values fail datetime parsing; consider numeric seconds.",
                    )
                )

    # Heuristic sample-size warnings (global)
    n = len(df)
    if n < 200:
        warns.append(
            SchemaWarning(
                "low_total_samples",
                f"Only {n} total rows. Many fracture estimates will be underpowered; "
                "consider collecting more trials or reducing binning resolution.",
            )
        )

    # Distribution per intervention
    try:
        counts = df.groupby("intervention_id")["trial_id"].count()
        if (counts < 50).any():
            warns.append(
                SchemaWarning(
                    "low_samples_per_intervention",
                    "Some intervention_id groups have <50 samples. Consider pooling interventions or collecting more data.",
                )
            )
    except Exception:
        pass

    # Many-to-one collapse suspicion: identical states with many unique outcomes may be fine; warn only if extreme.
    if state_cols:
        try:
            key = df[state_cols].astype(str).agg("|".join, axis=1)
            avg_group = float(df.groupby(key)["trial_id"].count().mean())
            if avg_group < 5.0:
                warns.append(
                    SchemaWarning(
                        "high_state_cardinality",
                        "state_* appears high-cardinality (few repeats per state). Consider discretizing/rounding "
                        "state variables for fracture estimation.",
                    )
                )
        except Exception:
            pass

    # history_* columns exist but are all-NaN
    if hist_cols:
        all_nan = [c for c in hist_cols if df[c].isna().all()]
        if all_nan:
            warns.append(
                SchemaWarning(
                    "history_all_nan",
                    f"Some history_* columns are entirely NaN: {all_nan}. These features cannot improve fracture.",
                )
            )

    return warns