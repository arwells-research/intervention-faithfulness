"""
core/fracture.py — Continuation fracture computation (v0.1)

This module implements the core diagnostic metric:
    continuation fracture

Conceptual role:
- Compare conditional outcome distributions under intervention
- Test invariance of distributions given reduced state vs full history
- Quantify representational failure, not dynamical error

This module is deliberately model-agnostic.
It returns both a scalar fracture score and a per-bin breakdown.

v0.1 scope:
- Jensen–Shannon divergence (default)
- Optional permutation testing
- Tail-restricted comparisons
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List, Literal
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance


@dataclass
class FractureResult:
    """
    Internal container for continuation fracture results.
    This object is adapted into user-facing structures by reporting.py.
    """

    fracture_value: float
    # If metric="both", store both components here (and fracture_value follows the chosen primary metric)
    metrics: Optional[Dict[str, float]] = None
    metric: Optional[str] = None    
    fracture_ci: Optional[Tuple[float, float]] = None
    ci_method: Optional[str] = None
    n_effective: Optional[int] = None
    p_value: Optional[float] = None
    n_permutations: Optional[int] = None
    recommended_features: Optional[List[Dict[str, Any]]] = None
    safe_regions: Optional[List[Dict[str, Any]]] = None
    breakdown: Optional[pd.DataFrame] = None
    warnings: Optional[List[str]] = None
    significance_warnings: Optional[List[str]] = None


# ---------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------

def compute_continuation_fracture(
    *,
    trials_df: pd.DataFrame,
    metric: Literal["refinement", "pairwise", "both"] = "refinement",
    divergence: str = "js",
    min_samples: int = 50,
    tail_mode: bool = False,
    quantile_focus: float = 0.95,
    n_bins: int = 30,
    n_pairwise_pairs: int = 50,
    n_permutations: int = 0,
    random_state: Optional[int] = None,
) -> FractureResult:
    """
    Compute continuation fracture for a canonical trials table.

    Parameters
    ----------
    trials_df:
        Canonical trials table.
    metric:
        'refinement' (current v0.1 behavior), 'pairwise' (history-vs-history),
        or 'both' (compute both; fracture_value follows refinement unless you choose otherwise).        
    divergence:
        'js' (Jensen–Shannon) or 'wasserstein'.
    min_samples:
        Minimum samples per conditional distribution.
    tail_mode:
        If True, restrict comparison to upper outcome quantiles.
    quantile_focus:
        Quantile threshold used when tail_mode=True.
    n_bins:
        Histogram bins for JS divergence.
    n_permutations:
        If >0, perform permutation testing.
    random_state:
        Optional RNG seed.

    Returns
    -------
    FractureResult
    """

    rng = np.random.default_rng(random_state)

    warnings: List[str] = []
    if metric not in ("refinement", "pairwise", "both"):
        raise ValueError(f"Unsupported metric: {metric}")

    # Extract columns
    outcome = pd.to_numeric(trials_df["outcome"], errors="coerce")
    intervention = trials_df["intervention_id"]

    state_cols = [c for c in trials_df.columns if c.startswith("state_")]
    history_cols = [
        c for c in trials_df.columns
        if isinstance(c, str) and (c.startswith("history_") or c == "history_h")
    ]

    if not state_cols:
        warnings.append("No state_* columns detected; fracture compares intervention-only conditioning.")

    # Build conditioning labels
    state_key = _row_key(trials_df, state_cols)
    history_key = _row_key(trials_df, state_cols + history_cols)

    # Add this guard:
    def _has_any_valid_state_bin() -> bool:
        for iv in intervention.unique():
            m = intervention == iv
            # group sizes inside this intervention
            vc = state_key[m].value_counts(dropna=False)
            if (vc >= min_samples).any():
                return True
        return False

    if state_cols and not _has_any_valid_state_bin():
        warnings.append(
            "All state_* bins are below min_samples; collapsing state conditioning (state_key='_'). "
            "Consider discretizing state_* columns or lowering min_samples."
        )
        state_key = pd.Series(["_"] * len(trials_df), index=trials_df.index)
        history_key = _row_key(trials_df, history_cols)  # keep history signal if present

    # Compute observed fracture + breakdown
    obs_ref, obs_pair, n_effective, breakdown = _compute_over_interventions(
        outcome, intervention, state_key, history_key,
        divergence, min_samples, tail_mode, quantile_focus, n_bins,
        n_pairwise_pairs=n_pairwise_pairs,
        rng=rng,
    )

    if np.isnan(obs_ref) and np.isnan(obs_pair):
        return FractureResult(
            fracture_value=float("nan"),
            warnings=["Insufficient data to compute fracture in any intervention regime."],
        )

    # Choose primary metric for fracture_value (keep v0.1 behavior by default)
    if metric == "refinement":
        fracture_value = float(obs_ref)
    elif metric == "pairwise":
        fracture_value = float(obs_pair)
    else:  # both
        # Default to refinement as primary unless you want to change policy later.
        fracture_value = float(obs_ref)

    # Optional permutation testing
    p_value = None
    if n_permutations > 0:
        perm_vals = []
        for _ in range(n_permutations):
            permuted_history = _permute_within_bins(
                history_key, intervention, state_key, rng
            )
            pr, pp, _, _ = _compute_over_interventions(
                outcome, intervention, state_key, permuted_history,
                divergence, min_samples, tail_mode, quantile_focus, n_bins,
                n_pairwise_pairs=n_pairwise_pairs,
                rng=rng,
            )

        if metric == "refinement":
            pv = pr
        elif metric == "pairwise":
            pv = pp
        else:
            pv = pr
        if not np.isnan(pv):
            perm_vals.append(float(pv))
        if perm_vals:
            p_value = float(np.mean(np.array(perm_vals) >= fracture_value))

    return FractureResult(
        fracture_value=fracture_value,
        metrics=({"refinement": float(obs_ref), "pairwise": float(obs_pair)} if metric == "both" else None),
        metric=str(metric),        
        n_effective=n_effective,
        p_value=p_value,
        n_permutations=n_permutations if n_permutations > 0 else None,
        warnings=warnings or None,
    )


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _compute_over_interventions(
    outcome: pd.Series,
    intervention: pd.Series,
    state_key: pd.Series,
    history_key: pd.Series,
    divergence: str,
    min_samples: int,
    tail_mode: bool,
    quantile_focus: float,
    n_bins: int,
    *,
    n_pairwise_pairs: int,
    rng: np.random.Generator,
) -> Tuple[float, float, int, pd.DataFrame]:
    """
    Compute mean refinement fracture and mean pairwise fracture over interventions and collect a breakdown table.
    """

    rows = []
    ref_vals: List[float] = []
    pair_vals: List[float] = []
    n_eff_total = 0

    for iv in intervention.unique():
        mask_iv = intervention == iv

        f_ref, f_pair, n_eff, per_state = _fracture_for_intervention(
            outcome[mask_iv],
            state_key[mask_iv],
            history_key[mask_iv],
            divergence=divergence,
            min_samples=min_samples,
            tail_mode=tail_mode,
            quantile_focus=quantile_focus,
            n_bins=n_bins,
            n_pairwise_pairs=n_pairwise_pairs,
            rng=rng,
        )

        if f_ref is None and f_pair is None:
            continue

        if f_ref is not None:
            ref_vals.append(float(f_ref))
        if f_pair is not None:
            pair_vals.append(float(f_pair))
        n_eff_total += n_eff

        for r in per_state:
            r["intervention_id"] = iv
            rows.append(r)

    if not ref_vals:
        ref_mean = float("nan")
    else:
        ref_mean = float(np.mean(ref_vals))

    if not pair_vals:
        pair_mean = float("nan")
    else:
        pair_mean = float(np.mean(pair_vals))

    if np.isnan(ref_mean) and np.isnan(pair_mean):
        return float("nan"), float("nan"), 0, pd.DataFrame()

    return ref_mean, pair_mean, int(n_eff_total), pd.DataFrame(rows)


def _fracture_for_intervention(
    outcome: pd.Series,
    state_key: pd.Series,
    history_key: pd.Series,
    *,
    divergence: str,
    min_samples: int,
    tail_mode: bool,
    quantile_focus: float,
    n_bins: int,
    n_pairwise_pairs: int,
    rng: np.random.Generator,
) -> Tuple[Optional[float], Optional[float], int, List[Dict[str, Any]]]:
    """
    Compute both refinement and pairwise fracture for a single intervention value.

    - Refinement fracture (v0.1): compare P(Y|S) vs P(Y|S,H_k) within each state bin.
    - Pairwise fracture: compare P(Y|S,H_k1) vs P(Y|S,H_k2) within each state bin.
    """

    ref_vals = []
    pair_vals = []
    n_eff = 0
    per_state_rows: List[Dict[str, Any]] = []

    # Group by reduced state
    for sk in state_key.unique():
        mask_s = state_key == sk
        y_s = outcome[mask_s]

        if len(y_s) < min_samples:
            continue

        # Collect valid history subgroups within this state
        groups: List[Tuple[str, np.ndarray]] = []
        for hk in history_key[mask_s].unique():
            mask_h = (history_key == hk)
            y_h = outcome[mask_s & mask_h].dropna().to_numpy()
            if len(y_h) >= min_samples:
                groups.append((str(hk), y_h))

        if len(groups) < 1:
            continue

        # --- refinement fracture: mean_k D( pooled(y_s), y_hk )
        sub_ref = []
        for _, y_h in groups:
            d = _divergence_np(
                y_s.dropna().to_numpy(),
                y_h,
                divergence=divergence,
                tail_mode=tail_mode,
                quantile_focus=quantile_focus,
                n_bins=n_bins,
            )
            if d is not None:
                sub_ref.append(d)
                n_eff += int(len(y_h))

        f_ref_sk: Optional[float] = None
        if sub_ref:
            f_ref_sk = float(np.mean(sub_ref))
            ref_vals.append(f_ref_sk)

        # --- pairwise fracture: mean_{(k1,k2)} D( y_hk1, y_hk2 )
        f_pair_sk: Optional[float] = None
        if len(groups) >= 2:
            sub_pair = _pairwise_group_divergences(
                groups,
                divergence=divergence,
                tail_mode=tail_mode,
                quantile_focus=quantile_focus,
                n_bins=n_bins,
                n_pairs=n_pairwise_pairs,
                rng=rng,
            )
            if sub_pair:
                f_pair_sk = float(np.mean(sub_pair))
                pair_vals.append(f_pair_sk)

        if f_ref_sk is not None or f_pair_sk is not None:
            per_state_rows.append(
                {
                    "state_key": sk,
                    "fracture_refinement": f_ref_sk,
                    "fracture_pairwise": f_pair_sk,
                    "n_state": int(len(y_s)),
                    "n_hist_groups": int(len(groups)),
                    "n_effective": int(n_eff),
                }
            )        

    f_ref = None if not ref_vals else float(np.mean(ref_vals))
    f_pair = None if not pair_vals else float(np.mean(pair_vals))

    if f_ref is None and f_pair is None:
        return None, None, 0, []

    return f_ref, f_pair, n_eff, per_state_rows


def _permute_within_bins(
    history_key: pd.Series,
    intervention: pd.Series,
    state_key: pd.Series,
    rng: np.random.Generator,
) -> pd.Series:
    """
    Permute history labels within (intervention, state) bins.
    """
    permuted = history_key.copy()
    for iv in intervention.unique():
        mask_iv = intervention == iv
        for sk in state_key[mask_iv].unique():
            mask = mask_iv & (state_key == sk)
            permuted.loc[mask] = rng.permutation(history_key[mask].values)
    return permuted

def _divergence(
    a: pd.Series,
    b: pd.Series,
    *,
    divergence: str,
    tail_mode: bool,
    quantile_focus: float,
    n_bins: int,
) -> Optional[float]:
    """
    Compute divergence between two outcome samples.
    """

    a = a.dropna().to_numpy()
    b = b.dropna().to_numpy()

    if tail_mode:
        qa = np.quantile(a, quantile_focus)
        qb = np.quantile(b, quantile_focus)
        a = a[a >= qa]
        b = b[b >= qb]

    if len(a) < 2 or len(b) < 2:
        return None

    if divergence == "js":
        hist_range = (min(a.min(), b.min()), max(a.max(), b.max()))
        pa, _ = np.histogram(a, bins=n_bins, range=hist_range, density=True)
        pb, _ = np.histogram(b, bins=n_bins, range=hist_range, density=True)
        pa = pa + 1e-12
        pb = pb + 1e-12
        return float(jensenshannon(pa, pb) ** 2)

    if divergence == "wasserstein":
        return float(wasserstein_distance(a, b))

    raise ValueError(f"Unsupported divergence: {divergence}")


def _divergence_np(
    a: np.ndarray,
    b: np.ndarray,
    *,
    divergence: str,
    tail_mode: bool,
    quantile_focus: float,
    n_bins: int,
) -> Optional[float]:
    """
    Numpy version of divergence to avoid repeated pandas conversions.
    """
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]

    if len(a) < 2 or len(b) < 2:
        return None

    if tail_mode:
        qa = np.quantile(a, quantile_focus)
        qb = np.quantile(b, quantile_focus)
        a = a[a >= qa]
        b = b[b >= qb]

    if len(a) < 2 or len(b) < 2:
        return None

    if divergence == "js":
        lo = float(min(a.min(), b.min()))
        hi = float(max(a.max(), b.max()))

        # Degenerate range -> both distributions are point masses in same place.
        # Treat as zero divergence (no evidence of fracture from outcome distribution shape).
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return 0.0

        pa, _ = np.histogram(a, bins=n_bins, range=(lo, hi), density=False)
        pb, _ = np.histogram(b, bins=n_bins, range=(lo, hi), density=False)

        pa = pa.astype(float)
        pb = pb.astype(float)

        # Smooth + normalize to proper probability vectors.
        pa = np.nan_to_num(pa, nan=0.0, posinf=0.0, neginf=0.0) + 1e-12
        pb = np.nan_to_num(pb, nan=0.0, posinf=0.0, neginf=0.0) + 1e-12
        pa = pa / pa.sum()
        pb = pb / pb.sum()

        d = float(jensenshannon(pa, pb, base=2))
        if not np.isfinite(d):
            return 0.0
        return d * d

    if divergence == "wasserstein":
        return float(wasserstein_distance(a, b))

    raise ValueError(f"Unsupported divergence: {divergence}")


def _pairwise_group_divergences(
    groups: List[Tuple[str, np.ndarray]],
    *,
    divergence: str,
    tail_mode: bool,
    quantile_focus: float,
    n_bins: int,
    n_pairs: int,
    rng: np.random.Generator,
) -> List[float]:
    """
    Compute divergences between history subgroups, sampling up to n_pairs distinct pairs.
    """
    m = len(groups)
    if m < 2:
        return []

    # Enumerate all pairs if small; otherwise sample.
    all_pairs = [(i, j) for i in range(m) for j in range(i + 1, m)]
    if len(all_pairs) > n_pairs:
        idx = rng.choice(len(all_pairs), size=n_pairs, replace=False)
        pairs = [all_pairs[k] for k in idx]
    else:
        pairs = all_pairs

    out: List[float] = []
    for i, j in pairs:
        _, a = groups[i]
        _, b = groups[j]
        d = _divergence_np(
            a, b,
            divergence=divergence,
            tail_mode=tail_mode,
            quantile_focus=quantile_focus,
            n_bins=n_bins,
        )
        if d is not None:
            out.append(float(d))
    return out

def _row_key(df: pd.DataFrame, cols: List[str]) -> pd.Series:
    """
    Build a hashable row-wise key from selected columns.
    """
    if not cols:
        return pd.Series(["_"] * len(df), index=df.index)
    return df[cols].astype(str).agg("|".join, axis=1)