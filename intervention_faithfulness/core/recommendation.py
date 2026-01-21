"""
core/recommendation.py — Minimal completion (feature ranking) (v0.1)

This module provides a lightweight "recommendation engine":
- Given a baseline trials_df and a set of candidate features (history variables),
  estimate which feature(s) most reduce continuation fracture.
- Rank candidates by:
    - delta fracture (improvement)
    - mutual information proxy (optional, v0.1 simple)
- Keep core model-agnostic: features are just columns or feature plugins.

v0.1 scope:
- Evaluate existing history_* columns already present in trials_df
- Evaluate feature plugins by computing their columns on a copy of trials_df
- Use compute_continuation_fracture as the scoring function
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union, Set
import numpy as np
import pandas as pd

from intervention_faithfulness.core.fracture import compute_continuation_fracture, FractureResult
from intervention_faithfulness.plugins.registry import get_feature_plugin


# ---------------------------------------------------------------------
# Internal helpers (v0.1)
# ---------------------------------------------------------------------

def _augment_state_with_history(df: pd.DataFrame, history_cols: List[str]) -> pd.DataFrame:
    """
    Return a copy of df where each provided history_* column is *copied* into a new state_* column.

    Design intent:
    - Fracture compares distributions conditioned on reduced state (state_*) vs full history (state_* + history_*).
    - A candidate completion should therefore *refine the state representation* by promoting selected
      history information into state_* while keeping the original history_* available for the full-history key.

    This function does **not** drop history_* columns.
    """
    out = df.copy()
    for c in history_cols:
        if c not in out.columns:
            continue
        out[f"state__aug__{c}"] = out[c]
    return out

@dataclass(frozen=True)
class FeatureScore:
    name: str
    delta_fracture: float
    baseline_fracture: float
    augmented_fracture: float
    mutual_info: Optional[float] = None
    params: Optional[Dict[str, Any]] = None
    data_requirements: Optional[str] = None


@dataclass(frozen=True)
class FeatureSetScore:
    """
    Score for a *set* of promoted history features evaluated together.
    """
    name: str
    features: List[str]
    delta_fracture: float
    baseline_fracture: float
    augmented_fracture: float
    mutual_info: Optional[float] = None
    params: Optional[Dict[str, Any]] = None
    data_requirements: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "features": list(self.features),
            "delta_fracture": float(self.delta_fracture),
            "baseline_fracture": float(self.baseline_fracture),
            "augmented_fracture": float(self.augmented_fracture),
            "mutual_info": self.mutual_info,
            "params": dict(self.params or {}),
            "data_requirements": self.data_requirements,
        }


def _promote_history_to_state(df: pd.DataFrame, history_cols: List[str]) -> pd.DataFrame:
    """
    Promote selected history_* columns into the reduced state by copying them into new state_* columns.

    Important:
    - The fracture metric defines reduced state via columns that start with "state_".
    - Therefore, to test a minimal completion, we must *add* state_* columns corresponding to the selected history labels.
    """
    out = df.copy()
    for h in history_cols:
        if h not in out.columns:
            continue
        # Stable, collision-resistant promoted name.
        promoted = f"state__promoted__{h}"
        out[promoted] = out[h]
    return out



@dataclass(frozen=True)
class _Candidate:
    """
    Internal representation of a candidate history feature to add.
    """
    display_name: str          # e.g. "history_h" or "time_since_last:history_tsl"
    col_name: str              # actual column name to materialize in aug_df (must start with history_)
    series: pd.Series          # column values aligned to trials_df index
    params: Optional[Dict[str, Any]] = None
    data_requirements: Optional[str] = None



def _build_candidates(
    *,
    df_full: pd.DataFrame,
    base_df: pd.DataFrame,
    candidate_history_cols: Optional[List[str]],
    candidate_feature_plugins: Optional[List[Tuple[str, Dict[str, Any]]]],
) -> List[Tuple[str, pd.Series, Optional[Dict[str, Any]], Optional[str]]]:
    """
    Build a list of candidate features as (name, series, params, data_requirements).
    Candidates are *single columns* that can be added onto base_df.
    """
    cands: List[Tuple[str, pd.Series, Optional[Dict[str, Any]], Optional[str]]] = []

    if candidate_history_cols is None:
        candidate_history_cols = [c for c in df_full.columns if c.startswith("history_")]

    for col in candidate_history_cols:
        if col in df_full.columns:
            cands.append((col, df_full[col], None, f"Requires column '{col}'."))

    if candidate_feature_plugins:
        for plugin_name, params in candidate_feature_plugins:
            plugin = get_feature_plugin(plugin_name)
            aug_df = plugin.compute(df_full.copy(), **params)

            new_cols = [
                c for c in aug_df.columns
                if c.startswith("history_") and c not in df_full.columns
            ]
            for new_col in new_cols:
                req = None
                md = getattr(plugin, "metadata", None)
                if md is not None:
                    req = getattr(md, "expected_format", None)
                cands.append((f"{plugin_name}:{new_col}", aug_df[new_col], dict(params), req))

    # De-dup by name (keep first)
    seen = set()
    out: List[Tuple[str, pd.Series, Optional[Dict[str, Any]], Optional[str]]] = []
    for name, s, p, r in cands:
        if name in seen:
            continue
        seen.add(name)
        out.append((name, s, p, r))
    return out


def rank_minimal_completions(
    *,
    trials_df: pd.DataFrame,
    candidate_history_cols: Optional[List[str]] = None,
    candidate_feature_plugins: Optional[List[Tuple[str, Dict[str, Any]]]] = None,
    divergence: str = "js",
    min_samples: int = 50,
    tail_mode: bool = False,
    quantile_focus: float = 0.95,
    n_bins: int = 30,
    top_k: int = 10,
    mode: str = "single",               # "single" | "greedy"
    greedy_k: int = 2,                  # only used by greedy
    min_delta: float = 0.0,             # filter: only return improvements >= min_delta    
) -> List[FeatureScore]:
    """
    Rank candidate history features by their ability to reduce continuation fracture.

    Parameters
    ----------
    trials_df:
        Canonical trials table.
    candidate_history_cols:
        List of existing history_* columns to consider. If None, uses all history_* columns found.
    candidate_feature_plugins:
        List of (plugin_name, params) pairs to evaluate. Each plugin will be computed on a copy.
    divergence, min_samples, tail_mode, quantile_focus, n_bins:
        Passed to fracture computation.
    top_k:
        Return at most top_k candidates.

    Returns
    -------
    List[FeatureScore]
        Sorted by delta_fracture descending (best improvement first).
    """

    df = trials_df.copy()

    # Baseline for minimal completion MUST retain history_* columns so fracture exists:
    # state_key uses state_*; history_key uses state_* + history_*.
    base_df = df
    
    base_res = compute_continuation_fracture(
        trials_df=base_df,
        divergence=divergence,
        min_samples=min_samples,
        tail_mode=tail_mode,
        quantile_focus=quantile_focus,
        n_bins=n_bins,
        n_permutations=0,
    )
    baseline = float(base_res.fracture_value)

    # Build candidate single-column additions
    candidates = _build_candidates(
        df_full=df,
        base_df=base_df,
        candidate_history_cols=candidate_history_cols,
        candidate_feature_plugins=candidate_feature_plugins,
    )

    # ------------------------------------------------------------
    # Mode A: independent single-feature scoring (existing behavior)
    # ------------------------------------------------------------
    # Back-compat aliases (in case earlier experiments used different names)
    if mode == "independent":
        mode = "single"
    if mode in ("greedy2", "greedy_k2"):
        mode = "greedy"

    if mode == "single":
        scores: List[FeatureScore] = []

        for name, series, params, req in candidates:
            aug_df = base_df.copy()
            colname = name.split(":", 1)[-1] if ":" in name else name
            aug_df[colname] = series

            aug_res = compute_continuation_fracture(
                trials_df=aug_df,
                divergence=divergence,
                min_samples=min_samples,
                tail_mode=tail_mode,
                quantile_focus=quantile_focus,
                n_bins=n_bins,
                n_permutations=0,
            )
            aug = float(aug_res.fracture_value)
            delta = baseline - aug

            if float(delta) < float(min_delta):
                continue

            scores.append(
                FeatureScore(
                    name=name,
                    delta_fracture=float(delta),
                    baseline_fracture=baseline,
                    augmented_fracture=aug,
                    mutual_info=None,
                    params=params,
                    data_requirements=req,
                )
            )

        scores.sort(key=lambda s: s.delta_fracture, reverse=True)
        return scores[:top_k]

    # ------------------------------------------------------------
    # Mode B: greedy2 (pair-aware) — finds best 1- or 2-feature set
    # ------------------------------------------------------------
    if mode != "greedy":
        raise ValueError(f"Unsupported recommendation mode: {mode}")

    if greedy_k < 1:
        return []

    # Evaluate best single first
    best_single: Optional[Tuple[str, float, float, Optional[Dict[str, Any]], Optional[str]]] = None
    for name, series, params, req in candidates:
        aug_df = base_df.copy()
        colname = name.split(":", 1)[-1] if ":" in name else name
        aug_df[colname] = series
        aug_res = compute_continuation_fracture(
            trials_df=aug_df,
            divergence=divergence,
            min_samples=min_samples,
            tail_mode=tail_mode,
            quantile_focus=quantile_focus,
        n_bins=n_bins,
            n_permutations=0,
        )
        aug = float(aug_res.fracture_value)
        delta = baseline - aug
        if best_single is None or delta > best_single[1]:
            best_single = (name, float(delta), float(aug), params, req)

    # If we only want 1 feature, return it (if it helps enough)
    if greedy_k == 1:
        if best_single is None or best_single[1] < float(min_delta):
            return []
        name, delta, aug, params, req = best_single
        return [
            FeatureScore(
                name=name,
                delta_fracture=float(delta),
                baseline_fracture=baseline,
                augmented_fracture=aug,
                mutual_info=None,
                params=params,
                data_requirements=req,
            )
        ]

    # Pair search (needed for synergy cases where no single helps)
    best_pair: Optional[Tuple[int, int, float, float]] = None  # (i,j, delta, aug)
    n = len(candidates)
    for i in range(n):
        name_i, s_i, _p_i, _r_i = candidates[i]
        col_i = name_i.split(":", 1)[-1] if ":" in name_i else name_i
        for j in range(i + 1, n):
            name_j, s_j, _p_j, _r_j = candidates[j]
            col_j = name_j.split(":", 1)[-1] if ":" in name_j else name_j

            aug_df = base_df.copy()
            aug_df[col_i] = s_i
            aug_df[col_j] = s_j

            aug_res = compute_continuation_fracture(
                trials_df=aug_df,
                divergence=divergence,
                min_samples=min_samples,
                tail_mode=tail_mode,
                quantile_focus=quantile_focus,
                n_bins=n_bins,
                n_permutations=0,
            )
            aug = float(aug_res.fracture_value)
            delta = baseline - aug

            if best_pair is None or delta > best_pair[2]:
                best_pair = (i, j, float(delta), float(aug))

    if best_pair is None or best_pair[2] < float(min_delta):
        # Fall back: if the best single meets threshold, return it
        if best_single is not None and best_single[1] >= float(min_delta):
            name, delta, aug, params, req = best_single
            return [
                FeatureScore(
                    name=name,
                    delta_fracture=float(delta),
                    baseline_fracture=baseline,
                    augmented_fracture=aug,
                    mutual_info=None,
                    params=params,
                    data_requirements=req,
                )
            ]
        return []

    i, j, delta2, aug2 = best_pair
    name_i, _s_i, params_i, req_i = candidates[i]
    name_j, _s_j, params_j, req_j = candidates[j]

    # Return two entries so the caller can display “picked set”.
    # We attach the *pair* augmented_fracture to both, but keep delta split readable:
    # each feature reports the same net pair delta (simple, test-friendly).
    return [
        FeatureScore(
            name=name_i,
            delta_fracture=float(delta2),
            baseline_fracture=baseline,
            augmented_fracture=float(aug2),
            mutual_info=None,
            params=params_i,
            data_requirements=req_i,
        ),
        FeatureScore(
            name=name_j,
            delta_fracture=float(delta2),
            baseline_fracture=baseline,
            augmented_fracture=float(aug2),
            mutual_info=None,
            params=params_j,
            data_requirements=req_j,
        ),
    ]


def rank_minimal_completion_sets(
    *,
    trials_df: pd.DataFrame,
    candidate_history_cols: Optional[List[str]] = None,
    candidate_feature_plugins: Optional[List[Tuple[str, Dict[str, Any]]]] = None,
    divergence: str = "js",
    min_samples: int = 50,
    tail_mode: bool = False,
    quantile_focus: float = 0.95,
    n_bins: int = 30,
    top_k: int = 10,
    max_set_size: int = 3,
    greedy_k: int = 2,
    min_delta: float = 0.0,
) -> List[FeatureSetScore]:
    """
    Rank *sets* of candidate history features by their ability to reduce continuation fracture.

    Motivation:
    Some systems require *multiple* history variables to faithfully refine state.
    A greedy / beam search over feature sets captures cases where no single feature
    is sufficient but pairs (or small sets) are.

    Semantics:
    - Baseline fracture is computed on the provided table as-is. History columns remain
      available for the full-history key; reduced state is determined by state_* columns.
    - A candidate feature is evaluated by copying its history_* column into a new state_* column
      (via _augment_state_with_history), refining the reduced state.
    - We return FeatureSetScore objects sorted by delta_fracture descending.

    Parameters
    ----------
    greedy_k:
        Beam width (number of partial sets retained at each size).
    min_delta:
        Minimum delta_fracture required to keep/return a set.
    """

    df_full = trials_df.copy()

    # Discover candidates -------------------------------------------------
    if candidate_history_cols is None:
        candidate_history_cols = [c for c in df_full.columns if c.startswith("history_")]

    candidates: List[_Candidate] = []
    for col in candidate_history_cols:
        if col in df_full.columns:
            candidates.append(
                _Candidate(
                    col_name=col,
                    display_name=col,
                    series=df_full[col],
                    params=None,
                    data_requirements=f"Requires column '{col}'.",
                )
            )

    if candidate_feature_plugins:
        for plugin_name, params in candidate_feature_plugins:
            plugin = get_feature_plugin(plugin_name)
            aug_df_full = plugin.compute(df_full.copy(), **params)
            new_cols = [c for c in aug_df_full.columns if c.startswith("history_") and c not in df_full.columns]
            for new_col in new_cols:
                candidates.append(
                    _Candidate(
                        col_name=new_col,
                        display_name=f"{plugin_name}:{new_col}",
                        series=aug_df_full[new_col],
                        params=dict(params),
                        data_requirements=getattr(plugin, "metadata", None).expected_format
                        if getattr(plugin, "metadata", None) is not None
                        else None,
                    )
                )

    if not candidates:
        return []

    # Baseline ------------------------------------------------------------
    base_df = df_full
    base_res = compute_continuation_fracture(
        trials_df=base_df,
        divergence=divergence,
        min_samples=min_samples,
        tail_mode=tail_mode,
        quantile_focus=quantile_focus,
        n_bins=n_bins,
        n_permutations=0,
    )
    baseline = float(base_res.fracture_value)

    def eval_set(chosen: List[_Candidate]) -> float:
        aug = base_df.copy()
        for cand in chosen:
            aug[cand.col_name] = cand.series
        aug = _augment_state_with_history(aug, [c.col_name for c in chosen])

        res = compute_continuation_fracture(
            trials_df=aug,
            divergence=divergence,
            min_samples=min_samples,
            tail_mode=tail_mode,
            quantile_focus=quantile_focus,
            n_bins=n_bins,
            n_permutations=0,
        )
        return float(res.fracture_value)

    # Beam / greedy search ------------------------------------------------
    scored: List[FeatureSetScore] = []

    # Seed with all singletons (keep top greedy_k for expansion)
    singleton_scores: List[Tuple[float, _Candidate, float]] = []
    for cand in candidates:
        val = eval_set([cand])
        delta = baseline - val
        if delta >= float(min_delta):
            singleton_scores.append((delta, cand, val))
            scored.append(
                FeatureSetScore(
                    name=cand.col_name,
                    features=[cand.col_name],
                    delta_fracture=float(delta),
                    baseline_fracture=float(baseline),
                    augmented_fracture=float(val),
                )
            )

    singleton_scores.sort(key=lambda t: t[0], reverse=True)
    beam: List[Tuple[List[_Candidate], float]] = [
        ([c], v) for (_d, c, v) in singleton_scores[: max(1, int(greedy_k))]
    ]

    # IMPORTANT: even if no singleton passes min_delta, still try pairs;
    # this is needed for XOR-like cases where *only* combinations help.
    if not beam:
        beam = [([], baseline)]

    for size in range(2, int(max_set_size) + 1):
        next_beam: List[Tuple[List[_Candidate], float]] = []
        seen = set()

        for chosen, _chosen_val in beam:
            chosen_names = {c.col_name for c in chosen}
            for cand in candidates:
                if cand.col_name in chosen_names:
                    continue
                new_set = chosen + [cand]
                key = tuple(sorted(c.col_name for c in new_set))
                if key in seen:
                    continue
                seen.add(key)

                val = eval_set(new_set)
                delta = baseline - val
                if delta < float(min_delta):
                    continue

                scored.append(
                    FeatureSetScore(
                        name="+".join(key),
                        features=list(key),
                        delta_fracture=float(delta),
                        baseline_fracture=float(baseline),
                        augmented_fracture=float(val),
                    )
                )
                next_beam.append((new_set, val))

        # Keep best greedy_k partial sets by improvement (delta)
        next_beam.sort(key=lambda t: baseline - t[1], reverse=True)
        beam = next_beam[: max(1, int(greedy_k))]
        if not beam:
            break

    scored.sort(key=lambda s: s.delta_fracture, reverse=True)
    return scored[: int(top_k)]