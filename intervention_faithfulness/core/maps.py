"""
core/maps.py — Faithfulness map visualization (v0.1)

This module produces "boundary of validity" maps:
- x-axis: intervention strength (or chosen intervention variable)
- y-axis: history depth / feature value / another intervention dimension
- color: continuation fracture score (or derived faithfulness metric)

v0.1 scope:
- Static matplotlib heatmap
- Works from FractureResult plus trials_df
- Two modes:
    1) If intervention_id is numeric: use it as x-axis.
    2) If intervention_id is categorical: map to ordered categories.

Notes:
- This is a visualization helper only. It does not define the fracture metric.
- We avoid hard domain assumptions. Users can supply which columns map to axes.

Future:
- Interactive maps (plotly) and click-through drilldowns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from intervention_faithfulness.core.fracture import compute_continuation_fracture


@dataclass
class MapConfig:
    x_col: str = "intervention_id"
    y_col: Optional[str] = None  # e.g., "history_integrated_current" or "history_depth"
    bins_x: int = 25
    bins_y: int = 25
    min_samples: int = 50
    metric: str = "refinement"  # "refinement" | "pairwise" | "both"
    divergence: str = "js"
    tail_mode: bool = False
    quantile_focus: float = 0.95
    n_bins: int = 30  # histogram bins for JS divergence in fracture computation
    n_pairwise_pairs: int = 50  # only used when metric is pairwise/both (if supported by fracture.py)
    normalize_quantile: float = 0.95  # robust max quantile for faithfulness normalization
    try_parse_numeric: bool = True  # treat numeric-looking object columns as numeric for binning
 
def compute_faithfulness_grid(
    *,
    trials_df: pd.DataFrame,
    config: Optional[MapConfig] = None,
    faithfulness: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], np.ndarray, List[str], MapConfig]:
    """
    Compute the (y_bins x x_bins) grid used for faithfulness maps.

    Returns
    -------
    grid : np.ndarray
        Either raw fracture values (faithfulness=False) or faithfulness values in [0,1] (faithfulness=True),
        with NaN for underpowered/undefined cells.
    counts : np.ndarray
        Sample count per cell.
    x_edges, x_labels, y_edges, y_labels :
        Bin edges and labels for axes.
    cfg_used : MapConfig
        The resolved config actually used (including default y_col selection).
    """
    cfg = config or MapConfig()

    if cfg.x_col not in trials_df.columns:
        raise ValueError(f"x_col '{cfg.x_col}' not found in trials_df")

    x = trials_df[cfg.x_col]

    # Resolve default y_col if not provided
    if cfg.y_col is None:
        y_col = _default_y_col(trials_df)
        cfg = MapConfig(
            x_col=cfg.x_col,
            y_col=y_col,
            bins_x=cfg.bins_x,
            bins_y=cfg.bins_y,
            min_samples=cfg.min_samples,
            metric=cfg.metric,
            divergence=cfg.divergence,
            tail_mode=cfg.tail_mode,
            quantile_focus=cfg.quantile_focus,
            n_bins=cfg.n_bins,
            n_pairwise_pairs=cfg.n_pairwise_pairs,
            normalize_quantile=cfg.normalize_quantile,
            try_parse_numeric=cfg.try_parse_numeric,
        )

    if cfg.y_col is not None and cfg.y_col not in trials_df.columns:
        raise ValueError(f"y_col '{cfg.y_col}' not found in trials_df")

    # Bin axes
    x_bin, x_edges, x_labels = _bin_axis(x, bins=cfg.bins_x, try_parse_numeric=cfg.try_parse_numeric)
    if cfg.y_col is not None:
        y = trials_df[cfg.y_col]
        y_bin, y_edges, y_labels = _bin_axis(y, bins=cfg.bins_y, try_parse_numeric=cfg.try_parse_numeric)
    else:
        y_bin = pd.Series([0] * len(trials_df), index=trials_df.index)
        y_edges = np.array([0.0, 1.0], dtype=float)
        y_labels = ["all"]

    grid_raw = np.full((len(y_labels), len(x_labels)), np.nan, dtype=float)
    counts = np.zeros((len(y_labels), len(x_labels)), dtype=int)

    for yi in range(len(y_labels)):
        for xi in range(len(x_labels)):
            cell_mask = (y_bin == yi) & (x_bin == xi)
            cell = trials_df[cell_mask]
            counts[yi, xi] = int(len(cell))

            if len(cell) < cfg.min_samples:
                continue

            res = compute_continuation_fracture(
                trials_df=cell,
                metric=cfg.metric,
                divergence=cfg.divergence,
                min_samples=cfg.min_samples,
                tail_mode=cfg.tail_mode,
                quantile_focus=cfg.quantile_focus,
                n_bins=cfg.n_bins,
                n_pairwise_pairs=cfg.n_pairwise_pairs,
                n_permutations=0,
            )
            grid_raw[yi, xi] = float(res.fracture_value)

    if not bool(faithfulness):
        return grid_raw, counts, x_edges, x_labels, y_edges, y_labels, cfg

    # Contract: faithfulness grid is a monotone transform of fracture:
    # faithfulness = clamp(1 - fracture, 0, 1)
    plot_grid = np.array(grid_raw, dtype=float, copy=True)
    finite_mask = np.isfinite(plot_grid)
    plot_grid[finite_mask] = 1.0 - plot_grid[finite_mask]
    plot_grid[finite_mask] = np.clip(plot_grid[finite_mask], 0.0, 1.0)

    return plot_grid, counts, x_edges, x_labels, y_edges, y_labels, cfg



def plot_faithfulness_map(
    *,
    trials_df: pd.DataFrame,
    config: Optional[MapConfig] = None,
    title: Optional[str] = None,
    faithfulness: bool = True,
) -> plt.Figure:
    """
    Create a static faithfulness map as a matplotlib Figure.

    Parameters
    ----------
    trials_df:
        Canonical trials table.
    config:
        MapConfig controlling axes and fracture settings.
    title:
        Optional title.
    faithfulness:
        If True, plot "faithfulness" = 1 - normalized fracture
        If False, plot raw fracture.

    Returns
    -------
    matplotlib.figure.Figure
    """
    plot_grid, counts, _x_edges, x_labels, _y_edges, y_labels, cfg = compute_faithfulness_grid(
        trials_df=trials_df,
        config=config,
        faithfulness=faithfulness,
    )

    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(plot_grid, aspect="auto", origin="lower")

    ax.set_xlabel(cfg.x_col)
    ax.set_ylabel(cfg.y_col if cfg.y_col is not None else "all")

    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=45, ha="right")

    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels)

    if title is None:
        title = "Faithfulness Map" if faithfulness else "Continuation Fracture Map"
    ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("faithfulness" if faithfulness else "fracture")

    # Overlay sample counts as faint text
    for yi in range(counts.shape[0]):
        for xi in range(counts.shape[1]):
            n = counts[yi, xi]
            if n > 0:
                ax.text(xi, yi, str(n), ha="center", va="center", fontsize=7)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _default_y_col(df: pd.DataFrame) -> Optional[str]:
    """
    Choose a default y-axis column if available:
    - Prefer numeric history_* columns
    - Else None
    """
    hist_cols = [c for c in df.columns if c.startswith("history_")]
    for c in hist_cols:
        try:
            s = pd.to_numeric(df[c], errors="coerce")
            if np.isfinite(s).sum() > 0:
                return c
        except Exception:
            continue
    return None


def _bin_axis(
    series: pd.Series,
    *,
    bins: int,
    try_parse_numeric: bool = True,
) -> Tuple[pd.Series, np.ndarray, List[str]]:
    """
    Bin an axis into integer bins [0..k-1] and produce labels.
    """
    # Numeric (or numeric-looking)?
    s_num = None
    if pd.api.types.is_numeric_dtype(series):
        s_num = pd.to_numeric(series, errors="coerce")
    elif try_parse_numeric:
        # Treat object/string columns as numeric if "enough" values parse.
        candidate = pd.to_numeric(series, errors="coerce")
        n_finite = int(np.isfinite(candidate.to_numpy(dtype=float)).sum())
        # Heuristic: if at least half parse as finite numbers, treat as numeric axis.
        if n_finite >= max(1, len(series) // 2):
            s_num = candidate

    if s_num is not None:
        s = s_num
        finite = s[np.isfinite(s)]
        if finite.empty:
            # Degenerate: everything in one bin
            b = pd.Series([0] * len(series), index=series.index)
            edges = np.array([0.0, 1.0])
            labels = ["nan"]
            return b, edges, labels

        lo = float(np.quantile(finite, 0.01))
        hi = float(np.quantile(finite, 0.99))
        if hi <= lo:
            lo = float(finite.min())
            hi = float(finite.max())
        if hi <= lo:
            b = pd.Series([0] * len(series), index=series.index)
            edges = np.array([lo, hi if hi > lo else lo + 1.0])
            labels = [f"{lo:.3g}"]
            return b, edges, labels

        edges = np.linspace(lo, hi, bins + 1)
        b = pd.cut(s, bins=edges, labels=False, include_lowest=True)
        b = b.fillna(-1).astype(int)
        # Map -1 (nan/outside) to 0th bin for visualization
        b = b.replace(-1, 0)

        labels = []
        for i in range(len(edges) - 1):
            labels.append(f"{edges[i]:.3g}-{edges[i+1]:.3g}")
        return b, edges, labels

    # Categorical: stable ordering by sorted unique
    cats = series.astype(str)
    uniq = sorted(cats.unique().tolist())
    mapping = {u: i for i, u in enumerate(uniq)}
    b = cats.map(mapping).astype(int)
    edges = np.arange(len(uniq) + 1, dtype=float)
    labels = uniq
    return b, edges, labels