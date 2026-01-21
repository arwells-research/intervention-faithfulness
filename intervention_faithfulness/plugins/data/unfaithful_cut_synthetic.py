"""
unfaithful_cut_synthetic.py — Built-in positive-control data plugin (Σ₂-I battery)

Goal
----
Generate a canonical trials table exhibiting an Unfaithful Cut:
histories collapse to the same reduced state S, but under intervention I the
continuation distribution depends on hidden history H.

Construction
------------
We generate:
- Reduced state: state_s  (observed, DISCRETIZED for stable conditioning bins)
- Hidden history: history_h (provided as a candidate completion feature, DISCRETIZED)

Outcome model (uses continuous latent H_cont, but exposes binned history_h):
  outcome = a*S_cont + b*I + c*(H_cont * g(I)) + noise

Where g(I) makes the history effect intervention-dependent; this is important so the
fracture is specifically "under intervention" rather than just state insufficiency.

Expected behavior (battery):
- Baseline (state_s only): fracture should be materially > 0
- Augmented (state_s + promote history_h into state): fracture should drop sharply

This plugin is intentionally in-memory and deterministic given seed + config.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import pandas as pd

from intervention_faithfulness.plugins.registry import (
    DataPlugin,
    PluginMetadata,
    register_data_plugin,
)


@dataclass(frozen=True)
class UnfaithfulCutSyntheticConfig:
    seed: int = 2025
    n_trials: int = 5000

    intervention_ids: tuple[str, ...] = ("I0", "I1", "I2")

    # continuous latent state + hidden history
    state_mu: float = 0.0
    state_sigma: float = 1.0
    hidden_sigma: float = 1.0

    # DISCRETIZATION (critical for fracture.py grouping)
    # state_s bins must be small enough that each bin has >= min_samples
    n_state_bins: int = 10
    # history_h bins must be small enough that each subgroup has >= min_samples
    n_history_bins: int = 2  # 2 = sign-binning; good default for stability

    # outcome model coefficients
    a_state: float = 1.10
    b_intervention: tuple[float, ...] = (0.0, 0.60, -0.40)

    # hidden-history strength; higher => easier detection
    c_hidden: float = 1.40

    # intervention-dependent multiplier for hidden history term
    # (must be same length as intervention_ids)
    g_intervention: tuple[float, ...] = (0.0, 1.0, -1.0)

    noise_sigma: float = 0.60


class UnfaithfulCutSyntheticDataPlugin(DataPlugin):
    metadata = PluginMetadata(
        name="unfaithful_cut_synthetic",
        description="Positive control: unfaithful cut under intervention (Σ₂-I battery).",
        expected_format="source may be None or a dict overriding config keys (seed, n_trials, etc.).",
        example_usage=(
            "from intervention_faithfulness import FaithfulnessTest\n"
            "test = FaithfulnessTest.from_plugin('unfaithful_cut_synthetic', None)\n"
            "res = test.diagnose(recommend=True)\n"
            "print(res.fracture_score)\n"
        ),
        tags=["synthetic", "positive-control", "sigma2", "unfaithful-cut"],
    )

    def load(self, source: Any, **kwargs) -> Dict[str, Any]:
        if source is None:
            return {}
        if isinstance(source, dict):
            return dict(source)
        raise TypeError("unfaithful_cut_synthetic expects source=None or source=dict of config overrides.")

    def to_trials(self, raw: Dict[str, Any], **kwargs) -> pd.DataFrame:
        cfg = _merge_config(UnfaithfulCutSyntheticConfig(), raw)
        rng = np.random.default_rng(int(cfg.seed))

        n = int(cfg.n_trials)
        if n < 400:
            raise ValueError("unfaithful_cut_synthetic: n_trials must be >= 400 for a meaningful positive control.")

        intervention_ids = list(cfg.intervention_ids)
        if len(intervention_ids) < 2:
            raise ValueError("unfaithful_cut_synthetic: need >=2 interventions.")

        I_idx = rng.integers(0, len(intervention_ids), size=n, endpoint=False)
        intervention = np.asarray([intervention_ids[i] for i in I_idx], dtype=object)

        # Continuous latent variables
        S_cont = rng.normal(loc=float(cfg.state_mu), scale=float(cfg.state_sigma), size=n)
        H_cont = rng.normal(loc=0.0, scale=float(cfg.hidden_sigma), size=n)

        b = np.array(list(cfg.b_intervention), dtype=float)
        g = np.array(list(cfg.g_intervention), dtype=float)
        if b.size != len(intervention_ids):
            raise ValueError("unfaithful_cut_synthetic: b_intervention length must match intervention_ids length.")
        if g.size != len(intervention_ids):
            raise ValueError("unfaithful_cut_synthetic: g_intervention length must match intervention_ids length.")

        # --- Discretize state_s into quantile bins for stable conditioning bins
        n_state_bins = int(cfg.n_state_bins)
        if n_state_bins < 2:
            n_state_bins = 2
        # Use qcut for near-equal bin counts (deterministic given seed)
        state_bins = pd.qcut(S_cont, q=n_state_bins, labels=False, duplicates="drop")
        state_s = state_bins.astype(int)

        # --- Discretize history_h into a small number of bins
        n_hist_bins = int(cfg.n_history_bins)
        if n_hist_bins <= 2:
            # Stable, interpretable: sign binning -> {0,1}
            history_h = (H_cont >= 0.0).astype(int)
        else:
            history_bins = pd.qcut(H_cont, q=n_hist_bins, labels=False, duplicates="drop")
            history_h = history_bins.astype(int)

        # Outcome uses continuous latent variables (so signal remains rich),
        # but fracture grouping uses discretized state/history columns above.
        Y = (
            float(cfg.a_state) * S_cont
            + b[I_idx]
            + float(cfg.c_hidden) * (H_cont * g[I_idx])
            + rng.normal(0.0, float(cfg.noise_sigma), size=n)
        )

        df = pd.DataFrame(
            {
                "trial_id": np.arange(n, dtype=int),
                "intervention_id": intervention,
                "outcome": Y.astype(float),
                "state_s": state_s.astype(int),     # DISCRETE state bins (state_*)
                "history_h": history_h.astype(int), # DISCRETE history groups (history_*)
                "regime_name": "unfaithful_cut_synthetic",
            }
        )
        return df

    def defaults(self) -> Dict[str, Any]:
        # Slightly stricter to make the positive control very stable.
        return {
            "divergence": "js",
            "min_samples": 60,
            "tail_mode": False,
            "quantile_focus": 0.95,
        }


def _merge_config(base: UnfaithfulCutSyntheticConfig, overrides: Dict[str, Any]) -> UnfaithfulCutSyntheticConfig:
    d = base.__dict__.copy()
    for k, v in (overrides or {}).items():
        if k in d:
            d[k] = v
    return UnfaithfulCutSyntheticConfig(**d)


register_data_plugin(UnfaithfulCutSyntheticDataPlugin)
