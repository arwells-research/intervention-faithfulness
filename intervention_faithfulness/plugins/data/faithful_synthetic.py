"""
faithful_synthetic.py — Built-in negative-control data plugin (Σ₂-I battery)

Goal
----
Generate a canonical trials table for a regime that is intervention-faithful:
Y ⟂ H | (S, I)

Important implementation detail (for v0.1 fracture grouping)
-----------------------------------------------------------
core/fracture.py builds state_key by string-joining state_* columns. If state_s is
continuous, nearly every row becomes its own state bin and min_samples filtering
makes fracture undefined (NaN). Therefore this plugin exposes DISCRETIZED state_s.

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
class FaithfulSyntheticConfig:
    seed: int = 1337
    n_trials: int = 4000

    # interventions (categorical labels)
    intervention_ids: tuple[str, ...] = ("I0", "I1", "I2")

    # latent continuous state distribution
    state_mu: float = 0.0
    state_sigma: float = 1.0

    # DISCRETIZATION (critical for fracture.py grouping)
    n_state_bins: int = 10

    # outcome model (faithful by construction):
    # outcome = a*S_cont + b*I + noise
    a_state: float = 1.25
    b_intervention: tuple[float, ...] = (0.0, 0.75, -0.50)
    noise_sigma: float = 0.60


class FaithfulSyntheticDataPlugin(DataPlugin):
    metadata = PluginMetadata(
        name="faithful_synthetic",
        description="Negative control: intervention-faithful synthetic regime (Σ₂-I battery).",
        expected_format="source may be None or a dict overriding config keys (seed, n_trials, etc.).",
        example_usage=(
            "from intervention_faithfulness import FaithfulnessTest\n"
            "test = FaithfulnessTest.from_plugin('faithful_synthetic', None)\n"
            "res = test.diagnose()\n"
            "print(res.fracture_score)\n"
        ),
        tags=["synthetic", "negative-control", "sigma2", "faithfulness"],
    )

    def load(self, source: Any, **kwargs) -> Dict[str, Any]:
        if source is None:
            return {}
        if isinstance(source, dict):
            return dict(source)
        raise TypeError("faithful_synthetic expects source=None or source=dict of config overrides.")

    def to_trials(self, raw: Dict[str, Any], **kwargs) -> pd.DataFrame:
        cfg = _merge_config(FaithfulSyntheticConfig(), raw)
        rng = np.random.default_rng(int(cfg.seed))

        n = int(cfg.n_trials)
        if n < 200:
            raise ValueError("faithful_synthetic: n_trials must be >= 200 for a meaningful negative control.")

        intervention_ids = list(cfg.intervention_ids)
        if len(intervention_ids) < 2:
            raise ValueError("faithful_synthetic: need >=2 interventions.")

        # assign interventions uniformly
        I_idx = rng.integers(0, len(intervention_ids), size=n, endpoint=False)
        intervention = np.asarray([intervention_ids[i] for i in I_idx], dtype=object)

        # latent continuous state (used in outcome)
        S_cont = rng.normal(loc=float(cfg.state_mu), scale=float(cfg.state_sigma), size=n)

        # expose discretized state for conditioning bins
        n_state_bins = int(cfg.n_state_bins)
        if n_state_bins < 2:
            n_state_bins = 2
        state_bins = pd.qcut(S_cont, q=n_state_bins, labels=False, duplicates="drop")
        state_s = state_bins.astype(int)

        # faithful outcome depends only on S_cont and I (no hidden H)
        b = np.array(list(cfg.b_intervention), dtype=float)
        if b.size != len(intervention_ids):
            raise ValueError("faithful_synthetic: b_intervention length must match intervention_ids length.")

        Y = float(cfg.a_state) * S_cont + b[I_idx] + rng.normal(0.0, float(cfg.noise_sigma), size=n)

        df = pd.DataFrame(
            {
                "trial_id": np.arange(n, dtype=int),
                "intervention_id": intervention,
                "outcome": Y.astype(float),
                "state_s": state_s.astype(int),  # DISCRETE state bin (state_*)
                "regime_name": "faithful_synthetic",
            }
        )
        return df

    def defaults(self) -> Dict[str, Any]:
        return {
            "divergence": "js",
            "min_samples": 50,
            "tail_mode": False,
            "quantile_focus": 0.95,
        }


def _merge_config(base: FaithfulSyntheticConfig, overrides: Dict[str, Any]) -> FaithfulSyntheticConfig:
    d = base.__dict__.copy()
    for k, v in (overrides or {}).items():
        if k in d:
            d[k] = v
    return FaithfulSyntheticConfig(**d)


register_data_plugin(FaithfulSyntheticDataPlugin)
