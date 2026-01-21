# intervention_faithfulness/plugins/data/sigma2i_unfaithful_cut_linear.py
# - This file is intentionally self-contained and deterministic.
# - It produces a canonical trials table per docs/plugin_spec.md:
#     trial_id, intervention_id, outcome, and at least one state_* column.
#
# Purpose (Σ₂-I):
# - Provide two regime slices:
#     FAITHFUL_BASELINE  -> should PASS (low continuation fracture)
#     UNFAITHFUL_CUT     -> should be refused / BOUNDARY (high continuation fracture)
#
# How we force the "UNFAITHFUL_CUT" to trip the fracture diagnostic:
# - Continuation fracture (v0.1) compares P(Y|S) vs P(Y|S,H_k) within each (intervention, state) bin.
# - Therefore UNFAITHFUL_CUT must contain a history_* variable that changes the outcome distribution
#   within the same state bin, with enough samples per subgroup to clear min_samples.
#
# Note:
# - Do NOT redefine PluginMetadata/DataPlugin here; import them from registry to satisfy isinstance checks.

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from intervention_faithfulness.plugins.registry import DataPlugin, PluginMetadata, register_data_plugin


class Sigma2IUnfaithfulCutLinear(DataPlugin):
    """
    Synthetic dataset for Σ₂-I (intervention faithfulness) guard tests.

    Design:
    - FAITHFUL_BASELINE:
        outcome is well-explained by the reduced state (state_x), and history_h is present
        but does not materially change outcome within a state bin -> low fracture -> PASS.

    - UNFAITHFUL_CUT:
        state_x is constant within each intervention (do_x=-1 and do_x=+1),
        and outcome depends strongly on history_h within that state bin
        -> P(Y|S) differs from P(Y|S,H) -> high fracture -> BOUNDARY/REFUSE.
    """

    metadata = PluginMetadata(
        name="sigma2i_unfaithful_cut_linear",
        description="Σ₂-I negative control: baseline passes; unfaithful cut fractures via history dependence.",
        expected_format="No input required (synthetic). Produces canonical trials table.",
        example_usage="FaithfulnessTest.from_plugin('sigma2i_unfaithful_cut_linear', source=None).run()",
        tags=["sigma2", "negative-control", "intervention", "unfaithful-cut"],
        links={"spec": "docs/SIGMA2_I_0001.md"},
    )

    def load(self, source: Any = None, **kwargs: Any) -> Dict[str, Any]:
        # No external source required. Accept kwargs for future extension.
        return {"source": source, "params": dict(kwargs)}

    def defaults(self) -> Dict[str, Any]:
        # Conservative defaults; align with v0.1 config keys.
        return {
            "divergence": "js",
            "min_samples": 50,
            "tail_mode": False,
        }

    def to_trials(self, raw: Any, **kwargs: Any) -> pd.DataFrame:
        seed = int(kwargs.get("seed", 210001))
        n_obs = int(kwargs.get("n_obs", 1200))
        n_do = int(kwargs.get("n_do", 600))

        rng = np.random.default_rng(seed)

        # ------------------------------------------------------------------
        # FAITHFUL_BASELINE
        # ------------------------------------------------------------------
        # Goal: Y ⟂ history_h | (state_x, intervention_id="obs")
        # So: outcome depends ONLY on state_x + noise. history_h is present but inert.
        def _make_obs(n: int) -> pd.DataFrame:
            n_states = int(kwargs.get("n_states_obs", 6))
            state_vals = rng.integers(0, n_states, size=n, dtype=int)

            # CRITICAL: history_h must be constant within each state_x bin
            # so there is only one history group per state.
            # (Keeps history_* present but removes within-state subgrouping noise.)
            history_h = (state_vals % 2).astype(int)

            state_norm = state_vals.astype(float) / max(1.0, float(n_states - 1))
            noise_sigma = float(kwargs.get("obs_noise_sigma", 0.25))
            y = state_norm + noise_sigma * rng.standard_normal(size=n)

            return pd.DataFrame(
                {
                    "trial_id": np.arange(n, dtype=int),
                    "intervention_id": ["obs"] * n,
                    "state_x": state_vals.astype(float),
                    "history_h": history_h,
                    "outcome": y.astype(float),
                    "regime_slice": ["FAITHFUL_BASELINE"] * n,
                }
            )

        # ------------------------------------------------------------------
        # UNFAITHFUL_CUT
        # ------------------------------------------------------------------
        # Within each do(x=c): state_x constant, history_h splits outcome distributions.
        def _make_do(n: int, c: float, start_id: int, label: str) -> pd.DataFrame:
            state_x = np.full(shape=n, fill_value=float(c), dtype=float)

            # Ensure both history groups appear with good probability
            history_h = rng.integers(0, 2, size=n, dtype=int)

            # Strong separation by history (within same state bin)
            # Keep noise moderate so fracture is reliably large.
            do_noise_sigma = float(kwargs.get("do_noise_sigma", 0.25))
            sep = float(kwargs.get("do_history_separation", 1.0))
            y = (sep * (2.0 * history_h - 1.0)) + do_noise_sigma * rng.standard_normal(size=n)

            return pd.DataFrame(
                {
                    "trial_id": np.arange(start_id, start_id + n, dtype=int),
                    "intervention_id": [f"do_x={c:g}"] * n,
                    "state_x": state_x,
                    "history_h": history_h.astype(int),
                    "outcome": y.astype(float),
                    "regime_slice": [label] * n,
                }
            )

        df_obs = _make_obs(n_obs)

        # Two intervention points to ensure distinct interventions exist.
        n_low = n_do // 2
        n_high = n_do - n_low
        df_do_low = _make_do(n_low, c=-1.0, start_id=n_obs, label="UNFAITHFUL_CUT")
        df_do_high = _make_do(n_high, c=+1.0, start_id=n_obs + n_low, label="UNFAITHFUL_CUT")

        df = pd.concat([df_obs, df_do_low, df_do_high], ignore_index=True)

        cols = ["trial_id", "intervention_id", "outcome", "state_x", "history_h", "regime_slice"]
        return df[cols]

    def validate(self, df: pd.DataFrame) -> list[str]:
        warnings: list[str] = []
        if not any(str(c).startswith("state_") for c in df.columns):
            warnings.append("No state_* columns present; intervention faithfulness may be ill-posed.")
        if df["intervention_id"].nunique() < 2:
            warnings.append("Only one intervention_id present; cannot test intervention fracture.")
        if not any(str(c).startswith("history_") for c in df.columns):
            warnings.append("No history_* columns present; fracture may be near-zero by construction.")
        return warnings


def plugin() -> Sigma2IUnfaithfulCutLinear:
    # Factory hook (optional).
    return Sigma2IUnfaithfulCutLinear()


# Register at import time (built-in plugin behavior)
register_data_plugin(Sigma2IUnfaithfulCutLinear)
