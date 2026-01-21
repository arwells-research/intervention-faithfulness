from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Optional

import pandas as pd
import pytest

from intervention_faithfulness import FaithfulnessTest


def _get_fracture_value(res: Any) -> float:
    """
    Extract a scalar fracture value from DiagnosisResult without assuming exact field names.
    This keeps the battery stable across minor UX refactors.

    Expected common patterns:
      - res.fracture_score -> float or dict-like with 'value'
      - res.fracture_value -> float
      - res.fracture -> float
    """
    for attr in ("fracture_score", "fracture_value", "fracture"):
        if hasattr(res, attr):
            v = getattr(res, attr)
            if isinstance(v, (int, float)):
                return float(v)
            if isinstance(v, dict) and "value" in v:
                return float(v["value"])
            # dataclass-like with .value
            if hasattr(v, "value"):
                return float(v.value)  # type: ignore[attr-defined]
    raise AssertionError("Could not extract fracture scalar from DiagnosisResult (unexpected API surface).")


def _diagnose_with_overrides(test: FaithfulnessTest, **cfg_overrides: Any) -> Any:
    """
    Run diagnose in a way that works whether diagnose() accepts kwargs or a config object.
    """
    try:
        # Many implementations allow diagnose(**kwargs)
        return test.diagnose(**cfg_overrides)
    except TypeError:
        # Otherwise: diagnose(config=DiagnoseConfig(...)) pattern
        if not hasattr(test, "default_config"):
            raise
        base_cfg = test.default_config()  # type: ignore[attr-defined]
        cfg = base_cfg
        if hasattr(base_cfg, "__dataclass_fields__"):
            cfg = replace(base_cfg, **cfg_overrides)
        else:
            # fallback: mutate copy of dict-like config
            d = dict(getattr(base_cfg, "__dict__", {}))
            d.update(cfg_overrides)
            cfg = base_cfg.__class__(**d)
        return test.diagnose(cfg)  # type: ignore[misc]


def test_sigma2I_battery_faithful_vs_unfaithful_cut() -> None:
    # ------------------------------------------------------------------
    # Negative control: faithful regime should have low fracture
    # ------------------------------------------------------------------
    test_faithful = FaithfulnessTest.from_plugin("faithful_synthetic", None)

    res_faithful = _diagnose_with_overrides(
        test_faithful,
        # keep it lightweight + deterministic
        divergence="js",
        min_samples=50,
        tail_mode=False,
        # If permutations are supported, keep small for CI; otherwise ignored.
        n_permutations=64,
    )

    F_faithful = _get_fracture_value(res_faithful)
    assert math.isfinite(F_faithful)
    assert F_faithful <= 0.08, f"faithful_synthetic should be near zero; got F={F_faithful:.4g}"

    # ------------------------------------------------------------------
    # Positive control: unfaithful cut should fracture with state-only
    # ------------------------------------------------------------------
    test_unfaithful = FaithfulnessTest.from_plugin("unfaithful_cut_synthetic", None)
    res_unfaithful = _diagnose_with_overrides(
        test_unfaithful,
        divergence="js",
        min_samples=60,
        tail_mode=False,
        n_permutations=64,
    )

    F_base = _get_fracture_value(res_unfaithful)
    assert math.isfinite(F_base)
    assert F_base >= 0.12, f"unfaithful_cut_synthetic should fracture; got F={F_base:.4g}"

    # ------------------------------------------------------------------
    # Repair check: promote history_h into state and re-run
    # ------------------------------------------------------------------
    # We avoid assuming a feature-plugin call exists; instead we directly augment the trials table.
    # This also directly tests the DESIGN.md contract: completion must be promoted to state_*.
    trials_df = getattr(test_unfaithful, "trials_df", None)
    if trials_df is None:
        # Most implementations keep canonical table on the test instance; otherwise, try pulling from result.
        trials_df = getattr(res_unfaithful, "trials_df", None) or getattr(res_unfaithful, "canonical_trials_df", None)
    assert isinstance(trials_df, pd.DataFrame), "Could not locate canonical trials table for augmentation."

    assert "history_h" in trials_df.columns, "Battery requires history_h to exist as a candidate completion feature."
    assert "state_h" not in trials_df.columns, "state_h should not already exist in baseline."

    df_aug = trials_df.copy()
    df_aug["state_h"] = df_aug["history_h"]

    test_aug = FaithfulnessTest(df_aug)
    res_aug = _diagnose_with_overrides(
        test_aug,
        divergence="js",
        min_samples=60,
        tail_mode=False,
        n_permutations=64,
        # Disable recommendations to keep run deterministic and fast; we are explicitly augmenting.
        recommend=False,
    )

    F_aug = _get_fracture_value(res_aug)
    assert math.isfinite(F_aug)

    # Require a strong reduction (this is the core Σ₂-I executable claim).
    assert F_aug <= 0.40 * F_base, f"Expected augmentation to reduce fracture strongly: base={F_base:.4g}, aug={F_aug:.4g}"

    # ------------------------------------------------------------------
    # Optional: if recommendation surface exists, require it to point at history_h
    # ------------------------------------------------------------------
    if hasattr(res_unfaithful, "recommended_features"):
        rec = getattr(res_unfaithful, "recommended_features")
        # common patterns: list[str] or list[dict{name:..}]
        top_name: Optional[str] = None
        if isinstance(rec, list) and rec:
            if isinstance(rec[0], str):
                top_name = rec[0]
            elif isinstance(rec[0], dict) and "name" in rec[0]:
                top_name = str(rec[0]["name"])
        if top_name is not None:
            assert "history_h" in top_name, f"Expected top recommendation to include history_h; got {top_name}"
