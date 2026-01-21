# tests/test_sigma2I_guard_policy.py
"""
Guard-policy acceptance tests (Σ₂-I).

These are *policy* tests (diagnostic -> decision), not algorithm tests.

Non-negotiable invariant enforced here:
- No false OK:
  - underpowered => BOUNDARY
  - intervention-unfaithful => REFUSE

We intentionally keep these deterministic and light.
"""

from __future__ import annotations

import math
from typing import Any, Dict

import pandas as pd

from intervention_faithfulness.core.FaithfulnessTest import FaithfulnessTest, DiagnoseConfig
from intervention_faithfulness.core.guard import GuardConfig, decide


def _get_slice(results: Any, name: str):
    """
    Normalize slice access for FaithfulnessTest.run().

    run() returns either:
      - DiagnosisResult (single regime)
      - dict[str, DiagnosisResult] keyed by slice label
    """
    if isinstance(results, dict):
        if name in results:
            return results[name]
        # tolerate string normalization
        for k, v in results.items():
            if str(k) == str(name):
                return v
    raise KeyError(f"slice '{name}' not found in results keys={list(results.keys()) if isinstance(results, dict) else type(results)}")


def test_guard_refuses_unfaithful_cut_slice() -> None:
    """
    Canonical guard behavior: the unfaithful-cut slice must be REFUSE.
    """
    test = FaithfulnessTest.from_plugin("sigma2i_unfaithful_cut_linear", source=None)

    # Use diagnose config consistent with your synthetic defaults / CI stability.
    cfg = DiagnoseConfig(
        divergence="js",
        min_samples=50,
        tail_mode=False,
        n_permutations=0,
        recommend=False,
        safe_envelope=True,  # ok if available; decision falls back if not
    )

    results = test.run(cfg)

    base = _get_slice(results, "FAITHFUL_BASELINE")
    cut = _get_slice(results, "UNFAITHFUL_CUT")

    # Policy: conservative defaults; keep thresholds explicit here.
    gcfg = GuardConfig(
        fracture_threshold=0.12,
        min_effective_samples=200,  # uses power proxy (n_effective or len(df))
        require_significance=False,
        use_safe_envelope_if_available=True,
        max_uncertain_fraction=0.50,
    )

    d_base = decide(base, gcfg)
    d_cut = decide(cut, gcfg)

    # Baseline should not be refused; ideally OK, but allow BOUNDARY if envelope missing.
    assert d_base.status in {"OK", "BOUNDARY"}, f"baseline decision unexpected: {d_base.status} ({d_base.reason})"

    # Unfaithful slice must be refused (the core objective).
    assert d_cut.status == "REFUSE", f"expected REFUSE; got {d_cut.status} ({d_cut.reason})"


def test_guard_boundary_when_underpowered() -> None:
    """
    Underpowered must never be OK.

    We construct a tiny canonical trials table that will fail the guard's power floor.
    """
    n = 40  # intentionally below typical power floors (e.g. 200)
    df = pd.DataFrame(
        {
            "trial_id": range(n),
            "intervention_id": ["I0"] * (n // 2) + ["I1"] * (n - n // 2),
            "outcome": [0.0] * n,
            "state_s": [0.0] * n,
        }
    )

    test = FaithfulnessTest(df)
    res = test.diagnose(
        DiagnoseConfig(
            divergence="js",
            min_samples=10,   # keep algorithm from trivially NaN-ing due to min_samples=50
            tail_mode=False,
            n_permutations=0,
            recommend=False,
            safe_envelope=False,
        )
    )

    gcfg = GuardConfig(
        fracture_threshold=0.12,
        min_effective_samples=200,  # power proxy should be len(df)=40 => underpowered
        require_significance=False,
        use_safe_envelope_if_available=True,
        max_uncertain_fraction=0.50,
    )

    d = decide(res, gcfg)
    assert d.status == "BOUNDARY", f"expected BOUNDARY; got {d.status} ({d.reason})"


def test_guard_ok_on_faithful_synthetic() -> None:
    """
    The faithful synthetic battery case should be OK under conservative guard thresholds.
    """
    test = FaithfulnessTest.from_plugin("faithful_synthetic", None)
    res = test.diagnose(
        DiagnoseConfig(
            divergence="js",
            min_samples=50,
            tail_mode=False,
            n_permutations=0,
            recommend=False,
            safe_envelope=False,
        )
    )

    # sanity: fracture should be finite for the faithful control
    f = float(getattr(res.fracture_score, "value", float("nan")))
    assert math.isfinite(f), f"faithful_synthetic produced non-finite fracture: {f}"

    gcfg = GuardConfig(
        fracture_threshold=0.12,
        min_effective_samples=200,
        require_significance=False,
        use_safe_envelope_if_available=True,
        max_uncertain_fraction=0.50,
    )

    d = decide(res, gcfg)
    assert d.status == "OK", f"expected OK; got {d.status} ({d.reason})"
