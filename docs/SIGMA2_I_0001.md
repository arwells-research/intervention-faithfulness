## S-0001: Σ₂-I Unfaithful-cut intervention-faithfulness guard (negative control)

**Claim:**
A mapping that appears valid in observational data but breaks under a declared intervention menu must be refused (BOUNDARY), even if in-sample observational fit is high.

**Assumptions:**
- Canonical trials table schema holds: `trial_id`, `intervention_id`, `outcome`, and at least one `state_*` column.
- Intervention faithfulness is evaluated by comparing outcome distributions across interventions at fixed (or binned) state.
- A “faithful” regime is one where outcome distributions are intervention-invariant up to declared tolerance (core-defined).
- An “unfaithful cut” regime is one where observational correlations exist but do not survive interventions (i.e., continuation / intervention changes the conditional outcome structure).
- No tuning: all parameters are fixed in the plugin defaults and/or test; no scanning.

**Repro:**
- Command(s):
  - `pytest -q tests/test_sigma2I_0001_unfaithful_cut_guard.py`
  - Optional (if CLI supports plugins): `python -m intervention_faithfulness.cli from-plugin sigma2i_unfaithful_cut_linear`
- Inputs:
  - None (synthetic dataset from plugin; deterministic seed declared in plugin).
- Outputs:
  - Test PASS/FAIL (primary gate).
  - If your core exports artifacts, the test may emit CSV artifacts under a temp dir (optional; depends on your reporting hooks).

**Expected output:**
- The faithful baseline slice must be certified as faithful (PASS / OK).
- The unfaithful-cut slice must be refused (BOUNDARY) with an explicit fracture/instability flag.
- If a slice is underpowered (per `min_samples`), it must be labeled INCONCLUSIVE (not forced PASS).

**Failure interpretation:**
- If the faithful baseline is refused, then the guard is too strict or the defaults/metric are miscalibrated for the declared synthetic construction.
- If the unfaithful-cut slice is certified as faithful, then the Σ₂-I guard fails to detect intervention fracture (hard failure of the intervention-faithfulness criterion).
- If results flip across repeated deterministic runs (same seed), then the implementation is nondeterministic and violates reproducibility constraints.
