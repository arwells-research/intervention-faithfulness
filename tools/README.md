# tools/

This directory contains **executable guard/harness scripts** for `intervention-faithfulness`.

These are the Σ₂-style “DFT modules” for this repo:

- **Core library (`intervention_faithfulness/`)**:
  computes metrics and produces reporting artifacts, without policy thresholds.

- **Tools (`tools/`)**:
  apply *explicit refusal policy* (PASS / BOUNDARY / INCONCLUSIVE) on top of core metrics,
  in a reproducible, audit-friendly, command-line runnable form.

The tools layer exists so that:

1) the library stays stable and domain-agnostic, and
2) DFT-style refusal logic can be operationalized and regression-tested as scripts.

## Conventions

- Tools must be runnable as standalone scripts:
  - `python tools/<script>.py ...`
- Tools must write an audit bundle into `tools/outputs/` (or a user-provided output dir):
  - `*_audit.json`
  - `*_breakdown.csv`
  - optionally: `*_certificate.json`, `*_trials.csv`

## Σ₂-I (Intervention Faithfulness)

Σ₂-I is implemented as an executable guard harness:

- it loads data via plugin or canonical CSV,
- runs `FaithfulnessTest.diagnose()`,
- applies policy thresholds to decide:
  - PASS (faithful)
  - BOUNDARY (fracture detected / not intervention-faithful)
  - INCONCLUSIVE (underpowered / invalid)

Policy thresholds belong in tools, not in core.

## Outputs are deterministic

Tools must strive for deterministic outputs given fixed inputs/config.
Timestamps are allowed but must be UTC and timezone-aware (library contract).

## Files

- `s0001_sigma2_i_guard_harness.py`
  First Σ₂-I guard harness for CI regression and “DFT module” execution.
