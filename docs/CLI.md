# CLI.md — Command Line Interface
Version: v0.1  
Status: **Authoritative** for CLI behavior and flag → config/export mapping.
The CLI is a thin wrapper around the library. DiagnosisResult and reporting.py are threshold-free (metrics + provenance + export). Σ₂-I policy (OK/BOUNDARY/REFUSE and “no false OK”) lives only in core/guard.py and is exercised by faithfulness guard.

This document describes the `faithfulness` CLI exposed by `intervention_faithfulness`.
It is written to prevent drift between `cli.py`, `DiagnoseConfig`, and `DiagnosisResult`
export surfaces.

> Note: Examples in this document avoid nested code fences by using indented blocks.

---

## 1. Installation and Entry Point

Typical usage assumes the package installs a console script:

    faithfulness --help

If you are running from a repo checkout without an installed entry point, you may also use:

    python -m intervention_faithfulness.cli --help

(Exact module path depends on packaging; the preferred interface is the `faithfulness` command.)

---

## 2. Commands Overview

The CLI is organized around a small number of commands:

- diagnose — run FaithfulnessTest.diagnose() and (optionally) export an artifact bundle
- certify — render certificate outputs (HTML/PDF) from a diagnosis result / run (no recompute if supported)
- guard — run diagnose + apply policy (core/guard.py) and emit JSON decision + exit code
- plugins — list available data + feature plugins
- plugin-help NAME — show plugin metadata and usage

Not all commands may be present in all builds; `diagnose` is the required command.

---

## 3. `diagnose` Command

### 3.1 Purpose

`diagnose` runs the full pipeline:

1. Load input trials data (CSV by default; plugin loaders optional)
2. Build a canonical trials table
3. Run the diagnostic via `FaithfulnessTest.diagnose(DiagnoseConfig)`
4. Export a reproducible artifact bundle via `DiagnosisResult.export_artifacts(...)`
5. Optionally render a certificate PDF/HTML via `DiagnosisResult.export_certificate(...)`

### 3.2 Typical usage

Minimal run (writes diagnosis JSON + breakdown CSV + curated certificate JSON by default, depending on `export_artifacts` settings):

    faithfulness diagnose --csv data/trials.csv --out-dir out/run_001

Include a map output (if implemented by reporting/maps integration):

    faithfulness diagnose --csv data/trials.csv --out-dir out/run_001 --map

Also export a canonical trials snapshot:

    faithfulness diagnose --csv data/trials.csv --out-dir out/run_001 --trials

Render a PDF certificate (separate from the JSON bundle certificate):

    faithfulness diagnose --csv data/trials.csv --out-dir out/run_001 --pdf

Render an HTML certificate:

    faithfulness diagnose --csv data/trials.csv --out-dir out/run_001 --html

Recommended “full practical bundle”:

    faithfulness diagnose --csv data/trials.csv --out-dir out/run_001 --map --trials --pdf

---

## 4. Input Options

### 4.1 CSV input

    --csv PATH

Reads a CSV file into a DataFrame. The CSV must already be in (or convertible to) the
Canonical Trials Table schema (see `docs/DESIGN.md`).

Expected columns at minimum:

- trial_id
- intervention_id
- outcome
- one or more state_* columns (or provide `--state-cols` if supported)

### 4.2 Plugin input (optional)

If data plugins are enabled, the CLI may support:

    --plugin NAME
    --source PATH

Example:

    faithfulness diagnose --plugin nanowire_switching --source data/runs.h5 --out-dir out/run_001

Plugins are responsible for mapping raw sources into the canonical trials table.

---

## 5. State / Outcome Column Handling

### 5.1 Canonical requirement

Core fracture computation identifies reduced state variables via the `state_*` prefix.

If your dataset uses raw column names like `current`, `voltage`, you have two options:

A) Pre-rename your columns to `state_current`, `state_voltage` before running CLI, or  
B) Use a CLI convenience flag (if implemented):

    --state-cols current,voltage

Contract if `--state-cols` exists:
- CLI must rename/clone these columns into `state_current`, `state_voltage`
  before calling `FaithfulnessTest`.

If `--state-cols` is not implemented, the user must supply `state_*` columns directly
or use a plugin that produces them.

### 5.2 Outcome

Default outcome column is assumed to be `outcome`. If CLI supports an override:

    --outcome-col NAME

Contract:
- CLI must rename/clone this into `outcome` for canonical processing.

---

## 6. Diagnostic Configuration Flags

The CLI should avoid duplicating defaults that already exist in `DiagnoseConfig`.
Preferred pattern:
- CLI flags default to `None`
- CLI passes only user-provided values into `DiagnoseConfig(...)`

Below are the intended flags and their meaning. Some may be omitted if not implemented.

### 6.1 Core knobs

    --min-samples N
        Minimum sample count required per equivalence class/cell.
        Maps to DiagnoseConfig.min_samples.

    --divergence {js,wasserstein,kl}
        Divergence measure for distribution comparisons.
        Maps to DiagnoseConfig.divergence.

    --tail-mode
        Enable tail restriction mode.
        Maps to DiagnoseConfig.tail_mode = True.

    --quantile-focus Q
        Tail quantile (e.g., 0.95).
        Maps to DiagnoseConfig.quantile_focus.

    --n-bins N
        Histogram bins for JS divergence mode.
        Maps to DiagnoseConfig.n_bins.

    --metric {refinement,pairwise,both}
        Metric family:
        - refinement: compares P(Y|S) vs P(Y|S,h)
        - pairwise: compares P(Y|S,h1) vs P(Y|S,h2)
        - both: compute both (if supported)
        Maps to DiagnoseConfig.metric (or equivalent).

### 6.2 Significance / permutation test knobs

    --permutations N
        Number of permutations for significance testing (0 disables).
        Maps to DiagnoseConfig.n_permutations (or equivalent).

### 6.3 Recommendations knobs

    --recommend
        Enable recommendations.
        Maps to DiagnoseConfig.recommend = True.

    --recommend-mode {single,greedy}
        Recommendation mode:
        - single: independent feature ranking (Mode A)
        - greedy: set search (Mode B)
        Maps to DiagnoseConfig.recommend_mode.

    --recommend-top-k N
        Number of top suggestions to return.
        Maps to DiagnoseConfig.recommend_top_k.

    --recommend-max-set-size N
        Max size of feature sets for greedy search.
        Maps to DiagnoseConfig.recommend_max_set_size.

    --recommend-greedy-k N
        Optional lookahead/beam width for greedy search if supported.
        Maps to DiagnoseConfig.recommend_greedy_k.

    --recommend-min-delta X
        Minimum delta threshold for acceptance/pruning.
        Maps to DiagnoseConfig.recommend_min_delta.

---

## 7. Maps and Safe Operating Envelope

Maps and envelope are computed from the canonical trials table by binning `x` and `y`.

### 7.1 Map controls (if exposed)

    --map
        Include a faithfulness map artifact in the bundle (if implemented).
        Controls export only; computation may still occur for safe envelope.

    --map-x-col NAME
        X axis column for maps (defaults to intervention_id).
        Typically maps to safe_envelope/map config x_col.

    --map-y-col NAME
        Y axis column for maps (optional).
        If omitted, a default history_* numeric column may be chosen.

    --bins-x N
    --bins-y N
        Map binning resolution.

    --threshold T
        Safe/unsafe threshold for envelope labeling.
        - if faithfulness=True: safe if value >= T
        - if faithfulness=False: safe if value <= T

    --faithfulness
        Plot/threshold on derived faithfulness score rather than raw fracture.
        (If supported; often default True in visualization.)

### 7.2 Safe envelope export

If the CLI supports a flag to compute/attach safe envelope:

    --safe-envelope

Contract:
- Envelope must be computed before returning/exporting results.
- Underpowered cells (< min_samples) must be labeled "uncertain".

---

## 8. Artifact Export Flags (Bundle)

### 8.1 Output directory

    --out-dir PATH

All artifacts from `export_artifacts` are written under this directory.

### 8.2 Prefix

    --prefix NAME

Filename prefix for artifact outputs. If omitted, CLI may choose a stable default.

### 8.3 Bundle toggles (authoritative mapping)

The CLI must call:

- `DiagnosisResult.export_artifacts(out_dir=..., include_trials=..., include_map=..., include_certificate=...)`

The CLI must not invent keyword arguments that the method does not accept.

Suggested CLI flags:

    --trials
        Export canonical trials CSV snapshot.
        Maps to include_trials=True.

    --map
        Export map artifact (if implemented).
        Maps to include_map=True.

    --certificate-json
        Export curated JSON certificate (bundle artifact).
        Maps to include_certificate=True.

Default behavior is implementation-defined, but the mapping must remain consistent.

### 8.4 PDF/HTML certificate rendering

The PDF/HTML certificate renderers are separate from the bundle JSON certificate.

Flags:

    --pdf
        Render PDF certificate.
        Calls:
            res.export_certificate(out_dir/<prefix>_certificate.pdf, format="pdf")

    --html
        Render HTML certificate.
        Calls:
            res.export_certificate(out_dir/<prefix>_certificate.html, format="html")

Important:
- `export_artifacts` may export `*_certificate.json` even when `--pdf` is requested.
- `--pdf` and `--html` control rendering only, not whether the bundle includes JSON.

---

---

## 9. `guard` Command (Σ₂-I Policy Enforcement)

### 9.1 Purpose

`guard` is the operational enforcement layer of Σ₂-I.

It consumes the same diagnostic pipeline as `diagnose`, but **does not classify or soften results**.
Instead, it enforces a conservative policy with the invariant:

> **No false OK.**

The guard emits:
- a single machine-readable JSON decision record (to stdout)
- a process exit code suitable for CI / pipelines

### 9.2 Typical usage

Minimal usage on a canonical trials table:

    faithfulness guard --csv data/trials.csv

Explicit thresholds:

    faithfulness guard \
      --csv data/trials.csv \
      --fracture-threshold 0.12 \
      --min-effective-samples 200

### 9.3 Guard decisions

The guard produces exactly one of:

- **OK**
    - Faithful under declared policy
    - Sufficient power
    - No unsafe envelope regions
    - Fracture below threshold

- **BOUNDARY**
    - Insufficient statistical power
    - Missing or non-finite fracture
    - Excessively uncertain envelope
    - Optional significance requirements not met

- **REFUSE**
    - Structural intervention-unfaithfulness detected
    - Fracture exceeds threshold
    - Unsafe envelope regions present

### 9.4 Policy logic (v0.1, conservative)

The guard applies the following gates **in order**:

1. Non-finite fracture → BOUNDARY  
2. Insufficient power proxy → BOUNDARY  
3. Unsafe envelope regions → REFUSE  
4. Excessive envelope uncertainty → BOUNDARY  
5. (Optional) significance failure → BOUNDARY  
6. Fracture ≥ threshold → REFUSE  
7. Otherwise → OK  

Thresholds are **policy-only** and do not affect reporting outputs.

### 9.5 Guard-specific flags

    --fracture-threshold T
        Fracture value above which REFUSE is issued.

    --min-effective-samples N
        Hard power floor (uses n_effective or trial count proxy).
        Below this, guard returns BOUNDARY.

    --require-significance
        Require permutation significance for OK/REFUSE.
        Missing or non-significant → BOUNDARY.

    --p-value-threshold P
        Significance cutoff (used only if --require-significance).

    --no-envelope
        Ignore safe operating envelope even if available.

    --max-uncertain-fraction F
        Maximum allowed fraction of uncertain envelope cells.
        Above this → BOUNDARY.

### 9.6 Output contract

- **stdout**: a single JSON decision record
- **stderr**: optional progress messages
- **no artifacts are written unless explicitly requested**

The JSON record includes:
- decision status and reason
- fracture metrics
- envelope summary (if available)
- provenance (hashes, config, metadata)

---

## 10. Expected Outputs

Assuming `--out-dir out/run_001` and `--prefix demo`:

Bundle outputs (typical):
- out/run_001/demo_diagnosis.json
- out/run_001/demo_certificate.json
- out/run_001/demo_breakdown.csv

Optional:
- out/run_001/demo_trials.csv (if `--trials`)
- out/run_001/demo_map.png (or equivalent) (if `--map` and map export implemented)
- out/run_001/demo_certificate.pdf (if `--pdf`)
- out/run_001/demo_certificate.html (if `--html`)

All outputs should be deterministic given the same input data and configuration,
except for timestamps which must be UTC and timezone-aware.

---

## 11. Exit Codes

### 11.1 Diagnose command

Recommended (v0.1):
- 0: success
- 2: CLI usage error (argparse default)
- 1: runtime error (exceptions, schema failure, export failure)

### 11.2 Guard command (Σ₂-I)

Authoritative (v0.1):

- **0** — OK  
- **2** — BOUNDARY (insufficient evidence; do not proceed)  
- **3** — REFUSE (structural intervention-unfaithfulness detected)  

These exit codes are stable and intended for CI, pipelines, and deployment guards.

---

## 12. Minimal UX Guarantees

The CLI should print progress markers:

- Loaded N trials from <source>
- Diagnosing...
- Diagnosis complete.
- Wrote artifacts:
  - <path1>
  - <path2>
  - ...

These prints are intentionally minimal to keep CI logs readable.

---

## 13. Troubleshooting

### 13.1 “No state columns found”
Cause:
- Your dataset lacks `state_*` columns and no plugin/flag promoted them.

Fix:
- Rename state columns to `state_*` in your CSV, or
- Use `--state-cols ...` if supported, or
- Use a plugin that produces `state_*`.

### 13.2 “All cells uncertain / underpowered”
Cause:
- `min_samples` too high for the binning resolution / dataset size.

Fix:
- Increase sample size, or
- Lower `min_samples`, or
- Reduce `bins_x`/`bins_y`, or
- Reduce the number of equivalence classes via coarser state binning.

### 13.3 Recommendation returns nothing
Interpretation:
- Candidate `history_*` features may not be present, or
- The system is already faithful in this regime, or
- The dataset is underpowered.

Fix:
- Add candidate features via feature plugins, or
- Increase trials in the fractured regime, or
- Confirm augmentation promotes features into `state_*` (core requirement).

---