# DESIGN.md — Intervention Faithfulness / Continuation Fracture
Version: v0.1 (design contract)  
Status: **Authoritative** for module boundaries, semantics, and stable external interfaces.

This document defines the **intended architecture and behavioral contracts** of the
`intervention_faithfulness` library and CLI. It exists to prevent drift across
implementations, contributors, and future refactors.

---

## 1. Problem Statement and Positioning

### What users are trying to do (their language)
Most users are looking for:
- **Causal model validation** / causal sufficiency testing
- **Model validity domains** (control engineers)
- **Hidden state detection** / non-Markovianity detection (physics)
- **Sim-to-real model mismatch** / epistemic uncertainty (robotics/AI)
- **Calibration validity** and **QCVV** regime validation (quantum hardware)

### What the tool does (our contribution)
We provide an empirical diagnostic for **intervention faithfulness** of a chosen reduced
state representation, using measured data only.

Core concept:
- **Continuation Fracture**: histories that map to the same reduced state produce
  **statistically distinguishable** outcome distributions under the same intervention.

Key framing sentence (contract):
> The diagnostic does **not** test determinism. It tests **invariance of conditional
> outcome distributions** under intervention.

---

## 2. Canonical Data Model (Core Contract)

All core algorithms operate on a single normalized internal table: the **Canonical Trials Table**.

### 2.1 Required columns
- `trial_id` : unique identifier (int/str)
- `intervention_id` : intervention label (str/int/float)
- `outcome` : outcome scalar (numeric) *or* an object/array payload
- `state_*` : one column per *reduced state* variable used by the model interface

### 2.2 Optional columns
- `history_*` : candidate history/completion features (for recommendations)
- `regime_*` : labels for slicing (device_id, temperature, run_id, etc.)
- `timestamp` : optional

### 2.3 Outcome storage guidance (important)
The canonical table may store outcomes in one of two ways:

1. **Scalar outcome** in `outcome` (preferred for v0.1)
2. **Pointer-based outcome** for heavy objects (e.g., trace blobs):
   - `outcome` may contain a small struct (path, key, index), or
   - the trials table contains a pointer column (e.g., `outcome_ref`) and a plugin supplies
     an adapter for divergence estimation.

**Core v0.1 guarantee:** `compute_continuation_fracture` must work on scalar outcomes.
Support for trace outcomes is allowed via plugins/adapters, but not required by core.

---

## 3. Semantics

### 3.1 Candidate state mapping
A practitioner supplies a reduced state representation implicitly via columns `state_*`.

Let `S` denote the reduced state vector and `I` the intervention label.

### 3.2 Continuation invariance criterion (ideal)
For histories `h1, h2` mapping to the same reduced state `S`:
\[
P(Y_{\mathrm{future}} \mid S, I, h_1) = P(Y_{\mathrm{future}} \mid S, I, h_2)
\]
In practice we test **distinguishability**, not equality (finite sample).

### 3.3 Continuation Fracture metric
We estimate a divergence between conditional continuation distributions, aggregated across
state-equivalence classes and interventions.

Core divergences (v0.1):
- Jensen–Shannon divergence (`js`) as the default bounded/symmetric option
- Wasserstein distance (`wasserstein`) when outcome has a metric structure

### 3.4 Tail mode (risk/safety)
Tail mode restricts outcomes to a high-quantile subset (e.g., `q ≥ 0.95`) before computing
divergences.

Contract:
- Tail restriction must be applied **consistently** across baseline and augmented runs.
- Tail restriction is part of the **definition of the metric**, not a visualization trick.

---

## 4. Public API Surface (Stable Interfaces)

### 4.1 `FaithfulnessTest`
Primary entry point.

**Constructor**
- `FaithfulnessTest(trials_df: pd.DataFrame, *, metadata: Optional[dict] = None, ...)`

Contract:
- If `trials_df` contains no `state_*` columns, the implementation may:
  - warn and proceed with an empty state (valid but often uninformative), OR
  - accept an explicit `state_cols=[...]` argument and promote them to `state_*`.
  This behavior must be documented and tested if implemented.

**Core method**
- `diagnose(config: Optional[DiagnoseConfig] = None) -> DiagnosisResult`

Contract:
- `diagnose()` must:
  1) validate and normalize the trials table
  2) compute fracture metric(s)
  3) compute significance (if enabled)
  4) compute recommendations (if enabled)
  5) compute maps/envelopes (if enabled)
  6) return a `DiagnosisResult` containing:
     - results + reporting bundles
     - the canonical trials table reference
     - configuration + metadata + schema warnings

### 4.2 `DiagnoseConfig`
Dataclass controlling the run.

Contract:
- Library defaults live in `DiagnoseConfig` (single source of truth).
- CLI should prefer `default=None` and defer to `DiagnoseConfig`.

Required stability:
- `min_samples`, `divergence`, `tail_mode`, `quantile_focus`, `n_bins`
- `recommend` and associated knobs (`recommend_mode`, etc.)
- map/envelope knobs (`bins_x`, `bins_y`, thresholds, etc.) if exposed

### 4.3 `DiagnosisResult` (Reporting / UX surface)
Encapsulates computed results and export functions.

Contract:
- Must expose:
  - `fracture_score` (value + CI/metadata)
  - `significance` (p-value + method)
  - `recommended_features` (list)
  - `safe_operating_regions` (list)
  - `breakdown_df` (cell/group breakdown suitable for CSV export)
- Must support deterministic exports (see §7).

---

## 5. Core Modules and Responsibilities

### 5.1 `core/schema.py`
Responsibilities:
- Validate canonical trials table schema.
- Provide warnings for missing columns or suspicious conditions.
- Provide lightweight normalization utilities (e.g., ensuring column existence).

Contract:
- Must not import heavy visualization deps (matplotlib/reportlab) if avoidable.

### 5.2 `core/fracture.py`
Responsibilities:
- Implement fracture metrics.
- Provide a single canonical function:

`compute_continuation_fracture(trials_df, *, metric, divergence, min_samples, tail_mode, quantile_focus, n_bins, n_pairwise_pairs, n_permutations, ...)`

Contracts:
- Must identify state columns via `state_*` prefix.
- May identify history columns via `history_*` prefix.
- Must return a structured result object with:
  - `fracture_value` (primary scalar)
  - any additional per-metric breakdown needed by reporting
  - (optionally) permutation statistics if requested

### 5.3 `core/recommendation.py`
Responsibilities:
- Rank minimal completions / missing features that reduce fracture.

Two conceptual modes:
- **Mode A (single feature / independent ranking)**:
  - Evaluate each candidate history feature individually as an augmentation.
- **Mode B (greedy / set search)**:
  - Search for small sets of features whose joint addition reduces fracture.

Critical contract (prevents a known bug):
- When a candidate feature is evaluated as a completion, it must be **promoted to state**
  for the augmented fracture evaluation.
  - i.e., augmentation must create `state__aug__<feature>` or equivalent `state_*` column,
    because fracture computation selects state by prefix.

If recommendations return “no useful features”:
- This is allowed and should be interpreted as:
  - insufficient candidate features, OR
  - insufficient power, OR
  - genuinely faithful regime

### 5.4 `core/maps.py`
Responsibilities:
- Compute grids (not only plots).
- Provide a function that returns a grid suitable for:
  - plotting
  - safe envelope derivation

Contract:
- Separate **grid computation** from **rendering**.

Required functions:
- `compute_faithfulness_grid(trials_df, config: MapConfig, *, faithfulness: bool) -> (grid, counts, x_edges, x_labels, y_edges, y_labels, cfg_used)`
- `plot_faithfulness_map(...) -> matplotlib.figure.Figure`

### 5.5 `core/reporting.py`
Responsibilities:
- Provide `DiagnosisResult` class.
- Provide export functions:
  - JSON records
  - curated certificate payloads
  - artifact bundles (directory export)
  - optionally certificate PDF/HTML renderers

Contracts:
- Exports must be deterministic and auditable.
- Must include cryptographic hashes:
  - hash of the canonical trials table serialization
  - hash of the diagnosis payload (or a stable subset)

Time handling contract:
- Use timezone-aware UTC timestamps (`datetime.now(UTC)`), not deprecated naive `utcnow()`.

---

## 6. Plugins (Data and Feature)

The plugin layer exists to keep the core **domain-agnostic** while enabling frictionless
usage across domains.

### 6.1 Data plugins
Convert raw lab logs / domain formats into the canonical trials table.

Suggested interface:
- `load(source, **kwargs) -> RawData`
- `to_trials(raw, **kwargs) -> pd.DataFrame`
- `defaults() -> dict` (domain defaults for DiagnoseConfig)

### 6.2 Feature plugins
Add `history_*` candidate features to an existing canonical trials table.

Suggested interface:
- `compute(trials_df, **params) -> pd.DataFrame` (adds columns)

Contract:
- Plugins may be opinionated.
- Core must remain clean and not depend on any specific plugin.

---

## 7. Artifacts, Certificates, and Auditability

### 7.1 Artifact bundle export
`DiagnosisResult.export_artifacts(out_dir, *, include_trials, include_map, include_certificate, prefix, ...) -> Dict[str, str]`

Bundle intent:
- a stable “audit kit” suitable for CI, nightly runs, and attachments.

Expected outputs (v0.1):
- `*_diagnosis.json`  (full diagnosis record)
- `*_certificate.json` (curated certificate payload)
- `*_breakdown.csv`   (breakdown table)
- optional `*_trials.csv` (canonical trials table snapshot)
- optional `*_map.png` or saved figure (if implemented)

Contract:
- Export function keyword args are **the canonical interface**.
- CLI must match this interface exactly (see §8).

### 7.2 Certificate export
Two layers:
1) **Curated JSON certificate** (always available; light-weight)
2) Optional **PDF/HTML renderers** (may be called separately)

Contract:
- If `export_artifacts()` exports certificate JSON, it must do so via a method that
  actually exists on `DiagnosisResult` (e.g., `export_certificate_json()`).
- If PDF/HTML are desired, CLI should call `export_certificate(..., format="pdf")`
  separately (unless export_artifacts explicitly supports it).

---

## 8. CLI Design and Contract

### 8.1 CLI goals
- Provide a reliable “one command” path:
  - load data
  - run diagnose
  - export artifact bundle

- Serve automation use cases:
  - CI pipelines
  - nightly regression checks
  - compliance/audit trails

### 8.2 CLI module: `cli.py`
Responsibilities:
- Parse arguments
- Build a `DiagnoseConfig`
- Instantiate `FaithfulnessTest`
- Call `diagnose()`
- Export artifacts

**Critical contract: CLI must not invent keyword arguments**
Known failure mode:
- CLI used `write_trials_csv=True`, `write_map=True`, etc.
- Reporting expected `include_trials`, `include_map`, `include_certificate`

**Contract (authoritative mapping):**
- CLI must call:

```python
written = res.export_artifacts(
    out_dir=str(out_dir),
    include_trials=args.write_trials,     # or equivalent CLI flag
    include_map=args.map,
    include_certificate=args.certificate, # certificate JSON bundle
)
```

If PDF/HTML requested:

python
Copy code
if args.pdf:
    res.export_certificate(str(out_dir / f"{prefix}_certificate.pdf"), format="pdf")
if args.html:
    res.export_certificate(str(out_dir / f"{prefix}_certificate.html"), format="html")
### 8.3 CLI defaults policy
CLI should set most defaults to None and rely on DiagnoseConfig defaults.

CLI may implement convenience toggles (e.g., --map, --certificate, --trials)
but must not replicate numeric defaults that exist in config.

### 8.4 Minimal UX guarantees
At minimum, CLI should print progress markers:

“Loaded N trials”

“Diagnosing…”

“Exported artifacts to …”

Optional later:

tqdm progress bars inside inner loops (not required for v0.1).

## 9. Failure Modes and Guardrails (Must Be Tested)
### 9.1 Underpowered regimes
Any cell / class with < min_samples must be treated as “uncertain” for envelope labeling.

Tool must warn when large fractions of the grid are underpowered.

### 9.2 Continuous state equivalence
Exact equality for continuous state_* is rare.
Contract:

Core must implement binning/discretization (or clustering) to construct equivalence classes.

Documentation must explicitly state that “state equivalence” is operationalized via binning.

### 9.3 Stochastic but faithful vs fractured
Fracture is not “variance.” It is conditional distribution non-invariance.
Contract:

The metric must compare conditional distributions, not point predictions.

Negative controls must show F≈0 in faithful regimes.

## 10. Example Battery (Release Contract)
The tool should ship with:

Positive case (fracture present): nanowire-like protocol dependence

Negative control (faithful regime): same pipeline, shows F≈0

Pedagogical example: simple synthetic system (RC, oscillator, etc.)

Contract:

Examples must run end-to-end from CLI and from Python API.

Examples must export a bundle artifact that includes hashes.

## 11. Naming and Messaging (Practical Adoption)
### 11.1 Search terms (external)
Project README/docs should include these phrases prominently:

Causal model validation

Causal sufficiency test

Model validity domain

Hidden state detection

Non-Markovian effects

Intervention robustness / intervention validity

### 11.2 Core terms (internal)
Continuation Fracture

Unfaithful Cut

Safe Operating Envelope

Minimal Completion / Feature Promotion

Contract:

Keep the math and internal terms, but always provide a “translation” section in docs.

## 12. Non-Goals (Scope Discipline)
Explicitly out of scope (v0.1):

Microscopic physics models (TDGL/RSJ/etc.)

Automatic root cause inference

Full dynamics reconstruction

“Hallucinating sensors” (recommendations only select from measured/provided candidates)

## 13. Change Control (How to prevent future drift)
Any PR or change must:

Update this DESIGN.md if it changes a contract surface

Add/adjust unit tests if it fixes a known failure mode

Maintain backwards compatibility for:

FaithfulnessTest.diagnose

DiagnosisResult.export_artifacts

canonical trials schema

## 14. Appendix: Quick Reference
Canonical trials columns
Required: trial_id, intervention_id, outcome, state_*

Optional: history_*, regime_*, timestamp

Major entry points
Python:

FaithfulnessTest(df).diagnose(cfg)

DiagnosisResult.export_artifacts(...)

CLI:

faithfulness diagnose --csv data.csv --out-dir ./report --map --certificate
