# Plugin Specification (v0.1)

This document specifies the **plugin layer** for `intervention-faithfulness`.

Plugins solve the tension between:

- keeping the **core** diagnostic stable and domain-agnostic, and
- making the tool **frictionless** for real users with real data.

This spec is intentionally minimal. It is expected to evolve; changes should preserve:
1) backwards compatibility for the public API whenever possible, and  
2) a strict separation: **core never depends on domain logic**.

---

## Status

- **Version:** v0.1 (initial)
- **Stability goal:** stable enough to build the first 3 plugins and the public API.
- **Expected evolution:** moderate. The canonical schema should remain stable; metadata fields may expand.

---

## Terms

- **Core:** model-agnostic implementation of continuation fracture, statistics, maps, ranking, reporting.
- **Data plugin:** converts raw domain data (CSV/HDF5/custom) into the canonical *trials table*.
- **Feature plugin:** adds candidate `history_*` columns to the trials table for minimal completion ranking.
- **Trials table:** canonical normalized DataFrame consumed by core algorithms.

---

## Non-goals (by design)

Plugins do **not** implement:
- microscopic models (TDGL, RSJ, vortex/hotspot simulators),
- causal root-cause inference,
- full dynamics reconstruction,
- automatic physical interpretation of features.

Plugins are a **translation and defaults layer**, not a physics engine.

---

## Canonical Trials Table

All core algorithms operate on a single normalized table.

### Required columns

- `trial_id`  
  Identifier for a trial/run/shot. May be string or integer.

- `intervention_id`  
  Identifier for the intervention applied at decision time.  
  May be a scalar, string label, or a hashable object encoding multiple parameters.

- `outcome`  
  Post-intervention measured outcome.  
  May be scalar (e.g., switching time), categorical (e.g., switched/not), or a vector (stored as an object).

### Recommended columns

- At least one `state_*` column  
  Example: `state_I`, `state_V`, `state_T`

### Optional columns

- `timestamp`  
  If available, used for some feature plugins and ordering checks.

- `regime_*` columns  
  Metadata for slicing / stratifying (e.g., `regime_device`, `regime_temp`).

- `history_*` columns  
  Candidate history features produced by feature plugins.

### Column naming conventions

- State variables must be prefixed with `state_`
- History-derived features must be prefixed with `history_`
- Regime / metadata columns should be prefixed with `regime_`

This enables wildcard selection and safe extension.

---

## Schema Validation

Core provides a single validator used by all plugins.

### Errors (hard failure)
- Missing required columns: `trial_id`, `intervention_id`, `outcome`

### Warnings (soft)
- No `state_*` columns present
- Non-hashable `intervention_id`
- Mixed dtypes in `outcome` (unless explicitly allowed)
- Suspected underpowered sampling (see below)

Data plugins may add additional validation warnings, but must not weaken core requirements.

---

## Plugin Registry and Discovery

### Registration & imports (deterministic discovery)

Built-in plugins are registered **by import side effects**: plugin modules call
`register_*` functions or registry decorators at import time.

To prevent order-dependent failures (e.g. “plugin not found unless I imported X first”)
and to avoid multiple competing plugin loaders, the core guarantees deterministic
registration as follows:

- `FaithfulnessTest` calls `_ensure_builtin_plugins_registered()` (or equivalent)
  before resolving plugin names.
- `intervention_faithfulness.plugins.data.__init__` and
  `intervention_faithfulness.plugins.features.__init__` **must import the concrete
  plugin implementations**, so registration occurs on import.
- Plugin modules must expose the **actual symbols imported by those `__init__.py` files**
  (avoid mismatches like defining `FooPlugin` but importing `BarPlugin`).

If a plugin does not appear in `list_plugins()`, the fix is **import/registration wiring**,
not adding a new plugin discovery mechanism.

### Built-in registry (v0.1)
Plugins shipped in the repo register themselves in a runtime registry.

User-facing functions:

- `list_plugins()`  
  Prints/returns available data and feature plugins with summaries.

- `FaithfulnessTest.plugin_help(name)`  
  Displays plugin metadata and example usage.

### Optional extension mechanism (planned)
Support third-party plugins via Python packaging entry points (v0.2+), without changing the user-facing API.

---

## Plugin Metadata

All plugins must provide a `PluginMetadata` object.

### Required metadata fields

- `name` (str)  
  Unique key used in `from_plugin()` and `add_feature()`.

- `description` (str)  
  One-line summary suitable for `list_plugins()`.

- `expected_format` (str)  
  Human-readable description of expected input schema.

- `example_usage` (str)  
  A minimal example demonstrating typical usage.

### Optional metadata fields (v0.1)
- `tags` (list[str])  
  Used for discovery/search in `list_plugins()`.

- `links` (dict[str, str])  
  Documentation links.

---

## Data Plugin Interface

Data plugins convert raw sources into the canonical trials table and supply sensible defaults.

### Base class: `DataPlugin`

Required methods:

- `load(source, **kwargs) -> Any`  
  Load raw data from a file, directory, handle, or preloaded object.

- `to_trials(raw, **kwargs) -> pd.DataFrame`  
  Convert raw data into the canonical trials table.

Optional methods:

- `defaults() -> dict`  
  Recommended diagnostic settings for this domain.

- `validate(df: pd.DataFrame) -> list[str]`  
  Additional domain-specific validations. Returns warnings (strings).

### Required behavior

- `to_trials()` must return a DataFrame satisfying required schema.
- Plugins must not mutate global state.
- Defaults should be conservative and explainable.

### Typical defaults keys (non-exhaustive)

- `divergence`: `"js"` (recommended default)
- `min_samples`: `50`
- `tail_mode`: `True|False`
- `quantile_focus`: e.g., `0.95` if tail-focused
- `binning`: recommended binning strategy (if used)

Plugins should not introduce new default keys without also documenting them in `plugin_help()` output.

---

## Feature Plugin Interface

Feature plugins add candidate `history_*` columns used for minimal completion ranking.

### Base class: `FeaturePlugin`

Required methods:

- `compute(trials_df: pd.DataFrame, **params) -> pd.DataFrame`  
  Returns a DataFrame with one or more `history_*` columns added.

Optional methods:

- `parameters() -> dict`  
  Describes tunable parameters and defaults for introspection/help.

- `requires() -> list[str]`  
  Declares required input columns (e.g., `["state_I"]`). If missing, core raises an actionable error.

### Required behavior

- Must not overwrite existing columns unless explicitly allowed.
- Must prefix newly created columns with `history_`.
- Should be deterministic given the same inputs and parameters.

### Two execution modes for history features (recommended)

Because many datasets are “one row per trial,” feature plugins may operate in:

1) **Sequence mode** (preferred)  
   Uses within-trial time series if available (e.g., stored as arrays in `outcome` or additional columns).

2) **Proxy mode** (fallback)  
   Uses shot-to-shot summaries and computes history across trials (e.g., EWMA of recent dissipation).

Feature plugins must document which modes they support.

---

## Underpowered Data Handling

Core should warn (not block) when estimates are likely underpowered.

### Minimum sample rule (default)
- `min_samples = 50` per history-equivalence grouping (or per bin)

If below minimum, core emits a warning including suggested remedies:
- pool history classes,
- coarsen state binning,
- collect more data in that regime.

Data plugins may set a different `min_samples` if justified (e.g., cryogenic constraints), but must document it.

---

## `from_plugin()` and `add_feature()` Semantics

### `FaithfulnessTest.from_plugin(...)`

- Looks up a named **data plugin**
- Calls `plugin.load(source, ...)`
- Calls `plugin.to_trials(raw, ...)`
- Runs schema validation
- Applies plugin defaults unless overridden by user

### `test.add_feature(name, **params)`

- Looks up a named **feature plugin**
- Checks `requires()` against current trials table
- Calls `compute(trials_df, **params)`
- Validates resulting columns (must be `history_*`)
- Records feature provenance for reporting

---

## Reporting Integration

Certificates and summaries must include:

- data plugin name and version (if available),
- feature plugins applied + parameter settings,
- defaults used and user overrides.

This supports reproducibility and “auditability” for engineering workflows.

---

## Starter Plugins (v0.1)

### Data plugins
- `nanowire_switching`  
  Switching trials from superconducting nanowires / JJs (CSV/HDF5).

- `faithful_regime`  
  Negative control regime loader/slicer using the same schema.

- `pedagogical_rc`  
  Synthetic tutorial dataset.

### Feature plugins
- `integrated_current`  
  ∫I(τ)dτ history proxy (sequence mode if time series exists; proxy mode otherwise).

- `prev_switch_count`  
  Recent event count/timing.

- `time_since_last`  
  Time since last event.

- `ewma_dissipation`  
  Exponentially-weighted recent power/energy proxy.

---

## Compatibility & Evolution Policy

This spec will evolve. To keep the ecosystem stable:

- The canonical required columns will remain stable.
- New optional metadata fields may be added freely.
- Plugin interfaces may gain optional methods but should not break existing plugins.
- Deprecated behaviors should remain supported for at least one minor release.

---

## Appendix: Minimal Example Canonical Table

    trial_id  intervention_id  outcome  state_I  state_V  history_prev_switches
    -------  ---------------  -------  -------  -------  ----------------------
    1        ramp_2GHz        12.4     7.1      0.02     0
    2        ramp_2GHz        10.8     7.1      0.02     1
    3        ramp_4GHz         3.2     7.1      0.02     0
    4        ramp_4GHz         2.9     7.1      0.02     1

This table is the only contract core requires.
Everything else is plugin responsibility.