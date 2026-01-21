# API Overview

This document provides a **high-level overview of the public API** for
*intervention-faithfulness*. It is intended for users who want to go beyond
the tutorials but do not need contributor-level details.

For step-by-step usage, see `tutorials/`.
For contributor requirements, see `docs/plugin_spec.md`.

---

## Core Concepts

The API is organized around a single primary object:

- **FaithfulnessTest** — orchestrates data ingestion, diagnostics, and reporting.

All other components exist to support this object.

---

## Primary Entry Point

### FaithfulnessTest

Create a test from either:
- a plugin (recommended), or
- a canonical trials table.

Examples:

    from intervention_faithfulness import FaithfulnessTest

    test = FaithfulnessTest.from_plugin("nanowire_switching", "data.csv")

or

    test = FaithfulnessTest(trials_df)

---

## Running the Diagnostic

### diagnose()

Runs the continuation fracture diagnostic.

    results = test.diagnose()

Optional configuration can be supplied to control divergence choice,
sample thresholds, tail sensitivity, and recommendation behavior.

The output is a **DiagnosisResult** object.

---

## Feature Augmentation

### add_feature()

Attach a history feature plugin before running the diagnostic.

    test.add_feature("integrated_current", window_ns=50)

Multiple features may be chained.

Feature plugins add `history_*` columns to the trials table.

---

## Results Object

### DiagnosisResult

Encapsulates all outputs of the diagnostic.

Common attributes:
- fracture score
- statistical significance (if enabled)
- warnings
- recommended features (optional)

Common methods:
- export_certificate(...)
- plot_faithfulness_map(...)
- export_json(...)

The results object contains **no interpretation policy**.

---

## Faithfulness Maps

Faithfulness maps visualize where a representation is valid or invalid
across intervention or history dimensions.

Maps are generated from a DiagnosisResult and reflect only tested regimes.

---

## Plugins

Plugins are optional and provide:
- data loading and canonicalization,
- sensible defaults for diagnostics,
- reusable feature generators.

Plugins are discoverable at runtime and do not modify core behavior.

---

## Schema Contract

All diagnostics operate on a **canonical trials table** with required fields:
- trial_id
- intervention_id
- outcome

Optional fields include:
- state_*
- history_*

This schema is enforced consistently across the API.

---

## What the API Guarantees

- Deterministic execution given fixed data and configuration
- No hidden domain assumptions in core logic
- Clear separation between diagnostics and interpretation

What it does not guarantee is defined in `CERTIFICATION.md`.

---

## When to Use This API

Use the API when you want to:
- validate a reduced model under intervention,
- identify hidden history dependence,
- map safe operating regimes,
- communicate model validity clearly.

Do not use it to infer physical mechanisms or causal structure.

---

## One-sentence summary

**The API gives you a reliable way to test whether your state representation
survives intervention.**