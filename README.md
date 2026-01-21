# intervention-faithfulness

**Empirical certification of model validity under intervention.**

`intervention-faithfulness` is a tool for determining whether a chosen **state representation**
remains predictive when you change the protocol.

It answers a single, practical question:

> *Does my model still tell the truth when I intervene?*

The tool does **not** assume a microscopic theory, does **not** reconstruct dynamics,
and does **not** try to explain mechanisms.  
It certifies whether a reduced state is **intervention-faithful**, identifies **where it fails**,
and recommends the **minimal additional structure** needed to restore validity.

---

## Why this exists

Many systems behave perfectly under steady conditions, then fail when you change how you drive them:

- superconducting nanowires that switch unpredictably under fast ramps  
- controllers that work in testing but fail under new maneuvers  
- materials with hysteresis or aging effects  
- models that “need memory kernels” to fit new protocols  

In these cases the problem is often **not the equations**, but the **state definition**.

This tool detects when:
- different histories collapse to the same state,
- but diverge immediately under intervention.

That condition is called an **Unfaithful Cut**.

---

## What the tool does

Given experimental or simulated trial data, the tool automatically:

- computes a **Continuation Fracture** score (with confidence intervals),
- tests statistical significance via permutation testing,
- generates **Faithfulness Maps** (safe vs unsafe operating regimes),
- ranks **minimal state augmentations** (history features) that restore validity,
- produces a **Model Validity Certificate** (JSON, with optional PDF/HTML rendering).

It works on real data, with sensible defaults, in minutes.

---

## Σ₂-I guard (policy layer)

`diagnose()` is **threshold-free reporting**: it computes fracture, envelope, recommendations, and certificate payloads.

`intervention_faithfulness/core/guard.py` is **policy**: it consumes a `DiagnosisResult` and returns an enforceable decision:
- **OK**
- **BOUNDARY**
- **REFUSE**

Guard policy is conservative by design:
- **Underpowered ⇒ BOUNDARY (never OK).**
- **Unsafe envelope or high fracture ⇒ REFUSE.**

Python usage:

    from intervention_faithfulness.core.guard import GuardConfig, decide

    result = test.diagnose()
    decision = decide(
        result,
        GuardConfig(fracture_threshold=0.12, min_effective_samples=200)
    )
    print(decision.status, decision.reason)

If using the CLI guard command, the exit codes are:
- `0` = OK
- `2` = BOUNDARY
- `3` = REFUSE

Certificates are exported as **curated JSON** by default, with optional **PDF/HTML rendering** via `export_certificate(...)`.

---

## Quick start (5 minutes)

### Install

    pip install intervention-faithfulness

### Run on switching data (no configuration required)

    from intervention_faithfulness import FaithfulnessTest

    test = FaithfulnessTest.from_plugin(
        "nanowire_switching",
        "data/switching_runs.csv"
    )

    results = test.diagnose()
    results.export_certificate("model_validity.pdf")

That’s it.

You now have:
- a fracture score with uncertainty,
- a significance test,
- recommended missing state variables,
- and a safe/unsafe operating envelope.

---

## What problem this solves (and what it doesn’t)

### This tool **does**
- certify whether a state representation is valid **under intervention**
- distinguish *stochastic but faithful* behavior from *structural failure*
- tell you **when your model breaks**, not just that it breaks
- recommend **low-dimensional fixes** instead of complex kernels

### This tool **does not**
- assume a microscopic mechanism (vortices, hotspots, etc.)
- replace simulators or domain models
- infer root causes
- reconstruct full dynamics

It sits **above** existing models as a certification layer.

---

## Plugin system (how it stays frictionless)

The core diagnostic is **domain-agnostic**.  
Plugins handle domain-specific data formats, defaults, and feature libraries.

### List available plugins

    from intervention_faithfulness import list_plugins
    list_plugins()

Example output:

    Data Plugins:
      - nanowire_switching    Superconducting nanowire switching experiments
      - faithful_regime       Negative control (slow / far-below-Ic operation)
      - pedagogical_rc        Synthetic RC circuit (tutorial)

    Feature Plugins:
      - integrated_current    ∫I(τ)dτ history proxy
      - prev_switch_count    Recent switching events
      - time_since_last      Time since last event

### Get help for a plugin

    FaithfulnessTest.plugin_help("nanowire_switching")

Plugins choose sensible defaults so new users don’t have to.

---

## Typical workflows

### 1. Novice user (just wants it to work)

    test = FaithfulnessTest.from_plugin("nanowire_switching", "data.csv")
    results = test.diagnose()
    results.export_certificate()

### 2. Intermediate user (override defaults + add features)

    test = FaithfulnessTest.from_plugin(
        "nanowire_switching",
        "data.h5",
        divergence="wasserstein"
    )

    test.add_feature("integrated_current", window_ns=50)
    results = test.diagnose()

### 3. Advanced user (no plugins)

    import pandas as pd

    df = pd.DataFrame({
        "trial_id": [...],
        "intervention_id": [...],
        "outcome": [...],
        "state_I": [...],
        "state_V": [...]
    })

    test = FaithfulnessTest(df)
    results = test.diagnose()

---

## Output artifacts

### Model Validity Certificate (auto-generated)

    MODEL VALIDITY CERTIFICATE
    ━━━━━━━━━━━━━━━━━━━━━━━━━━
    System: NbN nanowire
    State: (I, V)
    Intervention: Current ramp

    STATUS: UNFAITHFUL
    Continuation Fracture: F = 0.42 ± 0.08
    Significance: p < 0.001

    RECOMMENDED FIX:
    Add ∫I(τ)dτ over 50 ns
    Expected fracture reduction: 85%

    SAFE OPERATING REGIME:
    ✓ Ramp rate < 2 GHz
    ✗ Ramp rate > 3 GHz

This is the artifact people actually share.

---

## Faithfulness Maps

Faithfulness maps show **where your model is safe** and **where it breaks**:

- x-axis: intervention strength (e.g. ramp rate)
- y-axis: history depth / protocol intensity
- color: continuation fracture

Bright = certified  
Dark = intervention-invalid

---

## Negative controls (why you can trust it)

The repository includes **faithful regimes** where reduced models are known to work.

In these cases the diagnostic correctly reports:
- fracture ≈ 0
- stable bright regions
- no recommended state augmentation

The tool does not “cry wolf.”

---

## Supported investigation areas

The API applies anywhere you have:
- repeated trials,
- a proposed state summary,
- and controlled interventions.

Examples:
- superconducting devices (nanowires, JJs, qubits)
- control systems and robotics
- hysteretic materials
- chemical reactors
- neuroscience stimulation protocols
- network and congestion systems
- ML system monitoring under policy change

If your model works until you change the protocol, this tool applies.

---

## Project philosophy

- **Tool-first, not paper-first**
- **Certification, not explanation**
- **Minimal fixes, not complexity**
- **Engineers first**

The accompanying paper documents the method; the tool is the product.

---

## Contributing

- New **data plugins** welcome
- New **feature plugins** welcome
- Core algorithms are intentionally conservative

See `docs/plugin_spec.md` for details.

---

## Citation

If you use this tool, please cite the accompanying paper:

    [Title TBD]
    [Authors TBD]
    [Journal / arXiv TBD]

Software citation instructions will be added upon release.

---

## License

MIT License.

---

## One-sentence summary

**`intervention-faithfulness` tells you whether your model is valid when you actually use it.**