# Quick Start Tutorial (30 minutes)

This tutorial shows how to use **intervention-faithfulness** end-to-end on real data.
No background in the underlying theory is required.

If you can run Python and load a CSV, you can complete this in one sitting.

---

## Goal

By the end of this tutorial, you will be able to:

- run the intervention-faithfulness diagnostic on your data,
- determine whether your current state representation is **faithful under intervention**,
- identify **where** it breaks,
- and obtain a **concrete recommendation** for how to fix it (if needed).

---

## Installation

    pip install intervention-faithfulness

Verify installation:

    python -c "import intervention_faithfulness; print('OK')"

---

## What data you need

The tool works on **repeated trials** with:

- a proposed **state** (what your model thinks matters now),
- an **intervention** (what you change),
- an **outcome** (what happens next).

Typical examples:
- switching experiments,
- pulse protocols,
- control maneuvers,
- stimulus–response trials.

---

## Option A: Use a built-in plugin (recommended)

If your data matches a supported domain, plugins handle everything.

### Example: superconducting switching data

Assume you have a CSV like:

    trial_id, current, voltage, ramp_rate, time_to_switch
    1, 7.1, 0.02, 2.0, 12.4
    2, 7.1, 0.02, 2.0, 10.8
    3, 7.1, 0.02, 4.0, 3.2
    ...

### Run the diagnostic

    from intervention_faithfulness import FaithfulnessTest

    test = FaithfulnessTest.from_plugin(
        "nanowire_switching",
        "switching_runs.csv"
    )

    results = test.diagnose()

### Inspect results

    print(results.fracture_score)
    print(results.significance)

### Export a certificate

    results.export_certificate("model_validity.pdf")

That file is the **primary output** most users share.

---

## Interpreting the result

### Case 1: fracture ≈ 0

Your state representation is **faithful** in this regime.

- No hidden history is required.
- The model is safe to use under tested interventions.

This is a **certification**, not just a pass.

### Case 2: fracture > 0 and significant

Your state representation collapses distinct histories.

This means:
- two trials look identical to your model,
- but respond differently when you intervene.

This is an **Unfaithful Cut**.

---

## Minimal completion (the fix)

If fracture is detected, the tool ranks candidate history features.

    results.recommended_features

Example output:

    Feature                     ΔF
    --------------------------  ----
    integrated_current_50ns     0.38
    prev_switch_count           0.12
    time_since_last             0.05

Interpretation:
- adding `integrated_current` would reduce fracture by ~85%
- you do **not** need a high-order memory kernel

---

## Add a history feature and re-run

    test.add_feature("integrated_current", window_ns=50)
    results = test.diagnose()

Compare fracture before and after:

    print(results.fracture_score)

If fracture collapses, the fix worked.

---

## Faithfulness maps (where models break)

Faithfulness maps show **safe vs unsafe operating regions**.

    fig = results.faithfulness_map()
    fig.savefig("faithfulness_map.png")

Bright regions:
- your current state is valid

Dark regions:
- intervention reveals hidden history

These maps guide **protocol design**.

---

## Option B: Use your own data (no plugins)

Advanced users can bypass plugins entirely.

### Build the canonical trials table

    import pandas as pd

    df = pd.DataFrame({
        "trial_id": [...],
        "intervention_id": [...],
        "outcome": [...],
        "state_x": [...],
        "state_y": [...]
    })

### Run the diagnostic

    test = FaithfulnessTest(df)
    results = test.diagnose()

The core does not care about domain or physics.

---

## Common warnings (and what to do)

### ⚠ Underpowered estimate

    Only 18 trials per history class detected.
    Recommended minimum: 50.

Suggestions:
- pool similar histories,
- coarsen state binning,
- collect more data in this regime.

The tool warns; it does not block.

---

## Negative controls (sanity check)

The repository includes **faithful regimes** where fracture ≈ 0 is expected.

Run one before trusting your first result:

    test = FaithfulnessTest.from_plugin("faithful_regime", "data.csv")
    test.diagnose()

If this reports high fracture, something is wrong with your setup.

---

## Mental model (important)

You are **not** testing determinism.

You are testing:

> Do identical reduced states imply identical future distributions
> under the same intervention?

If not, the state is insufficient.

---

## What to do next

- Add domain-specific feature plugins
- Compare different state definitions
- Use faithfulness maps to redesign protocols
- Attach certificates to reports or grant updates

---

## One-sentence takeaway

**If your model works until you change the protocol, this tool tells you why—and how to fix it.**