# Plugins Guide

This guide lists the **built-in plugins** available in *intervention-faithfulness*, what data they expect,
and when you should use them. Plugins are optional convenience layers: they load data, choose sensible
defaults, and (optionally) provide feature generators.

If your data doesn’t match any plugin, you can always construct the canonical trials table manually
(see the Quick Start).

---

## How to use a plugin

Typical usage:

    from intervention_faithfulness import FaithfulnessTest

    test = FaithfulnessTest.from_plugin(
        "plugin_name",
        "path/to/data"
    )

    results = test.diagnose()

Plugins set **defaults** (divergence choice, tail sensitivity, minimum samples) appropriate to the domain.
You can override any default via `diagnose()` configuration or by adding features.

---

## Available Data Plugins

### nanowire_switching

**Domain:** Superconducting nanowire switching experiments  
**Use when:** You have repeated switching trials under current or pulse interventions.

**Expected inputs (CSV / HDF5):**
- `trial_id`
- `current`
- `voltage`
- `ramp_rate` (or equivalent intervention variable)
- `time_to_switch` (numeric outcome)

**What the plugin does:**
- Constructs the canonical trials table
- Uses `(current, voltage)` as the default state
- Treats ramp rate as the intervention
- Enables tail-sensitive analysis (late/rare switches matter)

**Typical questions it answers:**
- Does instantaneous current suffice, or is ramp history required?
- Why does switching behavior change under faster pulses?
- Where is my reduced model valid vs unsafe?

---

### faithful_regime

**Domain:** Negative control / sanity check  
**Use when:** You want to verify that the diagnostic does *not* report false positives.

**Expected inputs:**
- Same structure as the primary domain plugin
- Data collected in a regime known to be well-behaved (e.g., slow ramps, far below critical thresholds)

**What the plugin does:**
- Runs the identical pipeline as the main plugin
- Should produce fracture ≈ 0 if everything is configured correctly

**Why this matters:**
- Demonstrates that detected fracture is *regime-dependent*
- Confirms you are not just measuring noise or undersampling

---

### pedagogical_rc

**Domain:** Tutorial / teaching example  
**Use when:** You are learning the tool or teaching the concept.

**Inputs:**
- None required (synthetic data generated internally)

**What the plugin does:**
- Generates a simple RC-like system
- Produces identical instantaneous states with different histories
- Demonstrates continuation fracture under intervention

**Why this exists:**
- Lets you understand the diagnostic without domain knowledge
- Useful for onboarding students and new users

---

## Feature Plugins (History Augmentations)

Feature plugins add candidate **history variables** (`history_*`) that may restore faithfulness.

You attach them explicitly:

    test.add_feature("integrated_current", window_ns=50)

### integrated_current

**Purpose:** Proxy for thermal or phase-winding history  
**Adds:** `history_integrated_current`

**Parameters:**
- `window_ns` — integration window length

**Use when:**
- Switching or response depends on how fast or how long you approached the intervention
- Thermal or dissipative memory is suspected

---

### previous_switches

**Purpose:** Event-history memory  
**Adds:** `history_prev_switch_count` (or similar)

**Use when:**
- Outcomes depend on recent switching or reset events
- You suspect incomplete relaxation between trials

---

## Choosing the right plugin

Start here:

- **Have switching or pulse data?** → `nanowire_switching`
- **Want a sanity check?** → `faithful_regime`
- **Just learning the tool?** → `pedagogical_rc`
- **None fit?** → Build the canonical table manually (no plugin)

---

## When not to use a plugin

Do **not** use a plugin if:
- your data schema is very different,
- you already have a well-defined canonical trials table,
- you want full control over state and history variables.

In that case:

    test = FaithfulnessTest(trials_df)

Plugins are conveniences, not requirements.

---

## Extending with your own plugin

If you want to support a new domain:
- see `docs/plugin_spec.md` for the contributor-facing specification,
- implement a data plugin (loader + canonicalizer),
- optionally add feature plugins for domain-relevant histories.

Community-contributed plugins are encouraged.

---

## One-sentence takeaway

**Plugins remove setup friction, but the diagnostic itself is always domain-agnostic.**