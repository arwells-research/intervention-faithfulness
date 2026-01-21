# Troubleshooting Guide

This guide lists common problems, why they occur, and what to do next when using
intervention-faithfulness on real data.

If something looks confusing or wrong, start here.

---

## Installation issues

### ImportError or ModuleNotFoundError

Symptom:

    ModuleNotFoundError: No module named 'intervention_faithfulness'

Fix:
- Verify installation:

      pip install intervention-faithfulness

- Verify environment:

      python -c "import intervention_faithfulness; print('OK')"

- Make sure you are using the same Python environment where you installed the package.

---

## Schema errors

### Missing required columns

Symptom:

    ValueError: Missing required columns: ['trial_id', 'intervention_id', 'outcome']

Cause:
Your DataFrame is not in canonical trials format.

Fix:
Ensure your table contains at least:
- trial_id
- intervention_id
- outcome

State and history variables should be prefixed with:
- state_*
- history_*

---

### Warnings about low sample size

Symptom:

    ⚠ Only 18 trials per intervention detected.

Cause:
Fracture estimates are underpowered.

Fix options:
- Collect more trials
- Pool similar interventions
- Reduce binning resolution
- Focus on fewer candidate histories

The tool warns but does not block execution.

---

## Unexpected high fracture

### My model works fine — why is fracture high?

Cause:
Your model may be valid under passive observation but fail under intervention.

High fracture often indicates one or more of:
- hidden history dependence (non-Markovianity at the chosen state level)
- hysteresis or metastability
- incomplete reset between trials
- rate-dependent effects (dI/dt or pulse timing matters)
- unmeasured local states (e.g., hotspots, vortex configurations, quasiparticles)

What to do:
1) Run a negative control regime (known-faithful operating region).
2) Enable minimal completion ranking and try the top 1–3 suggested features.
3) Reduce state cardinality (coarsen/round state variables) to increase repeats per state.
4) Check that intervention labels are correct and not leaking hidden variables.

---

## Unexpected low fracture

### I expected memory effects, but fracture is near zero

Possible causes:
- You are operating in a slow or well-relaxed regime where reduced models are valid.
- Your intervention is too weak to expose hidden history.
- Your outcome variable is too coarse to show divergence.

What to do:
- Increase intervention strength or ramp rate (carefully, within safe limits).
- Use a more sensitive outcome (e.g., time-to-switch instead of binary switched/not).
- Increase sample size to resolve tail behavior.

---

## Recommendation engine returns nothing useful

### Recommended features do not reduce fracture

Possible causes:
- The missing variable is not present in your measured channels.
- You need deeper history than provided by current feature candidates.
- The system is genuinely high-dimensional in the tested regime.

What to do:
- Add new history feature candidates (domain-informed).
- Increase history window depth.
- Consider collecting an additional measurement channel that plausibly encodes the hidden state.

---

## Faithfulness map looks noisy or patchy

Cause:
Cell counts are too low for stable estimation, or binning is too fine.

Fix:
- Increase min_samples per cell
- Reduce bins_x / bins_y
- Pool adjacent interventions (e.g., group ramp rates into bands)
- Collect more trials in sparse regions

---

## Outcomes are not numeric

Symptom:
The tool warns that outcome is non-numeric.

Fix:
Convert outcomes into a numeric form:
- time-to-switch (preferred)
- peak voltage
- integrated dissipated energy proxy
- or a numeric code for categorical outcomes (if you must)

---

## One-minute sanity checklist

Before trusting a result:
- Can you reproduce low fracture on a negative control regime?
- Do you have enough trials per intervention/history class?
- Are state variables repeated enough to compare distributions?
- Are interventions labeled correctly?
- Does adding the top recommended feature reduce fracture meaningfully?

If yes, your result is likely actionable.