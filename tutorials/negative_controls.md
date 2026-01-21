# Negative Controls Tutorial

This tutorial explains **why** and **how** to run a negative control with
intervention-faithfulness, and how to interpret the result.

A negative control is not optional if you want confidence in your conclusions.

---

## What is a negative control?

A negative control is a regime where **reduced models are already known to work**.

In such regimes:
- history dependence should be negligible,
- conditional outcome distributions should be invariant,
- continuation fracture should be near zero.

If the diagnostic reports high fracture here, something is wrong with the setup.

---

## Why negative controls matter

They demonstrate three critical properties of the tool:

1) The diagnostic does **not** hallucinate failure  
2) Fracture is **regime-dependent**, not ubiquitous  
3) Low fracture is a **certification**, not just “absence of evidence”

Without a negative control, high fracture could be blamed on noise, binning, or misuse.

---

## Common negative control regimes

Choose a regime where interventions are gentle and relaxation is complete.

Examples:
- slow ramps far below critical thresholds,
- quasi-static control protocols,
- overdamped or strongly relaxed devices,
- long reset times between trials.

You should already believe these regimes are well-behaved.

---

## Running a negative control (plugin)

If a negative-control plugin is available:

    from intervention_faithfulness import FaithfulnessTest

    test = FaithfulnessTest.from_plugin(
        "faithful_regime",
        "negative_control_data.csv"
    )

    results = test.diagnose()

Expected outcome:
- fracture close to zero,
- no strong feature recommendations,
- bright faithfulness map.

---

## Running a negative control (manual)

You can also define a negative control manually.

Example approach:
- restrict your dataset to slow interventions,
- restrict to low-amplitude pulses,
- increase wait time between trials,
- then rerun the diagnostic.

Compare fracture in:
- full dataset vs
- restricted (negative control) dataset.

Fracture should drop in the control regime.

---

## Interpreting outcomes

### Case 1: Fracture ≈ 0

This is the desired result.

It means:
- the tool is behaving correctly,
- your data is sufficient,
- high fracture elsewhere is meaningful.

You now have a certified baseline.

---

### Case 2: Fracture remains high

This indicates a problem.

Possible causes:
- insufficient sample size,
- overly fine binning,
- mis-labeled interventions,
- state variables not repeated enough,
- outcome variable too coarse.

Do not proceed to interpretation until this is resolved.

---

## What a negative control is NOT

It is not:
- proof your physics model is correct,
- proof the system is Markovian everywhere,
- proof that no memory exists.

It only certifies:
- representational sufficiency in this regime.

---

## Recommended workflow

1) Run a negative control first  
2) Confirm fracture ≈ 0  
3) Run the diagnostic in the regime of interest  
4) Interpret high fracture as structural, not noise  

This order prevents false conclusions.

---

## One-sentence takeaway

**If the diagnostic can certify faithfulness where you expect it, you can trust it when it flags failure elsewhere.**