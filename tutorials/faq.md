# Frequently Asked Questions (FAQ)

This FAQ addresses common questions and failure modes encountered when first using
**intervention-faithfulness**. It is written for users running the tool on real data,
not for contributors.

If you are looking for implementation details, see `docs/`.

---

## What problem does this tool actually solve?

It tells you **whether your chosen state representation is sufficient to predict outcomes
under intervention**.

It does **not** try to explain *why* a system behaves as it does.
It diagnoses **when a reduced model collapses distinct histories** and therefore fails
under protocol changes.

---

## Is this a test of determinism?

No.

The diagnostic evaluates **conditional outcome distributions**, not point predictions.

A system can be:
- stochastic and faithful, or
- precise and unfaithful.

Randomness is allowed. History-dependent collapse is not.

---

## My model fits the data well. Why is fracture high?

Because good fit under one protocol does **not** guarantee validity under another.

High fracture means:
- two trials look identical to your model,
- but diverge when you intervene.

This is a **representational failure**, not a fitting error.

---

## Does high fracture mean my physics model is wrong?

No.

It means your **state variables are incomplete** for the tested intervention regime.

Multiple microscopic models may all be compatible with a faithful representation.
This tool does not adjudicate between them.

---

## What does “minimal completion” actually mean?

It means:
> the smallest amount of additional information required to restore faithfulness.

Often this is **one low-dimensional history variable**, not a complex memory kernel.

The recommendation engine ranks candidates by how much they reduce fracture.

---

## Are the recommended features unique or physically meaningful?

No.

They are:
- representationally useful,
- not guaranteed to be unique,
- not guaranteed to map to a specific physical mechanism.

They answer *what must be tracked*, not *why it matters*.

---

## How much data do I need?

Rule of thumb:
- ≥ 50 trials per relevant history / intervention class

The tool will warn you if estimates are underpowered.
It will not block execution.

If data is limited:
- pool similar interventions,
- reduce binning resolution,
- focus on fewer candidate histories.

---

## I see fracture only in rare events (tails). Is that real?

Yes — and that’s often where it matters most.

Many real failures (dark counts, late switches, reliability issues)
appear in distribution tails.

The tool supports tail-sensitive divergence measures for this reason.

---

## The diagnostic reports fracture in one regime but not another. Is that expected?

Yes.

Continuation fracture is **regime-dependent**.

This is why the tool produces **faithfulness maps** instead of a single score.
Models can be valid in one operating region and unsafe in another.

---

## What is a negative control and why should I run one?

A negative control is a regime where reduced models are known to work
(e.g., slow ramps, far from critical thresholds).

Running one demonstrates that:
- the diagnostic does not hallucinate failure,
- high fracture is not just noise or undersampling.

If a negative control shows high fracture, check your setup.

---

## When should I not use this tool?

Do not use it to:
- claim physical truth,
- infer microscopic mechanisms,
- assert global validity,
- replace domain-specific theory.

Use it to:
- validate reduced models,
- identify unsafe intervention regimes,
- guide protocol design.

---

## Where are the formal guarantees defined?

- **What the tool certifies:** `CERTIFICATION.md`
- **How the system is structured:** `ARCHITECTURE.md`

These documents define the scope and limits of all results.

---

## One-sentence reminder

**This tool certifies representations under intervention, not reality itself.**