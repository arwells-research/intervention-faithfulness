# tests/README.md — Σ₂-I Guard Acceptance Matrix

This document defines the **canonical acceptance tests** for the Σ₂-I guard.
These tests are **non-negotiable** and form the **spine of the repository**.

If these tests fail, the system is not safe to use as a certification guard.

---

## Purpose

The Σ₂-I guard is a **policy layer**, not a metric.

Its job is to take the outputs of `diagnose()` and enforce conservative,
machine-actionable decisions:

- **OK**
- **BOUNDARY**
- **REFUSE**

The guard must satisfy one overriding principle:

> **No false OK. Ever.**

---

## Guard Acceptance Matrix (Authoritative)

Every implementation, refactor, or optimization **must** preserve the following
decision outcomes.

| Scenario | Data Power | Structural Truth | Expected Guard Status |
|--------|-----------|------------------|-----------------------|
| Faithful regime | Sufficient | Faithful | **OK** |
| Faithful regime | Underpowered | Faithful but uncertain | **BOUNDARY** |
| Unfaithful cut | Sufficient | Structurally broken | **REFUSE** |
| Unfaithful cut | Underpowered | Broken but uncertain | **BOUNDARY** |
| Masquerade / confounded | Any | Ambiguous | **BOUNDARY** |

This table is the **ground truth** for guard behavior.

---

## Definitions

### Faithful regime
A dataset where:
- continuation fracture ≈ 0
- outcome distributions are invariant under intervention
- any stochasticity is explainable without hidden state

### Unfaithful cut
A dataset where:
- histories collapse to the same reduced state
- but diverge immediately under intervention
- fracture is materially above threshold when powered

### Underpowered
Any case where:
- effective sample size < required minimum, or
- envelope uncertainty exceeds policy tolerance

Underpowered **never returns OK**, regardless of apparent fracture value.

### Masquerade / confounded
Cases that *look* unfaithful but are actually explained by:
- regime mixing
- label leakage
- insufficient stratification
- observational confounds

These must land **BOUNDARY**, not OK and not REFUSE.

---

## Required Guard Invariants

All tests must enforce the following invariants:

1. **Underpowered ⇒ BOUNDARY**
   - Never OK
   - Never silently passed

2. **Unsafe envelope ⇒ REFUSE**
   - If any unsafe region exists and data are powered

3. **High fracture (powered) ⇒ REFUSE**
   - Even if envelope is missing or degenerate

4. **Ambiguity ⇒ BOUNDARY**
   - Missing significance
   - Conflicting signals
   - Partial envelope coverage

5. **OK is rare and earned**
   - Only when powered *and* faithful *and* envelope is safe

---

## Test Structure Expectations

The following tests must exist and remain passing:

### Core policy tests
- `test_guard_ok_when_faithful_and_powered`
- `test_guard_boundary_when_underpowered`
- `test_guard_refuses_unfaithful_cut`
- `test_guard_boundary_when_unfaithful_but_underpowered`
- `test_guard_boundary_for_masquerade_case`

### CLI integration tests
- `faithfulness guard` exit codes:
  - OK → exit 0
  - BOUNDARY → exit 2
  - REFUSE → exit 3

### Regression rule
Any change that causes a **false OK** must:
- add a failing test reproducing the issue
- fix the guard logic
- keep the test permanently

---

## Philosophy (Why this exists)

Σ₂-I is not a predictor.
It is not an explainer.
It is not an optimizer.

It is a **certification guard**.

If a system is not safe to certify,
the correct answer is **BOUNDARY** or **REFUSE** — never optimism.

---

## If You Are Adding a New Test

Ask yourself:

> “Could this ever cause the guard to say OK when it shouldn’t?”

If yes, it belongs here.

If not, it probably belongs elsewhere.

This file is the **last line of defense** against silent failure.
