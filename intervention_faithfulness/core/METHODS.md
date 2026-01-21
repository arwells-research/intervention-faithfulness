# METHODS — Intervention Faithfulness & Continuation Fracture (v0.1)

This document defines the **method-level invariants** and **operational meaning**
of the Intervention Faithfulness diagnostic implemented in this repository.

It is authoritative for **what the method claims**, **what it does not claim**,
and **what must never change without a breaking revision**.

---

## Method Invariants (Non-Negotiable)

These invariants define the scientific contract of the tool.

### I1) Purely empirical, model-agnostic

The diagnostic operates **only on measured trial data**.

- No simulation is required
- No microscopic physical assumptions are made
- No curve fitting or parametric modeling is assumed

The method evaluates **observed conditional outcome distributions** under intervention.

---

### I2) The target is intervention faithfulness of a reduced state representation

We test whether a chosen reduced state representation is sufficient to support
**prediction under intervention**.

Formally, a reduced state is any mapping of past observations:

```
s_t = R(y_0:t)
```

In practice, this corresponds to the columns prefixed with `state_*`
in the canonical trials table.

The reduced state may be:
- whatever a controller exposes,
- whatever a simulator uses internally,
- whatever a learned model reports at its interface.

The method does **not** require the state to be “correct” — only that it be
**faithful under intervention**.

---

### I3) What is tested is distributional invariance, not determinism

The criterion is **not** whether outcomes are deterministic.

For histories `h₁, h₂` that collapse to the same reduced state `s`,
and for the same intervention `I`:

```
P(y | h₁, I)  ≈  P(y | h₂, I)
```

Randomness is allowed.

Stochastic systems pass **if and only if** the conditional distributions
are invariant under the reduction.

This explicitly defuses “the system is noisy” objections.

---

### I4) Continuation fracture is a regime-dependent signature of representational collapse

Continuation fracture is **not**:
- noise,
- generic non-Markovianity,
- lack of determinism.

It is the empirical signature that the reduced state representation
**collapsed histories that are continuation-relevant under intervention**.

Fracture is:
- local (regime-dependent),
- conditional (state × intervention),
- observable only through intervention.

---

### I5) The primary output is operational, not interpretive

The tool is designed to answer:

- **When is my state representation good enough?**
- **Where does it fail?**
- **What minimal augmentation restores validity?**

Outputs (maps, envelopes, certificates) are **engineering artifacts**,
not explanations of underlying mechanisms.

---

## Canonical Data Contract

The core diagnostic operates on a single canonical table.

Everything else (file formats, plugins, loaders) is an adapter.

### Required columns

- `trial_id`
- `intervention_id` (categorical or numeric)
- `outcome` (scalar in v0.1)
- One or more `state_*` columns

### Optional columns

- `history_*` columns  
  Full-history descriptors, engineered features, or labels used for refinement.

### Invariant

> **The core operates exclusively on this canonical table.**

No other schema is assumed.

---

## Metrics: What “Fracture” Means (v0.1)

Two equivalent operationalizations are supported.

Both are consistent with the original method definition.

---

### M1) Refinement Fracture (State vs History Refinement)

**Interpretation:**  
“How much additional predictive structure exists inside the collapsed state
when conditioning on a finer key?”

Compare:

```
P(y | s, I)    vs    P(y | h, I)
```

where `h` refines `s`.

This matches the original **state-key vs history-key refinement** formulation.

---

### M2) Pairwise Fracture (Within-State Pairwise Divergence)

**Interpretation:**  
“Do different history classes inside the same state yield different
continuation distributions?”

For a fixed `(s, I)`:

- sample pairs of histories `hᵢ, hⱼ ⊂ H(s)`
- compute divergence:

```
D( P(y | hᵢ, I)  ||  P(y | hⱼ, I) )
```

This matches the **pairwise fracture** implementation and supports
bounded sampling via `n_pairwise_pairs`.

---

### Metric Invariant

Both M1 and M2 are valid operationalizations of **continuation fracture**.

The paper may present one as primary and the other as robustness;
the tool supports both.

---

## Recommendations (Minimal Completion) Invariants

### R1) Recommendations are repair suggestions, not causal explanations

Recommended features are **augmentation candidates** that reduce fracture.

They do **not** claim:
- physical truth,
- microscopic relevance,
- causal primacy.

---

### R2) Two recommendation modes

- **Single-feature ranking**  
  Rank candidate features by fracture reduction `ΔF`.

- **Greedy / set-based search**  
  Identify small feature sets that jointly reduce fracture.

Both modes are bounded and auditable.

---

### R3) Valid negative-control behavior exists

In a faithful regime, the recommender should often return:

- empty results,
- near-zero deltas,
- explicit “no action needed”.

This is a required property, not a failure mode.

---

## Maps, Safe Envelope, and Certificates

### V1) Faithfulness maps are boundary-of-validity views

Maps visualize fracture (or normalized faithfulness) over a 2D grid,
such as intervention strength × history depth.

They are **diagnostic aids**, not decision engines.

---

### V2) Safe envelope is a summary, not a new metric

The safe envelope reduces the map into segments:

- **Safe**
- **Unsafe**
- **Uncertain** (explicitly underpowered or NaN)

It does not introduce new statistical meaning.

---

### V3) Certificates must be auditable artifacts

A certificate must include:

- diagnosis JSON
- curated certificate JSON
- hashes of trials and diagnosis
- configuration and provenance

Certificates are designed to be **rerunnable and verifiable**.

---

## Paper Structure Implied by the Method

Minimum section arc:

1. Recognized failure mode (protocol / regime dependence)
2. Method: invariance of conditional continuation distributions
3. Metric: continuation fracture + significance + power warnings
4. Repair: minimal completion (single + greedy)
5. Operationalization: maps → envelope → certificate
6. Validation of the validator (negative controls)

This structure supports nanowires as a canonical example
while remaining a general methods paper.

---

## Final Invariant (Pin This)

> **We are not predicting the future.**  
> **We are testing whether the future predicted by a reduced model  
> is invariant under intervention.**

Any change that weakens this statement is a breaking change.
