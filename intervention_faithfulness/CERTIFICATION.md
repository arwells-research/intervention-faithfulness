# CERTIFICATION.md

## Intervention Faithfulness Certification Contract

This document defines **exactly what the intervention-faithfulness diagnostic certifies**,  
and **what it explicitly does not claim**.

It is a binding contract between:
- tool developers,
- tool users (engineers, scientists),
- reviewers and stakeholders.

This contract exists to prevent **representational overreach**, **semantic drift**, and
misinterpretation of diagnostic results.

---

## 1. What This Tool Certifies

The diagnostic certifies properties of a **state representation**, not of nature itself.

### 1.1 Representation Faithfulness Under Intervention

The tool certifies whether a chosen state representation is **faithful under intervention**, defined as:

> A representation is *faithful* if conditional outcome distributions remain invariant
> across admissible histories when subjected to the same intervention.

Operationally:
- Two trials that are indistinguishable at the level of the supplied state variables
- must produce statistically indistinguishable outcome distributions
- when the same intervention is applied.

Violation of this invariance is called **continuation fracture**.

---

### 1.2 Conditional Distribution Invariance (Not Determinism)

The diagnostic evaluates **distributions**, not point predictions.

Certified faithfulness means:
- stochastic outcomes are allowed,
- randomness is allowed,
- thermal activation and noise are allowed,

**provided** that the *conditional distributions* are invariant with respect to hidden history.

This explicitly distinguishes:
- *stochastic but faithful systems*  
from
- *fractured representations that collapse distinct histories*.

---

### 1.3 Validity Within a Tested Intervention Envelope

All certification is **local**.

Faithfulness is certified only:
- for the tested intervention protocols,
- within the tested parameter ranges,
- at the tested history depths.

The diagnostic produces **boundary-of-validity maps**, not universal guarantees.

---

## 2. What This Tool Does NOT Certify

The following claims are **explicitly out of scope** and must not be inferred.

### 2.1 No Microscopic Mechanism Claims

The diagnostic does **not** certify:
- vortex dynamics,
- hotspot models,
- quasiparticle mechanisms,
- TDGL correctness,
- RSJ correctness,
- or any other microscopic theory.

Multiple incompatible microscopic models may all be compatible with a faithful representation.

---

### 2.2 No Determinism or Noise Elimination

The diagnostic does **not**:
- eliminate stochasticity,
- identify noise sources,
- claim reduced variance implies physical correctness.

A model can be faithful and noisy.

A model can be precise and unfaithful.

---

### 2.3 No Global or Future Guarantees

Certification does **not** imply:
- validity under stronger interventions,
- validity under faster protocols,
- validity after hardware changes,
- validity outside tested regimes.

Extrapolation beyond certified regions is the responsibility of the user.

---

### 2.4 No Causal Completeness or Control Optimality

The diagnostic does **not**:
- prove causal sufficiency,
- infer causal graphs,
- guarantee optimal control policies,
- recommend physical explanations.

It diagnoses **representation adequacy**, not causal truth.

---

## 3. Interpretation of Diagnostic Outcomes

### 3.1 Low Fracture (Faithful)

Indicates:
- the chosen state representation is sufficient **in this regime**,
- no additional history variables are required for predictive validity under intervention,
- Markovian modeling at this representation level is justified.

---

### 3.2 High Fracture (Unfaithful)

Indicates:
- the representation collapses distinct histories,
- predictive failure under intervention is structural, not noise-related,
- at least one missing historical degree of freedom exists.

The diagnostic may suggest **minimal completion candidates**, but:
- it does not claim uniqueness,
- it does not claim physical interpretation.

---

### 3.3 Negative Control (Certified Faithfulness)

When fracture remains low in regimes where reduced models are known to work,
this certifies that the diagnostic:
- does not hallucinate failure,
- distinguishes stochastic variability from representational collapse.

---

## 4. Minimal Completion Guidance

When enabled, the tool may recommend candidate history features that reduce fracture.

These recommendations:
- identify *representational degrees of freedom*,
- do not identify physical causes,
- do not imply mechanistic interpretation.

They answer:
> “What information must be retained to restore faithfulness?”

Not:
> “Why the system behaves this way.”

---

## 5. Responsibility and Use

This tool is intended as **model engineering infrastructure**.

Appropriate uses:
- validating reduced models before deployment,
- identifying unsafe intervention regimes,
- guiding protocol design,
- certifying representational sufficiency.

Inappropriate uses:
- claiming physical truth,
- replacing domain theory,
- asserting causal explanations,
- making safety claims outside tested regimes.

---

## 6. Design Invariant

This certification contract is **binding**.

Future development of the tool must:
- preserve these guarantees,
- avoid expanding claims without explicit revision of this document,
- treat any change to certified scope as a breaking change.

---

## Summary

**This tool certifies representations, not reality.**  
It identifies when a model’s state collapses history in ways that break predictive validity
under intervention — and when it does not.

Nothing more. Nothing less.