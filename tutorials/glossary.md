# Glossary

This glossary defines **terms as they are used in intervention-faithfulness**.
Meanings here are operational and may differ from informal usage in physics,
statistics, or machine learning.

If a term appears ambiguous elsewhere, this document is authoritative.

---

## continuation fracture

A measurable divergence between **conditional outcome distributions**
that arises when a reduced state representation collapses distinct histories
and an intervention is applied.

Fracture indicates a **structural representational failure**, not noise.

---

## faithfulness

The absence of continuation fracture.

A representation is *faithful* in a given regime if conditional outcome
distributions remain invariant under intervention for indistinguishable states.

Faithfulness is **local** to tested regimes.

---

## state (state_*)

The set of variables treated as the **instantaneous reduced description**
of the system.

State variables are assumed to define what “matters now” for prediction.

---

## history (history_*)

Variables encoding **past information** not captured by the instantaneous state.

History variables are candidates for restoring faithfulness when fracture occurs.

---

## Σ₁ (Sigma-1 representation)

A **state-only** representation.

Σ₁ assumes Markovian behavior at the chosen state level.
It is sufficient only if fracture ≈ 0 under intervention.

---

## Σ₂ (Sigma-2 representation)

A **state + history** representation.

Σ₂ augments Σ₁ with minimal historical information required to restore
faithfulness under intervention.

---

## intervention

An externally imposed change applied to the system, such as:
- a control pulse,
- a ramp rate,
- a protocol change,
- a stimulus.

Interventions are explicitly labeled and conditioned upon in the diagnostic.

---

## outcome

The measured result following an intervention.

Outcomes are typically numeric (e.g., time-to-switch, peak voltage),
but may be categorical in future extensions.

---

## minimal completion

The smallest set of additional history variables that significantly
reduces continuation fracture.

Minimal completion identifies **what must be tracked**, not why it matters.

---

## negative control

A regime where reduced models are already expected to be valid.

Negative controls demonstrate that:
- the diagnostic does not hallucinate failure,
- fracture is regime-dependent,
- low fracture is a certification, not a null result.

---

## faithfulness map

A visualization showing where a representation is valid or invalid
across intervention strength, history depth, or other parameters.

Bright regions indicate faithfulness.
Dark regions indicate continuation fracture.

---

## tail-sensitive analysis

An analysis mode that emphasizes rare or extreme outcomes
(e.g., late switches, dark counts).

Used when failures are known to occur in distribution tails.

---

## certification

A statement about **representational sufficiency under intervention**
within a tested regime.

Certification does **not** imply physical truth, determinism,
or global validity.

---

## unfaithful cut

A many-to-one reduction that collapses distinct histories into a single state
in a way that breaks predictive validity under intervention.

Continuation fracture is the operational signature of an unfaithful cut.

---

## One-sentence summary

**The tool certifies whether your representation preserves the distinctions that interventions reveal.**