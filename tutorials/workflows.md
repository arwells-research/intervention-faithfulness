# Common Workflows

This guide describes **standard workflows** for using intervention-faithfulness in practice.
It focuses on *what order to do things in* and *why*, not on theory or implementation details.

Use this after completing the Quick Start.

---

## Workflow 1: First-Time Use on New Data

**Goal:** Determine whether your current state representation is safe to use.

Steps:
1) Prepare repeated-trial data with:
   - trial_id
   - intervention_id
   - outcome
   - candidate state_* variables
2) Run the diagnostic with no added history features.
3) Inspect the fracture score and warnings.
4) Run a negative control if possible.

Interpretation:
- fracture ≈ 0 → representation is faithful in this regime
- fracture > 0 → proceed to Workflow 2

---

## Workflow 2: Diagnosing Representational Failure

**Goal:** Identify what information is missing.

Steps:
1) Enable minimal completion ranking.
2) Inspect the top-ranked candidate features.
3) Add the top 1–2 candidates as history_* variables.
4) Re-run the diagnostic and compare fracture.

Stop when:
- fracture collapses meaningfully, or
- added features stop improving results.

Outcome:
- You have identified the **minimal information** required for faithfulness.

---

## Workflow 3: Mapping Safe vs Unsafe Regimes

**Goal:** Understand where your model can be trusted.

Steps:
1) Run the diagnostic across multiple intervention strengths.
2) Generate a faithfulness map.
3) Identify bright (safe) and dark (unsafe) regions.

Use this to:
- define operating envelopes,
- redesign protocols,
- avoid regimes where hidden history dominates.

---

## Workflow 4: Protocol Comparison

**Goal:** Choose between competing intervention protocols.

Steps:
1) Label each protocol as a distinct intervention_id.
2) Run the diagnostic on each protocol.
3) Compare fracture scores and maps.

Interpretation:
- Lower fracture → protocol better respects your representation
- Higher fracture → protocol exposes hidden state

---

## Workflow 5: Regression / Validation After Changes

**Goal:** Ensure changes did not invalidate your model.

Run this when:
- hardware changes,
- temperature changes,
- timing changes,
- protocol changes.

Steps:
1) Re-run a known negative control.
2) Re-run the diagnostic in the target regime.
3) Compare fracture to previous results.

If fracture increases:
- the representation may no longer be sufficient.

---

## Workflow 6: Reporting and Communication

**Goal:** Communicate results clearly to others.

Recommended outputs:
- Validity certificate (for managers / reports)
- Faithfulness map (for engineers)
- Fracture + recommendations (for researchers)

Always include:
- tested regime description,
- intervention envelope,
- whether a negative control was run.

---

## What these workflows prevent

Following these workflows helps prevent:
- mistaking good fit for validity,
- blaming noise for structural failure,
- over-interpreting a single score,
- extrapolating beyond tested regimes.

---

## One-sentence reminder

**Always certify your representation before trusting your predictions under intervention.**