# TODO.md — Σ₂-I Execution Roadmap & Stabilization Checklist
Project: **Intervention Faithfulness / Σ₂-I Certification Guard**  
Status: **Authoritative execution tracker** (living document)

This file sequences work to make **Σ₂-I a trustworthy certification guard** that can
be dropped into real workflows and *resists misuse, underpower, and false confidence*.

---

## North Star

A user can run:

- `diagnose` (threshold-free):  
  “what is the fracture, where, with what uncertainty, and what minimal completion fixes it?”

- `guard` (policy):  
  “OK / BOUNDARY / REFUSE, with a machine-readable reason + exit code”

…and trust that:

- **underpowered cases never return OK**
- **obvious unfaithfulness never slips through**

This invariant dominates all design decisions.

---

## Legend

- [ ] not started  
- [~] in progress  
- [x] complete  
- 🔒 contract surface (must update DESIGN.md + tests before changing)

---

## Phase 0 — Lock the Contracts (API stability first) 🔒

**Goal:** prevent drift, re-invention, and parallel interfaces.

### 0.1 Authoritative docs as contracts
- [x] `docs/DESIGN.md` — module boundaries, semantics, invariants
- [x] `docs/CLI.md` — CLI → config → export mapping
- [x] `docs/plugin_spec.md` — plugin registration & discovery
- [ ] Cross-check docs against code:
  - public class / method names
  - return types
  - default behaviors

### 0.2 Public surface freeze (v0.1)
🔒 Must not change without updating DESIGN.md + tests

- `FaithfulnessTest.from_plugin`
- `FaithfulnessTest.add_feature`
- `FaithfulnessTest.diagnose`
- `DiagnosisResult` public properties:
  - `fracture_score`
  - `significance`
  - `recommended_features`
  - `safe_operating_regions`
- Exports:
  - `export_artifacts`
  - `export_certificate_json`
  - `export_certificate`

### 0.3 No duplicate interfaces rule
- [x] One CLI entry point
- [x] One plugin registry path
- [ ] Add regression test that fails if a second CLI or registry is introduced

Deliverable: docs locked + short architecture map in README.

---

## Phase 1 — Core Metric Correctness & Invariances (Σ₂-I trustworthiness)

**Goal:** ensure continuation fracture behaves sanely before scaling features.

### 1.1 Negative controls must pass
- [x] Faithful synthetic → fracture ≈ 0
- [x] Guard returns OK when powered
- [x] Underpowered → guard returns BOUNDARY (never OK)

### 1.2 Positive controls must fail
- [x] Explicit unfaithful-cut synthetic → fracture high
- [x] Guard returns REFUSE
- [ ] “Masquerade” cases (confounds, label leakage):
  - must land BOUNDARY, not OK

### 1.3 Invariance tests
- [x] Row order does not matter
- [x] Label renaming does not matter (except prefix semantics)
- [x] Deterministic outputs with fixed `random_state`

Deliverable: focused test suite pinning these behaviors.

---

## Phase 2 — Safe Envelope & Maps (Derived artifacts, never semantics)

**Goal:** maps help engineers, but must never alter core truth.

### 2.1 Safe envelope invariants
- [x] Underpowered cells → “uncertain”
- [x] Stable contiguous segments & bounded labels
- [x] Envelope computation cannot crash `diagnose()`

### 2.2 Guard integration (conservative)
- [x] Any unsafe region → REFUSE
- [x] Too uncertain → BOUNDARY
- [x] Envelope never causes false OK

### 2.3 Break-it tests
- [ ] Categorical vs numeric axes
- [ ] Sparse bins
- [ ] Missing axis columns
- [ ] Degenerate single-bin cases

Deliverable: 3–5 targeted tests that intentionally stress map inputs.

---

## Phase 3 — Recommendations That Are Honest (Minimal Completion Search)

**Goal:** recommendations must not become a hallucination engine.

### 3.1 Required recommendation outputs
- [ ] Baseline fracture vs augmented fracture
- [ ] Data requirements (columns needed)
- [ ] Parameters used

### 3.2 Modes
- [x] Mode A: single-feature ranking (safe default)
- [ ] Mode B: greedy feature sets
  - bounded max set size
  - minimum delta threshold

### 3.3 Explicit “no improvement” outcome
- [ ] If nothing reduces fracture materially:
  - say so explicitly
  - guard remains BOUNDARY / REFUSE as appropriate

Deliverable: synthetic suite where the correct feature is known and recovered.

---

## Phase 4 — CLI as Execution Wrapper (Not a Second Product)

**Goal:** operational usability without duplicating logic.

### 4.1 CLI responsibilities only
- [x] Load trials / invoke plugin loader
- [x] Call `FaithfulnessTest.diagnose`
- [x] Call `DiagnosisResult.export_artifacts`
- [x] Optionally call guard and emit JSON + exit code

### 4.2 Drift tripwires
- [x] CLI help compliance test
- [x] CLI bundle output compliance test
- [ ] Golden run artifact checked in CI

Deliverable: CLI smoke tests + one golden output bundle.

---

## Phase 5 — Guard Acceptance Test Matrix (Project Spine)

**Goal:** prove “no false OK” under all realistic conditions.

Acceptance matrix (must exist as tests):

- [x] Faithful + powered → OK
- [x] Faithful + underpowered → BOUNDARY
- [x] Unfaithful + powered → REFUSE
- [x] Unfaithful + underpowered → BOUNDARY
- [ ] Masquerade / confounded → BOUNDARY

This matrix is the **highest-priority invariant** in the repo.

---

## Phase 6 — Real Domain Plugins (After Guard Is Hard to Fool)

**Goal:** extend reach while keeping core stable.

### 6.1 Data plugins
- [ ] `nanowire_switching`
- [ ] faithful negative-control regime
- [ ] pedagogical synthetic (RC / oscillator)

Each plugin must include:
- metadata completeness
- tiny fixture dataset for tests

### 6.2 Feature plugins
- [ ] integrated current
- [ ] EWMA dissipation
- [ ] previous switch count
- [ ] time since last event

Deliverable: plugin CI ensuring `list_plugins()` stability and help text renders.

---

## Phase 7 — Certification Artifacts (What People Share)

**Goal:** make the output bundle audit-grade and rerunnable.

- [x] Stable curated JSON certificate payload
- [x] PDF / HTML rendering consumes curated JSON
- [ ] Bundle manifest with hashes:
  - trials
  - diagnosis
  - certificate

Deliverable: certificate an engineer can email and reproduce.

---

## Operating Principle (Pinned)

**No false OK dominates everything:**

- underpowered ⇒ BOUNDARY  
- missing / ambiguous ⇒ BOUNDARY  
- strong evidence of failure ⇒ REFUSE  

If a change weakens this, it is wrong.

---

## Explicit Non-Goals (Do Not Drift)

Out of scope by design:

- ❌ Microscopic physical modeling
- ❌ Root-cause inference
- ❌ Full dynamics reconstruction
- ❌ Automated sensor synthesis
- ❌ End-to-end control synthesis

If any appear in issues or PRs → reject or defer.
