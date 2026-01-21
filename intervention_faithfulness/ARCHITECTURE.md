# ARCHITECTURE.md

## System Architecture Overview

This document describes the **structural architecture** of the intervention-faithfulness project.
It exists to preserve design intent, prevent scope creep, and avoid future unfaithful cuts
to the system’s representational geometry.

This is a **maintainer-facing document**. It complements `CERTIFICATION.md`, which is user-facing.

---

## 1. Architectural Goal

The system provides **model-agnostic certification of representation faithfulness under intervention**.

It sits **above domain models**, not in competition with them.

The architecture enforces a strict separation between:
- what is measured (core),
- how data is interpreted (plugins),
- how results are communicated (reporting).

---

## 2. Layered Design

### 2.1 Core (Invariant, Domain-Agnostic)

Location: `core/`

Responsibilities:
- Define the canonical trials schema
- Compute continuation fracture
- Rank minimal representational completions
- Produce faithfulness maps
- Return structured results

Non-responsibilities (hard constraints):
- No domain logic
- No physical interpretation
- No policy thresholds
- No causal claims

Core files are **API-stable** once published.

---

### 2.2 Plugins (Domain-Specific, Optional)

Location: `plugins/`

Responsibilities:
- Load domain-specific data formats
- Provide sensible defaults for diagnostics
- Define reusable history/feature generators

Characteristics:
- Pluggable
- Discoverable
- Non-invasive to core
- Community-extensible

The core never imports domain knowledge directly.

---

### 2.3 Reporting (Presentation, Not Interpretation)

Location: `core/reporting.py`

Responsibilities:
- Package results for humans and machines
- Export certificates and summaries
- Produce visualizations

Non-responsibilities:
- No thresholds for “good/bad”
- No claims about physical mechanisms
- No decision policy

All interpretation is deferred to the user or the paper.

---

## 3. Canonical Data Flow

Raw Data  
→ Data Plugin  
→ Canonical Trials Table  
→ Feature Plugins (optional)  
→ Continuation Fracture Diagnostic  
→ Minimal Completion Ranking (optional)  
→ DiagnosisResult  
→ Reports / Maps / Certificates  

This flow is **one-directional**.
No step retroactively alters upstream semantics.

---

## 4. Representation Semantics (Frozen)

The following meanings are fixed:

- state_* : Instantaneous reduced representation
- history_* : Candidate memory / completion variables
- Σ₁ : State-only representation
- Σ₂ : State + history representation
- fracture : Divergence of conditional outcome distributions
- faithfulness : Absence of continuation fracture

Changing these meanings is a **breaking change**.

---

## 5. Change Discipline

### Allowed changes
- New plugins
- New visualization helpers
- New export formats
- New divergence metrics (as additions)

### Disallowed without explicit revision
- Rewriting core semantics
- Adding causal language
- Embedding physical interpretation
- Expanding certification claims

---

## 6. Design Principle (Non-Negotiable)

The system diagnoses representational failure.  
It does not explain the universe.

This principle overrides convenience, novelty, and completeness.

---

## 7. Relationship to CERTIFICATION.md

- `CERTIFICATION.md` defines what users are allowed to infer
- `ARCHITECTURE.md` defines what maintainers are allowed to build

Both must remain consistent.

---

## Closing Note

This architecture exists to ensure that as the system grows,
history is not collapsed, intent is not lost, and
diagnostic power is preserved.

Any future contribution should be evaluated against this document
before code is written.