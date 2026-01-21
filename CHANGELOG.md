# CHANGELOG

All notable changes to **intervention-faithfulness** will be documented here.

This project follows **semantic versioning**, with special emphasis on
preserving certified scope and representational guarantees.

---

## v0.1.0 — Initial Release

### Added
- Continuation fracture diagnostic for testing representation faithfulness under intervention
- Canonical trials table schema
- Minimal completion (history feature) ranking
- Faithfulness maps for boundary-of-validity visualization
- Plugin system for domain-specific data and features
- Policy guard layer (Σ₂-I) producing OK / BOUNDARY / REFUSE decisions with machine-readable reasons and exit codes
- User-facing tutorials:
  - Quick Start
  - Plugins Guide
  - FAQ
  - Troubleshooting
  - Negative Controls
  - Common Workflows
  - Glossary
- Project guardrails and contracts:
  - CERTIFICATION.md
  - ARCHITECTURE.md
  - DESIGN.md
  - CLI.md
  - plugin_spec.md
  - api_overview.md
  - tests/README.md (guard acceptance matrix)

### Certified Scope (Unchanged)
- Certifies **representational sufficiency under intervention**
- Evaluates **conditional outcome distributions**, not determinism
- Provides **local, regime-dependent** validity only
- Makes **no claims** about microscopic mechanisms, causality, or physical truth

### Explicit Non-Goals
- No automatic causal inference
- No mechanism identification
- No global validity guarantees
- No embedded interpretation or policy thresholds

### Notes
This release establishes the **canonical meaning** of:
- continuation fracture
- faithfulness
- Σ₁ / Σ₂ representations
- minimal completion

Any future change that alters these meanings is a **breaking change**
and must be accompanied by an explicit update to `CERTIFICATION.md`.

---

## Future Releases

Future entries must:
- list added, changed, and removed features,
- explicitly state whether certified scope changed,
- reference any updates to certification or architecture documents.

If certified scope did not change, that fact must be stated explicitly.

