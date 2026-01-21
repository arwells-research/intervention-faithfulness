# intervention_faithfulness/core/guard.py
"""
core/guard.py — Σ₂-I guard policy layer (v0.1)

Purpose
-------
Consume a DiagnosisResult (threshold-free reporting) and produce an enforceable
guard decision:

- OK       : faithful enough under declared policy
- BOUNDARY : insufficient evidence / underpowered / mixed / uncertain
- REFUSE   : structurally unfaithful under intervention (fracture too high or unsafe envelope)

Non-negotiables
---------------
- No false OK: underpower => BOUNDARY, not OK
- Thresholds live here, not in reporting.py
"""
# Acceptance/contract tests for this policy live in: tests/README.md

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, List
import json
import math

import pandas as pd

from intervention_faithfulness.core.reporting import DiagnosisResult

# Exit codes (guard contract)
EXIT_OK = 0
EXIT_BOUNDARY = 2
EXIT_REFUSE = 3


@dataclass(frozen=True)
class GuardConfig:
    """
    Policy thresholds and guard preferences.

    Notes:
    - fracture_threshold is interpreted on the reported fracture value (fs.value).
    - min_effective_samples is a hard power floor using a power proxy (see _power_proxy).
    """
    fracture_threshold: float = 0.12
    min_effective_samples: int = 200

    require_significance: bool = False
    p_value_threshold: float = 0.05

    # Safe-envelope usage:
    # - If any "unsafe" region exists, we REFUSE.
    # - If too much "uncertain", we BOUNDARY (no false OK).
    use_safe_envelope_if_available: bool = True
    max_uncertain_fraction: float = 0.50

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GuardDecision:
    status: str  # {"OK","BOUNDARY","REFUSE"}
    reason: str
    exit_code: int
    details: Dict[str, Any]
    certificate: Optional[Dict[str, Any]] = None

    def to_record(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "decision": {
                "status": self.status,
                "reason": self.reason,
                "exit_code": int(self.exit_code),
            },
            "details": self.details,
        }
        if self.certificate is not None:
            out["certificate"] = self.certificate
        return out

def export_decision_certificate(decision: GuardDecision, out_path: str) -> str:
    """
    Write a minimal, deterministic JSON certificate representing the guard decision.
    """
    from pathlib import Path
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = decision.to_record()
    p.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
    return str(p)

def _envelope_summary(result: DiagnosisResult) -> Optional[Dict[str, Any]]:
    """
    Summarize safe/unsafe/uncertain from DiagnosisResult.safe_operating_regions.
    Returns None if envelope isn't present.
    """
    try:
        regs = list(result.safe_operating_regions or [])
    except Exception:
        regs = []

    if not regs:
        return None

    n_safe = 0
    n_unsafe = 0
    n_uncertain = 0
    labels: List[str] = []

    for r in regs:
        status = str(getattr(r, "status", "")).lower().strip()
        label = str(getattr(r, "label", "")).strip()
        if label:
            labels.append(label)

        if status == "safe":
            n_safe += 1
        elif status == "unsafe":
            n_unsafe += 1
        else:
            n_uncertain += 1

    n_total = n_safe + n_unsafe + n_uncertain
    frac_uncertain = (n_uncertain / n_total) if n_total > 0 else 1.0

    return {
        "n_safe": int(n_safe),
        "n_unsafe": int(n_unsafe),
        "n_uncertain": int(n_uncertain),
        "n_total": int(n_total),
        "uncertain_fraction": float(frac_uncertain),
        "labels": labels[:25],  # bounded
    }


def _power_proxy(fracture_score: Any, trials_df: pd.DataFrame) -> Optional[int]:
    """
    Policy power proxy.

    Prefer fracture_score.n_effective if populated by core.
    Otherwise fall back to total trial count (conservative, but avoids perma-BOUNDARY).

    The guard still enforces "no false OK" because we hard-gate on min_effective_samples.
    """
    try:
        n_eff = getattr(fracture_score, "n_effective", None)
        if n_eff is not None:
            n_eff_i = int(n_eff)
            if n_eff_i >= 0:
                return n_eff_i
    except Exception:
        pass

    try:
        return int(len(trials_df))
    except Exception:
        return None


def decide(result: DiagnosisResult, config: GuardConfig) -> GuardDecision:
    """
    Turn a DiagnosisResult into an enforceable guard decision.

    Policy (v0.1, conservative):
    - If fracture is NaN/non-finite => BOUNDARY
    - If power proxy is missing or below min_effective_samples => BOUNDARY
    - If safe envelope exists and any unsafe => REFUSE
    - If safe envelope exists and uncertain_fraction too high => BOUNDARY
    - If require_significance and p_value missing or > threshold => BOUNDARY
    - If fracture >= fracture_threshold => REFUSE
    - else OK
    """
    fs = result.fracture_score
    f_val = float(getattr(fs, "value", float("nan")))
    f_warn = [str(w) for w in (list(getattr(fs, "warnings", None) or []))]

    sig = result.significance
    p_value = getattr(sig, "p_value", None)
    n_perm = getattr(sig, "n_permutations", None)
    s_warn = [str(w) for w in (list(getattr(sig, "warnings", None) or []))]

    df = result.trials_df
    trials_sha256 = result.trials_table_sha256
    env = _envelope_summary(result)

    # recommendations (bounded + machine-friendly)
    try:
        recs = list(result.recommended_features or [])
    except Exception:
        recs = []

    power_n = _power_proxy(fs, df)

    details: Dict[str, Any] = {
        "metrics": {
            "fracture": {
                "value": f_val,
                "ci_low": getattr(fs, "ci_low", None),
                "ci_high": getattr(fs, "ci_high", None),
                "metric": getattr(fs, "metric", None),
                "metrics": getattr(fs, "metrics", None),
                "n_effective": getattr(fs, "n_effective", None),
                "power_proxy_n": power_n,
                "warnings": f_warn,
            },
            "significance": {
                "p_value": p_value,
                "n_permutations": n_perm,
                "warnings": s_warn,
            },
        },
        "envelope_summary": env,
        "recommendations_top": [
            {
                "name": getattr(r, "name", None),
                "delta_fracture": getattr(r, "delta_fracture", None),
                "mutual_info": getattr(r, "mutual_info", None),
                "params": getattr(r, "params", None),
                "data_requirements": getattr(r, "data_requirements", None),
            }
            for r in recs[:10]
        ],
        "provenance": {
            "trials_table_sha256": trials_sha256,
            "config": result.config,
            "metadata": result.metadata,
        },
        "policy": config.to_dict(),
    }

    # ---- gates ----
    if not math.isfinite(f_val):
        return GuardDecision(
            status="BOUNDARY",
            reason="fracture_nonfinite",
            exit_code=EXIT_BOUNDARY,
            details=details,
        )

    if power_n is None:
        return GuardDecision(
            status="BOUNDARY",
            reason="power_unknown",
            exit_code=EXIT_BOUNDARY,
            details=details,
        )

    if int(power_n) < int(config.min_effective_samples):
        return GuardDecision(
            status="BOUNDARY",
            reason="underpowered",
            exit_code=EXIT_BOUNDARY,
            details=details,
        )

    # Envelope gating (policy v0.1):
    # - "no false OK": uncertainty can block OK, but must NOT mask a clear REFUSE.
    # - If envelope says unsafe => REFUSE (it is additional evidence of structural failure).
    # - If both envelope uncertainty is too high AND fracture is clearly high => REFUSE (fracture wins).
    # - Otherwise too-uncertain => BOUNDARY.
    if config.use_safe_envelope_if_available and isinstance(env, dict):
        n_unsafe = int(env.get("n_unsafe", 0))
        if n_unsafe > 0:
            return GuardDecision(
                status="REFUSE",
                reason="unsafe_regions",
                exit_code=EXIT_REFUSE,
                details=details,
            )

        unc_frac = float(env.get("uncertain_fraction", 1.0))
        if unc_frac > float(config.max_uncertain_fraction):
            # Do not allow envelope underpower/uncertainty to override a hard fracture refusal.
            if f_val >= float(config.fracture_threshold):
                return GuardDecision(
                    status="REFUSE",
                    reason="fracture_high",
                    exit_code=EXIT_REFUSE,
                    details=details,
                )
            return GuardDecision(
                status="BOUNDARY",
                reason="too_uncertain",
                exit_code=EXIT_BOUNDARY,
                details=details,
            )

    # Significance gate (optional, conservative).
    if bool(config.require_significance):
        if p_value is None:
            return GuardDecision(
                status="BOUNDARY",
                reason="p_value_missing",
                exit_code=EXIT_BOUNDARY,
                details=details,
            )
        try:
            pv = float(p_value)
        except Exception:
            pv = 1.0
        if (not math.isfinite(pv)) or (pv > float(config.p_value_threshold)):
            return GuardDecision(
                status="BOUNDARY",
                reason="not_significant",
                exit_code=EXIT_BOUNDARY,
                details=details,
            )

    # Primary refusal threshold.
    if f_val >= float(config.fracture_threshold):
        return GuardDecision(
            status="REFUSE",
            reason="fracture_high",
            exit_code=EXIT_REFUSE,
            details=details,
        )

    return GuardDecision(
        status="OK",
        reason="faithful_enough",
        exit_code=EXIT_OK,
        details=details,
    )

def decision_record_json(decision: GuardDecision) -> str:
    """
    Stable JSON for stdout emission (CLI contract).
    """
    return json.dumps(
        {
            "tool": {"name": "intervention_faithfulness.guard"},
            **decision.to_record(),
        },
        sort_keys=True,
        indent=2,
        default=str,
    )
