#!/usr/bin/env python3
"""
tools/s0001_sigma2_i_guard_harness.py — Σ₂-I executable guard harness (v0.1)

Goal:
- Provide a DFT-style executable refusal gate for intervention-faithfulness.

This script:
1) loads data via a data plugin OR from a canonical CSV
2) optionally applies feature plugins
3) runs FaithfulnessTest.diagnose()
4) applies a *policy threshold* to emit PASS / BOUNDARY / INCONCLUSIVE
5) exports an audit bundle (JSON + CSV), using DiagnosisResult exports when available

Important:
- Policy thresholds live HERE, not in core.
- The library remains policy-free and reports computed metrics + provenance.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from intervention_faithfulness.core.FaithfulnessTest import FaithfulnessTest, DiagnoseConfig


def _parse_kv_pairs(items: List[str]) -> Dict[str, Any]:
    """
    Parse repeated KEY=VALUE items. Values are parsed as:
    - int, float, bool, or left as string.
    """
    out: Dict[str, Any] = {}
    for it in items:
        if "=" not in it:
            raise ValueError(f"Expected KEY=VALUE, got: {it}")
        k, v = it.split("=", 1)
        vv: Any = v
        lv = v.lower()
        if lv in ("true", "false"):
            vv = (lv == "true")
        else:
            try:
                vv = int(v)
            except Exception:
                try:
                    vv = float(v)
                except Exception:
                    vv = v
        out[k] = vv
    return out


def _parse_feature_steps(items: List[str]) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Parse repeated feature specs:

      --feature integrated_current window_ns=50
      --feature ewma_dissipation alpha=0.2

    Implemented as:
      --feature NAME [KEY=VALUE ...]
    """
    steps: List[Tuple[str, Dict[str, Any]]] = []
    i = 0
    while i < len(items):
        name = items[i]
        i += 1
        params: List[str] = []
        while i < len(items) and ("=" in items[i]) and (not items[i].startswith("--")):
            params.append(items[i])
            i += 1
        steps.append((name, _parse_kv_pairs(params)))
    return steps


def _classify_guard(
    *,
    fracture_value: Optional[float],
    p_value: Optional[float],
    underpowered_fraction: Optional[float],
    warnings: List[str],
    max_fracture_pass: float,
    max_p_reject: float,
    max_underpowered_frac: float,
) -> Tuple[str, str]:
    """
    Return (verdict, reason).

    Conservative policy:
    - INCONCLUSIVE if obviously underpowered or fracture_value missing
    - PASS if fracture <= threshold AND (p-value is None OR p >= max_p_reject)
    - otherwise BOUNDARY
    """
    if fracture_value is None:
        return ("INCONCLUSIVE", "missing_fracture_value")

    if underpowered_fraction is not None and underpowered_fraction > max_underpowered_frac:
        return ("INCONCLUSIVE", f"underpowered_fraction>{max_underpowered_frac}")

    # If core emitted strong warnings about sampling, treat as inconclusive.
    for w in warnings:
        wl = w.lower()
        if "underpowered" in wl or "min_samples" in wl:
            return ("INCONCLUSIVE", "core_warning_underpowered")

    pass_fracture = fracture_value <= max_fracture_pass
    pass_sig = (p_value is None) or (p_value >= max_p_reject)

    if pass_fracture and pass_sig:
        return ("PASS", "fracture_below_threshold_and_not_significant")
    return ("BOUNDARY", "fracture_or_significance_violation")


def main() -> int:
    ap = argparse.ArgumentParser(description="Σ₂-I guard harness (intervention-faithfulness).")

    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--plugin", type=str, help="Data plugin name (e.g. nanowire_switching).")
    src.add_argument("--csv", type=str, help="Canonical trials CSV path.")

    ap.add_argument("--source", type=str, default=None, help="Plugin source (path/dir/etc). Required if --plugin.")
    ap.add_argument("--plugin-kw", action="append", default=[], help="Plugin load kwargs KEY=VALUE (repeatable).")
    ap.add_argument("--to-trials-kw", action="append", default=[], help="Plugin to_trials kwargs KEY=VALUE (repeatable).")

    ap.add_argument(
        "--feature",
        nargs="+",
        action="append",
        default=[],
        help="Feature step: NAME [KEY=VALUE ...]. Repeatable.",
    )

    ap.add_argument("--out-dir", type=str, default="tools/outputs", help="Output directory.")
    ap.add_argument("--prefix", type=str, default="s0001_sigma2_i", help="Output filename prefix.")

    # Guard policy thresholds (live in tools)
    ap.add_argument("--max-fracture-pass", type=float, default=0.10, help="PASS if fracture <= this.")
    ap.add_argument("--max-p-reject", type=float, default=0.05, help="Significant if p < this.")
    ap.add_argument("--max-underpowered-frac", type=float, default=0.35, help="INCONCLUSIVE if underpowered frac > this.")

    # Config overrides (optional; if omitted DiagnoseConfig defaults apply)
    ap.add_argument("--divergence", type=str, default=None, help="Override divergence (js/wasserstein/...).")
    ap.add_argument("--min-samples", type=int, default=None, help="Override min_samples.")
    ap.add_argument("--tail-mode", type=str, default=None, help="Override tail_mode true/false.")
    ap.add_argument("--quantile-focus", type=float, default=None, help="Override quantile_focus.")
    ap.add_argument("--n-bins", type=int, default=None, help="Override n_bins.")
    ap.add_argument("--n-permutations", type=int, default=None, help="Override n_permutations.")

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build DiagnoseConfig with optional overrides
    cfg = DiagnoseConfig()
    cfg_dict = cfg.to_dict()

    def _maybe_set(k: str, v: Any) -> None:
        if v is not None:
            cfg_dict[k] = v

    _maybe_set("divergence", args.divergence)
    _maybe_set("min_samples", args.min_samples)
    if args.tail_mode is not None:
        _maybe_set("tail_mode", args.tail_mode.strip().lower() == "true")
    _maybe_set("quantile_focus", args.quantile_focus)
    _maybe_set("n_bins", args.n_bins)
    _maybe_set("n_permutations", args.n_permutations)

    cfg = DiagnoseConfig(**cfg_dict)

    # Load data
    if args.plugin:
        if not args.source:
            raise SystemExit("--source is required when using --plugin.")
        plugin_kwargs = _parse_kv_pairs(args.plugin_kw)
        to_trials_kwargs = _parse_kv_pairs(args.to_trials_kw)

        test = FaithfulnessTest.from_plugin(
            args.plugin,
            args.source,
            plugin_kwargs=plugin_kwargs,
            to_trials_kwargs=to_trials_kwargs,
            metadata={"guard_harness": "s0001_sigma2_i"},
        )
    else:
        df = pd.read_csv(args.csv)
        test = FaithfulnessTest(df, metadata={"guard_harness": "s0001_sigma2_i", "data_source": args.csv})

    # Apply feature steps
    flat_steps: List[str] = []
    for group in args.feature:
        flat_steps.extend(group)
    steps = _parse_feature_steps(flat_steps)
    for name, params in steps:
        test.add_feature(name, **params)

    # Run
    res = test.diagnose(cfg)

    # Pull key fields (robust to partial implementations)
    fracture_value = getattr(res, "fracture_score", None)
    if isinstance(fracture_value, dict) and "value" in fracture_value:
        fracture_value = fracture_value.get("value", None)

    p_value = getattr(res, "significance", None)
    if isinstance(p_value, dict) and "p_value" in p_value:
        p_value = p_value.get("p_value", None)

    warnings = list(getattr(res, "warnings", []) or [])
    underpowered_fraction = None
    try:
        breakdown_df = getattr(res, "breakdown_df", None)
        if breakdown_df is not None and hasattr(breakdown_df, "__len__"):
            # Convention: if breakdown has an "underpowered" boolean column, estimate fraction.
            if "underpowered" in breakdown_df.columns:
                underpowered_fraction = float(breakdown_df["underpowered"].mean())
    except Exception:
        underpowered_fraction = None

    verdict, reason = _classify_guard(
        fracture_value=float(fracture_value) if fracture_value is not None else None,
        p_value=float(p_value) if p_value is not None else None,
        underpowered_fraction=underpowered_fraction,
        warnings=[str(w) for w in warnings],
        max_fracture_pass=float(args.max_fracture_pass),
        max_p_reject=float(args.max_p_reject),
        max_underpowered_frac=float(args.max_underpowered_frac),
    )

    audit: Dict[str, Any] = {
        "guard": "Sigma2-I",
        "script": "tools/s0001_sigma2_i_guard_harness.py",
        "verdict": verdict,
        "reason": reason,
        "fracture_value": fracture_value,
        "p_value": p_value,
        "underpowered_fraction": underpowered_fraction,
        "policy": {
            "max_fracture_pass": float(args.max_fracture_pass),
            "max_p_reject": float(args.max_p_reject),
            "max_underpowered_frac": float(args.max_underpowered_frac),
        },
        "config": getattr(res, "config", None),
        "metadata": getattr(res, "metadata", None),
    }

    # Export artifacts
    audit_path = out_dir / f"{args.prefix}_audit.json"
    with audit_path.open("w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, sort_keys=True)

    # Export breakdown if present
    try:
        breakdown_df = getattr(res, "breakdown_df", None)
        if breakdown_df is not None:
            bd_path = out_dir / f"{args.prefix}_breakdown.csv"
            breakdown_df.to_csv(bd_path, index=False)
    except Exception:
        pass

    # Prefer library export bundle if available
    try:
        export_artifacts = getattr(res, "export_artifacts", None)
        if callable(export_artifacts):
            export_artifacts(
                out_dir=str(out_dir),
                include_trials=False,
                include_map=False,
                include_certificate=True,
                prefix=args.prefix,
            )
    except Exception:
        pass

    print(f"[S-Σ2I] verdict={verdict} reason={reason} fracture={fracture_value} p={p_value}")
    print(f"[S-Σ2I] wrote {audit_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
