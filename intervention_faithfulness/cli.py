# intervention_faithfulness/cli.py — CLI entry point (v0.1)
"""
Commands:
- diagnose: run fracture + optional recommendations + optional envelope; write artifact bundle.
- map: render a faithfulness map to PNG (or PDF/SVG via matplotlib).
- certify: write certificate.html and/or certificate.pdf.
- plugins: list available plugins (data + features).
- plugin-help: show help for a specific plugin.
- guard: run diagnose + enforce Σ₂-I policy; emit JSON decision record; exit with guard code.

This CLI is intentionally small and dependency-light.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import pandas as pd

from intervention_faithfulness.core.FaithfulnessTest import FaithfulnessTest, DiagnoseConfig
from intervention_faithfulness.core.maps import MapConfig, plot_faithfulness_map
from intervention_faithfulness.plugins.registry import list_plugins, plugin_help

# Guard is optional but expected in Σ₂-I builds.
try:
    from intervention_faithfulness.core.guard import GuardConfig, decide, decision_record_json
except Exception:
    GuardConfig = None  # type: ignore
    decide = None  # type: ignore
    decision_record_json = None  # type: ignore


def _read_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {p}")
    return pd.read_csv(p)


def _build_diagnose_config(args: argparse.Namespace) -> DiagnoseConfig:
    # v0.1: keep it direct and explicit (avoid None-default complexity for now).
    return DiagnoseConfig(
        metric=getattr(args, "metric", "refinement"),
        divergence=args.divergence,
        min_samples=args.min_samples,
        tail_mode=args.tail_mode,
        quantile_focus=args.quantile_focus,
        n_bins=args.n_bins,
        n_pairwise_pairs=getattr(args, "n_pairwise_pairs", 50),
        n_permutations=args.n_permutations,
        random_state=args.random_state,
        recommend=(not args.no_recommend),
        recommend_top_k=args.top_k,
        recommend_mode=args.recommend_mode,
        recommend_max_set_size=args.max_set_size,
    )


def cmd_diagnose(args: argparse.Namespace) -> int:
    df = _read_csv(args.csv)
    test = FaithfulnessTest(df)
    cfg = _build_diagnose_config(args)
    res = test.diagnose(cfg)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix = args.prefix if args.prefix else None

    # Bundle exports (JSON + CSV + optional map). Certificate here means JSON certificate.
    written = res.export_artifacts(
        out_dir=str(out_dir),
        prefix=prefix,
        include_trials=bool(args.trials),
        include_map=bool(args.map),
        include_certificate=bool(args.certificate_json),
        include_recommendations=(not args.no_recommendations_csv),
    )

    # Render certificate(s) (separate from bundle JSON certificate)
    if args.html:
        html_path = out_dir / f"{(prefix or 'certificate')}_certificate.html"
        res.export_certificate(str(html_path), format="html")
        written["certificate_html"] = str(html_path)

    if args.pdf:
        pdf_path = out_dir / f"{(prefix or 'certificate')}_certificate.pdf"
        res.export_certificate(str(pdf_path), format="pdf")
        written["certificate_pdf"] = str(pdf_path)

    print("Wrote artifacts:")
    for k, v in written.items():
        print(f"  {k}: {v}")
    return 0


def cmd_map(args: argparse.Namespace) -> int:
    df = _read_csv(args.csv)

    cfg = MapConfig(
        x_col=args.x_col,
        y_col=args.y_col,
        bins_x=args.bins_x,
        bins_y=args.bins_y,
        min_samples=args.min_samples,
        metric=args.metric,
        divergence=args.divergence,
        tail_mode=args.tail_mode,
        quantile_focus=args.quantile_focus,
        n_bins=args.n_bins,
        n_pairwise_pairs=args.n_pairwise_pairs,
        normalize_quantile=args.normalize_quantile,
        try_parse_numeric=(not args.no_parse_numeric),
    )

    fig = plot_faithfulness_map(
        trials_df=df,
        config=cfg,
        title=args.title,
        faithfulness=(not args.raw_fracture),
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    print(f"Wrote map: {out}")
    return 0


def cmd_certify(args: argparse.Namespace) -> int:
    df = _read_csv(args.csv)
    test = FaithfulnessTest(df)
    cfg = _build_diagnose_config(args)
    res = test.diagnose(cfg)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    html_path = out_dir / "certificate.html"
    res.export_certificate(str(html_path), format="html")
    print(f"Wrote: {html_path}")

    if args.pdf:
        pdf_path = out_dir / "certificate.pdf"
        res.export_certificate(str(pdf_path), format="pdf")
        print(f"Wrote: {pdf_path}")

    return 0


def cmd_plugins(_args: argparse.Namespace) -> int:
    print(list_plugins())
    return 0


def cmd_plugin_help(args: argparse.Namespace) -> int:
    print(plugin_help(args.name))
    return 0


def cmd_guard(args: argparse.Namespace) -> int:
    if GuardConfig is None or decide is None or decision_record_json is None:
        raise RuntimeError(
            "Guard is not available: intervention_faithfulness.core.guard could not be imported."
        )

    if getattr(args, "out", None):
        from intervention_faithfulness.core.guard import export_decision_certificate
        cert_path = export_decision_certificate(decision, args.out)
        print(f"Wrote decision certificate: {cert_path}", file=sys.stderr)

    df = _read_csv(args.csv)
    test = FaithfulnessTest(df)

    cfg = _build_diagnose_config(args)
    res = test.diagnose(cfg)

    gcfg = GuardConfig(
        fracture_threshold=args.fracture_threshold,
        min_effective_samples=args.min_effective_samples,
        require_significance=bool(args.require_significance),
        p_value_threshold=args.p_value_threshold,
        use_safe_envelope_if_available=(not args.no_envelope),
        max_uncertain_fraction=args.max_uncertain_fraction,
    )

    decision = decide(res, gcfg)

    # Emit stable JSON to stdout (for CI + wrappers)
    print(decision_record_json(decision))

    # Exit with policy code (0 OK, 2 boundary, 3 refuse)
    return int(decision.exit_code)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="faithfulness", description="intervention-faithfulness CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    # Shared diagnose args
    def add_common(subp: argparse.ArgumentParser) -> None:
        subp.add_argument("--csv", required=True, help="Input canonical trials CSV")

        subp.add_argument("--metric", default="refinement", choices=["refinement", "pairwise", "both"])
        subp.add_argument("--divergence", default="js", choices=["js", "wasserstein"])
        subp.add_argument("--min-samples", type=int, default=50)
        subp.add_argument("--tail-mode", action="store_true", default=False)
        subp.add_argument("--quantile-focus", type=float, default=0.95)
        subp.add_argument("--n-bins", type=int, default=30)
        subp.add_argument("--n-pairwise-pairs", type=int, default=50)
        subp.add_argument("--n-permutations", type=int, default=0)
        subp.add_argument("--random-state", type=int, default=None)

        subp.add_argument("--no-recommend", action="store_true", default=False)
        subp.add_argument("--top-k", type=int, default=10)

        subp.add_argument(
            "--recommend-mode",
            default="single",
            choices=["single", "greedy"],
            help="Recommendation mode: evaluate single features, or greedy feature-sets.",
        )
        subp.add_argument(
            "--max-set-size",
            type=int,
            default=3,
            help="Max set size for greedy recommendations (ignored for single).",
        )

    # diagnose
    d = sub.add_parser("diagnose", help="Run diagnosis and write artifact bundle")
    add_common(d)
    d.add_argument("--out-dir", required=True, help="Output directory for artifacts")
    d.add_argument("--prefix", default=None, help="Filename prefix for outputs (optional)")

    # Bundle toggles
    d.add_argument("--trials", action="store_true", default=False, help="Also export canonical trials CSV snapshot")
    d.add_argument("--map", action="store_true", default=False, help="Also export map PNG via export_artifacts")
    d.add_argument(
        "--certificate-json",
        dest="certificate_json",
        action="store_true",
        default=True,
        help="Export curated JSON certificate as part of bundle (default: on)",
    )
    d.add_argument(
        "--no-certificate-json",
        dest="certificate_json",
        action="store_false",
        help="Disable JSON certificate export in the bundle",
    )
    d.add_argument(
        "--no-recommendations-csv",
        action="store_true",
        default=False,
        help="Disable recommended_features CSV export in the bundle",
    )

    # Rendered certificates
    d.add_argument("--pdf", action="store_true", default=False, help="Also render certificate PDF (reportlab)")
    d.add_argument("--html", action="store_true", default=False, help="Also render certificate HTML")
    d.set_defaults(func=cmd_diagnose)

    # map
    m = sub.add_parser("map", help="Render a faithfulness map image")
    m.add_argument("--csv", required=True)
    m.add_argument("--out", required=True, help="Output image path (png/pdf/svg)")
    m.add_argument("--title", default=None)
    m.add_argument("--raw-fracture", action="store_true", default=False)

    m.add_argument("--x-col", default="intervention_id")
    m.add_argument("--y-col", default=None)
    m.add_argument("--bins-x", type=int, default=25)
    m.add_argument("--bins-y", type=int, default=25)
    m.add_argument("--min-samples", type=int, default=50)
    m.add_argument("--metric", default="refinement", choices=["refinement", "pairwise", "both"])
    m.add_argument("--divergence", default="js", choices=["js", "wasserstein"])
    m.add_argument("--tail-mode", action="store_true", default=False)
    m.add_argument("--quantile-focus", type=float, default=0.95)
    m.add_argument("--n-bins", type=int, default=30)
    m.add_argument("--n-pairwise-pairs", type=int, default=50)
    m.add_argument("--normalize-quantile", type=float, default=0.95)
    m.add_argument("--no-parse-numeric", action="store_true", default=False)
    m.set_defaults(func=cmd_map)

    # certify
    c = sub.add_parser("certify", help="Write certificate.html (and optionally certificate.pdf)")
    add_common(c)
    c.add_argument("--out-dir", required=True)
    c.add_argument("--pdf", action="store_true", default=False)
    c.set_defaults(func=cmd_certify)

    # plugins
    pl = sub.add_parser("plugins", help="List available data and feature plugins")
    pl.set_defaults(func=cmd_plugins)

    # plugin-help
    ph = sub.add_parser("plugin-help", help="Show help for a specific plugin")
    ph.add_argument("name", help="Plugin name")
    ph.set_defaults(func=cmd_plugin_help)

    # guard
    g = sub.add_parser("guard", help="Run Σ₂-I guard decision (OK/BOUNDARY/REFUSE)")
    add_common(g)
    g.add_argument("--fracture-threshold", type=float, default=0.12)
    g.add_argument("--min-effective-samples", type=int, default=200)
    g.add_argument("--require-significance", action="store_true", default=False)
    g.add_argument("--p-value-threshold", type=float, default=0.05)
    g.add_argument("--no-envelope", action="store_true", default=False)
    g.add_argument("--max-uncertain-fraction", type=float, default=0.50)
    g.add_argument("--out", default=None, help="Optional path to write decision certificate JSON")
    g.set_defaults(func=cmd_guard)

    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
