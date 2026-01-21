"""
core/reporting.py — Results object + reporting helpers (v0.1)

This module defines the public-facing results container returned by:
    FaithfulnessTest.diagnose()

Goals:
- Provide a stable, ergonomic surface for users.
- Keep "what users need" (plots, certificates, summaries) close to results.
- Avoid domain assumptions: reporting is generic and metadata-driven.
- Maintain auditability: include plugin provenance, config, and warnings.

This is a skeleton. The data structures are stable; implementations may evolve.

Stability note:
- Public attributes/properties of DiagnosisResult are part of the v0.1 API surface.
- Internal result object shape may change; adapters below should preserve the surface.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Optional
from typing import Any, Dict, Optional, List, Tuple, Union, Iterable
import json
import datetime as _dt
from datetime import UTC
import hashlib
import io
import numpy as np

import pandas as pd

# ---------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------
 
def _sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def _sha256_file(path: Union[str, Path], *, chunk_size: int = 1024 * 1024) -> str:
    """
    Compute SHA-256 of a file (streaming).
    """
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def _to_csv_stable_newlines(df: pd.DataFrame, buf_or_path, *, index: bool = False) -> None:
    """
    Write CSV with stable newline behavior across pandas versions.

    Pandas historically uses `lineterminator` (no underscore). Some environments may
    not support `line_terminator`. We try the common spelling first and fall back.
    """
    try:
        # Common pandas spelling
        df.to_csv(buf_or_path, index=index, lineterminator="\n")
    except TypeError:
        # Fallback: no lineterminator support (rare) -> accept pandas default
        df.to_csv(buf_or_path, index=index)


def _canonical_trials_csv_bytes(df: pd.DataFrame) -> bytes:
    """
    Deterministically serialize trials_df to canonical CSV bytes for hashing.

    Rules:
    - columns sorted lexicographically
    - index excluded
    - consistent line endings
    - represent NaN consistently via pandas defaults
    """
    # Sort columns for stable ordering across runs
    cols = sorted(df.columns.tolist())
    buf = io.StringIO()
    # Ensure consistent output across pandas versions
    _to_csv_stable_newlines(df.loc[:, cols], buf, index=False)
    return buf.getvalue().encode("utf-8")


def _sha256_trials_table(df: pd.DataFrame) -> str:
    """
    SHA-256 over canonical CSV serialization of trials table.
    """
    return _sha256_bytes(_canonical_trials_csv_bytes(df))


def hash_trials_table(df: pd.DataFrame) -> str:
    import hashlib

    d = df.copy()
    cols_sorted = sorted(list(d.columns), key=lambda x: str(x))
    d = d[cols_sorted]
    csv = d.to_csv(index=False, lineterminator="\n", float_format="%.12g")
    return hashlib.sha256(csv.encode("utf-8")).hexdigest()
    

@dataclass(frozen=True)
class FractureScore:
    """
    Continuation fracture metric summary.

    Attributes
    ----------
    value:
        Point estimate of continuation fracture (e.g., Jensen–Shannon divergence aggregated).
    ci_low, ci_high:
        Confidence interval bounds (if computed).
    method:
        CI estimation method (e.g., bootstrap, asymptotic).
    n_effective:
        Effective sample size used for estimate (or None if not computed).
    warnings:
        Human-readable warnings related to statistical power or estimation.
    """

    value: float
    # Which metric produced `value` (e.g., "refinement", "pairwise", "both")
    metric: Optional[str] = None
    # Optional: include both components when available (e.g., {"refinement": ..., "pairwise": ...})
    metrics: Optional[Dict[str, float]] = None    
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None
    method: Optional[str] = None
    n_effective: Optional[int] = None
    warnings: Optional[List[str]] = None

    def __float__(self) -> float:
        """
        Allow FractureScore to be used as a numeric in downstream tooling.
 
        This preserves the richer structure while supporting simple consumers:
             float(results.fracture_score)
        """
        return float(self.value)

@dataclass(frozen=True)
class SignificanceTest:
    """
    Significance testing summary.

    Attributes
    ----------
    p_value:
        P-value from permutation testing or other method.
    method:
        Test method used (e.g., permutation).
    n_permutations:
        Number of permutations used (if applicable).
    """

    p_value: Optional[float] = None
    method: Optional[str] = None
    n_permutations: Optional[int] = None
    warnings: Optional[List[str]] = None


@dataclass(frozen=True)
class RecommendedFeature:
    """
    Minimal completion recommendation entry.

    Attributes
    ----------
    name:
        Feature plugin name (or feature label for user-provided history variable).
    delta_fracture:
        Estimated reduction in fracture if the feature is added (positive is good).
    mutual_info:
        Mutual information score or similar relevance metric.
    params:
        Parameters used when computing the feature (if known).
    data_requirements:
        Human-readable note about what the feature requires (columns, time-series, etc.)
    """

    name: str
    delta_fracture: float
    mutual_info: Optional[float] = None
    params: Optional[Dict[str, Any]] = None
    data_requirements: Optional[str] = None


@dataclass(frozen=True)
class SafeRegion:
    """
    A simple safe/unsafe envelope summary entry.

    Attributes
    ----------
    label:
        Human-readable label (e.g., "ramp_rate < 2 GHz").
    status:
        One of: "safe", "unsafe", "uncertain"
    details:
        Optional details or caveats.
    """

    label: str
    status: str
    details: Optional[str] = None


class DiagnosisResult:
    """
    Immutable results container returned by FaithfulnessTest.diagnose().

    Parameters
    ----------
    result:
        Raw result object returned by core algorithms (opaque to users).
    trials_df:
        Canonical trials table used for diagnosis (for reproducibility).
    config:
        Resolved configuration used for diagnosis (divergence, min_samples, etc.).
    metadata:
        Metadata for reporting, including plugin provenance and user-supplied system info.
    """

 
    @property
    def fracture(self) -> float:
        """
        Numeric alias for the continuation fracture point estimate.
 
        This is an ergonomic convenience for users who want a plain float.
        """
        return float(self.fracture_score)

    def __init__(
        self,
        *,
        result: Any,
        trials_df: pd.DataFrame,
        config: Dict[str, Any],
        metadata: Dict[str, Any],
    ):
        self._result = result
        self._trials_df = trials_df
        self._config = config
        self._metadata = metadata

    @property
    def trials_df(self) -> pd.DataFrame:
        return self._trials_df.copy()

    @property
    def fracture(self) -> float:
        # Numeric alias for convenience
        return float(self.fracture_score)
 
    @property
    def score(self) -> float:
        # Back-compat / convenience alias for simple consumers
        return float(self.fracture_score)

    @property
    def tag(self) -> str:
        f = getattr(self.result, "fracture_value", float("nan"))
        if not np.isfinite(f):
            return "BOUNDARY"
        # lightweight default; can be refined later by CERTIFICATION.md rules
        return "PASS" if f <= 0.08 else "BOUNDARY"

    @property
    def status(self) -> str:
        return self.tag

    @property
    def verdict(self) -> str:
        return self.tag        

    # ------------------------------------------------------------------
    # Primary surfaced properties (these should remain stable)
    # ------------------------------------------------------------------

    @property
    def trials_df(self) -> pd.DataFrame:
        return self._trials_df.copy()

    @property
    def config(self) -> Dict[str, Any]:
        return dict(self._config)

    @property
    def metadata(self) -> Dict[str, Any]:
        return dict(self._metadata)

    @property
    def trials_table_sha256(self) -> str:
        from intervention_faithfulness.core.reporting import hash_trials_table
        return hash_trials_table(self._trials_df)

    @property
    def fracture_score(self) -> FractureScore:
        """
        Return the continuation fracture score with uncertainty if available.

        This adapter converts internal result fields into a stable, user-facing structure.
        """
        # TODO: map from internal result format
        value = float(getattr(self._result, "fracture_value", getattr(self._result, "F", 0.0)))
        ci = getattr(self._result, "fracture_ci", None)

        ci_low = ci[0] if isinstance(ci, (tuple, list)) and len(ci) == 2 else None
        ci_high = ci[1] if isinstance(ci, (tuple, list)) and len(ci) == 2 else None

        warnings = getattr(self._result, "warnings", None)
        if warnings is not None and not isinstance(warnings, list):
            warnings = [str(warnings)]

        return FractureScore(
            value=value,
            metric=getattr(self._result, "metric", None),
            metrics=getattr(self._result, "metrics", None),            
            ci_low=ci_low,
            ci_high=ci_high,
            method=getattr(self._result, "ci_method", None),
            n_effective=getattr(self._result, "n_effective", None),
            warnings=warnings,
        )

    @property
    def significance(self) -> SignificanceTest:
        """
        Return significance testing results (p-value and test metadata).
        """
        # TODO: map from internal result format
        p_value = getattr(self._result, "p_value", None)
        method = getattr(self._result, "test_method", "permutation")
        n_perm = getattr(self._result, "n_permutations", None)

        warnings = getattr(self._result, "significance_warnings", None)
        if warnings is not None and not isinstance(warnings, list):
            warnings = [str(warnings)]

        return SignificanceTest(
            p_value=p_value,
            method=method,
            n_permutations=n_perm,
            warnings=warnings,
        )

    @property
    def recommended_features(self) -> List[RecommendedFeature]:
        """
        Ranked list of recommended minimal completion features.

        If the core did not run feature ranking, returns an empty list.
        """
        # TODO: map from internal result format
        recs = getattr(self._result, "recommended_features", None)
        if not recs:
            return []

        out: List[RecommendedFeature] = []
        for r in recs:
            # Accept either dict-like or object-like records
            if isinstance(r, dict):
                out.append(
                    RecommendedFeature(
                        name=str(r.get("name")),
                        delta_fracture=float(r.get("delta_fracture", 0.0)),
                        mutual_info=r.get("mutual_info"),
                        params=r.get("params"),
                        data_requirements=r.get("data_requirements"),
                    )
                )
            else:
                out.append(
                    RecommendedFeature(
                        name=str(getattr(r, "name")),
                        delta_fracture=float(getattr(r, "delta_fracture", 0.0)),
                        mutual_info=getattr(r, "mutual_info", None),
                        params=getattr(r, "params", None),
                        data_requirements=getattr(r, "data_requirements", None),
                    )
                )
        return out

    # ------------------------------------------------------------------
    # Breakdown / diagnostics
    # ------------------------------------------------------------------

    @property
    def breakdown_df(self) -> pd.DataFrame:
        """
        Per-intervention, per-state breakdown of fracture contributions.

        Returns an empty DataFrame if breakdown was not computed.
        """
        bd = getattr(self._result, "breakdown", None)
        if bd is None:
            return pd.DataFrame()
        return bd.copy()


    @property
    def safe_operating_regions(self) -> List[SafeRegion]:
        """
        Human-readable safe/unsafe/uncertain regime summary.

        If core did not compute envelopes, returns an empty list.
        """
        # TODO: map from internal result format
        regs = getattr(self._result, "safe_regions", None)
        if not regs:
            return []

        out: List[SafeRegion] = []
        for r in regs:
            if isinstance(r, dict):
                out.append(
                    SafeRegion(
                        label=str(r.get("label")),
                        status=str(r.get("status")),
                        details=r.get("details"),
                    )
                )
            else:
                out.append(
                    SafeRegion(
                        label=str(getattr(r, "label")),
                        status=str(getattr(r, "status")),
                        details=getattr(r, "details", None),
                    )
                )
        return out

    # ------------------------------------------------------------------
    # Plots / maps
    # ------------------------------------------------------------------

    def faithfulness_map(self, **kwargs) -> Any:
        """
        Generate a faithfulness map visualization.

        Returns a matplotlib Figure by default (or a plotly Figure if enabled).
        The exact plotting backend is implemented in core/maps.py.

        kwargs are forwarded to the plotting backend.
        """
        # Import lazily to keep base import light (matplotlib can be slow / optional).
        from intervention_faithfulness.core.maps import plot_faithfulness_map

        # Core maps operate on the canonical trials table; internal result object is not required.
        # (This keeps maps strictly a visualization of "fracture computed on subsets of trials_df".)
        return plot_faithfulness_map(trials_df=self._trials_df, **kwargs)


    def safe_envelope(
        self,
        *,
        x_col: str = "intervention_id",
        y_col: Optional[str] = None,
        bins_x: int = 25,
        bins_y: int = 25,
        min_samples: int = 50,
        metric: str = "refinement",
        divergence: str = "js",
        tail_mode: bool = False,
        quantile_focus: float = 0.95,
        n_bins: int = 30,
        n_pairwise_pairs: int = 50,
        normalize_quantile: float = 0.95,
        try_parse_numeric: bool = True,
        threshold: float = 0.7,
        faithfulness: bool = True,
        max_categories_in_label: int = 6,
    ) -> List[SafeRegion]:
        """
        Compute a simple safe/unsafe/uncertain envelope summary from a faithfulness map.

        Semantics (v0.1):
        - Cells with < min_samples (or NaN) are labeled "uncertain".
        - If faithfulness=True: cell is "safe" if value >= threshold else "unsafe".
        - If faithfulness=False (raw fracture): cell is "safe" if value <= threshold else "unsafe".

        The envelope is summarized along x for each y-bin (or a single row if y_col is None).
        """
        # Lazy import to avoid importing matplotlib on pure-report paths.
        from intervention_faithfulness.core.maps import MapConfig

        cfg = MapConfig(
            x_col=x_col,
            y_col=y_col,
            bins_x=bins_x,
            bins_y=bins_y,
            min_samples=min_samples,
            metric=metric,
            divergence=divergence,
            tail_mode=tail_mode,
            quantile_focus=quantile_focus,
            n_bins=n_bins,
            n_pairwise_pairs=n_pairwise_pairs,
            normalize_quantile=normalize_quantile,
            try_parse_numeric=try_parse_numeric,
        )

        # Freeze caller intent early; never reference the outer name again.
        faith = bool(faithfulness)
        thr = float(threshold)
        ms = int(min_samples)  # use the function argument, not cfg_used

        import intervention_faithfulness.core.maps as _maps

        grid, counts, x_edges, x_labels, _y_edges, y_labels, cfg_used = _maps.compute_faithfulness_grid(
            trials_df=self._trials_df,
            config=cfg,
            faithfulness=faith,
            )

        # Determine whether x should be treated as numeric for envelope labeling.
        # Do NOT infer this from x_edges alone (categorical edges are often numeric arange).
        x_series = self._trials_df[cfg_used.x_col]
        x_is_numeric = False
        if pd.api.types.is_numeric_dtype(x_series):
            x_is_numeric = True
        elif bool(cfg_used.try_parse_numeric):
            cand = pd.to_numeric(x_series, errors="coerce")
            n_finite = int(np.isfinite(cand.to_numpy(dtype=float)).sum())
            # Heuristic: if at least half parse as finite numbers, treat as numeric.
            if n_finite >= max(1, len(x_series) // 2):
                x_is_numeric = True

        # Also require a consistent edge array for numeric bin labeling.
        x_edges_ok = (
            x_is_numeric
            and isinstance(x_edges, np.ndarray)
            and x_edges.ndim == 1
            and len(x_edges) == (len(x_labels) + 1)
            and np.issubdtype(x_edges.dtype, np.number)
        )        

        def cell_status(v: float, n: int) -> str:
            if n < ms or not np.isfinite(v):
                return "uncertain"
            if faith:
                return "safe" if float(v) >= thr else "unsafe"
            return "safe" if float(v) <= thr else "unsafe"

        def fmt_num(x: float) -> str:
            # Compact numeric formatting for labels.
            try:
                return f"{float(x):.3g}"
            except Exception:
                return str(x)

        def x_segment_label(i0: int, i1: int) -> str:
            """
            Produce a human-readable x-range label for bins [i0..i1] inclusive.
            Uses numeric edges when available; otherwise uses categorical labels.
            """
            if x_edges_ok and len(x_edges) >= 2:
                lo = x_edges[i0]
                hi = x_edges[i1 + 1]
                return f"{cfg_used.x_col}: {fmt_num(lo)}–{fmt_num(hi)}"

            # Categorical axis
            labs = x_labels[i0 : i1 + 1]
            if len(labs) <= max_categories_in_label:
                return f"{cfg_used.x_col}: {', '.join(labs)}"
            return f"{cfg_used.x_col}: {labs[0]}–{labs[-1]} ({len(labs)} bins)"

        regions: List[SafeRegion] = []

        # IMPORTANT:
        # - compute_faithfulness_grid(..., faithfulness=...) determines what "grid" means.
        # - If faithfulness=True, grid is already a faithfulness score (higher is safer).
        # - If faithfulness=False, grid is fracture (lower is safer).
        #
        # Therefore: never remap grid based on the local "faithfulness" flag again.

        for yi, ylab in enumerate(y_labels):
            statuses: List[str] = []

            for xi in range(len(x_labels)):
                v = float(grid[yi, xi]) if np.isfinite(grid[yi, xi]) else float("nan")
                n = int(counts[yi, xi])
                statuses.append(cell_status(v, n))

            # Segment contiguous runs in *x_labels order* (do not sort)
            start = 0
            while start < len(statuses):
                st = str(statuses[start])
                end = start
                while end + 1 < len(statuses) and str(statuses[end + 1]) == st:
                    end += 1

                base_label = x_segment_label(start, end)
                if cfg_used.y_col is not None and ylab != "all":
                    label = f"{cfg_used.y_col}={ylab} | {base_label}"
                else:
                    label = base_label

                details = None
                if st == "uncertain":
                    try:
                        seg_counts = counts[yi, start : end + 1]
                        seg_min = int(np.min(seg_counts)) if seg_counts.size else 0
                    except Exception:
                        seg_min = 0
                    details = f"Underpowered: min cell n={seg_min} (< {cfg_used.min_samples})."

                regions.append(SafeRegion(label=label, status=st, details=details))

                start = end + 1

        return regions

    def safe_envelope_table(
        self,
        *,
        x_col: str = "intervention_id",
        y_col: Optional[str] = None,
        bins_x: int = 25,
        bins_y: int = 25,
        min_samples: int = 50,
        metric: str = "refinement",
        divergence: str = "js",
        tail_mode: bool = False,
        quantile_focus: float = 0.95,
        n_bins: int = 30,
        n_pairwise_pairs: int = 50,
        normalize_quantile: float = 0.95,
        try_parse_numeric: bool = True,
        threshold: float = 0.7,
        faithfulness: bool = True,
        max_categories_in_label: int = 6,
    ) -> pd.DataFrame:
        """
        Tabular form of safe_envelope(), suitable for CSV export and programmatic use.

        Returns a DataFrame with columns:
            - label
            - status  (safe | unsafe | uncertain)
            - details

        This is intentionally "tidy" and domain-agnostic: labels are human-readable
        strings produced by safe_envelope().
        """
        regs = self.safe_envelope(
            x_col=x_col,
            y_col=y_col,
            bins_x=bins_x,
            bins_y=bins_y,
            min_samples=min_samples,
            metric=metric,
            divergence=divergence,
            tail_mode=tail_mode,
            quantile_focus=quantile_focus,
            n_bins=n_bins,
            n_pairwise_pairs=n_pairwise_pairs,
            normalize_quantile=normalize_quantile,
            try_parse_numeric=try_parse_numeric,
            threshold=threshold,
            faithfulness=faithfulness,
            max_categories_in_label=max_categories_in_label,
        )

        # Keep it simple and stable: no parsing of labels into columns (users can if they want).
        rows: List[Dict[str, Any]] = []
        for r in regs:
            rows.append(
                {
                    "label": r.label,
                    "status": r.status,
                    "details": r.details,
                }
            )

        return pd.DataFrame(rows, columns=["label", "status", "details"])

    def export_csv(
        self,
        path: str,
        *,
        table: str = "safe_envelope",
        index: bool = False,
        **kwargs: Any,
    ) -> str:
        """
        Export a CSV artifact for quick operational use.

        Parameters
        ----------
        path:
            Output CSV file path.
        table:
            Which table to export:
                - "safe_envelope" (default): safe_envelope_table(**kwargs)
                - "recommended_features": recommendations list as a table
                - "trials": the canonical trials table used for diagnosis
        index:
            Whether to write the DataFrame index to CSV.
        **kwargs:
            Forwarded to the relevant table builder (currently only used for "safe_envelope").

        Returns
        -------
        str:
            The path written.
        """
        tbl = (table or "").strip().lower()
        if not tbl:
            tbl = "safe_envelope"

        if not path.lower().endswith(".csv"):
            # Keep behavior obvious and ergonomic.
            path = path + ".csv"

        # Build DataFrame
        if tbl == "safe_envelope":
            df = self.safe_envelope_table(**kwargs)
        elif tbl == "recommended_features":
            rows: List[Dict[str, Any]] = []
            for r in self.recommended_features:
                rows.append(
                    {
                        "name": r.name,
                        "delta_fracture": float(r.delta_fracture),
                        "mutual_info": r.mutual_info,
                        "data_requirements": r.data_requirements,
                        "params": json.dumps(r.params, sort_keys=True) if r.params is not None else None,
                    }
                )
            df = pd.DataFrame(
                rows,
                columns=["name", "delta_fracture", "mutual_info", "data_requirements", "params"],
            )
        elif tbl == "trials":
            df = self._trials_df.copy()
        else:
            raise ValueError(
                f"Unsupported table='{table}'. Use one of: 'safe_envelope', 'recommended_features', 'trials'."
            )

        # Ensure directory exists if user provided one
        out_dir = os.path.dirname(path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        df.to_csv(path, index=index)
        return path

    def export_artifacts(
        self,
        out_dir: str,
        *,
        prefix: Optional[str] = None,
        # Envelope defaults
        envelope_table_kwargs: Optional[Dict[str, Any]] = None,
        # Map defaults
        include_map: bool = True,
        map_kwargs: Optional[Dict[str, Any]] = None,
        # Optional extra exports
        include_certificate: bool = True,
        include_trials: bool = False,
        include_recommendations: bool = True,
    ) -> Dict[str, str]:
        """
        Export a small bundle of operational artifacts into a directory.

        Artifacts (v0.1):
        - diagnosis JSON:      <prefix>_diagnosis.json
        - safe envelope CSV:   <prefix>_safe_envelope.csv
        - recommendations CSV: <prefix>_recommended_features.csv (optional)
        - map PNG:             <prefix>_faithfulness_map.png (optional)
        - trials CSV:          <prefix>_trials.csv (optional)
        - breakdown CSV:       <prefix>_breakdown.csv

        Parameters
        ----------
        out_dir:
            Directory to write outputs into (created if missing).
        prefix:
            Filename prefix. If None, uses a timestamp-based prefix.
        envelope_table_kwargs:
            Forwarded to safe_envelope_table(**kwargs).
        include_map:
            Whether to export a PNG faithfulness map.
        map_kwargs:
            Forwarded to plot_faithfulness_map(trials_df=..., config=..., faithfulness=..., title=...).
        include_trials:
            Whether to export the full canonical trials_df to CSV.
        include_recommendations:
            Whether to export recommended_features to CSV.

        Returns
        -------
        Dict[str, str]:
            Keys -> file paths written.
        """
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        if not prefix:
            prefix = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d_%H%M%S")

        written: Dict[str, str] = {}

        # --------------------------------------------------------------
        # 1) Diagnosis JSON
        # --------------------------------------------------------------
        diagnosis_path = out / f"{prefix}_diagnosis.json"
        self.export_json(str(diagnosis_path))
        written["diagnosis_json"] = str(diagnosis_path)

        # --------------------------------------------------------------
        # 1a) Trials CSV (optional)
        # --------------------------------------------------------------
        trials_path = None
        if include_trials:
            trials_path = out / f"{prefix}_trials.csv"
            # Deterministic-ish export: sort columns
            cols = sorted(self._trials_df.columns.tolist())
            _to_csv_stable_newlines(self._trials_df.loc[:, cols], str(trials_path), index=False)
            written["trials_csv"] = str(trials_path)

        # Compute artifact hashes for “both” (diagnosis + trials if present)
        artifact_hashes: Dict[str, str] = {}
        try:
            artifact_hashes["diagnosis_json_sha256"] = _sha256_file(diagnosis_path)
        except Exception:
            # keep best-effort; certificate still exports
            pass

        if trials_path is not None:
            try:
                artifact_hashes["trials_csv_sha256"] = _sha256_file(trials_path)
            except Exception:
                pass

        # --------------------------------------------------------------
        # 1b) Certificate JSON (curated) (optional)
        # --------------------------------------------------------------
        if include_certificate:
            cert_path = out / f"{prefix}_certificate.json"
            self.export_certificate_json(str(cert_path), artifact_hashes=artifact_hashes)
            written["certificate_json"] = str(cert_path)

        # --------------------------------------------------------------
        # 1c) Breakdown CSV (canonical)
        # --------------------------------------------------------------
        breakdown_path = out / f"{prefix}_breakdown.csv"
        bd = None
        try:
            bd = self.breakdown_df
        except Exception:
            bd = None

        if bd is not None:
            bd.to_csv(breakdown_path, index=False)
            written["breakdown_csv"] = str(breakdown_path)

        # --------------------------------------------------------------
        # 2) Safe envelope CSV
        # --------------------------------------------------------------
        env_kwargs = dict(envelope_table_kwargs or {})
        env_path = out / f"{prefix}_safe_envelope.csv"
        # safe_envelope_table already returns a DataFrame; keep export_csv plumbing consistent
        df_env = self.safe_envelope_table(**env_kwargs)
        df_env.to_csv(str(env_path), index=False)
        written["safe_envelope_csv"] = str(env_path)

        # --------------------------------------------------------------
        # 3) Recommendations CSV (optional)
        # --------------------------------------------------------------
        if include_recommendations:
            rec_path = out / f"{prefix}_recommended_features.csv"
            # uses export_csv helper (stable formatting incl params JSON)
            self.export_csv(str(rec_path), table="recommended_features")
            written["recommended_features_csv"] = str(rec_path)

        # --------------------------------------------------------------
        # 4) Faithfulness map PNG (optional)
        # --------------------------------------------------------------
        if include_map:
            # Lazy import so users can run in headless/report-only paths
            from intervention_faithfulness.core.maps import MapConfig, plot_faithfulness_map

            mk = dict(map_kwargs or {})

            # MapConfig fields are explicit; keep this forgiving:
            # - if user passed a "config" object, prefer it
            cfg_obj = mk.pop("config", None)
            if cfg_obj is None:
                cfg_obj = MapConfig(
                    x_col=mk.pop("x_col", "intervention_id"),
                    y_col=mk.pop("y_col", None),
                    bins_x=int(mk.pop("bins_x", 25)),
                    bins_y=int(mk.pop("bins_y", 25)),
                    min_samples=int(mk.pop("min_samples", 50)),
                    metric=str(mk.pop("metric", "refinement")),
                    divergence=str(mk.pop("divergence", "js")),
                    tail_mode=bool(mk.pop("tail_mode", False)),
                    quantile_focus=float(mk.pop("quantile_focus", 0.95)),
                    n_bins=int(mk.pop("n_bins", 30)),
                    n_pairwise_pairs=int(mk.pop("n_pairwise_pairs", 50)),
                    normalize_quantile=float(mk.pop("normalize_quantile", 0.95)),
                    try_parse_numeric=bool(mk.pop("try_parse_numeric", True)),
                )

            faith = bool(mk.pop("faithfulness", True))
            title = mk.pop("title", "Faithfulness Map")

            # Any remaining mk keys are ignored intentionally (UX > strictness).
            fig = plot_faithfulness_map(
                trials_df=self._trials_df,
                config=cfg_obj,
                title=title,
                faithfulness=faith,
            )

            map_path = out / f"{prefix}_faithfulness_map.png"
            fig.savefig(str(map_path), dpi=150)
            try:
                import matplotlib.pyplot as _plt
                _plt.close(fig)
            except Exception:
                pass

            written["faithfulness_map_png"] = str(map_path)

        return written

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------

    def certificate_dict(
        self,
        *,
        artifact_hashes: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Produce a *curated* certificate payload intended for downstream presentation.

        Design intent:
        - Stable, small surface (safe to email, attach, archive).
        - Avoid embedding the full trials table.
        - Include enough config + provenance to be auditable.

        Notes:
        - JSON-first in v0.1; PDF/HTML rendering can consume this payload.
        """
        fs = self.fracture_score
        sig = self.significance

        # Keep recommendations + envelope readable and bounded.
        recs = [
            {
                "name": r.name,
                "delta_fracture": r.delta_fracture,
                "mutual_info": r.mutual_info,
                "data_requirements": r.data_requirements,
                "params": r.params,
            }
            for r in self.recommended_features
        ]

        env = [
            {
                "label": r.label,
                "status": r.status,
                "details": r.details,
            }
            for r in self.safe_operating_regions
        ]

        meta = dict(self._metadata or {})

        art_hashes = dict(artifact_hashes or {})
        # Keep only string hashes
        art_hashes = {str(k): str(v) for k, v in art_hashes.items() if v is not None}

        return {
            "certificate": {
                "tool": "intervention-faithfulness",
                "version": meta.get("version", "v0.1"),
                "issued_utc": _utcnow_iso(),
            },
            "fracture_score": {
                "value": fs.value,
                "ci_low": fs.ci_low,
                "ci_high": fs.ci_high,
                "method": fs.method,
                "n_effective": fs.n_effective,
                "warnings": fs.warnings or [],
            },
            "significance": {
                "p_value": sig.p_value,
                "method": sig.method,
                "n_permutations": sig.n_permutations,
                "warnings": sig.warnings or [],
            },
            "safe_operating_regions": env,
            "recommended_features": recs,
            "config": dict(self._config or {}),
            "provenance": {
                # keep this flexible; avoid committing to exact keys beyond "applied_features"
                "applied_features": meta.get("applied_features", []),
                "schema_warnings": meta.get("schema_warnings", []),
                # user may add: dataset_id, git_sha, run_id, hostname, etc.
                "metadata": {
                    k: v
                    for k, v in meta.items()
                    if k not in ("applied_features", "schema_warnings")
                },
                "artifact_hashes": art_hashes,
            },
        }

    def export_certificate_json(
        self,
        path: str,
        *,
        artifact_hashes: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Export a curated certificate JSON payload.
        """
        payload = self.certificate_dict(artifact_hashes=artifact_hashes)
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return str(out_path)

    def export_certificate(self, path: Optional[str] = None, format: Optional[str] = None) -> str:
        """
        Export a Model Validity Certificate as PDF or HTML.

        Parameters
        ----------
        path:
            Output path. If None, a default filename is chosen in the current directory.
        format:
            'pdf' or 'html'. If None, inferred from path extension (or defaults to 'pdf').

        Returns
        -------
        str:
            Final output path written.
        """
        fmt = (format or "").lower().strip() if format else None
        if path and not fmt:
            lowered = path.lower()
            if lowered.endswith(".pdf"):
                fmt = "pdf"
            elif lowered.endswith(".html") or lowered.endswith(".htm"):
                fmt = "html"
        if not fmt:
            fmt = "pdf"

        if not path:
            ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"model_validity_{ts}.{fmt}"

        if fmt not in ("pdf", "html"):
            raise ValueError(f"Unsupported certificate format: {fmt}")

        payload = self.to_dict()
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if fmt == "html":
            html = _render_certificate_html(payload)
            out_path.write_text(html, encoding="utf-8")
            return str(out_path)

        # PDF
        _write_certificate_pdf(out_path, payload)
        return str(out_path)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the diagnosis result to a dict suitable for JSON export.
        """
        fs = self.fracture_score
        sig = self.significance

        recs = [
            {
                "name": r.name,
                "delta_fracture": r.delta_fracture,
                "mutual_info": r.mutual_info,
                "params": r.params,
                "data_requirements": r.data_requirements,
            }
            for r in self.recommended_features
        ]

        regions = [
            {
                "label": r.label,
                "status": r.status,
                "details": r.details,
            }
            for r in self.safe_operating_regions
        ]

        return {
            "tool": {
                "name": "intervention-faithfulness",
                "timestamp_utc": _utcnow_iso(),
            },
            "trials_table_sha256": _sha256_trials_table(self._trials_df),
            "fracture_score": {
                "value": fs.value,
                "metric": fs.metric,
                "metrics": fs.metrics or {},                
                "ci_low": fs.ci_low,
                "ci_high": fs.ci_high,
                "method": fs.method,
                "n_effective": fs.n_effective,
                "warnings": fs.warnings or [],
            },
            "significance": {
                "p_value": sig.p_value,
                "method": sig.method,
                "n_permutations": sig.n_permutations,
                "warnings": sig.warnings or [],
            },
            "recommended_features": recs,
            "safe_operating_regions": regions,
            "breakdown": self.breakdown_df.to_dict(orient="records"),
            "config": dict(self._config),
            "metadata": dict(self._metadata),
            # These hashes are extremely useful for audit bundles.
            "hashes": {
                "diagnosis_sha256": _sha256_bytes(json.dumps(
                    {
                        "fracture_score": {
                            "value": fs.value,
                            "ci_low": fs.ci_low,
                            "ci_high": fs.ci_high,
                            "method": fs.method,
                            "n_effective": fs.n_effective,
                        },
                        "significance": {
                            "p_value": sig.p_value,
                            "method": sig.method,
                            "n_permutations": sig.n_permutations,
                        },
                        "recommended_features": recs,
                        "safe_operating_regions": regions,
                        "config": dict(self._config),
                        "metadata": dict(self._metadata),
                    },
                    sort_keys=True,
                    ensure_ascii=False,
                ).encode("utf-8")),
                "trials_table_sha256": _sha256_trials_table(self._trials_df),
            },            
        }

    def export_json(self, path: str) -> str:
        """
        Export a JSON record of the diagnosis (for reproducibility and audits).
        """
        payload = self.to_dict()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        return path

    @property
    def result(self):
        return self._result

# ------------------------------------------------------------------
# Hashing + certificate rendering helpers
# ------------------------------------------------------------------

def _utcnow_iso() -> str:
    # Avoid deprecated utcnow(); return UTC-aware ISO.
    return _dt.datetime.now(_dt.UTC).isoformat().replace("+00:00", "Z")


def _render_certificate_html(payload: Dict[str, Any]) -> str:
    """
    Minimal but useful HTML certificate.
    No external assets; easy to archive alongside JSON/CSV artifacts.
    """
    tool = payload.get("tool", {})
    fs = payload.get("fracture_score", {})
    sig = payload.get("significance", {})
    recs = payload.get("recommended_features", []) or []
    regs = payload.get("safe_operating_regions", []) or []
    cfg = payload.get("config", {}) or {}
    meta = payload.get("metadata", {}) or {}
    hashes = payload.get("hashes", {}) or {}

    def esc(x: Any) -> str:
        s = "" if x is None else str(x)
        return (
            s.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    rec_rows = ""
    for r in recs[:20]:
        rec_rows += (
            "<tr>"
            f"<td>{esc(r.get('name'))}</td>"
            f"<td>{esc(r.get('delta_fracture'))}</td>"
            f"<td>{esc(r.get('mutual_info'))}</td>"
            f"<td>{esc(r.get('data_requirements'))}</td>"
            "</tr>"
        )
    if not rec_rows:
        rec_rows = "<tr><td colspan='4'><em>None</em></td></tr>"

    reg_rows = ""
    for r in regs[:50]:
        reg_rows += (
            "<tr>"
            f"<td>{esc(r.get('status'))}</td>"
            f"<td>{esc(r.get('label'))}</td>"
            f"<td>{esc(r.get('details'))}</td>"
            "</tr>"
        )
    if not reg_rows:
        reg_rows = "<tr><td colspan='3'><em>None</em></td></tr>"

    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Model Validity Certificate</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 24px; }}
    h1 {{ margin: 0 0 8px 0; }}
    .meta {{ color: #444; margin-bottom: 20px; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
    .card {{ border: 1px solid #ddd; border-radius: 12px; padding: 14px; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ border-bottom: 1px solid #eee; padding: 8px; text-align: left; vertical-align: top; }}
    th {{ background: #fafafa; }}
    code {{ background: #f6f6f6; padding: 2px 6px; border-radius: 6px; }}
    .small {{ font-size: 12px; color: #555; }}
  </style>
</head>
<body>
  <h1>Model Validity Certificate</h1>
  <div class="meta">
    <div><b>Tool:</b> {esc(tool.get("name"))}</div>
    <div><b>Issued:</b> {esc(tool.get("timestamp_utc"))}</div>
  </div>

  <div class="grid">
    <div class="card">
      <h3>Score</h3>
      <div><b>fracture</b>: <code>{esc(fs.get("value"))}</code></div>
      <div class="small">CI: {esc(fs.get("ci_low"))} – {esc(fs.get("ci_high"))} ({esc(fs.get("method"))})</div>
      <div class="small">n_effective: {esc(fs.get("n_effective"))}</div>
      <div class="small">warnings: {esc(", ".join(fs.get("warnings", []) or []))}</div>
    </div>

    <div class="card">
      <h3>Significance</h3>
      <div><b>p_value</b>: <code>{esc(sig.get("p_value"))}</code></div>
      <div class="small">method: {esc(sig.get("method"))}</div>
      <div class="small">n_permutations: {esc(sig.get("n_permutations"))}</div>
      <div class="small">warnings: {esc(", ".join(sig.get("warnings", []) or []))}</div>
    </div>
  </div>

  <div class="card" style="margin-top:16px;">
    <h3>Safe envelope</h3>
    <table>
      <thead><tr><th>Status</th><th>Label</th><th>Details</th></tr></thead>
      <tbody>{reg_rows}</tbody>
    </table>
  </div>

  <div class="card" style="margin-top:16px;">
    <h3>Recommended minimal completions</h3>
    <table>
      <thead><tr><th>Feature</th><th>Δ fracture</th><th>MI</th><th>Requirements</th></tr></thead>
      <tbody>{rec_rows}</tbody>
    </table>
  </div>

  <div class="grid" style="margin-top:16px;">
    <div class="card">
      <h3>Config</h3>
      <pre class="small">{esc(json.dumps(cfg, indent=2, sort_keys=True))}</pre>
    </div>
    <div class="card">
      <h3>Provenance</h3>
      <pre class="small">{esc(json.dumps(meta, indent=2, sort_keys=True))}</pre>
    </div>
  </div>

  <div class="card" style="margin-top:16px;">
    <h3>Hashes</h3>
    <div class="small"><b>diagnosis_sha256</b>: <code>{esc(hashes.get("diagnosis_sha256"))}</code></div>
    <div class="small"><b>trials_table_sha256</b>: <code>{esc(hashes.get("trials_table_sha256"))}</code></div>
  </div>
</body>
</html>
"""


def _write_certificate_pdf(path: Path, payload: Dict[str, Any]) -> None:
    """
    PDF backend via reportlab. If reportlab isn't available, we fall back to HTML
    next to the requested PDF path (and raise a helpful error).
    """
    try:
        from reportlab.lib.pagesizes import LETTER
        from reportlab.lib.units import inch
        from reportlab.pdfgen import canvas
    except Exception as e:
        # Always generate an HTML sibling for utility.
        html_path = path.with_suffix(".html")
        html_path.write_text(_render_certificate_html(payload), encoding="utf-8")
        raise RuntimeError(
            "PDF certificate requires reportlab. Wrote HTML certificate instead: "
            f"{html_path}"
        ) from e

    tool = payload.get("tool", {})
    fs = payload.get("fracture_score", {})
    sig = payload.get("significance", {})
    regs = payload.get("safe_operating_regions", []) or []
    recs = payload.get("recommended_features", []) or []
    hashes = payload.get("hashes", {}) or {}

    c = canvas.Canvas(str(path), pagesize=LETTER)
    w, h = LETTER

    x0 = 0.75 * inch
    y = h - 0.85 * inch

    def line(txt: str, dy: float = 14) -> None:
        nonlocal y
        c.drawString(x0, y, txt)
        y -= dy
        if y < 0.75 * inch:
            c.showPage()
            y = h - 0.85 * inch

    c.setFont("Helvetica-Bold", 16)
    line("Model Validity Certificate", dy=20)

    c.setFont("Helvetica", 10)
    line(f"Tool: {tool.get('name', '')}")
    line(f"Issued (UTC): {tool.get('timestamp_utc', '')}")
    line("")

    c.setFont("Helvetica-Bold", 12)
    line("Score", dy=16)
    c.setFont("Helvetica", 10)
    line(f"fracture: {fs.get('value')}")
    line(f"CI: {fs.get('ci_low')} – {fs.get('ci_high')} ({fs.get('method')})")
    line(f"n_effective: {fs.get('n_effective')}")
    line("")

    c.setFont("Helvetica-Bold", 12)
    line("Significance", dy=16)
    c.setFont("Helvetica", 10)
    line(f"p_value: {sig.get('p_value')}")
    line(f"method: {sig.get('method')}, n_permutations: {sig.get('n_permutations')}")
    line("")

    c.setFont("Helvetica-Bold", 12)
    line("Safe envelope (first 20 rows)", dy=16)
    c.setFont("Helvetica", 9)
    for r in regs[:20]:
        line(f"[{r.get('status')}] {r.get('label')}", dy=12)
    if len(regs) > 20:
        line(f"... ({len(regs) - 20} more)", dy=12)
    line("")

    c.setFont("Helvetica-Bold", 12)
    line("Recommendations (first 10)", dy=16)
    c.setFont("Helvetica", 9)
    for r in recs[:10]:
        line(f"{r.get('name')}: Δ={r.get('delta_fracture')} (MI={r.get('mutual_info')})", dy=12)
    if len(recs) > 10:
        line(f"... ({len(recs) - 10} more)", dy=12)
    line("")

    c.setFont("Helvetica-Bold", 12)
    line("Hashes", dy=16)
    c.setFont("Helvetica", 9)
    line(f"diagnosis_sha256: {hashes.get('diagnosis_sha256')}", dy=12)
    line(f"trials_table_sha256: {hashes.get('trials_table_sha256')}", dy=12)

    c.save()
