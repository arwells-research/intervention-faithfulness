# intervention_faithfulness/core/FaithfulnessTest.py

"""
core/FaithfulnessTest.py — User-facing orchestration class (v0.1)

This is the primary user entry point.

Responsibilities:
- Accept a canonical trials table OR load via a data plugin
- Validate schema
- Apply feature plugins (optional)
- Run diagnostic (fracture, optional permutation testing)
- Optionally rank minimal-completion candidates
- Return DiagnosisResult (reporting wrapper)

Design principles:
- Core remains domain-agnostic
- Plugins handle domain formats and sensible defaults
- Reporting should not embed policy thresholds; it reports computed metrics + provenance
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any, Tuple, Literal

import pandas as pd

from intervention_faithfulness.core.schema import validate_trials_table
from intervention_faithfulness.core.fracture import compute_continuation_fracture
from intervention_faithfulness.core.recommendation import rank_minimal_completions
from intervention_faithfulness.core.reporting import DiagnosisResult
from intervention_faithfulness.plugins.registry import get_data_plugin, get_feature_plugin


def _ensure_builtin_plugins_registered() -> None:
    """
    Import built-in plugins exactly once, so decorator registration runs.

    This avoids relying on incidental import side-effects across the codebase.
    """
    # Local import so package import time stays light and deterministic.
    import intervention_faithfulness.plugins  # noqa: F401


@dataclass(frozen=True)
class DiagnoseConfig:
    # Fracture computation
    # Metric definition:
    # - "refinement" = compare P(Y|S) vs P(Y|S,H_k) within state bins (v0.1 baseline)
    # - "pairwise"   = compare P(Y|S,H_k1) vs P(Y|S,H_k2) within state bins
    # - "both"       = compute both; report both values
    metric: str = "refinement"
    divergence: str = "js"
    min_samples: int = 50
    tail_mode: bool = True
    quantile_focus: float = 0.95
    n_bins: int = 30
    n_pairwise_pairs: int = 50
    n_permutations: int = 0
    random_state: Optional[int] = None

    # Recommendations
    recommend: bool = True
    recommend_top_k: int = 10
    recommend_mode: Literal["single", "greedy"] = "single"
    recommend_greedy_k: int = 2           # used by greedy2
    recommend_min_delta: float = 0.0      # filter threshold
    recommend_max_set_size: int = 3

    # If None: use all history_* columns present after feature steps
    recommend_history_cols: Optional[List[str]] = None

    # Candidate feature plugins to *try* (name, params)
    recommend_feature_plugins: Optional[List[Tuple[str, Dict[str, Any]]]] = None

    # Safe envelope (derived from map grid)
    # If True, populate fracture_res.safe_regions so it appears in reporting/certificates.
    safe_envelope: bool = True
    safe_envelope_threshold: float = 0.7
    safe_envelope_faithfulness: bool = True
    safe_envelope_x_col: str = "intervention_id"
    safe_envelope_y_col: Optional[str] = None
    safe_envelope_bins_x: int = 25
    safe_envelope_bins_y: int = 25
    safe_envelope_normalize_quantile: float = 0.95
    safe_envelope_try_parse_numeric: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FaithfulnessTest:
    """
    Main user-facing class.

    Construct with a canonical trials table:
        test = FaithfulnessTest(trials_df)

    Or construct via plugin:
        test = FaithfulnessTest.from_plugin("nanowire_switching", "data.csv")

    Add feature plugins:
        test.add_feature("integrated_current", window_ns=50)

    Run:
        results = test.diagnose()
    """

    def __init__(
        self,
        trials_df: pd.DataFrame,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        _ensure_builtin_plugins_registered()

        self._base_trials_df = trials_df.copy()
        self._metadata = dict(metadata or {})
        self._feature_steps: List[Tuple[str, Dict[str, Any]]] = []

        # Validate immediately so schema errors appear early.
        self._schema_warnings = validate_trials_table(self._base_trials_df)

    @property
    def trials_df(self) -> pd.DataFrame:
        """
        Canonical trials table (base, pre-feature augmentation).

        Exposed for tests and for workflows that need to directly augment
        the canonical table (e.g., promoting history_* to state_*).
        """
        return self._base_trials_df.copy()
        
    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_plugin(
        cls,
        plugin_name: str,
        source: Any,
        *,
        plugin_kwargs: Optional[Dict[str, Any]] = None,
        to_trials_kwargs: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "FaithfulnessTest":
        """
        Construct FaithfulnessTest by loading and canonicalizing data using a registered data plugin.

        plugin_kwargs: passed to plugin.load(...)
        to_trials_kwargs: passed to plugin.to_trials(...)
        """
        _ensure_builtin_plugins_registered()

        plugin_kwargs = plugin_kwargs or {}
        to_trials_kwargs = to_trials_kwargs or {}

        plugin = get_data_plugin(plugin_name)
        raw = plugin.load(source, **plugin_kwargs)
        trials_df = plugin.to_trials(raw, **to_trials_kwargs)

        md = dict(metadata or {})
        md.setdefault("data_plugin", plugin_name)
        md.setdefault("data_source", str(source))

        # Capture domain warnings (non-fatal)
        domain_warns: List[str] = []
        if hasattr(plugin, "validate"):
            try:
                domain_warns = list(plugin.validate(trials_df) or [])
            except Exception:
                domain_warns = []
        if domain_warns:
            md["plugin_warnings"] = domain_warns

        # Capture plugin defaults for provenance (not mandatory)
        if hasattr(plugin, "defaults"):
            try:
                md["plugin_defaults"] = dict(plugin.defaults())
            except Exception:
                pass

        return cls(trials_df, metadata=md)

    # ------------------------------------------------------------------
    # Feature augmentation
    # ------------------------------------------------------------------

    def add_feature(self, feature_plugin_name: str, **params) -> "FaithfulnessTest":
        """
        Add a feature plugin step to be applied before diagnose().
        """
        _ensure_builtin_plugins_registered()
        self._feature_steps.append((feature_plugin_name, dict(params)))
        return self

    def with_feature(self, feature_plugin_name: str, **params) -> "FaithfulnessTest":
        """
        Alias for add_feature() for fluent usage.
        """
        return self.add_feature(feature_plugin_name, **params)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def run(self, config: "DiagnoseConfig" | None = None, **kwargs: object):
        """
        v0.1 convenience: run the test.

        If a regime/slice column is present, return dict[str, DiagnosisResult]
        keyed by slice name. Otherwise return a single DiagnosisResult.

        Tests expect slicing when `regime_slice` exists.
        """
        import pandas as pd

        df = self._base_trials_df  # or whatever your attribute is
        if not isinstance(df, pd.DataFrame):
            return self.diagnose(config=config, **kwargs)

        # Prefer explicit `regime_slice`, else any `regime_*` column.
        slice_col = None
        if "regime_slice" in df.columns:
            slice_col = "regime_slice"
        else:
            for c in df.columns:
                if isinstance(c, str) and c.startswith("regime_"):
                    slice_col = c
                    break

        if slice_col is None:
            return self.diagnose(config=config, **kwargs)

        results: dict[str, object] = {}
        for key, sub in df.groupby(slice_col, sort=True):
            sub_test = self.__class__(sub.copy(), metadata=dict(self._metadata))
            sub_test._feature_steps = list(self._feature_steps)
            results[str(key)] = sub_test.diagnose(config=config, **kwargs)

        # Optional convenience attribute (tests also check .by_slice)
        try:
            setattr(results, "by_slice", results)  # harmless no-op for dict (will fail)
        except Exception:
            pass

        return results

    def diagnose(self, config: "DiagnoseConfig" | None = None, **kwargs: object):
        """
        Run the continuation fracture diagnostic and return a reporting wrapper.

        Baseline semantics (important):
        - The diagnostic itself uses whatever columns are present in trials_df.
        - The recommendation engine defines its baseline Σ₁ representation as:
              (trial_id, intervention_id, outcome, state_*)
          i.e., it drops all history_* columns and tests which additions restore faithfulness.

        This makes "state vs history" an explicit operational distinction in the tool.
        """
        if config is None:
            # Allow diagnose(divergence=..., min_samples=..., ...) as a convenience API.
            if kwargs:
                config = DiagnoseConfig(**kwargs)
            else:
                config = DiagnoseConfig()
        else:
            if kwargs:
                raise TypeError("Pass either config=... or keyword overrides, not both.")

        cfg = config or DiagnoseConfig()

        # Start from base canonical table
        df = self._base_trials_df.copy()

        # Apply feature plugins in order
        applied_features: List[Dict[str, Any]] = []
        for name, params in self._feature_steps:
            plugin = get_feature_plugin(name)
            df = plugin.compute(df, **params)
            applied_features.append({"name": name, "params": dict(params)})

        # Validate again after features (collect warnings for provenance)
        post_feature_warnings = validate_trials_table(df)

        # Compute fracture
        fracture_res = compute_continuation_fracture(
            trials_df=df,
            metric=cfg.metric,
            divergence=cfg.divergence,
            min_samples=cfg.min_samples,
            tail_mode=cfg.tail_mode,
            quantile_focus=cfg.quantile_focus,
            n_bins=cfg.n_bins,
            n_pairwise_pairs=cfg.n_pairwise_pairs,
            n_permutations=cfg.n_permutations,
            random_state=cfg.random_state,
        )

        # Minimal completion ranking (optional)
        if cfg.recommend:
            recommend_mode = getattr(cfg, "recommend_mode", "single")
            recommend_greedy_k = int(getattr(cfg, "recommend_greedy_k", 2))
            recommend_min_delta = float(getattr(cfg, "recommend_min_delta", 0.0))

            if recommend_mode == "greedy":
                from intervention_faithfulness.core.recommendation import rank_minimal_completion_sets

                max_set_size = int(getattr(cfg, "recommend_max_set_size", 2))
                sets = rank_minimal_completion_sets(
                    trials_df=df,
                    candidate_history_cols=cfg.recommend_history_cols,
                    candidate_feature_plugins=cfg.recommend_feature_plugins,
                    divergence=cfg.divergence,
                    min_samples=cfg.min_samples,
                    tail_mode=cfg.tail_mode,
                    quantile_focus=cfg.quantile_focus,
                    n_bins=cfg.n_bins,
                    top_k=cfg.recommend_top_k,
                    max_set_size=max_set_size,
                    greedy_k=recommend_greedy_k,
                    min_delta=recommend_min_delta,
                )
                fracture_res.recommended_features = [s.to_dict() for s in sets]
            else:
                scores = rank_minimal_completions(
                    trials_df=df,
                    candidate_history_cols=cfg.recommend_history_cols,
                    candidate_feature_plugins=cfg.recommend_feature_plugins,
                    divergence=cfg.divergence,
                    min_samples=cfg.min_samples,
                    tail_mode=cfg.tail_mode,
                    quantile_focus=cfg.quantile_focus,
                    n_bins=cfg.n_bins,
                    top_k=cfg.recommend_top_k,
                )
                fracture_res.recommended_features = [
                    {
                        "name": s.name,
                        "delta_fracture": float(s.delta_fracture),
                        "mutual_info": s.mutual_info,
                        "params": s.params,
                        "data_requirements": s.data_requirements,
                        "baseline_fracture": float(s.baseline_fracture),
                        "augmented_fracture": float(s.augmented_fracture),
                    }
                    for s in scores
                ]

        # NOTE: safe envelope is a derived artifact attached to the fracture result for reporting.
        # This must never cause diagnose() to fail.
        if getattr(cfg, "safe_envelope", False):
            try:
                # DiagnosisResult is the reporting wrapper; it may implement helpers for map/envelope.
                # If not, this call should raise and we’ll convert to a warning.
                tmp_res = DiagnosisResult(
                    result=fracture_res,
                    trials_df=df,
                    config=cfg.to_dict(),
                    metadata=dict(self._metadata),
                )
                regions = tmp_res.safe_envelope(
                    x_col=getattr(cfg, "safe_envelope_x_col", "intervention_id"),
                    y_col=getattr(cfg, "safe_envelope_y_col", None),
                    bins_x=getattr(cfg, "safe_envelope_bins_x", 25),
                    bins_y=getattr(cfg, "safe_envelope_bins_y", 25),
                    min_samples=cfg.min_samples,
                    metric=getattr(cfg, "metric", "refinement"),
                    divergence=cfg.divergence,
                    tail_mode=cfg.tail_mode,
                    quantile_focus=cfg.quantile_focus,
                    n_bins=cfg.n_bins,
                    n_pairwise_pairs=getattr(cfg, "n_pairwise_pairs", 50),
                    normalize_quantile=getattr(cfg, "safe_envelope_normalize_quantile", 0.95),
                    try_parse_numeric=getattr(cfg, "safe_envelope_try_parse_numeric", True),
                    threshold=getattr(cfg, "safe_envelope_threshold", 0.7),
                    faithfulness=getattr(cfg, "safe_envelope_faithfulness", True),
                )
                fracture_res.safe_regions = [
                    {"label": r.label, "status": r.status, "details": r.details} for r in regions
                ]
            except Exception as e:
                ws = list(getattr(fracture_res, "warnings", []) or [])
                ws.append(f"safe_envelope failed: {type(e).__name__}: {e}")
                fracture_res.warnings = ws

        # Provenance for reporting (no policy thresholds here)
        report_config = cfg.to_dict()
        report_metadata = dict(self._metadata)
        report_metadata["applied_features"] = applied_features

        base_warns = [getattr(w, "message", str(w)) for w in (self._schema_warnings or [])]
        post_warns = [getattr(w, "message", str(w)) for w in (post_feature_warnings or [])]
        report_metadata["schema_warnings"] = sorted(set(base_warns + post_warns))

        return DiagnosisResult(
            result=fracture_res,
            trials_df=df,
            config=report_config,
            metadata=report_metadata,
        )
