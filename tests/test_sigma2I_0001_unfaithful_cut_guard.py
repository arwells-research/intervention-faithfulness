from __future__ import annotations

import pytest


def _get_result_slice(results, slice_name: str):
    """
    Helper to pull a named slice from result structures.
    Adjust this to match your core's actual return structure.
    """
    # Common patterns:
    # - results is a dict keyed by slice/regime name
    # - results has an attribute like .by_slice or .slices
    if isinstance(results, dict):
        return results[slice_name]
    if hasattr(results, "by_slice"):
        return results.by_slice[slice_name]
    if hasattr(results, "slices"):
        return results.slices[slice_name]
    raise TypeError("Unknown results structure; update _get_result_slice() for your core.")


def _tag_of(res) -> str:
    """
    Normalize tag/status extraction.

    Expected semantics:
    - faithful => PASS/OK/SAFE-like tag
    - unfaithful => BOUNDARY/REFUSE-like tag
    """
    # Common patterns:
    for k in ["tag", "status", "verdict", "label"]:
        if isinstance(res, dict) and k in res:
            return str(res[k])
        if hasattr(res, k):
            return str(getattr(res, k))
    raise KeyError("Could not extract tag/status from slice result; update _tag_of().")


def test_sigma2I_0001_unfaithful_cut_is_refused():
    # Import inside test so pytest shows clean import errors.
    from intervention_faithfulness.core.FaithfulnessTest import FaithfulnessTest

    # NOTE:
    # This assumes FaithfulnessTest has a from_plugin constructor per docs/plugin_spec.md:
    #   FaithfulnessTest.from_plugin(data_plugin_name, source, **kwargs)
    #
    # If your signature differs, adjust here only.
    test = FaithfulnessTest.from_plugin("sigma2i_unfaithful_cut_linear", source=None)

    # If your framework requires specifying a slice column, do it here.
    # This plugin provides `regime_slice` as an optional regime column.
    #
    # If your core auto-detects regime_* only, you may want to rename
    # `regime_slice` to `regime_slice` (it already is "regime_*"-prefixed? no).
    #
    # If needed, change plugin column to `regime_slice` -> `regime_slice` is NOT prefixed.
    # Consider updating plugin to `regime_slice` -> `regime_slice` only if your core supports it.
    #
    # For now, we assume the core can slice or the test config can request slicing.
    #
    # Example (if supported):
    # test = test.set_regime_columns(["regime_slice"])

    results = test.run()

    # Expect two regime slices:
    # - FAITHFUL_BASELINE
    # - UNFAITHFUL_CUT
    base = _get_result_slice(results, "FAITHFUL_BASELINE")
    cut = _get_result_slice(results, "UNFAITHFUL_CUT")

    base_tag = _tag_of(base).upper()
    cut_tag = _tag_of(cut).upper()

    # Faithful baseline must pass (allow a small set of synonyms).
    assert base_tag in {"OK", "PASS", "SAFE", "CERTIFIED", "OK_BASE", "OK_FAITHFUL"}, (
        f"Expected faithful baseline to pass; got tag={base_tag}"
    )

    # Unfaithful cut must be refused.
    assert cut_tag in {"BOUNDARY", "REFUSE", "REJECT", "FAIL_UNFAITHFUL"}, (
        f"Expected unfaithful cut to be refused; got tag={cut_tag}"
    )
