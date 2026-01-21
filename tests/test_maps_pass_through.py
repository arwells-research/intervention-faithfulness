import pandas as pd


def test_plot_faithfulness_map_passes_metric_and_pairwise_pairs(monkeypatch):
    """
    Contract test:
    plot_faithfulness_map must pass MapConfig.metric and MapConfig.n_pairwise_pairs
    to compute_continuation_fracture.
    """
    import intervention_faithfulness.core.maps as maps

    # Make a dataset that ensures at least one heatmap cell passes min_samples.
    n = 120
    df = pd.DataFrame(
        {
            "trial_id": list(range(n)),
            "intervention_id": [1.0] * n,
            "outcome": [0.1] * (n // 2) + [0.9] * (n // 2),
            "state_x": [1] * n,
            "history_h": [0] * (n // 2) + [1] * (n // 2),
        }
    )

    calls = []

    def fake_compute_continuation_fracture(**kwargs):
        calls.append(kwargs)

        class _R:
            fracture_value = 0.123

        return _R()

    # Patch the symbol *as used inside maps.py*
    monkeypatch.setattr(maps, "compute_continuation_fracture", fake_compute_continuation_fracture)

    cfg = maps.MapConfig(
        x_col="intervention_id",
        y_col="history_h",
        bins_x=1,
        bins_y=1,
        min_samples=50,
        metric="pairwise",
        n_pairwise_pairs=77,
        divergence="js",
        tail_mode=False,
        n_bins=19,
        normalize_quantile=0.95,
        try_parse_numeric=True,
    )

    fig = maps.plot_faithfulness_map(trials_df=df, config=cfg, faithfulness=False)
    assert fig is not None
    assert calls, "expected compute_continuation_fracture to be called at least once"

    last = calls[-1]
    assert last.get("metric") == "pairwise"
    assert last.get("n_pairwise_pairs") == 77
    assert last.get("n_bins") == 19