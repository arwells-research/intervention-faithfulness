import pandas as pd


def test_plot_faithfulness_map_categorical_intervention_smoke():
    import intervention_faithfulness.core.maps as maps

    # Enough samples so min_samples is satisfied for at least one cell.
    n = 120
    df = pd.DataFrame(
        {
            "trial_id": list(range(n)),
            "intervention_id": ["A"] * (n // 2) + ["B"] * (n // 2),
            "outcome": [0.1] * (n // 2) + [0.9] * (n // 2),
            "state_x": [1] * n,
            "history_h": [0] * (n // 2) + [1] * (n // 2),
        }
    )

    cfg = maps.MapConfig(
        x_col="intervention_id",
        y_col="history_h",
        bins_x=2,      # ignored for categorical but should not break
        bins_y=2,
        min_samples=50,
        metric="refinement",
        tail_mode=False,
    )

    fig = maps.plot_faithfulness_map(trials_df=df, config=cfg, faithfulness=True)
    assert fig is not None