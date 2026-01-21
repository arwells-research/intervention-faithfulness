import numpy as np
import pandas as pd


def test_plot_faithfulness_map_marks_underpowered_cells_nan(monkeypatch):
    """
    If a heatmap cell has < min_samples rows, maps.py should skip fracture computation
    and leave the grid entry as NaN.
    """
    import intervention_faithfulness.core.maps as maps

    # Build data with two X bins: one underpowered, one sufficiently powered.
    # We'll set bins_x=2 and min_samples=50.
    # - "low" bin: 40 samples (should be NaN)
    # - "high" bin: 80 samples (should be computed)
    n_low = 40
    n_high = 80

    df = pd.DataFrame(
        {
            "trial_id": list(range(n_low + n_high)),
            "intervention_id": (["low"] * n_low) + (["high"] * n_high),
            "outcome": ([0.1] * (n_low // 2) + [0.9] * (n_low - n_low // 2))
            + ([0.1] * (n_high // 2) + [0.9] * (n_high - n_high // 2)),
            "state_x": [1] * (n_low + n_high),
            "history_h": ([0] * (n_low // 2) + [1] * (n_low - n_low // 2))
            + ([0] * (n_high // 2) + [1] * (n_high - n_high // 2)),
        }
    )

    calls = []

    def fake_compute_continuation_fracture(**kwargs):
        calls.append(kwargs)

        class _R:
            fracture_value = 0.5

        return _R()

    # Patch the symbol used by maps.py so we can:
    # (a) count calls, (b) avoid depending on fracture implementation details.
    monkeypatch.setattr(maps, "compute_continuation_fracture", fake_compute_continuation_fracture)

    cfg = maps.MapConfig(
        x_col="intervention_id",
        y_col=None,          # single y row ("all")
        bins_x=2,
        bins_y=1,
        min_samples=50,
        metric="refinement",
        tail_mode=False,
    )

    fig = maps.plot_faithfulness_map(trials_df=df, config=cfg, faithfulness=False)
    assert fig is not None

    # Should have computed exactly one cell (the "high" bin).
    assert len(calls) == 1

    # Extract the image array from the Axes.
    ax = fig.axes[0]
    im = ax.images[0]
    grid = np.asarray(im.get_array())

    # Shape: (y_bins, x_bins) => (1, 2)
    assert grid.shape == (1, 2)

    # One cell should be NaN (underpowered), the other should be the fake value.
    flat = grid.ravel()
    assert np.isnan(flat).sum() == 1
    assert np.isfinite(flat).sum() == 1
    assert float(flat[np.isfinite(flat)][0]) == 0.5