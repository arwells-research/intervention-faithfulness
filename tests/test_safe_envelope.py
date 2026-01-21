import numpy as np
import pandas as pd

from intervention_faithfulness.core.FaithfulnessTest import FaithfulnessTest


def test_safe_envelope_returns_safe_unsafe_uncertain(monkeypatch):
    import intervention_faithfulness.core.maps as maps

    # Make a tiny dataset with 3 categorical interventions.
    # We'll patch compute_faithfulness_grid to control the grid deterministically.
    df = pd.DataFrame(
        {
            "trial_id": list(range(10)),
            "intervention_id": ["A"] * 4 + ["B"] * 3 + ["C"] * 3,
            "outcome": [0.0] * 10,
            "state_x": [1] * 10,
            "history_h": [0] * 10,
        }
    )

    def fake_compute_faithfulness_grid(*, trials_df, config=None, faithfulness=True):
        # 1 row (y), 3 cols (x=A,B,C)
        grid = np.array([[0.9, 0.2, np.nan]], dtype=float)  # safe, unsafe, uncertain
        counts = np.array([[100, 100, 10]], dtype=int)      # last is underpowered
        x_edges = np.arange(4, dtype=float)
        x_labels = ["A", "B", "C"]
        y_edges = np.array([0.0, 1.0])
        y_labels = ["all"]
        cfg_used = config
        return grid, counts, x_edges, x_labels, y_edges, y_labels, cfg_used

    monkeypatch.setattr(maps, "compute_faithfulness_grid", fake_compute_faithfulness_grid)

    res = FaithfulnessTest(df).diagnose()
    regions = res.safe_envelope(
        x_col="intervention_id",
        y_col=None,
        threshold=0.7,
        faithfulness=True,
        min_samples=50,
        bins_x=3,
        bins_y=1,
    )

    # We expect 3 contiguous segments: A safe, B unsafe, C uncertain.
    assert len(regions) == 3
    assert regions[0].status == "safe"
    assert regions[1].status == "unsafe"
    assert regions[2].status == "uncertain"