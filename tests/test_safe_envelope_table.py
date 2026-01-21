import pandas as pd

from intervention_faithfulness.core.FaithfulnessTest import FaithfulnessTest, DiagnoseConfig


def test_safe_envelope_table_returns_dataframe_with_expected_columns():
    # Construct a small dataset with numeric intervention and history partition so a map can form.
    n = 240
    df = pd.DataFrame(
        {
            "trial_id": list(range(n)),
            "intervention_id": ([0.0] * (n // 3)) + ([1.0] * (n // 3)) + ([2.0] * (n - 2 * (n // 3))),
            "state_x": [1] * n,
            "history_h": ([0] * (n // 2)) + ([1] * (n - (n // 2))),
        }
    )
    # Outcome depends on history label so there is something for fracture to detect in some bins.
    df["outcome"] = ([0.1] * (n // 2)) + ([0.9] * (n - (n // 2)))

    cfg = DiagnoseConfig(
        min_samples=20,   # keep small enough that per-cell can qualify
        tail_mode=False,
        recommend=False,
    )
    res = FaithfulnessTest(df).diagnose(cfg)

    tbl = res.safe_envelope_table(
        x_col="intervention_id",
        y_col=None,
        bins_x=6,
        bins_y=1,
        min_samples=20,
        metric="refinement",
        faithfulness=True,
        threshold=0.7,
    )

    assert isinstance(tbl, pd.DataFrame)
    assert list(tbl.columns) == ["label", "status", "details"]
    assert len(tbl) >= 1
    assert set(tbl["status"].unique()).issubset({"safe", "unsafe", "uncertain"})