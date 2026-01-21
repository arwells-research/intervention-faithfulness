import pandas as pd

from intervention_faithfulness.core.FaithfulnessTest import FaithfulnessTest, DiagnoseConfig


def test_export_csv_writes_safe_envelope_csv(tmp_path):
    n = 240
    df = pd.DataFrame(
        {
            "trial_id": list(range(n)),
            "intervention_id": ([0.0] * (n // 3)) + ([1.0] * (n // 3)) + ([2.0] * (n - 2 * (n // 3))),
            "state_x": [1] * n,
            "history_h": ([0] * (n // 2)) + ([1] * (n - (n // 2))),
        }
    )
    df["outcome"] = ([0.1] * (n // 2)) + ([0.9] * (n - (n // 2)))

    cfg = DiagnoseConfig(
        min_samples=20,
        tail_mode=False,
        recommend=False,
    )
    res = FaithfulnessTest(df).diagnose(cfg)

    out = tmp_path / "env.csv"
    written = res.export_csv(
        str(out),
        table="safe_envelope",
        x_col="intervention_id",
        y_col=None,
        bins_x=6,
        bins_y=1,
        min_samples=20,
        faithfulness=True,
        threshold=0.7,
    )

    assert str(out) == written
    assert out.exists()
    assert out.stat().st_size > 0