import pandas as pd

def test_guard_never_ok_when_underpowered() -> None:
    from intervention_faithfulness.core.FaithfulnessTest import FaithfulnessTest
    from intervention_faithfulness.core.guard import GuardConfig, decide

    n = 40
    df = pd.DataFrame({
        "trial_id": range(n),
        "intervention_id": ["I0"] * (n // 2) + ["I1"] * (n - n // 2),
        "outcome": [0.0] * n,
        "state_s": [0.0] * n,
    })

    test = FaithfulnessTest(df)
    res = test.diagnose()
    decision = decide(res, GuardConfig(min_effective_samples=200))

    assert decision.status != "OK"
