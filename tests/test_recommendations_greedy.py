import pandas as pd

from intervention_faithfulness.core.FaithfulnessTest import FaithfulnessTest, DiagnoseConfig


def test_greedy_recommendation_can_pick_two_feature_set():
    # Construct a case where two independent history labels explain the outcome,
    # and either one alone only partially reduces fracture.
    n = 400
    h1 = ([0] * (n // 2)) + ([1] * (n // 2))
    h2 = ([0, 1] * (n // 2))

    df = pd.DataFrame(
        {
            "trial_id": list(range(n)),
            "intervention_id": ["pulse"] * n,
            "state_I": [1.0] * n,
            "history_h1": h1,
            "history_h2": h2,
        }
    )

    # Outcome depends on (h1 XOR h2) as a crude way to require both labels
    y = []
    for a, b in zip(h1, h2):
        y.append(0.9 if (int(a) ^ int(b)) == 1 else 0.1)
    df["outcome"] = y

    cfg = DiagnoseConfig(
        min_samples=50,
        tail_mode=False,
        recommend=True,
        recommend_mode="greedy",
        recommend_max_set_size=3,
        recommend_top_k=10,
    )

    res = FaithfulnessTest(df).diagnose(cfg)

    recs = res.recommended_features
    assert recs, "Expected greedy recommendations"
    # We expect at least one set recommendation with >=2 features
    first = recs[0]
    # recommended_features can be dict-like (FeatureSetScore.to_dict) in greedy mode
    assert hasattr(first, "name") or isinstance(first, dict)
    if isinstance(first, dict):
        assert "features" in first
        assert len(first["features"]) >= 2
        assert float(first["delta_fracture"]) >= 0.0
    else:
        # fallback if adapter converts it into RecommendedFeature
        assert "+" in first.name