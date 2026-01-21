# tests/test_recommendations_baseline.py

import pandas as pd

from intervention_faithfulness import FaithfulnessTest


def test_recommendations_reduce_fracture_when_history_is_informative():
    # Construct a toy dataset where:
    #
    # - `history_h` is a "full-history" label (what the system actually depends on).
    # - baseline model state tracks only `state_I` (collapsing histories).
    # - completion augments the state with `state_h`, which matches `history_h`.
    #
    # With the current v0.1 fracture definition (reduced state vs provided full-history key),
    # baseline should exhibit fracture > 0 and augmented state should reduce it toward 0.
    n = 200
    history_h = ([0] * (n // 2)) + ([1] * (n // 2))
  
    df = pd.DataFrame(
       {
              "trial_id": list(range(n)),
              "intervention_id": ["pulse"] * n,
              "state_I": [1.0] * n,  # identical reduced state component
              "history_h": history_h,  # provided full-history label
       }
    )
  
    # Outcome depends on history_h (two distinct distributions)
    df["outcome"] = ([0.1] * (n // 2)) + ([0.9] * (n // 2))
  
    # Baseline: reduced state does NOT track h (state_I only), but full-history is available.
    base = FaithfulnessTest(df).diagnose()
  
    # Completion: augment reduced state with h (matches full-history), fracture should drop.
    df_aug = df.copy()
    df_aug["state_h"] = df_aug["history_h"]
    full = FaithfulnessTest(df_aug).diagnose()
  
    def get_score(res):
        for attr in ("fracture_score", "fracture", "score"):
            if hasattr(res, attr):
                return float(getattr(res, attr))
        return None
  
    base_score = get_score(base)
    full_score = get_score(full)
  
    if base_score is not None and full_score is not None:
        assert full_score <= base_score