# tests/test_faithfulness_test_smoke.py

import pandas as pd

from intervention_faithfulness.core.FaithfulnessTest import FaithfulnessTest, DiagnoseConfig


def test_faithfulness_test_smoke_runs_and_returns_results_object():
    # Minimal canonical trials table with repeated states under different interventions
    df = pd.DataFrame(
        {
            "trial_id": [1, 2, 3, 4, 5, 6],
            "intervention_id": ["A", "A", "A", "B", "B", "B"],
            "outcome": [0.10, 0.12, 0.11, 0.40, 0.42, 0.39],
            "state_x": [1, 1, 1, 1, 1, 1],
        }
    )

    test = FaithfulnessTest(df)
    # Smoke test: ensure pipeline runs on tiny toy data.
    # Use a relaxed min_samples so fracture computation doesn't prune everything.
    results = test.diagnose(
        DiagnoseConfig(
            min_samples=2,
            recommend=False,
            n_permutations=0,
        )
    )

    # Avoid over-constraining names; just ensure we got something with a score-like attribute.
    assert results is not None
    assert hasattr(results, "fracture_score") or hasattr(results, "fracture") or hasattr(results, "score")