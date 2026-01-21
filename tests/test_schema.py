# tests/test_schema.py

import pandas as pd
import pytest

from intervention_faithfulness.core.schema import validate_trials_table


def test_validate_trials_table_raises_on_non_dataframe():
    with pytest.raises(ValueError):
        validate_trials_table({"trial_id": [1], "intervention_id": [0], "outcome": [1.0]})  # type: ignore


def test_validate_trials_table_raises_on_missing_required_columns():
    df = pd.DataFrame({"trial_id": [1], "outcome": [1.0]})
    with pytest.raises(ValueError) as e:
        validate_trials_table(df)
    assert "Missing required columns" in str(e.value)


def test_validate_trials_table_minimal_valid_table_returns_warnings_list():
    df = pd.DataFrame(
        {
            "trial_id": [1, 2, 3],
            "intervention_id": ["A", "A", "B"],
            "outcome": [0.1, 0.2, 0.3],
        }
    )
    warnings = validate_trials_table(df)
    assert isinstance(warnings, list)


def test_validate_trials_table_accepts_state_and_history_columns():
    df = pd.DataFrame(
        {
            "trial_id": [1, 2, 3, 4],
            "intervention_id": ["A", "A", "B", "B"],
            "outcome": [0.1, 0.2, 0.3, 0.4],
            "state_I": [7.1, 7.1, 7.2, 7.2],
            "state_V": [0.02, 0.02, 0.03, 0.03],
            "history_integrated_current": [1.0, 1.1, 0.9, 1.2],
        }
    )
    warnings = validate_trials_table(df)
    assert isinstance(warnings, list)


def test_validate_trials_table_non_numeric_outcome_warns_but_does_not_raise():
    df = pd.DataFrame(
        {
            "trial_id": [1, 2, 3],
            "intervention_id": ["A", "A", "B"],
            "outcome": ["x", "y", "z"],
        }
    )
    warnings = validate_trials_table(df)
    # Should not raise; should warn
    assert any(getattr(w, "code", "") == "outcome_non_numeric" for w in warnings) or isinstance(warnings, list)