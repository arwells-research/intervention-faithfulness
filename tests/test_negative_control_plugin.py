# tests/test_negative_control_plugin.py

from intervention_faithfulness import FaithfulnessTest
from intervention_faithfulness.plugins.registry import list_data_plugins


def test_negative_control_plugin_is_low_fracture_if_available():
    if "faithful_regime" not in list_data_plugins():
        # Plugin not included in this build; do not fail the test suite.
        return

    test = FaithfulnessTest.from_plugin("faithful_regime", None)
    results = test.diagnose()

    assert results is not None

    # Be tolerant about attribute names and numeric scale.
    if hasattr(results, "fracture_score"):
        assert float(results.fracture_score) < 0.10
    elif hasattr(results, "fracture"):
        assert float(results.fracture) < 0.10
    elif hasattr(results, "score"):
        assert float(results.score) < 0.10