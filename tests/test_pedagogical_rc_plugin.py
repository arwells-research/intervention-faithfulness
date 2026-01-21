# tests/test_pedagogical_rc_plugin.py

from intervention_faithfulness import FaithfulnessTest
from intervention_faithfulness.plugins.registry import list_data_plugins


def test_pedagogical_rc_plugin_smoke_if_available():
    if "pedagogical_rc" not in list_data_plugins():
        # Plugin not included in this build; do not fail the test suite.
        return

    # For this plugin, "source" may be ignored or may accept config; keep it minimal.
    test = FaithfulnessTest.from_plugin("pedagogical_rc", None)
    results = test.diagnose()

    assert results is not None
    assert hasattr(results, "fracture_score") or hasattr(results, "fracture") or hasattr(results, "score")