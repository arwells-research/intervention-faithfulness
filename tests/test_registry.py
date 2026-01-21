# tests/test_registry.py

from intervention_faithfulness.plugins.registry import (
    list_data_plugins,
    list_feature_plugins,
    get_data_plugin,
    get_feature_plugin,
)


def test_list_data_plugins_returns_iterable():
    names = list_data_plugins()
    assert isinstance(names, (list, tuple))
    assert all(isinstance(n, str) for n in names)


def test_list_feature_plugins_returns_iterable():
    names = list_feature_plugins()
    assert isinstance(names, (list, tuple))
    assert all(isinstance(n, str) for n in names)


def test_get_data_plugin_pedagogical_rc_exists_if_registered():
    names = list_data_plugins()
    if "pedagogical_rc" in names:
        plugin = get_data_plugin("pedagogical_rc")
        assert plugin is not None


def test_get_data_plugin_nanowire_switching_exists_if_registered():
    names = list_data_plugins()
    if "nanowire_switching" in names:
        plugin = get_data_plugin("nanowire_switching")
        assert plugin is not None


def test_get_data_plugin_unknown_raises():
    try:
        get_data_plugin("__definitely_not_a_real_plugin__")
        assert False, "Expected get_data_plugin() to raise for unknown plugin"
    except Exception:
        pass


def test_get_feature_plugin_integrated_current_exists_if_registered():
    names = list_feature_plugins()
    if "integrated_current" in names:
        plugin = get_feature_plugin("integrated_current")
        assert plugin is not None


def test_get_feature_plugin_unknown_raises():
    try:
        get_feature_plugin("__definitely_not_a_real_feature__")
        assert False, "Expected get_feature_plugin() to raise for unknown feature plugin"
    except Exception:
        pass