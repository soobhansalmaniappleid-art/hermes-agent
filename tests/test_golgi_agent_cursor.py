"""Golgi fork: computer_use.agent_cursor styles the visible cua-driver cursor."""

from unittest.mock import Mock

from tools.computer_use import cua_backend


def _backend():
    backend = cua_backend.CuaDriverBackend.__new__(cua_backend.CuaDriverBackend)
    backend._session_id = "hermes-test"
    backend.set_agent_cursor_style = Mock()
    backend.set_agent_cursor_motion = Mock()
    return backend


def test_agent_cursor_config_is_applied(monkeypatch):
    monkeypatch.setattr(
        cua_backend,
        "_computer_use_cfg",
        lambda: {"agent_cursor": {"gradient_colors": ["#c4adff", "#785ce8"], "bloom_color": "#9d7bf7", "glide_ms": 420}},
    )
    backend = _backend()
    backend._apply_agent_cursor_config()
    backend.set_agent_cursor_style.assert_called_once_with(
        cursor_id="hermes-test", gradient_colors=["#c4adff", "#785ce8"], bloom_color="#9d7bf7"
    )
    backend.set_agent_cursor_motion.assert_called_once_with(cursor_id="hermes-test", glide_ms=420.0)


def test_missing_agent_cursor_config_changes_nothing(monkeypatch):
    monkeypatch.setattr(cua_backend, "_computer_use_cfg", lambda: {})
    backend = _backend()
    backend._apply_agent_cursor_config()
    backend.set_agent_cursor_style.assert_not_called()
    backend.set_agent_cursor_motion.assert_not_called()


def test_cursor_styling_failure_is_not_fatal(monkeypatch):
    monkeypatch.setattr(cua_backend, "_computer_use_cfg", lambda: {"agent_cursor": {"bloom_color": "#fff"}})
    backend = _backend()
    backend.set_agent_cursor_style.side_effect = RuntimeError("driver too old")
    backend._apply_agent_cursor_config()


def test_gradient_colors_saved_as_json_string_are_parsed(monkeypatch):
    monkeypatch.setattr(
        cua_backend, "_computer_use_cfg", lambda: {"agent_cursor": {"gradient_colors": '["#c4adff","#785ce8"]'}}
    )
    backend = _backend()
    backend._apply_agent_cursor_config()
    backend.set_agent_cursor_style.assert_called_once_with(
        cursor_id="hermes-test", gradient_colors=["#c4adff", "#785ce8"]
    )
