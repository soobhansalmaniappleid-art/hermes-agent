"""Golgi fork: OpenRouter keys can be saved/removed through the gateway."""

from unittest.mock import Mock

from tui_gateway import server


def _stub_payload(monkeypatch, providers):
    monkeypatch.setattr(server, "_model_picker_context", Mock(return_value=object()))
    monkeypatch.setattr(
        "hermes_cli.inventory.build_models_payload",
        Mock(return_value={"providers": providers}),
    )


def test_openrouter_key_is_saved_to_its_env_var(monkeypatch):
    monkeypatch.setattr("hermes_cli.auth.PROVIDER_REGISTRY", {})
    monkeypatch.setattr("hermes_cli.config.is_managed", lambda: False)
    save_credential = Mock()
    monkeypatch.setattr(
        "hermes_cli.credential_lifecycle.save_provider_env_credential", save_credential
    )
    _stub_payload(monkeypatch, [])
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    fake_key = "sk-or-" + "test"

    resp = server._methods["model.save_key"](1, {"slug": "openrouter", "api_key": fake_key})

    assert "result" in resp, resp
    assert resp["result"]["provider"]["slug"] == "openrouter"
    assert resp["result"]["provider"]["authenticated"] is True
    save_credential.assert_called_once_with("OPENROUTER_API_KEY", fake_key)


def test_unknown_provider_without_registry_entry_is_still_rejected(monkeypatch):
    monkeypatch.setattr("hermes_cli.auth.PROVIDER_REGISTRY", {})
    monkeypatch.setattr("hermes_cli.config.is_managed", lambda: False)

    resp = server._methods["model.save_key"](2, {"slug": "nope", "api_key": "x"})

    assert resp["error"]["code"] == 4002


def test_openrouter_disconnect_removes_its_env_var(monkeypatch):
    monkeypatch.setattr("hermes_cli.auth.PROVIDER_REGISTRY", {})
    remove = Mock(return_value={"found": True})
    monkeypatch.setattr(
        "hermes_cli.credential_lifecycle.remove_provider_env_credential", remove
    )
    monkeypatch.setattr("hermes_cli.auth.clear_provider_auth", lambda slug: False)

    resp = server._methods["model.disconnect"](3, {"slug": "openrouter"})

    assert resp["result"]["disconnected"] is True
    remove.assert_called_once_with("OPENROUTER_API_KEY")
