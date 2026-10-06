"""Golgi fork: a persistent Browserbase context keeps sign-ins between cloud sessions."""

from plugins.browser.browserbase import provider as bb


class _Response:
    ok = True
    status_code = 200

    def json(self):
        return {"id": "s1", "connectUrl": "wss://connect.example/s1"}


def _create(monkeypatch, env):
    sent = {}

    def post(url, json=None, headers=None, timeout=None):
        sent.update(json or {})
        return _Response()

    for key in ("BROWSERBASE_CONTEXT_ID", "BROWSERBASE_ADVANCED_STEALTH"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("BROWSERBASE_API_KEY", "bb_live_x")
    monkeypatch.setenv("BROWSERBASE_PROJECT_ID", "proj-1")
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(bb.requests, "post", post)
    bb.BrowserbaseBrowserProvider().create_session("task-1")
    return sent


def test_sessions_open_with_the_owner_context(monkeypatch):
    sent = _create(monkeypatch, {"BROWSERBASE_CONTEXT_ID": "ctx-9", "BROWSERBASE_ADVANCED_STEALTH": "true"})
    assert sent["browserSettings"] == {"advancedStealth": True, "context": {"id": "ctx-9", "persist": True}}


def test_no_context_no_browser_settings(monkeypatch):
    assert "browserSettings" not in _create(monkeypatch, {})
