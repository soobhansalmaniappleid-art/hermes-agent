"""Golgi fork: each of the owner's agents keeps its own cloud browser profile."""

import json

import pytest

from plugins.browser.browserbase import provider as bb


class _Response:
    ok = True
    status_code = 200
    text = ""

    def json(self):
        return {"id": "bb-1", "connectUrl": "wss://connect"}


@pytest.fixture
def sent(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("BROWSERBASE_API_KEY", "k")
    monkeypatch.setenv("BROWSERBASE_PROJECT_ID", "p")
    monkeypatch.setenv("BROWSERBASE_CONTEXT_ID", "shared")
    calls = []
    monkeypatch.setattr(bb.requests, "post", lambda url, headers, json, timeout: calls.append(json) or _Response())
    (tmp_path / "golgi").mkdir()
    (tmp_path / "golgi" / "browser_contexts.json").write_text(json.dumps({"20261008_abc": "ctx-research"}))
    return calls


def test_a_mapped_session_uses_its_agents_profile(sent):
    bb.BrowserbaseBrowserProvider().create_session("20261008_abc")
    assert sent[-1]["browserSettings"]["context"] == {"id": "ctx-research", "persist": True}


def test_other_sessions_keep_the_shared_profile(sent):
    bb.BrowserbaseBrowserProvider().create_session("someone-else")
    assert sent[-1]["browserSettings"]["context"]["id"] == "shared"
