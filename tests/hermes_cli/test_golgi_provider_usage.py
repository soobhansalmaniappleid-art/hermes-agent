"""Golgi fork: account limits over the dashboard API, and session cost in usage."""

from datetime import datetime, timezone
from types import SimpleNamespace

from fastapi.testclient import TestClient

from agent.account_usage import AccountUsageSnapshot, AccountUsageWindow
from hermes_cli.web_server import _SESSION_TOKEN, app

client = TestClient(app)
HEADERS = {"X-Hermes-Session-Token": _SESSION_TOKEN}


def test_provider_usage_reports_windows(monkeypatch):
    reset = datetime(2026, 10, 13, 18, 0, tzinfo=timezone.utc)
    snapshot = AccountUsageSnapshot(
        provider="openai-codex",
        source="usage_api",
        fetched_at=reset,
        plan="Pro",
        windows=(AccountUsageWindow("5h", used_percent=42.0, reset_at=reset),),
    )
    monkeypatch.setattr("agent.account_usage.fetch_account_usage", lambda provider: snapshot)
    data = client.get("/api/providers/usage?provider=openai-codex", headers=HEADERS).json()
    assert data["available"] is True
    assert data["plan"] == "Pro"
    assert data["windows"] == [
        {"label": "5h", "used_percent": 42.0, "reset_at": reset.isoformat(), "detail": None}
    ]


def test_provider_usage_without_an_account(monkeypatch):
    monkeypatch.setattr("agent.account_usage.fetch_account_usage", lambda provider: None)
    data = client.get("/api/providers/usage?provider=anthropic", headers=HEADERS).json()
    assert data == {"provider": "anthropic", "available": False, "windows": [], "details": []}


def test_session_usage_carries_its_cost():
    from tui_gateway import server

    agent = SimpleNamespace(model="m", session_total_tokens=10, session_estimated_cost_usd=0.0123)
    assert server._get_usage(agent)["cost_usd"] == 0.0123
    agent.session_estimated_cost_usd = 0
    assert "cost_usd" not in server._get_usage(agent)
