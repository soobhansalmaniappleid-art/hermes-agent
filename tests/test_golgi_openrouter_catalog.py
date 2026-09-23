"""Golgi fork: OpenRouter's full tool-calling catalog behind an opt-in flag.

Upstream intersects the live OpenRouter catalog with a curated allowlist, so
Golgi's Models page — which mirrors openrouter.ai — saw ~39 of ~450 models.
The flag turns the allowlist into a ranking instead of a filter.
"""

import json
from contextlib import contextmanager

import pytest

from hermes_cli import models as models_mod


def _live(model_id, *, tools=True, free=False):
    item = {"id": model_id, "pricing": {"prompt": "0" if free else "0.000003",
                                        "completion": "0" if free else "0.000015"}}
    item["supported_parameters"] = ["tools"] if tools else ["temperature"]
    return item


@pytest.fixture
def catalog(monkeypatch):
    """Serve a fake OpenRouter /v1/models and a fixed curated list."""
    live = [
        _live("anthropic/claude-opus-5"),
        _live("openai/gpt-5.5"),
        _live("zzz-lab/new-model"),
        _live("free-lab/tiny", free=True),
        _live("image-lab/diffusion", tools=False),
    ]

    class _Response:
        def read(self):
            return json.dumps({"data": live}).encode()

    @contextmanager
    def fake_open(req, timeout=8.0):
        yield _Response()

    monkeypatch.setattr(models_mod, "_urlopen_model_catalog_request", fake_open)
    monkeypatch.setattr(models_mod, "OPENROUTER_MODELS",
                        [("anthropic/claude-opus-5", ""), ("openai/gpt-5.5", "")])
    monkeypatch.setattr(models_mod, "_openrouter_catalog_cache", None)
    monkeypatch.setattr(models_mod, "_openrouter_catalog_full", False)
    monkeypatch.setattr(models_mod, "get_preferred_silent_default_model", lambda _slug: "")
    # Ignore the remotely-hosted manifest so the fake curated list is the one used.
    monkeypatch.setattr("hermes_cli.model_catalog.get_curated_openrouter_models",
                        lambda: None, raising=False)
    monkeypatch.delenv("HERMES_OPENROUTER_FULL_CATALOG", raising=False)
    return live


def test_upstream_default_keeps_only_the_curated_intersection(catalog, monkeypatch) -> None:
    monkeypatch.setattr(models_mod, "openrouter_full_catalog_enabled", lambda: False)
    ids = [mid for mid, _ in models_mod.fetch_openrouter_models(force_refresh=True)]
    assert ids == ["anthropic/claude-opus-5", "openai/gpt-5.5"]


def test_the_flag_adds_every_live_model_that_supports_tools(catalog, monkeypatch) -> None:
    monkeypatch.setenv("HERMES_OPENROUTER_FULL_CATALOG", "1")
    entries = models_mod.fetch_openrouter_models(force_refresh=True)
    ids = [mid for mid, _ in entries]

    # curated first (they are what pickers feature), then the rest, sorted
    assert ids[:2] == ["anthropic/claude-opus-5", "openai/gpt-5.5"]
    assert ids[2:] == ["free-lab/tiny", "zzz-lab/new-model"]
    # a model that cannot call tools is still hidden: the agent loop needs them
    assert "image-lab/diffusion" not in ids
    assert dict(entries)["free-lab/tiny"] == "free"


def test_the_cache_does_not_leak_across_modes(catalog, monkeypatch) -> None:
    monkeypatch.setenv("HERMES_OPENROUTER_FULL_CATALOG", "1")
    assert len(models_mod.fetch_openrouter_models()) == 4
    monkeypatch.setenv("HERMES_OPENROUTER_FULL_CATALOG", "0")
    assert len(models_mod.fetch_openrouter_models()) == 2


def test_the_flag_reads_the_hermes_config_when_no_env_var_is_set(monkeypatch) -> None:
    monkeypatch.delenv("HERMES_OPENROUTER_FULL_CATALOG", raising=False)
    monkeypatch.setattr("hermes_cli.config.load_config",
                        lambda: {"model_catalog": {"openrouter_full": True}})
    assert models_mod.openrouter_full_catalog_enabled() is True

    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {})
    assert models_mod.openrouter_full_catalog_enabled() is False

    # an explicit env var wins over the config either way
    monkeypatch.setenv("HERMES_OPENROUTER_FULL_CATALOG", "0")
    monkeypatch.setattr("hermes_cli.config.load_config",
                        lambda: {"model_catalog": {"openrouter_full": True}})
    assert models_mod.openrouter_full_catalog_enabled() is False
