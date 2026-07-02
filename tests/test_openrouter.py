"""Tests for the OpenRouter provider (OpenAI-compatible subclass).

No network calls: construction is enough to verify endpoint wiring, key
precedence, and env hygiene. Context-size lookup is exercised against a
patched model listing.
"""
import pytest

import archytas.models.openrouter as openrouter_mod
from archytas.exceptions import AuthenticationError
from archytas.models.openai import OpenAIModel
from archytas.models.openrouter import OPENROUTER_BASE_URL, DEFAULT_CONTEXT_SIZE, OpenRouterModel


def _make(model_name="openai/gpt-4o-mini", **config):
    return OpenRouterModel({"model_name": model_name, **config})


class TestAuth:
    def test_explicit_key_wins_over_env(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-env-key")
        model = _make(api_key="sk-or-config-key")
        assert model.config.api_key == "sk-or-config-key"

    def test_falls_back_to_env(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-env-key")
        model = _make()
        assert model.config.api_key == "sk-or-env-key"

    def test_no_key_raises(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with pytest.raises(AuthenticationError):
            _make()

    def test_does_not_touch_openai_env(self, monkeypatch):
        # The OpenRouter key must not leak into OPENAI_API_KEY (which may
        # belong to a different provider entry in the same process).
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        _make(api_key="sk-or-config-key")
        import os
        assert "OPENAI_API_KEY" not in os.environ


class TestClientWiring:
    def test_points_at_openrouter(self):
        model = _make(api_key="sk-or-key")
        client = model._model
        assert str(client.openai_api_base).startswith(OPENROUTER_BASE_URL)
        assert client.model_name == "openai/gpt-4o-mini"
        assert client.openai_api_key.get_secret_value() == "sk-or-key"


class TestContextSize:
    def test_uses_live_listing(self, monkeypatch):
        monkeypatch.setattr(
            openrouter_mod, "_openrouter_context_lengths",
            lambda: {"anthropic/claude-3.5-sonnet": 200_000},
        )
        model = _make(model_name="anthropic/claude-3.5-sonnet", api_key="sk-or-key")
        assert model.contextsize("anthropic/claude-3.5-sonnet") == 200_000

    def test_falls_back_to_default(self, monkeypatch):
        monkeypatch.setattr(openrouter_mod, "_openrouter_context_lengths", lambda: {})
        model = _make(api_key="sk-or-key")
        assert model.contextsize("someone/unknown-model") == DEFAULT_CONTEXT_SIZE


class TestOpenAIBaseUrlPassthrough:
    def test_base_url_forwarded_from_config(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-dummy")
        model = OpenAIModel({
            "model_name": "gpt-4o-mini",
            "api_key": "sk-dummy",
            "base_url": "http://localhost:4000/v1",
        })
        assert str(model._model.openai_api_base).startswith("http://localhost:4000/v1")

    def test_no_base_url_uses_default(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-dummy")
        model = OpenAIModel({"model_name": "gpt-4o-mini", "api_key": "sk-dummy"})
        assert model._model.openai_api_base in (None, "")
