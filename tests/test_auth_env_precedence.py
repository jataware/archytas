"""Regression tests for API-key precedence in OpenAI/Azure model auth.

These guard the subtle bug where an explicitly-configured key was shadowed by
a stale value left in ``os.environ`` (written there by an earlier model
instance via ``set_env_auth``'s ``setdefault``), so a freshly-entered key only
took effect after a process restart.

No network calls: constructing the model is enough to resolve the effective key.
"""
import pytest

from archytas.models.openai import OpenAIModel


def _client_key(model) -> str:
    secret = model.model.openai_api_key
    return secret.get_secret_value() if hasattr(secret, "get_secret_value") else str(secret)


def test_explicit_key_overrides_stale_env(monkeypatch):
    # Simulate a stale/bad key left in the environment by an earlier build.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-STALE-must-not-win")
    model = OpenAIModel({"api_key": "sk-explicit-config-key", "model_name": "gpt-4o-mini"})
    assert _client_key(model) == "sk-explicit-config-key"


def test_defers_to_env_when_no_key_configured(monkeypatch):
    # With no key configured, a user-provided environment variable should win.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-user-environment")
    model = OpenAIModel({"model_name": "gpt-4o-mini"})
    assert _client_key(model) == "sk-from-user-environment"
