"""Profile-backed context limits and reasoning configuration."""
from types import SimpleNamespace

import pytest

from archytas.models.anthropic import AnthropicModel
from archytas.models.bedrock import BedrockModel
from archytas.models.gemini import GeminiModel
from archytas.models.openai import OpenAIModel


@pytest.mark.parametrize("model_class,name,fallback", [
    (OpenAIModel, "gpt-5-custom", 400_000),
    (GeminiModel, "gemini-2.5-flash-custom", 1_048_576),
    (AnthropicModel, "claude-sonnet-custom", 200_000),
    (BedrockModel, "us.anthropic.claude-custom", 50_000),
])
def test_context_size_profile_and_fallback(model_class, name, fallback):
    model = object.__new__(model_class)
    model.config = SimpleNamespace(model_name=name)
    model._model = SimpleNamespace(profile={"max_input_tokens": 123_456})
    assert model.contextsize() == 123_456
    assert model.contextsize(name) == 123_456
    # A profile belongs to the active model, not an arbitrary requested model.
    assert model.contextsize(name + "-other") == fallback
    model.contextsize.cache_clear()
    model._model.profile = {}
    assert model.contextsize() == fallback


@pytest.mark.parametrize("model_class,model_name", [
    (OpenAIModel, "gpt-5"),
    (GeminiModel, "gemini-3-flash"),
])
def test_profile_reasoning_levels_override_family_fallback(model_class, model_name):
    model = object.__new__(model_class)
    assert model._get_supported_reasoning_efforts(model_name, {"reasoning_effort_levels": ["low"]}) == ["low"]
    assert model._get_supported_reasoning_efforts(model_name, {"reasoning_effort_levels": []}) is None
    assert model._get_supported_reasoning_efforts(model_name, {"reasoning_output": False}) is None
    assert "medium" in model._get_supported_reasoning_efforts(model_name, {})


@pytest.mark.parametrize("model_class,module,model_name,reasoning_key", [
    (OpenAIModel, "archytas.models.openai", "new-model", "reasoning"),
    (GeminiModel, "archytas.models.gemini", "new-model", "thinking_level"),
])
def test_initialization_uses_profile_to_configure_reasoning(monkeypatch, model_class, module, model_name, reasoning_key):
    calls = []

    class FakeChatModel:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            self.profile = {"reasoning_effort_levels": ["low"]}

    target = "ChatOpenAI" if model_class is OpenAIModel else "ChatGoogleGenerativeAI"
    monkeypatch.setattr(f"{module}.{target}", FakeChatModel)
    monkeypatch.setattr(f"{module}.REASONING_EFFORT_REQUEST", None)
    model = object.__new__(model_class)
    model.config = SimpleNamespace(model_name=model_name, model_extra=None, api_key="dummy")
    model.api_key = "dummy"
    model.initialize_model()
    assert len(calls) == 2
    if reasoning_key == "reasoning":
        assert calls[-1][reasoning_key]["effort"] == "low"
    else:
        assert calls[-1][reasoning_key] == "low"
        assert calls[-1]["include_thoughts"] is True


def test_gemini_budget_fallback():
    model = object.__new__(GeminiModel)
    assert model._get_thinking_config("gemini-2.5-flash", {}) == {
        "include_thoughts": True,
        "thinking_budget_token_limit": -1,
    }
