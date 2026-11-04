"""Tests for profile-based temperature handling across model providers."""
import asyncio
import inspect

import httpx
import pytest
from anthropic import BadRequestError
from langchain_core.messages import AIMessage, HumanMessage

from archytas.agent import Agent
from archytas.models.anthropic import AnthropicModel
from archytas.models.base import BaseArchytasModel
from archytas.models.gemini import GeminiModel
from archytas.models.ollama import OllamaModel
from archytas.models.openai import OpenAIModel


class _FakeChatModel:
    def __init__(self, profile):
        self.profile = profile
        self.calls = []
        self.openai_api_key = "dummy"

    async def ainvoke(self, messages, config=None, stop=None, **kwargs):
        self.calls.append(kwargs)
        return AIMessage(content="ok")


class _TestModel(BaseArchytasModel):
    def initialize_model(self, **kwargs):
        return _FakeChatModel(None)


@pytest.mark.parametrize("profile,temperature,expected", [
    ({"temperature": True}, 0.0, 0.0),
    ({"temperature": True}, None, None),
    ({"temperature": False}, 0.5, None),
    ({}, 0.5, None),
    (None, 0.5, None),
])
def test_temperature_requires_explicit_value_and_profile_support(profile, temperature, expected):
    model = _TestModel({"model_name": "test"})
    fake = _FakeChatModel(profile)
    model._model = fake
    asyncio.run(model.ainvoke([HumanMessage(content="hi")], temperature=temperature))
    if expected is None:
        assert "temperature" not in fake.calls[0]
    else:
        assert fake.calls[0]["temperature"] == expected


def test_agent_temperature_defaults_to_none():
    assert inspect.signature(Agent).parameters["temperature"].default is None


@pytest.mark.parametrize("model_class,config", [
    (OpenAIModel, {"api_key": "sk-dummy", "model_name": "gpt-4o"}),
    (AnthropicModel, {"api_key": "sk-dummy", "model_name": "claude-sonnet-5"}),
])
def test_provider_invocations_use_profile(model_class, config, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-dummy")
    model = model_class(config)
    fake = _FakeChatModel({"temperature": False})
    model._model = fake
    asyncio.run(model.ainvoke([HumanMessage(content="hi")], temperature=0.5))
    assert "temperature" not in fake.calls[0]


@pytest.mark.parametrize("model_class,key", [
    (GeminiModel, "generation_config"),
    (OllamaModel, "options"),
])
def test_provider_specific_temperature_format(model_class, key):
    model = object.__new__(model_class)
    model._model = _FakeChatModel({"temperature": True})
    result = model._prepare_invoke_kwargs({"temperature": 0.5, key: {"other": 1}})
    assert result == {key: {"temperature": 0.5, "other": 1}}
    model._model.profile = None
    assert model._prepare_invoke_kwargs({"temperature": 0.5}) == {}


def _bad_request(message):
    response = httpx.Response(400, request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"))
    return BadRequestError(message, response=response, body={"error": {"message": message}})


class _RejectingChatModel(_FakeChatModel):
    def __init__(self, error, profile=None):
        super().__init__(profile if profile is not None else {"temperature": True})
        self.error = error

    async def ainvoke(self, messages, config=None, stop=None, **kwargs):
        self.calls.append(kwargs)
        if self.error and "temperature" in kwargs:
            raise self.error
        return AIMessage(content="ok")


def test_anthropic_retries_rejected_temperature_and_remembers(monkeypatch):
    model = AnthropicModel({"api_key": "dummy", "model_name": "claude-sonnet-4-5-20250929"})
    fake = _RejectingChatModel(_bad_request("temperature: Extra inputs are not permitted"))
    model._model = fake
    seen_tail_counts = []
    monkeypatch.setattr(model, "_preprocess_messages", lambda messages: seen_tail_counts.append(model.tail_injection_count) or messages)
    model.tail_injection_count = 2
    result = asyncio.run(model.ainvoke([HumanMessage(content="hi")], temperature=0.0))
    assert result.content == "ok"
    assert fake.calls == [{"temperature": 0.0}, {}]
    assert seen_tail_counts == [2, 2]
    assert model.tail_injection_count == 0
    fake.calls.clear()
    asyncio.run(model.ainvoke([HumanMessage(content="hi")], temperature=0.0))
    assert fake.calls == [{}]


@pytest.mark.parametrize("message", [
    "max_tokens: Extra inputs are not permitted",
    "temperature must be between 0 and 1",
])
def test_anthropic_does_not_retry_other_errors(message):
    model = AnthropicModel({"api_key": "dummy", "model_name": "claude-sonnet-4-5-20250929"})
    fake = _RejectingChatModel(_bad_request(message))
    model._model = fake
    with pytest.raises(BadRequestError):
        asyncio.run(model.ainvoke([HumanMessage(content="hi")], temperature=0.0))
    assert len(fake.calls) == 1
    assert model._temperature_unsupported is False
