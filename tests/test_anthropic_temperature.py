"""Tests for profile-based temperature handling across model providers."""
import asyncio
import inspect

import pytest
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
def test_provider_invocations_use_profile(model_class, config):
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
