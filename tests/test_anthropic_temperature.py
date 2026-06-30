"""Regression tests for Anthropic temperature handling.

The newest Claude models (e.g. Sonnet 5) reject an explicit ``temperature``.
``Agent.execute()`` always passes one, and the model name gives no reliable
signal as to which models accept it, so ``AnthropicModel`` must learn from the
first rejection, retry without ``temperature``, and remember it thereafter --
without swallowing unrelated ``BadRequestError``s.
"""
import asyncio

import httpx
import pytest
from anthropic import BadRequestError
from langchain_core.messages import AIMessage, HumanMessage

from archytas.models.anthropic import AnthropicModel


def _bad_request(message: str) -> BadRequestError:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(400, request=request)
    return BadRequestError(message, response=response, body={"error": {"message": message}})


class _FakeChatAnthropic:
    """Stand-in for the langchain client. Records the kwargs of each call and
    raises a configurable error on the first call (optionally only when a
    temperature is present)."""

    def __init__(self, error: Exception | None = None, only_with_temperature: bool = True):
        self.calls: list[dict] = []
        self._error = error
        self._only_with_temperature = only_with_temperature

    async def ainvoke(self, messages, config=None, stop=None, **kwargs):
        self.calls.append(dict(kwargs))
        if self._error is not None and (not self._only_with_temperature or "temperature" in kwargs):
            raise self._error
        return AIMessage(content="ok")


@pytest.fixture(autouse=True)
def _disable_prompt_cache(monkeypatch):
    # Keep _preprocess_messages off the cache-control path so a minimal message
    # list is enough to exercise ainvoke.
    monkeypatch.setenv("ARCHYTAS_DISABLE_PROMPT_CACHE", "1")


def _model() -> AnthropicModel:
    return AnthropicModel({"api_key": "sk-ant-dummy", "model_name": "claude-sonnet-5"})


def test_retries_without_temperature_then_remembers():
    model = _model()
    fake = _FakeChatAnthropic(error=_bad_request("temperature is deprecated for this model."))
    model._model = fake

    result = asyncio.run(model.ainvoke([HumanMessage(content="hi")], temperature=0.0))

    assert result.content == "ok"
    assert model._temperature_unsupported is True
    # First attempt carried temperature; the retry dropped it.
    assert "temperature" in fake.calls[0]
    assert "temperature" not in fake.calls[1]

    # A subsequent call strips temperature up front -- no failed attempt.
    fake.calls.clear()
    asyncio.run(model.ainvoke([HumanMessage(content="hi")], temperature=0.0))
    assert len(fake.calls) == 1
    assert "temperature" not in fake.calls[0]


def test_unrelated_bad_request_is_not_retried():
    model = _model()
    fake = _FakeChatAnthropic(error=_bad_request("some other invalid parameter"), only_with_temperature=False)
    model._model = fake

    with pytest.raises(BadRequestError):
        asyncio.run(model.ainvoke([HumanMessage(content="hi")], temperature=0.0))

    assert model._temperature_unsupported is False
    assert len(fake.calls) == 1  # no retry
