import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

from archytas.chat_history import ChatHistory, MessageRecord
from archytas.models.base import BaseArchytasModel
from archytas.models.bedrock import BedrockModel


class OtherModel(BaseArchytasModel):
    def initialize_model(self, **kwargs):
        pass


def make_model(cls, name):
    model = object.__new__(cls)
    model._model = SimpleNamespace(model=name, profile={})
    model.config = SimpleNamespace(model_name=name)
    model.tail_injection_count = 0
    return model


def test_model_specific_content_is_filtered_without_mutating_history(monkeypatch):
    monkeypatch.setenv("ARCHYTAS_DISABLE_PROMPT_CACHE", "1")
    origin = make_model(BedrockModel, "model-a")
    history = ChatHistory(model=origin)
    message = AIMessage(content=[
        {"type": "reasoning_content", "reasoning": {"signature": bytearray(b"secret")}},
        {"type": "text", "text": "answer"},
    ])
    record = history.add_message(message)

    assert origin._prepare_messages([record.message])[-1].content == message.content
    other_name = make_model(BedrockModel, "model-b")
    other_class = make_model(OtherModel, "model-a")
    for model in (other_name, other_class):
        assert model._prepare_messages([record.message])[-1].content == [
            {"type": "text", "text": "answer"},
        ]
    assert record.message.content == message.content
    assert origin._prepare_messages([record.message])[-1].content == message.content

    for payload in (record.to_dict(), history.to_dict()):
        encoded = json.dumps(payload)
        assert "reasoning_content" not in encoded
        assert "secret" not in encoded
        assert "answer" in encoded
    restored = MessageRecord.from_dict(record.to_dict())
    assert restored.message.content == [{"type": "text", "text": "answer"}]
    assert record.message.content == message.content


@pytest.mark.asyncio
async def test_token_estimate_keeps_byte_safe_content(monkeypatch):
    monkeypatch.setenv("ARCHYTAS_DISABLE_PROMPT_CACHE", "1")
    model = make_model(BedrockModel, "model-a")
    seen = []

    async def count(messages, tools=None):
        seen.extend(messages)
        return 1

    model.get_num_tokens_from_messages = count
    message = model.mark_model_specific_content(AIMessage(content=[
        {"type": "reasoning_content", "signature": bytearray(b"secret\xff")},
    ]))
    assert await model.token_estimate([message]) == 1
    assert "c2VjcmV0/w==" in seen[-1].content[0]["text"]
