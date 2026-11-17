import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_aws.chat_models.bedrock_converse import _messages_to_bedrock

from archytas.models.providers.bedrock import BedrockProvider, STOP_REASON_MESSAGES


@pytest.mark.asyncio
@pytest.mark.parametrize("content, expected", [
    ("plain answer", "plain answer"),
    ([
        {"type": "reasoning_content", "reasoning_content": {"text": "private"}},
        {"type": "text", "text": "first"},
        {"type": "text", "text": "second"},
    ], "first\nsecond"),
])
async def test_end_turn_becomes_final_answer(content, expected):
    message = AIMessage(content=content, response_metadata={"stopReason": "end_turn"})

    async def invoke(*args, **kwargs):
        return message

    result = await BedrockProvider().ainvoke(None, invoke, [])

    assert result is message
    assert result.content == content
    assert len(result.tool_calls) == 1
    call = result.tool_calls[0]
    assert call["id"]
    assert call["name"] == "final_answer"
    assert call["args"] == {"response": expected}


@pytest.mark.asyncio
@pytest.mark.parametrize("metadata, calls", [
    ({"stopReason": "tool_use"}, []),
    ({}, []),
    ({"stopReason": "end_turn"}, [{"id": "existing", "name": "other", "args": {}}]),
])
async def test_other_responses_are_unchanged(metadata, calls):
    message = AIMessage(content="answer", response_metadata=metadata, tool_calls=calls)
    original_calls = list(message.tool_calls)

    async def invoke(*args, **kwargs):
        return message

    result = await BedrockProvider().ainvoke(None, invoke, [])

    assert result.tool_calls == original_calls


@pytest.mark.asyncio
@pytest.mark.parametrize("stop_reason", STOP_REASON_MESSAGES)
@pytest.mark.parametrize("content, expected_text", [
    ("", None),
    ("   ", None),
    ([{"type": "reasoning_content", "reasoning_content": {"text": "private"}}], None),
    ("partial response", "partial response"),
    ([{"type": "text", "text": "partial response"}], "partial response"),
])
async def test_interrupted_response_is_resumable(stop_reason, content, expected_text):
    message = AIMessage(
        content=content,
        response_metadata={"stopReason": stop_reason},
        tool_calls=[{"id": "partial", "name": "other", "args": {}}],
        invalid_tool_calls=[{"id": "invalid", "name": "broken", "args": "{"}],
    )

    async def invoke(*args, **kwargs):
        return message

    result = await BedrockProvider().ainvoke(None, invoke, [])
    call = result.tool_calls[0]
    assert len(result.tool_calls) == 1
    assert call["name"] == "final_answer"
    assert call["args"]["response"] == (expected_text or STOP_REASON_MESSAGES[stop_reason])
    assert not result.invalid_tool_calls
    messages, _ = _messages_to_bedrock([
        HumanMessage(content="request"),
        result,
        ToolMessage(content=call["args"]["response"], tool_call_id=call["id"]),
        HumanMessage(content="continue"),
    ])
    assert messages[1]["content"][-1]["toolUse"]["toolUseId"] == call["id"]
    assert messages[2]["content"][0]["toolResult"]["toolUseId"] == call["id"]


@pytest.mark.asyncio
async def test_interrupted_response_removes_tool_use_content():
    message = AIMessage(
        content=[
            {"type": "text", "text": "partial response"},
            {"type": "tool_use", "id": "partial", "name": "other", "input": {}},
        ],
        response_metadata={"stopReason": "malformed_tool_use"},
        tool_calls=[{"id": "partial", "name": "other", "args": {}}],
    )

    async def invoke(*args, **kwargs):
        return message

    result = await BedrockProvider().ainvoke(None, invoke, [])
    assert result.content == [{"type": "text", "text": "partial response"}]
    messages, _ = _messages_to_bedrock([result])
    tool_uses = [block["toolUse"] for block in messages[0]["content"] if "toolUse" in block]
    assert [call["name"] for call in tool_uses] == ["final_answer"]
