"""
Deterministic, offline tests for ReAct loop-control behavior.

These tests use a fake model so they exercise the real ``ReActAgent`` loop
without requiring a live LLM. They cover behaviors that are otherwise only
checked by the parameterized end-to-end suite, where they are sensitive to
provider-specific quirks (e.g. parallel tool calls collapsing multiple
actions into a single ReAct step).
"""
import uuid

import pytest
from langchain_core.messages import AIMessage

from archytas.react import ReActAgent, FailedTaskError
from archytas.models.base import BaseArchytasModel
from archytas.tool_utils import tool


class _FakeLCModel:
    """Minimal stand-in for the underlying LangChain chat model."""

    model = "fake-model"

    def bind_tools(self, tools):
        return self

    def get_num_tokens_from_messages(self, messages, tools=None):
        return 0


class LoopingToolModel(BaseArchytasModel):
    """
    A deterministic fake model that, on every invocation, asks to call a single
    named tool and never emits ``final_answer``. This forces the ReAct loop to
    run indefinitely so the step-limit guard can be exercised in isolation.
    """

    def __init__(self, tool_name: str, **kwargs):
        self._tool_name = tool_name
        self.invocation_count = 0
        super().__init__({"model_name": "fake-model"}, **kwargs)

    def initialize_model(self, **kwargs):
        return _FakeLCModel()

    async def get_num_tokens_from_messages(self, messages, tools=None) -> int:
        return 0

    def contextsize(self, model_name=None):
        # Large enough that summarization is never triggered during the test.
        return 1_000_000

    async def ainvoke(self, input, *, config=None, stop=None, agent_tools=None, **kwargs):
        self.invocation_count += 1
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": self._tool_name,
                    "args": {},
                    "id": uuid.uuid4().hex,
                    "type": "tool_call",
                }
            ],
        )


class FinalAnswerModel(BaseArchytasModel):
    """A fake model that immediately calls ``final_answer`` with a fixed value."""

    def __init__(self, answer: str, **kwargs):
        self._answer = answer
        super().__init__({"model_name": "fake-model"}, **kwargs)

    def initialize_model(self, **kwargs):
        return _FakeLCModel()

    async def get_num_tokens_from_messages(self, messages, tools=None) -> int:
        return 0

    def contextsize(self, model_name=None):
        return 1_000_000

    async def ainvoke(self, input, *, config=None, stop=None, agent_tools=None, **kwargs):
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "final_answer",
                    "args": {"response": self._answer},
                    "id": uuid.uuid4().hex,
                    "type": "tool_call",
                }
            ],
        )


@pytest.mark.asyncio
async def test_max_steps_exceeded_raises():
    """
    The loop must raise ``FailedTaskError`` once it takes more ReAct steps than
    ``max_react_steps`` allows. This is the deterministic counterpart to the
    live ``test_max_steps_exceeded`` e2e test, which is flaky because a model is
    free to satisfy a "call this N times" instruction with parallel tool calls
    inside a single ReAct step.
    """
    call_count = {"count": 0}

    @tool()
    def increment_counter() -> str:
        """
        Increment a counter.

        Returns:
            str: Current counter value
        """
        call_count["count"] += 1
        return str(call_count["count"])

    model = LoopingToolModel("increment_counter")
    agent = ReActAgent(
        model=model,
        tools=[increment_counter],
        max_react_steps=5,
        allow_ask_user=False,
        verbose=False,
    )

    with pytest.raises(FailedTaskError) as exc_info:
        await agent.react_async("Call increment_counter forever")

    assert "Too many steps" in str(exc_info.value)
    # The guard trips on the step *after* the limit: steps 1..5 run, step 6 raises.
    assert agent.steps == 6
    assert agent.max_react_steps == 5


@pytest.mark.asyncio
async def test_step_guard_is_synchronous_before_model_call():
    """
    The step-limit check happens before the model is invoked for the next step,
    so an agent already at its limit raises without an additional LLM round-trip.
    """
    model = LoopingToolModel("noop")
    agent = ReActAgent(
        model=model,
        tools=[],
        max_react_steps=3,
        allow_ask_user=False,
        verbose=False,
    )
    agent.steps = 3
    before = model.invocation_count

    with pytest.raises(FailedTaskError, match="Too many steps"):
        await agent.execute()

    assert model.invocation_count == before


class _ConcreteModel(BaseArchytasModel):
    """Concrete BaseArchytasModel usable for exercising shared, model-agnostic methods."""

    def initialize_model(self, **kwargs):
        return _FakeLCModel()

    async def get_num_tokens_from_messages(self, messages, tools=None) -> int:
        return 0


@pytest.mark.parametrize(
    "content",
    [
        # openai reasoning models (e.g. gpt-5): a reasoning block with summary
        # text and no top-level "text" key. This shape previously crashed the
        # thoughts e2e test's assertion with a KeyError.
        [
            {
                "id": "rs_x",
                "summary": [{"text": "planning", "type": "summary_text"}],
                "type": "reasoning",
                "content": [],
            }
        ],
        # anthropic / chat-style text block
        [{"type": "text", "text": "Let me calculate that."}],
        # gemini thinking block
        [{"type": "thinking", "thinking": "I should add the numbers."}],
    ],
)
def test_process_result_handles_reasoning_content_with_tool_calls(content):
    """
    ``process_result`` must not raise on any provider's list-shaped content when
    tool calls are present, even when no top-level ``text`` block exists (the
    openai reasoning-model case).
    """
    model = _ConcreteModel({"model_name": "fake-model"})
    msg = AIMessage(
        content=content,
        tool_calls=[
            {
                "name": "calculate",
                "args": {"a": 5, "b": 3, "operation": "add"},
                "id": "call_1",
                "type": "tool_call",
            }
        ],
    )

    result = model.process_result(msg)

    assert isinstance(result.text, str)
    assert len(result.text) > 0
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0]["name"] == "calculate"


@pytest.mark.asyncio
async def test_loop_completes_without_hitting_step_limit():
    """A model that finishes promptly must not trip the step-limit guard."""
    model = FinalAnswerModel("the answer is 42")
    agent = ReActAgent(
        model=model,
        tools=[],
        max_react_steps=5,
        allow_ask_user=False,
        verbose=False,
    )

    result = await agent.react_async("What is the answer?")
    assert result == "the answer is 42"
    assert agent.steps <= 5
