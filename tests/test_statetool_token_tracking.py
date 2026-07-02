"""
Tests for issue #86: statetool outputs are counted sporadically in token
estimates because their tail injections are ephemeral. ChatHistory now tracks
the last-known size of each statetool's output separately and folds it into
the token overhead, updating on fire and retaining (not zeroing) otherwise.

All tests run offline against FakeModel (tokens ~= chars // 4).
"""
import pytest

from archytas.agent import Agent
from archytas.chat_history import ChatHistory
from archytas.tool_utils import statetool, AgentRef

from .fake_model import FakeModel


def make_gated_statetool(flag: dict, output: dict, name: str = "gated_state"):
    """A statetool that fires when flag['fire'] and returns output['content']."""

    def gate_condition(agent: AgentRef) -> bool:
        return bool(flag.get("fire"))

    @statetool(condition=gate_condition, name=name)
    def gated_state() -> str:
        """
        ** INTERNAL ** Return canned state content for tests.

        Returns:
            str: The canned content.
        """
        return output["content"]

    return gated_state


class TestChatHistoryStatetoolAccounting:

    def test_statetool_estimate_sums_entries(self):
        history = ChatHistory(model=FakeModel())
        assert history.statetool_token_estimate == 0

        history.statetool_token_estimates["tool_a"] = 100
        history.statetool_token_estimates["tool_b"] = 250
        assert history.statetool_token_estimate == 350

    def test_token_overhead_includes_statetool_estimate(self):
        history = ChatHistory(model=FakeModel())
        history.base_tokens = 50
        history.tool_token_estimate = 200
        baseline = history.token_overhead

        history.statetool_token_estimates["tool_a"] = 100
        assert history.token_overhead == baseline + 100

    def test_serde_roundtrip_preserves_estimates(self):
        history = ChatHistory(model=FakeModel())
        history.statetool_token_estimates = {"tool_a": 100, "tool_b": 250}

        data = history.to_dict()
        restored = ChatHistory.from_dict(data, model=FakeModel())

        assert restored.statetool_token_estimates == {"tool_a": 100, "tool_b": 250}


class TestStatetoolEstimateTracking:

    @pytest.mark.asyncio
    async def test_estimate_recorded_when_fired(self):
        flag = {"fire": True}
        output = {"content": "s" * 400}  # ~100 tokens under FakeModel
        tool_fn = make_gated_statetool(flag, output)

        agent = Agent(model=FakeModel(), spinner=None)
        agent.statetools = {"gated_state": tool_fn}

        await agent.build_tail_injections()

        estimate = agent.chat_history.statetool_token_estimates.get("gated_state")
        assert estimate == 100

    @pytest.mark.asyncio
    async def test_estimate_retained_when_not_fired(self):
        """A statetool that doesn't fire keeps its last-known size instead of
        dropping back to zero — the 'jumping around' from the issue."""
        flag = {"fire": True}
        output = {"content": "s" * 400}
        tool_fn = make_gated_statetool(flag, output)

        agent = Agent(model=FakeModel(), spinner=None)
        agent.statetools = {"gated_state": tool_fn}

        await agent.build_tail_injections()
        overhead_after_fire = agent.chat_history.token_overhead
        assert agent.chat_history.statetool_token_estimates["gated_state"] == 100

        flag["fire"] = False
        tail = await agent.build_tail_injections()

        assert tail == []
        assert agent.chat_history.statetool_token_estimates["gated_state"] == 100
        assert agent.chat_history.token_overhead == overhead_after_fire

    @pytest.mark.asyncio
    async def test_estimate_updated_on_refire_with_new_size(self):
        flag = {"fire": True}
        output = {"content": "s" * 400}
        tool_fn = make_gated_statetool(flag, output)

        agent = Agent(model=FakeModel(), spinner=None)
        agent.statetools = {"gated_state": tool_fn}

        await agent.build_tail_injections()
        assert agent.chat_history.statetool_token_estimates["gated_state"] == 100

        output["content"] = "s" * 1200  # ~300 tokens
        await agent.build_tail_injections()
        assert agent.chat_history.statetool_token_estimates["gated_state"] == 300

    @pytest.mark.asyncio
    async def test_multiple_statetools_tracked_independently(self):
        flag_a = {"fire": True}
        flag_b = {"fire": True}
        tool_a = make_gated_statetool(flag_a, {"content": "a" * 400}, name="state_a")
        tool_b = make_gated_statetool(flag_b, {"content": "b" * 800}, name="state_b")

        agent = Agent(model=FakeModel(), spinner=None)
        agent.statetools = {"state_a": tool_a, "state_b": tool_b}

        await agent.build_tail_injections()
        assert agent.chat_history.statetool_token_estimates == {
            "state_a": 100, "state_b": 200,
        }

        # Only state_b fires next turn; state_a's size is retained.
        flag_a["fire"] = False
        await agent.build_tail_injections()
        assert agent.chat_history.statetool_token_estimates == {
            "state_a": 100, "state_b": 200,
        }
        assert agent.chat_history.statetool_token_estimate == 300
