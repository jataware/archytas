"""
Tests for the @statetool mechanism (plan §4.2).
"""
import pytest

from archytas.agent import STATE_INJECTION_PLACEHOLDER_TEXT
from archytas.tool_utils import statetool, AgentRef
from archytas.react import ReActAgent


def fabricated_injection_present(sent: list, tool_name: str) -> bool:
    """
    True iff the framework-fabricated AI→Tool injection for `tool_name` appears
    in the outgoing list. Uses the placeholder AIMessage text as a signature so
    LLM-initiated calls of the same statetool are not counted.
    """
    from langchain_core.messages import AIMessage
    for m in sent:
        if not isinstance(m, AIMessage):
            continue
        if m.content != STATE_INJECTION_PLACEHOLDER_TEXT:
            continue
        for tc in (m.tool_calls or []):
            if tc.get("name") == tool_name:
                return True
    return False


def fabricated_tool_response_present(sent: list, content_marker: str) -> bool:
    """
    True iff a ToolMessage paired with a framework-fabricated AIMessage carries
    the given marker in its content. Uses the fabricated AIMessage's tool_call
    ids to identify which ToolMessages are ours.
    """
    from langchain_core.messages import AIMessage, ToolMessage
    fabricated_ids: set[str] = set()
    for m in sent:
        if isinstance(m, AIMessage) and m.content == STATE_INJECTION_PLACEHOLDER_TEXT:
            for tc in (m.tool_calls or []):
                call_id = tc.get("id")
                if call_id:
                    fabricated_ids.add(call_id)
    if not fabricated_ids:
        return False
    for m in sent:
        if isinstance(m, ToolMessage) and m.tool_call_id in fabricated_ids:
            if content_marker in (m.content or ""):
                return True
    return False


# ---- Fixtures / helpers -----------------------------------------------------

def make_always_true_statetool(content: str, name: str = "always_on"):
    """Build a statetool whose condition always fires."""

    def always_on_condition(agent: AgentRef) -> bool:
        return True

    @statetool(condition=always_on_condition, name=name)
    def always_on_tool() -> str:
        """
        ** INTERNAL ** Return canned state content for tests.

        Returns:
            str: The canned content.
        """
        return content

    return always_on_tool


def make_gated_statetool(flag: dict, key: str = "fire", content: str = "gated-state"):
    """Build a statetool whose condition reads `flag[key]` at eval time."""

    def gate_condition(agent: AgentRef) -> bool:
        return bool(flag.get(key))

    @statetool(condition=gate_condition, name="gated_tool")
    def gated_tool() -> str:
        """
        ** INTERNAL ** Return canned gated state content for tests.

        Returns:
            str: The canned content.
        """
        return content

    return gated_tool


# ---- Decorator / registration -----------------------------------------------

class TestStateToolDecorator:

    def test_decorator_marks_function(self):
        """@statetool sets _is_tool, _is_statetool, and _statetool_condition."""
        cond = lambda agent: True
        @statetool(condition=cond)
        def my_tool() -> str:
            """
            ** INTERNAL ** Dummy.

            Returns:
                str: Value.
            """
            return "x"

        assert getattr(my_tool, "_is_tool", False) is True
        assert getattr(my_tool, "_is_statetool", False) is True
        assert getattr(my_tool, "_statetool_condition", None) is cond
        assert hasattr(my_tool, "run")

    def test_decorator_rejects_missing_condition(self):
        """@statetool() without a condition is a usage error."""
        with pytest.raises(TypeError):
            @statetool()  # type: ignore[call-arg]
            def bad() -> str:
                """
                Dummy.

                Returns:
                    str: Value.
                """
                return "x"


# ---- Registration through ReActAgent ---------------------------------------

class TestStateToolRegistration:

    @pytest.mark.asyncio
    async def test_statetools_populated_on_react_agent(self, model_fixture):
        """ReActAgent.statetools is populated from tools at init time."""
        tool_fn = make_always_true_statetool("reg-test-content", name="reg_test")
        agent = ReActAgent(
            model=model_fixture,
            tools=[tool_fn],
            temperature=0.0,
            verbose=False,
            allow_ask_user=False,
        )

        assert "reg_test" in agent.statetools
        assert "reg_test" in agent.tools  # also in the normal tool registry

    @pytest.mark.asyncio
    async def test_statetool_description_has_internal_marker(self, model_fixture):
        """Statetool descriptions in the bound schema carry the INTERNAL marker."""
        tool_fn = make_always_true_statetool("marker-test", name="marker_tool")
        agent = ReActAgent(
            model=model_fixture,
            tools=[tool_fn],
            temperature=0.0,
            verbose=False,
            allow_ask_user=False,
        )

        # The lc_tools list is what was bound via set_tools.
        lc_tools = agent.model.lc_tools or []
        matches = [t for t in lc_tools if t.name == "marker_tool"]
        assert len(matches) == 1, "statetool should be present in bound tool schema"
        assert matches[0].description.startswith("** INTERNAL **")


# ---- Execution-time injection behavior -------------------------------------

class TestStateToolInjection:

    @pytest.mark.asyncio
    async def test_always_on_statetool_injected_in_outgoing(self, model_fixture):
        """
        A statetool whose condition is True produces a framework-fabricated
        AI→Tool pair at the tail of `last_sent_messages`, carrying the tool's
        output.
        """
        marker = "STATE_MARKER_ALPHA_8219"
        tool_fn = make_always_true_statetool(marker, name="alpha_state")

        agent = ReActAgent(
            model=model_fixture,
            tools=[tool_fn],
            temperature=0.0,
            verbose=False,
            allow_ask_user=False,
        )

        await agent.react_async("Say hello.")
        sent = agent.chat_history.last_sent_messages or []

        assert fabricated_injection_present(sent, "alpha_state"), (
            "framework-fabricated AIMessage for alpha_state must be in outgoing list"
        )
        assert fabricated_tool_response_present(sent, marker), (
            "fabricated ToolMessage with marker content must be in outgoing list"
        )

    @pytest.mark.asyncio
    async def test_false_condition_suppresses_injection(self, model_fixture):
        """A statetool whose condition returns False is NOT auto-injected."""
        flag = {"fire": False}
        tool_fn = make_gated_statetool(flag, content="should-not-appear")

        agent = ReActAgent(
            model=model_fixture,
            tools=[tool_fn],
            temperature=0.0,
            verbose=False,
            allow_ask_user=False,
        )

        await agent.react_async("Say hello.")
        sent = agent.chat_history.last_sent_messages or []

        # The LLM may still *choose* to call the tool itself — that's expected
        # given the tool is in the schema. What we verify is that the
        # framework did not auto-inject.
        assert not fabricated_injection_present(sent, "gated_tool"), (
            "framework must not auto-inject when condition is False"
        )

    @pytest.mark.asyncio
    async def test_statetool_output_not_persisted(self, model_fixture):
        """
        The framework-fabricated state pair appears only in the transient
        outgoing list — specifically, the placeholder AIMessage is never
        written to raw_records. (The LLM may still persist its own responses
        that quote the state content; that's unrelated to our injection.)
        """
        marker = "STATE_EPHEMERAL_MARKER_ZW12"
        tool_fn = make_always_true_statetool(marker, name="ephemeral_state")

        agent = ReActAgent(
            model=model_fixture,
            tools=[tool_fn],
            temperature=0.0,
            verbose=False,
            allow_ask_user=False,
        )

        await agent.react_async("Say hello.")

        for record in agent.chat_history.raw_records:
            content = getattr(record.message, "content", "") or ""
            if isinstance(content, str):
                assert content != STATE_INJECTION_PLACEHOLDER_TEXT, (
                    f"fabricated placeholder AIMessage leaked into raw_records: {record.message!r}"
                )

    @pytest.mark.asyncio
    async def test_condition_reevaluated_each_call(self, model_fixture):
        """
        Turn 1 with condition=False → no auto-injection.
        Flip the gate, turn 2 with condition=True → auto-injection fires.

        Ordering chosen so turn 1 produces no framework-supplied state content
        the LLM could persist; the turn-2 assertion is specifically about our
        fabricated injection.
        """
        flag = {"fire": False}
        tool_fn = make_gated_statetool(flag, content="turn_by_turn_state")

        agent = ReActAgent(
            model=model_fixture,
            tools=[tool_fn],
            temperature=0.0,
            verbose=False,
            allow_ask_user=False,
        )

        # Turn 1: condition False, no injection.
        await agent.react_async("First query.")
        sent_1 = agent.chat_history.last_sent_messages or []
        assert not fabricated_injection_present(sent_1, "gated_tool"), (
            "turn 1 should have no framework injection (condition False)"
        )

        # Flip the gate, ask again.
        flag["fire"] = True
        await agent.react_async("Second query.")
        sent_2 = agent.chat_history.last_sent_messages or []
        assert fabricated_injection_present(sent_2, "gated_tool"), (
            "turn 2 should include framework injection (condition now True)"
        )


class TestStateAndInstructionTogether:

    @pytest.mark.asyncio
    async def test_state_precedes_instruction_in_tail(self, model_fixture):
        """When both are present, state pairs precede the instruction block."""
        from langchain_core.messages import AIMessage, HumanMessage

        state_marker = "CO_STATE_MARKER_9X3"
        instr_marker = "CO_INSTRUCTION_MARKER_Q7"
        tool_fn = make_always_true_statetool(state_marker, name="co_state")

        agent = ReActAgent(
            model=model_fixture,
            tools=[tool_fn],
            temperature=0.0,
            verbose=False,
            allow_ask_user=False,
        )
        agent.set_auto_context(
            default_content=instr_marker,
            content_updater=lambda: instr_marker,
            auto_update=True,
        )

        await agent.react_async("Say hello.")
        sent = agent.chat_history.last_sent_messages or []

        # Locate the fabricated state injection's AIMessage (by placeholder text).
        state_ai_idx = next(
            (
                i for i, m in enumerate(sent)
                if isinstance(m, AIMessage)
                and m.content == STATE_INJECTION_PLACEHOLDER_TEXT
                and any(tc.get("name") == "co_state" for tc in (m.tool_calls or []))
            ),
            None,
        )
        instruction_idx = next(
            (
                i for i, m in enumerate(sent)
                if isinstance(m, HumanMessage)
                and isinstance(m.content, str)
                and instr_marker in m.content
                and "<system_context_update>" in m.content
            ),
            None,
        )

        assert state_ai_idx is not None, "fabricated state AIMessage should be in outgoing list"
        assert instruction_idx is not None, "instruction HumanMessage should be in outgoing list"
        assert state_ai_idx < instruction_idx, (
            "state pair must precede the instruction block in the outgoing list"
        )

        # The instruction content and our placeholder AIMessage must never be persisted.
        # (The state marker content itself may end up in raw_records if the LLM
        # chose to call the tool on its own; that's unrelated to our injection.)
        for record in agent.chat_history.raw_records:
            content = getattr(record.message, "content", "") or ""
            if isinstance(content, str):
                assert instr_marker not in content, (
                    f"instruction marker leaked into raw_records: {record.message!r}"
                )
                assert content != STATE_INJECTION_PLACEHOLDER_TEXT, (
                    f"fabricated placeholder AIMessage leaked into raw_records: {record.message!r}"
                )
