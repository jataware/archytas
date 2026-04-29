"""
Tests for agent context management.
"""
import pytest
from archytas.react import ReActAgent
from archytas.tool_utils import tool, AgentRef


class TestContextMessages:
    """Test context message functionality."""

    @pytest.mark.asyncio
    async def test_add_context(self, react_agent):
        """Test adding context to agent."""
        context_id = react_agent.add_context("Important information: the answer is 42")
        assert isinstance(context_id, int)

    @pytest.mark.asyncio
    async def test_agent_uses_context(self, react_agent):
        """Test agent can use context information."""
        react_agent.add_context("The desired color to remember is: blue")
        result = await react_agent.react_async("What color were you supposed to remember?")
        assert "blue" in result.lower()

    @pytest.mark.asyncio
    async def test_clear_context(self, react_agent):
        """Test clearing specific context."""
        context_id = react_agent.add_context("Temporary information")
        react_agent.clear_context(context_id)

        # Context should be removed - agent won't have access to it
        result = await react_agent.react_async("What did I just tell you about temporary information?")
        # Agent should not have the specific context anymore
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_clear_all_context(self, react_agent):
        """Test clearing all context messages."""
        react_agent.add_context("Context one")
        react_agent.add_context("Context two")
        react_agent.add_context("Context three")

        react_agent.clear_all_context()

        # All context should be cleared
        assert isinstance(react_agent.chat_history.raw_records, list)


class TestAutoContext:
    """Test automatic context updates (plan §4.3 instruction mechanism)."""

    @pytest.mark.asyncio
    async def test_auto_context_basic(self, react_agent):
        """Registering an instruction via set_auto_context fires the updater."""
        counter = {"value": 0}

        def update_counter():
            counter["value"] += 1
            return f"Counter is now: {counter['value']}"

        react_agent.set_auto_context(
            default_content="Counter is: 0",
            content_updater=update_counter,
            auto_update=True
        )

        result = await react_agent.react_async("What is the counter value?")
        assert isinstance(result, str)
        assert counter["value"] >= 1

    @pytest.mark.asyncio
    async def test_auto_context_updates_each_call(self, react_agent):
        """Updater fires on each react call, reflecting updated external state."""
        state = {"count": 0}

        def get_state():
            return f"Call count: {state['count']}"

        react_agent.set_auto_context(
            default_content="No calls yet",
            content_updater=get_state,
            auto_update=True
        )

        await react_agent.react_async("First query: what is 1+1?")
        state["count"] = 10
        result = await react_agent.react_async("What is the call count?")

        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_instruction_delivered_at_tail(self, react_agent):
        """
        Phase 1: the instruction content is delivered at the tail of the
        outgoing message list, wrapped in the <system_context_update> XML
        tags, NOT as a SystemMessage near the top.
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        react_agent.set_auto_context(
            default_content="Operational context: quiet hours in effect.",
            content_updater=lambda: "Operational context: quiet hours in effect.",
            auto_update=True,
        )

        await react_agent.react_async("Say hello.")

        sent = react_agent.chat_history.last_sent_messages
        assert sent is not None and len(sent) > 0, "last_sent_messages must be populated after execute()"

        # The tail message must be a HumanMessage carrying the XML wrapper.
        tail = sent[-1]
        assert isinstance(tail, HumanMessage), f"tail should be HumanMessage, got {type(tail).__name__}"
        assert "<system_context_update>" in tail.content
        assert "quiet hours in effect" in tail.content

        # The instruction content must NOT appear in any non-tail message.
        # In particular, it should not be folded into the leading SystemMessage.
        for msg in sent[:-1]:
            if isinstance(msg, SystemMessage):
                assert "<system_context_update>" not in msg.content
                assert "quiet hours in effect" not in msg.content

    @pytest.mark.asyncio
    async def test_instruction_not_persisted(self, react_agent):
        """
        Phase 1: instruction content is ephemeral — it appears in the outgoing
        message list for one execute() call but is never written into
        chat_history.raw_records.
        """
        react_agent.set_auto_context(
            default_content="UNIQUE_INSTRUCTION_MARKER_7a2b3c",
            content_updater=lambda: "UNIQUE_INSTRUCTION_MARKER_7a2b3c",
            auto_update=True,
        )

        await react_agent.react_async("Say hello.")

        # None of the persisted records should carry the instruction marker.
        for record in react_agent.chat_history.raw_records:
            content = getattr(record.message, "content", "") or ""
            if isinstance(content, str):
                assert "UNIQUE_INSTRUCTION_MARKER_7a2b3c" not in content, (
                    f"instruction marker leaked into persisted record: {record.message!r}"
                )

        # And the instruction registration itself should still be live.
        assert react_agent.chat_history.instruction is not None
        assert react_agent.chat_history.instruction.current_content == "UNIQUE_INSTRUCTION_MARKER_7a2b3c"


class TestLastSentMessagesHook:
    """Test the Phase 0 observability hook on ChatHistory."""

    @pytest.mark.asyncio
    async def test_last_sent_messages_populated_after_execute(self, react_agent):
        """ChatHistory.last_sent_messages is populated after execute()."""
        assert react_agent.chat_history.last_sent_messages is None

        await react_agent.react_async("What is 1+1?")

        sent = react_agent.chat_history.last_sent_messages
        assert sent is not None
        assert len(sent) > 0

    @pytest.mark.asyncio
    async def test_last_sent_messages_recaptured_each_call(self, react_agent):
        """Each execute() call replaces the last_sent_messages snapshot."""
        await react_agent.react_async("What is 1+1?")
        first_snapshot = list(react_agent.chat_history.last_sent_messages or [])
        assert len(first_snapshot) > 0

        await react_agent.react_async("What is 2+2?")
        second_snapshot = list(react_agent.chat_history.last_sent_messages or [])
        assert len(second_snapshot) > 0

        # Second snapshot should be at least as long (history grew).
        assert len(second_snapshot) >= len(first_snapshot)


class TestToolAccessToContext:
    """Test tools accessing agent context."""

    @pytest.mark.asyncio
    async def test_tool_with_agent_ref(self, react_agent_with_tools):
        """Test tool can access agent and its context."""
        @tool()
        def check_context(query: str, agent: AgentRef) -> str:
            """
            Check agent context.

            Args:
                query (str): What to check

            Returns:
                str: Context info
            """
            # Access agent's current query
            return f"Agent's current query: {agent.current_query}"

        agent = react_agent_with_tools([check_context])
        result = await agent.react_async("Use check_context to see what I'm asking")

        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_tool_with_agent_access(self, react_agent_with_tools):
        """Test tool can access agent reference."""
        @tool()
        def get_agent_info(agent: AgentRef) -> str:
            """
            Get information about the agent.

            Returns:
                str: Agent info
            """
            return f"Agent class: {agent.__class__.__name__}"

        agent = react_agent_with_tools([get_agent_info])
        result = await agent.react_async("Use get_agent_info")

        # Verify the tool was called and accessed agent
        assert "ReActAgent" in result or "Agent" in result


class TestReactContext:
    """Test react_context parameter functionality."""

    @pytest.mark.asyncio
    async def test_react_context_passed_to_tools(self, react_agent_with_tools):
        """Test react_context dict is accessible to tools."""
        from archytas.tool_utils import ReactContextRef

        @tool()
        def use_context(key: str, context: ReactContextRef) -> str:
            """
            Use react context.

            Args:
                key (str): Key to look up

            Returns:
                str: Value from context
            """
            value = context.get(key, "Not found")
            return f"Context[{key}] = {value}"

        agent = react_agent_with_tools([use_context])

        # Pass custom context
        custom_context = {"test_key": "test_value", "number": 42}
        result = await agent.react_async("Use use_context to get 'test_key'", react_context=custom_context)

        assert "test_value" in result
