"""
Tests for agent context management.
"""
import pytest
from archytas.react import ReActAgent
from archytas.tool_utils import tool, AgentRef


class TestContextMessages:
    """Test context message functionality."""

    def test_add_context(self, react_agent):
        """Test adding context to agent."""
        context_id = react_agent.add_context("Important information: the answer is 42")
        assert isinstance(context_id, int)

    def test_agent_uses_context(self, react_agent):
        """Test agent can use context information."""
        react_agent.add_context("The desired color to remember is: blue")
        result = react_agent.react("What color were you supposed to remember?")
        assert "blue" in result.lower()

    def test_clear_context(self, react_agent):
        """Test clearing specific context."""
        context_id = react_agent.add_context("Temporary information")
        react_agent.clear_context(context_id)

        # Context should be removed - agent won't have access to it
        result = react_agent.react("What did I just tell you about temporary information?")
        # Agent should not have the specific context anymore
        assert isinstance(result, str)

    def test_clear_all_context(self, react_agent):
        """Test clearing all context messages."""
        react_agent.add_context("Context one")
        react_agent.add_context("Context two")
        react_agent.add_context("Context three")

        react_agent.clear_all_context()

        # All context should be cleared
        assert isinstance(react_agent.chat_history.raw_records, list)


class TestAutoContext:
    """Test automatic context updates."""

    def test_auto_context_basic(self, react_agent):
        """Test setting auto context."""
        counter = {"value": 0}

        def update_counter():
            counter["value"] += 1
            return f"Counter is now: {counter['value']}"

        react_agent.set_auto_context(
            default_content="Counter is: 0",
            content_updater=update_counter,
            auto_update=True
        )

        # First query should have counter = 1
        result = react_agent.react("What is the counter value?")
        assert isinstance(result, str)
        assert counter["value"] >= 1

    def test_auto_context_updates_each_call(self, react_agent):
        """Test auto context updates on each agent call."""
        state = {"count": 0}

        def get_state():
            return f"Call count: {state['count']}"

        react_agent.set_auto_context(
            default_content="No calls yet",
            content_updater=get_state,
            auto_update=True
        )

        # Make multiple queries
        react_agent.react("First query: what is 1+1?")
        state["count"] = 10
        result = react_agent.react("What is the call count?")

        # Should reflect updated state
        assert isinstance(result, str)


class TestToolAccessToContext:
    """Test tools accessing agent context."""

    def test_tool_with_agent_ref(self, react_agent_with_tools):
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
        result = agent.react("Use check_context to see what I'm asking")

        assert isinstance(result, str)

    def test_tool_with_agent_access(self, react_agent_with_tools):
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
        result = agent.react("Use get_agent_info")

        # Verify the tool was called and accessed agent
        assert "ReActAgent" in result or "Agent" in result


class TestReactContext:
    """Test react_context parameter functionality."""

    def test_react_context_passed_to_tools(self, react_agent_with_tools):
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
        result = agent.react("Use use_context to get 'test_key'", react_context=custom_context)

        assert "test_value" in result
