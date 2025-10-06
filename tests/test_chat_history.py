"""
Tests for chat history management.
"""
import pytest
from langchain_core.messages import HumanMessage, AIMessage
from archytas.react import ReActAgent


class TestChatHistory:
    """Test basic chat history functionality."""

    @pytest.mark.asyncio
    async def test_messages_are_stored(self, react_agent):
        """Test that messages are stored in chat history."""
        await react_agent.react_async("What is 2+2?")

        messages = await react_agent.all_messages()
        assert len(messages) > 0

        # Should have at least the user message
        human_messages = [m for m in messages if isinstance(m, HumanMessage)]
        assert len(human_messages) >= 1

    @pytest.mark.asyncio
    async def test_conversation_continuity(self, react_agent):
        """Test agent remembers previous messages."""
        await react_agent.react_async("The magic number is 70.")
        response = await react_agent.react_async("Repeat the magic number back.")

        assert "70" in response

    @pytest.mark.asyncio
    async def test_multiple_exchanges(self, react_agent):
        """Test multiple back-and-forth exchanges."""
        await react_agent.react_async("The magic number is 105.")
        await react_agent.react_async("The magic color is purple.")
        response = await react_agent.react_async("List the magic number and magic color.")

        assert "105" in response.lower()
        assert "purple" in response.lower()


class TestReActLoopHistory:
    """Test chat history during ReAct loops."""

    @pytest.mark.asyncio
    async def test_tool_calls_in_history(self, react_agent_with_tools):
        """Test that tool calls are recorded in history."""
        from archytas.tool_utils import tool

        @tool()
        def test_tool(value: int) -> str:
            """
            Test tool.

            Args:
                value (int): Input value

            Returns:
                str: Output
            """
            return f"Processed: {value}"

        agent = react_agent_with_tools([test_tool])
        await agent.react_async("Use test_tool with value 42")

        messages = await agent.all_messages()
        assert len(messages) > 0

    @pytest.mark.asyncio
    async def test_multiple_react_loops_separated(self, react_agent_with_tools):
        """Test multiple ReAct loops create separate conversation segments."""
        from archytas.tool_utils import tool

        @tool()
        def echo(text: str) -> str:
            """
            Echo text.

            Args:
                text (str): Text to echo

            Returns:
                str: Echoed text
            """
            return text

        agent = react_agent_with_tools([echo])

        # First loop
        result1 = await agent.react_async("Echo 'first'")
        assert "first" in result1.lower()

        # Second loop
        result2 = await agent.react_async("Echo 'second'")
        assert "second" in result2.lower()

        # Both should be in history
        messages = await agent.all_messages()
        assert len(messages) > 2


class TestHistorySummarization:
    """Test automatic history summarization."""

    @pytest.mark.asyncio
    async def test_summarization_threshold_config(self, openai_model):
        """Test configuring summarization threshold."""
        from archytas.models.openai import OpenAIModel

        # Create model with custom summarization settings
        model = OpenAIModel({
            "model_name": "gpt-5",
            "summarization_threshold": 1000
        })

        assert model.summarization_threshold == 1000

    @pytest.mark.asyncio
    async def test_long_conversation_management(self, react_agent):
        # TODO: cause a summarization in chat history
        pass


class TestHistoryInspection:
    """Test inspecting chat history."""

    @pytest.mark.asyncio
    async def test_all_messages(self, react_agent):
        """Test retrieving all messages."""
        await react_agent.react_async("What is 1+1?")
        await react_agent.react_async("What is 2+2?")

        messages = await react_agent.all_messages()
        assert len(messages) >= 2

    @pytest.mark.asyncio
    async def test_all_messages_sync(self, react_agent):
        """Test synchronous message retrieval."""
        await react_agent.react_async("What is 3+3?")

        messages = await react_agent.all_messages()
        assert len(messages) >= 1

        # Check message types
        human_msgs = [m for m in messages if isinstance(m, HumanMessage)]
        assert len(human_msgs) >= 1


class TestHistoryEdgeCases:
    """Test edge cases in history management."""

    @pytest.mark.asyncio
    async def test_empty_history_query(self, react_agent):
        """Test querying with fresh/empty history."""
        result = await react_agent.react_async("What is 2+2?")
        assert "4" in result

    @pytest.mark.asyncio
    async def test_query_after_error(self, react_agent_with_tools):
        """Test history is maintained even after errors."""
        from archytas.tool_utils import tool
        from archytas.react import FailedTaskError

        @tool()
        def failing_tool(x: int) -> str:
            """
            Failing tool.

            Args:
                x (int): Input

            Returns:
                str: Output
            """
            raise ValueError("Always fails")

        agent = react_agent_with_tools([failing_tool], max_errors=1)

        # First query will fail
        try:
            await agent.react_async("Use failing_tool with 5")
        except FailedTaskError:
            pass

        # Agent should still work after error
        result = await agent.react_async("Say hello.")
        assert isinstance(result, str)
        assert len(result) > 0
