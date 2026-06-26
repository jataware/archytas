"""
Tests for chat history management.
"""
import pytest
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from archytas.chat_history import ChatHistory, MessageRecord
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


class TestAllRecords:
    """Pure-unit tests for ChatHistory.all_records()/all_messages().

    These require no model or API key.

    Regression coverage for a bug where all_records() appended the system
    preamble to an undefined `records` name (it should be `messages`), raising
    NameError. The branch was previously dead because set_system_preamble_text
    misrouted to user_preamble, leaving system_preamble unset.
    """

    @pytest.mark.asyncio
    async def test_all_records_includes_system_preamble(self):
        history = ChatHistory()
        history.set_system_message("system message")
        history.set_system_preamble_text("system preamble")
        history.set_user_preamble_text("user preamble")
        history.add_message(HumanMessage(content="hello"))

        records = await history.all_records()

        contents = [r.message.content for r in records]
        assert contents == [
            "system message",
            "system preamble",
            "user preamble",
            "hello",
        ]

    @pytest.mark.asyncio
    async def test_all_records_with_only_system_preamble(self):
        # The exact branch that used to raise NameError: system_preamble set
        # while system_message and user_preamble are unset.
        history = ChatHistory()
        history.set_system_preamble_text("system preamble")

        records = await history.all_records()

        assert len(records) == 1
        assert records[0] is history.system_preamble
        assert isinstance(records[0].message, SystemMessage)
        assert records[0].message.content == "system preamble"

    @pytest.mark.asyncio
    async def test_all_messages_includes_system_preamble(self):
        history = ChatHistory()
        history.set_system_preamble_text("system preamble")
        history.add_message(HumanMessage(content="hello"))

        messages = await history.all_messages()

        assert [m.content for m in messages] == ["system preamble", "hello"]

    @pytest.mark.asyncio
    async def test_all_records_without_preamble(self):
        history = ChatHistory()
        history.add_message(HumanMessage(content="hello"))

        records = await history.all_records()

        assert [r.message.content for r in records] == ["hello"]

    @pytest.mark.asyncio
    async def test_setting_user_preamble_string(self):
        history = ChatHistory()
        history.set_system_message("system message")
        history.add_message(HumanMessage(content="hello"))
        history.set_user_preamble_text("user preamble")

        records = await history.all_records()

        contents = [r.message.content for r in records]
        assert contents == [
            "system message",
            "user preamble",
            "hello",
        ]

    @pytest.mark.asyncio
    async def test_setting_user_preamble_empty_string(self):
        history = ChatHistory()
        history.set_system_message("system message")
        history.set_user_preamble_text("user preamble")
        history.add_message(HumanMessage(content="hello"))
        
        initial_records = await history.all_records()

        history.set_user_preamble_text("")

        updated_records = await history.all_records()

        initial_contents = [r.message.content for r in initial_records]
        final_contents = [r.message.content for r in updated_records]

        assert initial_contents == [
            "system message",
            "user preamble",
            "hello",
        ]
        assert final_contents == [
            "system message",
            "hello",
        ]

    @pytest.mark.asyncio
    async def test_setting_user_preamble_blank_string(self):
        history = ChatHistory()
        history.set_system_message("system message")
        history.set_user_preamble_text("user preamble")
        history.add_message(HumanMessage(content="hello"))
        
        initial_records = await history.all_records()

        history.set_user_preamble_text("   ")

        updated_records = await history.all_records()

        initial_contents = [r.message.content for r in initial_records]
        final_contents = [r.message.content for r in updated_records]

        assert initial_contents == [
            "system message",
            "user preamble",
            "hello",
        ]
        assert final_contents == [
            "system message",
            "hello",
        ]

    @pytest.mark.asyncio
    async def test_setting_user_preamble_none(self):
        history = ChatHistory()
        history.set_system_message("system message")
        history.set_user_preamble_text("user preamble")
        history.add_message(HumanMessage(content="hello"))
        
        initial_records = await history.all_records()

        history.set_user_preamble_text(None)

        final_records = await history.all_records()

        initial_contents = [r.message.content for r in initial_records]
        final_contents = [r.message.content for r in final_records]

        assert initial_contents == [
            "system message",
            "user preamble",
            "hello",
        ]
        assert final_contents == [
            "system message",
            "hello",
        ]

    @pytest.mark.asyncio
    async def test_setting_user_preamble_valid_record(self):
        orig_user_preamble = MessageRecord(HumanMessage(content="original user preamble"), uuid="abc123", metadata={"preamble": True})

        history = ChatHistory()
        history.set_system_message("system message")
        history.add_message(HumanMessage(content="hello"))
        history.set_user_preamble_text(orig_user_preamble)

        records = await history.all_records()
        assert records[1] == orig_user_preamble
        assert records[1].uuid == orig_user_preamble.uuid

        contents = [r.message.content for r in records]
        assert contents == [
            "system message",
            "original user preamble",
            "hello",
        ]

    @pytest.mark.asyncio
    async def test_setting_user_preamble_other_record(self):
        human_user_preamble = MessageRecord(HumanMessage(content="human user preamble"), uuid="abc123")
        system_user_preamble = MessageRecord(SystemMessage(content="system user preamble"), uuid="system123")

        history = ChatHistory()
        history.set_system_message("system message")
        history.add_message(HumanMessage(content="hello"))
        history.set_user_preamble_text(human_user_preamble)

        records_1 = await history.all_records()
        assert records_1[1].uuid != human_user_preamble.uuid
        assert isinstance(records_1[1].message, HumanMessage)
        assert records_1[1].metadata.get("preamble", None) == True

        contents = [r.message.content for r in records_1]
        assert contents == [
            "system message",
            "human user preamble",
            "hello",
        ]

        history.set_user_preamble_text(system_user_preamble)

        records_2 = await history.all_records()
        assert records_2[1].uuid != human_user_preamble.uuid
        assert isinstance(records_2[1].message, HumanMessage)
        assert records_2[1].metadata.get("preamble", None) == True

        contents = [r.message.content for r in records_2]
        assert contents == [
            "system message",
            "system user preamble",
            "hello",
        ]

    @pytest.mark.asyncio
    async def test_setting_system_preamble_string(self):
        history = ChatHistory()
        history.set_system_message("system message")
        history.add_message(HumanMessage(content="hello"))
        history.set_system_preamble_text("system preamble")

        records = await history.all_records()

        contents = [r.message.content for r in records]
        assert contents == [
            "system message",
            "system preamble",
            "hello",
        ]

    @pytest.mark.asyncio
    async def test_setting_system_preamble_empty_string(self):
        history = ChatHistory()
        history.set_system_message("system message")
        history.set_system_preamble_text("system preamble")
        history.add_message(HumanMessage(content="hello"))

        initial_records = await history.all_records()

        history.set_system_preamble_text("")

        final_records = await history.all_records()

        initial_contents = [r.message.content for r in initial_records]
        final_contents = [r.message.content for r in final_records]

        assert initial_contents == [
            "system message",
            "system preamble",
            "hello",
        ]
        assert final_contents == [
            "system message",
            "hello",
        ]

    @pytest.mark.asyncio
    async def test_setting_system_preamble_blank_string(self):
        history = ChatHistory()
        history.set_system_message("system message")
        history.set_system_preamble_text("system preamble")
        history.add_message(HumanMessage(content="hello"))

        initial_records = await history.all_records()

        history.set_system_preamble_text("   ")

        updated_records = await history.all_records()

        initial_contents = [r.message.content for r in initial_records]
        final_contents = [r.message.content for r in updated_records]

        assert initial_contents == [
            "system message",
            "system preamble",
            "hello",
        ]
        assert final_contents == [
            "system message",
            "hello",
        ]

    @pytest.mark.asyncio
    async def test_setting_system_preamble_none(self):
        history = ChatHistory()
        history.set_system_message("system message")
        history.set_system_preamble_text("system preamble")
        history.add_message(HumanMessage(content="hello"))

        initial_records = await history.all_records()

        history.set_system_preamble_text(None)

        final_records = await history.all_records()

        initial_contents = [r.message.content for r in initial_records]
        final_contents = [r.message.content for r in final_records]

        assert initial_contents == [
            "system message",
            "system preamble",
            "hello",
        ]
        assert final_contents == [
            "system message",
            "hello",
        ]

    @pytest.mark.asyncio
    async def test_setting_system_preamble_valid_record(self):
        orig_system_preamble = MessageRecord(SystemMessage(content="original system preamble"), uuid="abc123", metadata={"preamble": True})

        history = ChatHistory()
        history.set_system_message("system message")
        history.add_message(HumanMessage(content="hello"))
        history.set_system_preamble_text(orig_system_preamble)

        records = await history.all_records()
        assert records[1] == orig_system_preamble
        assert records[1].uuid == orig_system_preamble.uuid

        contents = [r.message.content for r in records]
        assert contents == [
            "system message",
            "original system preamble",
            "hello",
        ]

    @pytest.mark.asyncio
    async def test_setting_system_preamble_other_record(self):
        system_system_preamble = MessageRecord(SystemMessage(content="system system preamble"), uuid="system123")
        human_system_preamble = MessageRecord(HumanMessage(content="human system preamble"), uuid="abc123")

        history = ChatHistory()
        history.set_system_message("system message")
        history.add_message(HumanMessage(content="hello"))
        history.set_system_preamble_text(system_system_preamble)

        records_1 = await history.all_records()
        assert records_1[1].uuid != system_system_preamble.uuid
        assert isinstance(records_1[1].message, SystemMessage)
        assert records_1[1].metadata.get("preamble", None) == True

        contents = [r.message.content for r in records_1]
        assert contents == [
            "system message",
            "system system preamble",
            "hello",
        ]

        history.set_system_preamble_text(human_system_preamble)

        records_2 = await history.all_records()
        assert records_2[1].uuid != human_system_preamble.uuid
        assert isinstance(records_2[1].message, SystemMessage)
        assert records_2[1].metadata.get("preamble", None) == True

        contents = [r.message.content for r in records_2]
        assert contents == [
            "system message",
            "human system preamble",
            "hello",
        ]
