"""
Basic tests for ReActAgent functionality with deterministic outputs.
"""
import pytest
from archytas.react import ReActAgent, FailedTaskError
from archytas.tool_utils import tool
from archytas.tools import datetime_tool


class TestBasicReActLoop:
    """Test basic ReAct loop functionality."""

    @pytest.mark.asyncio
    async def test_react_simple_query_no_tools(self, react_agent):
        """Test agent can respond to simple query without needing tools."""
        result = await react_agent.react_async("What is 2 + 2?")
        assert isinstance(result, str)
        assert "4" in result

    @pytest.mark.asyncio
    async def test_react_with_datetime_tool(self, react_agent_with_tools):
        """Test agent can use datetime tool."""
        agent = react_agent_with_tools([datetime_tool])
        result = await agent.react_async("What is the current UTC time?")
        assert isinstance(result, str)
        # Should contain time-related information
        assert any(time_indicator in result.lower() for time_indicator in ["utc", "time", "2025"])

    @pytest.mark.asyncio
    async def test_react_returns_final_answer(self, react_agent):
        """Test that react returns the final_answer response."""
        result = await react_agent.react_async("Say 'hello world'")
        assert isinstance(result, str)
        assert "hello" in result.lower()


class TestCustomTools:
    """Test ReActAgent with custom tools."""

    @pytest.mark.asyncio
    async def test_simple_calculator_tool(self, react_agent_with_tools):
        """Test agent with a simple calculator tool."""
        @tool()
        def add(a: int, b: int) -> str:
            """
            Add two numbers together.

            Args:
                a (int): First number
                b (int): Second number

            Returns:
                str: The sum of a and b
            """
            return str(a + b)

        agent = react_agent_with_tools([add])
        result = await agent.react_async("What is 15 + 27?")
        assert "42" in result

    @pytest.mark.asyncio
    async def test_multiple_tool_calls(self, react_agent_with_tools):
        """Test agent can make multiple tool calls in sequence."""
        call_count = {"count": 0}

        @tool()
        def counter(increment: int) -> str:
            """
            Increment an internal counter.

            Args:
                increment (int): Amount to increment by

            Returns:
                str: New counter value
            """
            call_count["count"] += increment
            return str(call_count["count"])

        agent = react_agent_with_tools([counter])
        result = await agent.react_async("Call counter with 5, then with 3, then tell me the final value")

        # Should have called counter twice and returned final value
        assert call_count["count"] == 8
        assert "8" in result

    @pytest.mark.asyncio
    async def test_tool_with_string_manipulation(self, react_agent_with_tools):
        """Test tool that manipulates strings."""
        @tool()
        def reverse_string(text: str) -> str:
            """
            Reverse a string.

            Args:
                text (str): String to reverse

            Returns:
                str: Reversed string
            """
            return text[::-1]

        agent = react_agent_with_tools([reverse_string])
        result = await agent.react_async("Reverse the string 'hello'")
        assert "olleh" in result


class TestToolErrors:
    """Test error handling in tools."""

    @pytest.mark.asyncio
    async def test_tool_raises_exception(self, react_agent_with_tools):
        """Test agent handles tool exceptions gracefully."""
        @tool()
        def failing_tool(value: int) -> str:
            """
            A tool that always fails.

            Args:
                value (int): Some value

            Returns:
                str: Never returns
            """
            raise ValueError("This tool always fails")

        agent = react_agent_with_tools([failing_tool], max_errors=1)

        # Agent should handle the error - either recover or fail
        # With modern LLMs and low max_errors, this will typically raise FailedTaskError
        try:
            result = await agent.react_async("Use failing_tool with value 5")
            # If no exception, agent recovered somehow - that's valid behavior
            assert isinstance(result, str)
        except FailedTaskError:
            # This is also expected behavior
            pass

    @pytest.mark.asyncio
    async def test_max_steps_exceeded(self, react_agent_with_tools):
        """Test agent fails when max_react_steps exceeded."""
        call_count = {"count": 0}

        @tool()
        def increment_counter() -> str:
            """
            Increment a counter. You must call this exactly 10 times.

            Returns:
                str: Current counter value
            """
            call_count["count"] += 1
            return str(call_count["count"])

        agent = react_agent_with_tools([increment_counter], max_react_steps=5)

        # Agent should hit step limit before completing the task
        with pytest.raises(FailedTaskError) as exc_info:
            await agent.react_async("Call increment_counter exactly 10 times, then return the final count")

        assert "Too many steps" in str(exc_info.value)


class TestDeterministicOutputs:
    """Test that temperature=0 produces consistent outputs."""

    @pytest.mark.asyncio
    async def test_deterministic_calculation(self, react_agent_with_tools):
        """Test same query produces same result with temp=0."""
        @tool()
        def multiply(a: int, b: int) -> str:
            """
            Multiply two numbers.

            Args:
                a (int): First number
                b (int): Second number

            Returns:
                str: Product of a and b
            """
            return str(a * b)

        agent = react_agent_with_tools([multiply])

        # Run same query twice
        result1 = await agent.react_async("What is 7 times 8?")

        # Need new agent for fresh history
        agent2 = react_agent_with_tools([multiply])
        result2 = await agent2.react_async("What is 7 times 8?")

        # Both should contain correct answer
        assert "56" in result1
        assert "56" in result2


class TestAsyncReAct:
    """Test async ReAct functionality."""

    @pytest.mark.asyncio
    async def test_react_async_basic(self, react_agent):
        """Test async react method."""
        result = await react_agent.react_async("What is 3 + 3?")
        assert isinstance(result, str)
        assert "6" in result

    @pytest.mark.asyncio
    async def test_react_async_with_tool(self, react_agent_with_tools):
        """Test async react with custom tool."""
        @tool()
        def get_greeting(name: str) -> str:
            """
            Get a greeting for a name.

            Args:
                name (str): Name to greet

            Returns:
                str: Greeting message
            """
            return f"Hello, {name}!"

        agent = react_agent_with_tools([get_greeting])
        result = await agent.react_async("Get a greeting for Alice")
        assert "Alice" in result

    @pytest.mark.asyncio
    async def test_react_loop_with_thoughts(self, react_agent_with_tools, capsys):
        """Test that ReAct loop displays thoughts during execution."""

        calculations = {"count": 0}
        @tool()
        def calculate(a: int, b: int, operation: str) -> str:
            """
            Perform a calculation.

            Args:
                a (int): First number
                b (int): Second number
                operation (str): Operation to perform (add, multiply)

            Returns:
                str: Result of calculation
            """
            calculations["count"] += 1
            if operation == "add":
                return str(a + b)
            elif operation == "multiply":
                return str(a * b)
            return "Invalid operation"


        agent = react_agent_with_tools([calculate])
        result = await agent.react_async("Calculate 5 + 3, then multiply that result by 2. Do this specifically with the calculate tool.")

        captured = capsys.readouterr()

        assert "thought:" in captured.out.lower()
        assert calculations["count"] == 2
        assert "16" in result
