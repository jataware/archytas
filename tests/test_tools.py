"""
Tests for tool creation, validation, and execution.
"""
import pytest
from archytas.tool_utils import tool, is_tool, make_tool_dict, AgentRef, LoopControllerRef
from archytas.react import LoopController


class TestToolDecorator:
    """Test the @tool decorator."""

    def test_tool_decorator_basic(self):
        """Test basic tool decoration."""
        @tool()
        def simple_tool(value: int) -> str:
            """
            A simple tool.

            Args:
                value (int): Some value

            Returns:
                str: The value as string
            """
            return str(value)

        assert is_tool(simple_tool)
        assert simple_tool._name == "simple_tool"
        assert hasattr(simple_tool, "run")

    def test_tool_decorator_with_name(self):
        """Test tool with custom name."""
        @tool(name="custom_name")
        def some_function(x: int) -> str:
            """
            Some function.

            Args:
                x (int): Input

            Returns:
                str: Output
            """
            return str(x)

        assert some_function._name == "custom_name"

    def test_tool_without_parentheses(self):
        """Test @tool without parentheses."""
        @tool
        def another_tool(value: str) -> str:
            """
            Another tool.

            Args:
                value (str): Input value

            Returns:
                str: Output value
            """
            return value

        assert is_tool(another_tool)
        assert another_tool._name == "another_tool"

    def test_tool_missing_docstring(self):
        """Test tool raises error without docstring."""
        with pytest.raises(AssertionError):
            @tool()
            def no_docstring(x: int) -> str:
                return str(x)

    def test_tool_docstring_mismatch(self):
        """Test tool raises error when docstring args don't match signature."""
        with pytest.raises(ValueError, match="Docstring argument names do not match"):
            @tool()
            def mismatched(x: int, y: int) -> str:
                """
                Mismatched tool.

                Args:
                    x (int): First arg
                    z (int): Wrong name

                Returns:
                    str: Result
                """
                return str(x + y)


class TestToolExecution:
    """Test tool execution via run method."""

    @pytest.mark.asyncio
    async def test_tool_run_method(self):
        """Test calling tool via run method."""
        @tool()
        def add_numbers(a: int, b: int) -> str:
            """
            Add two numbers.

            Args:
                a (int): First number
                b (int): Second number

            Returns:
                str: Sum
            """
            return str(a + b)

        result = await add_numbers.run(args={"a": 5, "b": 3})
        assert result == "8"

    @pytest.mark.asyncio
    async def test_tool_with_optional_args(self):
        """Test tool with optional arguments."""
        @tool()
        def greet(name: str, title: str = "Mr.") -> str:
            """
            Greet someone with optional title.

            Args:
                name (str): Person's name
                title (str): Optional title

            Returns:
                str: Greeting
            """
            return f"Hello, {title} {name}"

        result = await greet.run(args={"name": "Smith"})
        assert "Smith" in result

    @pytest.mark.asyncio
    async def test_tool_with_agent_injection(self, openai_model):
        """Test tool with AgentRef dependency injection."""
        @tool()
        def tool_with_agent(value: int, agent: AgentRef) -> str:
            """
            Tool that receives agent reference.

            Args:
                value (int): Some value

            Returns:
                str: Result
            """
            return f"Agent type: {type(agent).__name__}, value: {value}"

        from archytas.react import ReActAgent
        mock_agent = ReActAgent(model=openai_model, tools=[], allow_ask_user=False)

        result = await tool_with_agent.run(
            args={"value": 42},
            tool_context={"agent": mock_agent}
        )
        assert "ReActAgent" in result
        assert "42" in result

    @pytest.mark.asyncio
    async def test_tool_with_loop_controller(self):
        """Test tool can access loop controller."""
        @tool()
        def tool_with_controller(action: str, controller: LoopControllerRef) -> str:
            """
            Tool that can control loop.

            Args:
                action (str): What to do

            Returns:
                str: Result
            """
            if action == "stop":
                controller.set_state(LoopController.STOP_SUCCESS)
            return f"Action: {action}"

        controller = LoopController()
        result = await tool_with_controller.run(
            args={"action": "stop"},
            tool_context={"loop_controller": controller}
        )

        assert controller.state == LoopController.STOP_SUCCESS


class TestToolDict:
    """Test tool dictionary creation."""

    def test_make_tool_dict_functions(self):
        """Test creating tool dict from functions."""
        @tool()
        def tool_one(x: int) -> str:
            """
            Tool one.

            Args:
                x (int): Input

            Returns:
                str: Output
            """
            return str(x)

        @tool()
        def tool_two(y: str) -> str:
            """
            Tool two.

            Args:
                y (str): Input

            Returns:
                str: Output
            """
            return y

        tools_dict = make_tool_dict([tool_one, tool_two])
        assert "tool_one" in tools_dict
        assert "tool_two" in tools_dict
        assert len(tools_dict) == 2

    def test_make_tool_dict_with_class(self):
        """Test creating tool dict from class with tool methods."""
        class ToolClass:
            """A class with tool methods."""

            @tool()
            def method_one(self, x: int) -> str:
                """
                Method one.

                Args:
                    x (int): Input

                Returns:
                    str: Output
                """
                return str(x * 2)

            @tool()
            def method_two(self, y: int) -> str:
                """
                Method two.

                Args:
                    y (int): Input

                Returns:
                    str: Output
                """
                return str(y + 1)

        tools_dict = make_tool_dict([ToolClass])

        # Class methods should be in dict
        assert "method_one" in tools_dict
        assert "method_two" in tools_dict


class TestToolAutosummarize:
    """Test tool autosummarize functionality."""

    def test_tool_with_autosummarize(self):
        """Test tool can be marked for autosummarization."""
        @tool(autosummarize=True)
        def verbose_tool(x: int) -> str:
            """
            Verbose tool that should be summarized.

            Args:
                x (int): Input

            Returns:
                str: Long output
            """
            return "Very long output " * 100

        assert verbose_tool.autosummarize is True
        assert verbose_tool.summarizer is not None

    def test_tool_without_autosummarize(self):
        """Test tool without autosummarize."""
        @tool()
        def normal_tool(x: int) -> str:
            """
            Normal tool.

            Args:
                x (int): Input

            Returns:
                str: Output
            """
            return str(x)

        assert normal_tool.autosummarize is False
        assert normal_tool.summarizer is None
