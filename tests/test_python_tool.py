"""
Tests for PythonTool functionality.
"""
import pytest
from archytas.tools import PythonTool
from archytas.react import ReActAgent


class TestPythonToolBasic:
    """Test basic PythonTool functionality."""

    @pytest.mark.asyncio
    async def test_python_tool_simple_execution(self):
        """Test PythonTool can execute simple Python code."""
        py_tool = PythonTool()
        result = await py_tool.run.run(args={"code": "print(2 + 2)"}, self_ref=py_tool)
        assert "4" in result

    @pytest.mark.asyncio
    async def test_python_tool_persistent_state(self):
        """Test PythonTool maintains state between executions."""
        py_tool = PythonTool()

        # First execution sets variable
        await py_tool.run.run(args={"code": "x = 10"}, self_ref=py_tool)

        # Second execution uses the variable
        result = await py_tool.run.run(args={"code": "print(x * 2)"}, self_ref=py_tool)
        assert "20" in result

    @pytest.mark.asyncio
    async def test_python_tool_with_imports(self):
        """Test PythonTool can import and use modules."""
        py_tool = PythonTool()
        result = await py_tool.run.run(args={"code": "import math\nprint(math.pi)"}, self_ref=py_tool)
        assert "3.14" in result

    @pytest.mark.asyncio
    async def test_python_tool_with_prelude(self):
        """Test PythonTool with prelude code."""
        py_tool = PythonTool(prelude="import json\ndata = {'key': 'value'}")
        result = await py_tool.run.run(args={"code": "print(json.dumps(data))"}, self_ref=py_tool)
        assert "key" in result
        assert "value" in result

    @pytest.mark.asyncio
    async def test_python_tool_with_locals(self):
        """Test PythonTool with initial locals."""
        def helper_func(x):
            return x * 2

        py_tool = PythonTool(locals={"helper": helper_func})
        result = await py_tool.run.run(args={"code": "print(helper(21))"}, self_ref=py_tool)
        assert "42" in result

    @pytest.mark.asyncio
    async def test_python_tool_exception_handling(self):
        """Test PythonTool handles exceptions."""
        py_tool = PythonTool()

        with pytest.raises(Exception):
            await py_tool.run.run(args={"code": "raise ValueError('test error')"})


class TestPythonToolWithAgent:
    """Test PythonTool integrated with ReActAgent."""

    @pytest.mark.asyncio
    async def test_agent_with_python_tool_calculation(self, react_agent_with_tools):
        """Test agent can use PythonTool for calculations."""
        py_tool = PythonTool()
        agent = react_agent_with_tools([py_tool])

        result = await agent.react_async("Use Python to calculate the sum of numbers from 1 to 10")
        # Sum of 1 to 10 is 55
        assert "55" in result

    @pytest.mark.asyncio
    async def test_agent_with_python_tool_data_processing(self, react_agent_with_tools):
        """Test agent can use PythonTool for data processing."""
        py_tool = PythonTool()
        agent = react_agent_with_tools([py_tool])

        result = await agent.react_async(
            "Use Python to create a list of the first 5 even numbers (not including zero), then calculate their sum"
        )
        # First 5 even numbers: 2, 4, 6, 8, 10 -> sum = 30
        assert "30" in result

    @pytest.mark.asyncio
    async def test_agent_with_python_tool_string_manipulation(self, react_agent_with_tools):
        """Test agent can use PythonTool for string operations."""
        py_tool = PythonTool()
        agent = react_agent_with_tools([py_tool])

        result = await agent.react_async(
            "Use Python to reverse the string 'Archytas' and print it"
        )
        assert "satyhcrA" in result

    @pytest.mark.asyncio
    async def test_agent_python_tool_multiple_steps(self, react_agent_with_tools):
        """Test agent can use PythonTool across multiple steps."""
        py_tool = PythonTool()
        agent = react_agent_with_tools([py_tool])

        result = await agent.react_async(
            "Use Python to: 1) Create a variable x = 7, "
            "2) Create y = x * 6, "
            "3) Print the final value of y"
        )
        assert "42" in result


class TestPythonToolEdgeCases:
    """Test edge cases for PythonTool."""

    @pytest.mark.asyncio
    async def test_python_tool_multiline_code(self):
        """Test PythonTool with multiline code."""
        py_tool = PythonTool()
        code = """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

print(factorial(5))
"""
        result = await py_tool.run.run(args={"code": code}, self_ref=py_tool)
        assert "120" in result

    @pytest.mark.asyncio
    async def test_python_tool_no_output(self):
        """Test PythonTool with code that produces no output."""
        py_tool = PythonTool()
        result = await py_tool.run.run(args={"code": "x = 5"}, self_ref=py_tool)
        # Should return empty or minimal output
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_python_tool_with_comprehensions(self):
        """Test PythonTool with list comprehensions."""
        py_tool = PythonTool()
        result = await py_tool.run.run(args={
            "code": "print([x**2 for x in range(5)])"
        }, self_ref=py_tool)
        assert "0" in result
        assert "1" in result
        assert "4" in result
        assert "9" in result
        assert "16" in result
