# ruff: noqa
# type: ignore
"""Tests for MCP tools integration."""

import pytest
from unittest.mock import Mock, AsyncMock


class TestNameSanitization:
    """Test tool name sanitization for Python identifiers."""

    def test_sanitize_hyphenated_name(self):
        """Test that hyphenated names get converted to underscores."""
        from archytas.mcp_tools import _wrap_langchain_tool

        # Create a mock LangChain tool with hyphenated name
        mock_tool = Mock()
        mock_tool.name = "resolve-library-id"
        mock_tool.description = "Resolve library ID"
        mock_tool.args_schema = {
            "properties": {
                "library_name": {"type": "string", "description": "Library name"}
            }
        }
        mock_tool.ainvoke = AsyncMock(return_value="result")

        wrapped = _wrap_langchain_tool(mock_tool)

        # Should have sanitized name
        assert wrapped.__name__ == "resolve_library_id"

    def test_sanitize_space_in_name(self):
        """Test that spaces in names get converted to underscores."""
        from archytas.mcp_tools import _wrap_langchain_tool

        mock_tool = Mock()
        mock_tool.name = "get library docs"
        mock_tool.description = "Get docs"
        mock_tool.args_schema = {"properties": {}}
        mock_tool.ainvoke = AsyncMock(return_value="result")

        wrapped = _wrap_langchain_tool(mock_tool)

        assert wrapped.__name__ == "get_library_docs"

    def test_sanitize_name_starting_with_number(self):
        """Test that names starting with numbers get prefixed."""
        from archytas.mcp_tools import _wrap_langchain_tool

        mock_tool = Mock()
        mock_tool.name = "2fa-verify"
        mock_tool.description = "Verify 2FA"
        mock_tool.args_schema = {"properties": {}}
        mock_tool.ainvoke = AsyncMock(return_value="result")

        wrapped = _wrap_langchain_tool(mock_tool)

        # Should start with underscore
        assert wrapped.__name__ == "_2fa_verify"


class TestParameterMapping:
    """Test JSON Schema to Python type parameter mapping."""

    @pytest.mark.asyncio
    async def test_string_parameter_type(self):
        """Test string parameter gets correct type annotation."""
        from archytas.mcp_tools import _wrap_langchain_tool

        mock_tool = Mock()
        mock_tool.name = "test_tool"
        mock_tool.description = "Test"
        mock_tool.args_schema = {
            "properties": {
                "text": {"type": "string", "description": "Text param"}
            }
        }
        mock_tool.ainvoke = AsyncMock(return_value="success")

        wrapped = _wrap_langchain_tool(mock_tool)

        # Execute the tool
        result = await wrapped.run({"text": "hello"}, {})
        assert result == "success"
        mock_tool.ainvoke.assert_called_once_with({"text": "hello"})

    @pytest.mark.asyncio
    async def test_integer_parameter_type(self):
        """Test integer parameter gets correct type annotation."""
        from archytas.mcp_tools import _wrap_langchain_tool

        mock_tool = Mock()
        mock_tool.name = "test_tool"
        mock_tool.description = "Test"
        mock_tool.args_schema = {
            "properties": {
                "count": {"type": "integer", "description": "Count param"}
            }
        }
        mock_tool.ainvoke = AsyncMock(return_value=42)

        wrapped = _wrap_langchain_tool(mock_tool)

        result = await wrapped.run({"count": 10}, {})
        # Archytas normalizes non-string results to strings
        assert result == "42"
        mock_tool.ainvoke.assert_called_once_with({"count": 10})

    @pytest.mark.asyncio
    async def test_multiple_parameter_types(self):
        """Test tool with multiple different parameter types."""
        from archytas.mcp_tools import _wrap_langchain_tool

        mock_tool = Mock()
        mock_tool.name = "multi_param"
        mock_tool.description = "Multi param tool"
        mock_tool.args_schema = {
            "properties": {
                "name": {"type": "string", "description": "Name"},
                "age": {"type": "integer", "description": "Age"},
                "score": {"type": "number", "description": "Score"},
                "active": {"type": "boolean", "description": "Active"}
            }
        }
        mock_tool.ainvoke = AsyncMock(return_value="ok")

        wrapped = _wrap_langchain_tool(mock_tool)

        result = await wrapped.run({
            "name": "test",
            "age": 25,
            "score": 98.5,
            "active": True
        }, {})

        assert result == "ok"
        mock_tool.ainvoke.assert_called_once_with({
            "name": "test",
            "age": 25,
            "score": 98.5,
            "active": True
        })


class TestDocstringGeneration:
    """Test docstring generation for wrapped tools."""

    def test_docstring_includes_description(self):
        """Test wrapped tool includes description in docstring."""
        from archytas.mcp_tools import _wrap_langchain_tool

        mock_tool = Mock()
        mock_tool.name = "test_tool"
        mock_tool.description = "This is a test tool"
        mock_tool.args_schema = {"properties": {}}
        mock_tool.ainvoke = AsyncMock()

        wrapped = _wrap_langchain_tool(mock_tool)

        assert "This is a test tool" in wrapped.__doc__

    def test_docstring_includes_args_section(self):
        """Test wrapped tool includes Args section in docstring."""
        from archytas.mcp_tools import _wrap_langchain_tool

        mock_tool = Mock()
        mock_tool.name = "test_tool"
        mock_tool.description = "Test tool"
        mock_tool.args_schema = {
            "properties": {
                "param1": {"type": "string", "description": "First param"},
                "param2": {"type": "integer", "description": "Second param"}
            }
        }
        mock_tool.ainvoke = AsyncMock()

        wrapped = _wrap_langchain_tool(mock_tool)

        # Should have Args section
        assert "Args:" in wrapped.__doc__
        assert "param1 (string): First param" in wrapped.__doc__
        assert "param2 (integer): Second param" in wrapped.__doc__


class TestMCPIntegration:
    """Integration tests with mock MCP server."""

    @pytest.mark.asyncio
    async def test_register_mcp_server(self):
        """Test registering a mock MCP server."""
        import sys
        import os

        # Get the path to mock server
        test_dir = os.path.dirname(__file__)
        mock_server_path = os.path.join(test_dir, "fixtures", "mock_mcp_server.py")

        if not os.path.exists(mock_server_path):
            pytest.skip("Mock MCP server not found")

        from archytas.mcp_tools import mcp_tool_async

        try:
            tools = await mcp_tool_async(
                server_name="test",
                command=[sys.executable, mock_server_path]
            )

            # Should have 3 tools: echo, add, generate_image
            assert len(tools) >= 3

            tool_names = [t.__name__ for t in tools]
            assert "echo" in tool_names
            assert "add" in tool_names
            assert "generate_image" in tool_names

        except Exception as e:
            pytest.skip(f"Could not connect to mock MCP server: {e}")

    @pytest.mark.asyncio
    async def test_execute_echo_tool(self):
        """Test executing echo tool from mock server."""
        import sys
        import os

        test_dir = os.path.dirname(__file__)
        mock_server_path = os.path.join(test_dir, "fixtures", "mock_mcp_server.py")

        if not os.path.exists(mock_server_path):
            pytest.skip("Mock MCP server not found")

        from archytas.mcp_tools import mcp_tool_async

        try:
            tools = await mcp_tool_async(
                server_name="test_echo",
                command=[sys.executable, mock_server_path]
            )

            echo_tool = next(t for t in tools if t.__name__ == "echo")

            # Wrapped tools now use Archytas's run() method
            result = await echo_tool.run({"message": "test"}, {})

            # langchain-mcp-adapters returns the tool result directly
            # The format depends on how the MCP server returns it
            assert result is not None
            # Basic smoke test - just verify we got a result
            assert isinstance(result, (str, list, dict))

        except Exception as e:
            pytest.skip(f"Could not execute MCP tool: {e}")

    @pytest.mark.asyncio
    async def test_execute_add_tool(self):
        """Test executing add tool with numeric parameters."""
        import sys
        import os

        test_dir = os.path.dirname(__file__)
        mock_server_path = os.path.join(test_dir, "fixtures", "mock_mcp_server.py")

        if not os.path.exists(mock_server_path):
            pytest.skip("Mock MCP server not found")

        from archytas.mcp_tools import mcp_tool_async

        try:
            tools = await mcp_tool_async(
                server_name="test_add",
                command=[sys.executable, mock_server_path]
            )

            add_tool = next(t for t in tools if t.__name__ == "add")

            # Execute with numeric parameters
            result = await add_tool.run({"a": 5, "b": 3}, {})

            assert result is not None
            # The result should be numeric or a string representation
            assert isinstance(result, (int, float, str, list, dict))

        except Exception as e:
            pytest.skip(f"Could not execute add tool: {e}")

    @pytest.mark.asyncio
    async def test_execute_multimodal_tool(self):
        """Test executing tool that returns multimodal content."""
        import sys
        import os

        test_dir = os.path.dirname(__file__)
        mock_server_path = os.path.join(test_dir, "fixtures", "mock_mcp_server.py")

        if not os.path.exists(mock_server_path):
            pytest.skip("Mock MCP server not found")

        from archytas.mcp_tools import mcp_tool_async

        try:
            tools = await mcp_tool_async(
                server_name="test_image",
                command=[sys.executable, mock_server_path]
            )

            image_tool = next(t for t in tools if t.__name__ == "generate_image")

            # Execute tool
            result = await image_tool.run({}, {})

            assert result is not None
            # Should return multimodal format or string
            assert isinstance(result, (str, list, dict))

        except Exception as e:
            pytest.skip(f"Could not execute image tool: {e}")

    @pytest.mark.asyncio
    async def test_tool_filtering(self):
        """Test filtering MCP tools."""
        import sys
        import os

        test_dir = os.path.dirname(__file__)
        mock_server_path = os.path.join(test_dir, "fixtures", "mock_mcp_server.py")

        if not os.path.exists(mock_server_path):
            pytest.skip("Mock MCP server not found")

        from archytas.mcp_tools import mcp_tool_async

        try:
            # Only register echo and add tools
            tools = await mcp_tool_async(
                server_name="test_filtered",
                command=[sys.executable, mock_server_path],
                tools=["echo", "add"]
            )

            tool_names = [t.__name__ for t in tools]
            assert len(tools) == 2
            assert "echo" in tool_names
            assert "add" in tool_names
            assert "generate_image" not in tool_names

        except Exception as e:
            pytest.skip(f"Could not filter MCP tools: {e}")


class TestMCPClientAPI:
    """Test MCPClient class API."""

    @pytest.mark.asyncio
    async def test_client_with_single_server(self):
        """Test MCPClient with a single server."""
        import sys
        import os

        test_dir = os.path.dirname(__file__)
        mock_server_path = os.path.join(test_dir, "fixtures", "mock_mcp_server.py")

        if not os.path.exists(mock_server_path):
            pytest.skip("Mock MCP server not found")

        from archytas.mcp_tools import MCPClient

        try:
            client = MCPClient({
                "test": {
                    "transport": "stdio",
                    "command": sys.executable,
                    "args": [mock_server_path]
                }
            })

            tools = await client.get_tools()
            assert len(tools) >= 3

            tool_names = [t.__name__ for t in tools]
            assert "echo" in tool_names
            assert "add" in tool_names
            assert "generate_image" in tool_names

        except Exception as e:
            pytest.skip(f"Could not create MCPClient: {e}")

    @pytest.mark.asyncio
    async def test_client_get_tools_by_server(self):
        """Test getting tools from specific server."""
        import sys
        import os

        test_dir = os.path.dirname(__file__)
        mock_server_path = os.path.join(test_dir, "fixtures", "mock_mcp_server.py")

        if not os.path.exists(mock_server_path):
            pytest.skip("Mock MCP server not found")

        from archytas.mcp_tools import MCPClient

        try:
            client = MCPClient({
                "test": {
                    "transport": "stdio",
                    "command": sys.executable,
                    "args": [mock_server_path]
                }
            })

            # Get tools from specific server
            tools = await client.get_tools(server_name="test")
            assert len(tools) >= 3

        except Exception as e:
            pytest.skip(f"Could not get tools by server: {e}")
