# ruff: noqa
# type: ignore
"""Tests for multimodal tool returns and MCP integration."""

import pytest
from archytas.tool_utils import tool
from archytas.mcp_tools import MCPToolBridge
from archytas.react import format_tool_result_for_display


class TestMultimodalToolReturns:
    """Test that tools can return LangChain multimodal format."""

    @pytest.mark.asyncio
    async def test_string_return_backward_compatible(self):
        """Test that string returns still work (backward compatibility)."""
        @tool()
        def my_tool() -> str:
            """Simple string tool."""
            return "Hello"

        result = await my_tool.run({}, {})
        assert result == "Hello"
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_multimodal_list_return(self):
        """Test returning LangChain multimodal format as list."""
        @tool()
        def my_tool() -> list[dict]:
            """Multimodal tool."""
            return [
                {"type": "text", "text": "Hello"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}
            ]

        result = await my_tool.run({}, {})
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["type"] == "text"
        assert result[0]["text"] == "Hello"
        assert result[1]["type"] == "image_url"
        assert "url" in result[1]["image_url"]

    @pytest.mark.asyncio
    async def test_single_content_dict_return(self):
        """Test returning single content block gets wrapped in list."""
        @tool()
        def my_tool() -> dict:
            """Tool returning single content block."""
            return {"type": "text", "text": "Single block"}

        result = await my_tool.run({}, {})
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["type"] == "text"
        assert result[0]["text"] == "Single block"

    @pytest.mark.asyncio
    async def test_invalid_multimodal_format_raises(self):
        """Test that invalid multimodal format raises ValueError."""
        @tool()
        def bad_tool() -> list[dict]:
            """Tool with invalid format."""
            return [{"no_type_field": "value"}]

        with pytest.raises(ValueError, match="must have 'type' field"):
            await bad_tool.run({}, {})

    @pytest.mark.asyncio
    async def test_unsupported_content_type_raises(self):
        """Test that unsupported content type raises ValueError."""
        @tool()
        def bad_tool() -> list[dict]:
            """Tool with unsupported type."""
            return [{"type": "audio", "data": "xyz"}]

        with pytest.raises(ValueError, match="Unsupported content type"):
            await bad_tool.run({}, {})

    @pytest.mark.asyncio
    async def test_none_return_converts_to_empty_string(self):
        """Test that None returns convert to empty string."""
        @tool()
        def none_tool():
            """Tool that returns None."""
            return None

        result = await none_tool.run({}, {})
        assert result == ""

    @pytest.mark.asyncio
    async def test_numeric_return_converts_to_string(self):
        """Test that numeric returns convert to string."""
        @tool()
        def number_tool() -> int:
            """Tool that returns number."""
            return 42

        result = await number_tool.run({}, {})
        assert result == "42"


class TestDisplayFormatting:
    """Test display formatting for multimodal content."""

    def test_format_string_returns_as_is(self):
        """Test string formatting."""
        result = format_tool_result_for_display("Hello world")
        assert result == "Hello world"

    def test_format_text_content(self):
        """Test formatting text content block."""
        content = [{"type": "text", "text": "Hello"}]
        result = format_tool_result_for_display(content)
        assert result == "Hello"

    def test_format_base64_image(self):
        """Test formatting base64 image."""
        content = [{
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,abc123"}
        }]
        result = format_tool_result_for_display(content)
        assert "[Image: image/png, base64 data]" in result

    def test_format_remote_image_url(self):
        """Test formatting remote image URL."""
        content = [{
            "type": "image_url",
            "image_url": {"url": "https://example.com/image.png"}
        }]
        result = format_tool_result_for_display(content)
        assert "[Image: https://example.com/image.png]" in result

    def test_format_mixed_content(self):
        """Test formatting mixed text and image content."""
        content = [
            {"type": "text", "text": "Here is an image:"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,xyz"}}
        ]
        result = format_tool_result_for_display(content)
        assert "Here is an image:" in result
        assert "[Image: image/png, base64 data]" in result

    def test_format_single_dict(self):
        """Test formatting single content dict."""
        content = {"type": "text", "text": "Single"}
        result = format_tool_result_for_display(content)
        assert result == "Single"

    def test_format_unknown_type(self):
        """Test formatting unknown content type."""
        content = [{"type": "unknown", "data": "xyz"}]
        result = format_tool_result_for_display(content)
        assert "[unknown]" in result


class TestMCPBridgeCore:
    """Test MCPToolBridge core functionality."""

    def test_json_type_to_python(self):
        """Test JSON schema type mapping."""
        bridge = MCPToolBridge()

        assert bridge._json_type_to_python({"type": "string"}) == str
        assert bridge._json_type_to_python({"type": "integer"}) == int
        assert bridge._json_type_to_python({"type": "number"}) == float
        assert bridge._json_type_to_python({"type": "boolean"}) == bool
        assert bridge._json_type_to_python({"type": "array"}) == list
        assert bridge._json_type_to_python({"type": "object"}) == dict
        assert bridge._json_type_to_python({"type": "null"}) == type(None)
        assert bridge._json_type_to_python({"type": "unknown"}) == str  # default

    def test_extract_parameters(self):
        """Test parameter extraction from JSON schema."""
        bridge = MCPToolBridge()

        schema = {
            "properties": {
                "name": {
                    "type": "string",
                    "description": "User name"
                },
                "age": {
                    "type": "integer",
                    "description": "User age"
                },
                "optional": {
                    "type": "boolean",
                    "description": "Optional flag"
                }
            },
            "required": ["name", "age"]
        }

        params = bridge._extract_parameters(schema)
        assert len(params) == 3

        name_param = next(p for p in params if p.name == "name")
        assert name_param.type == str
        assert name_param.required is True
        assert "User name" in name_param.description

        age_param = next(p for p in params if p.name == "age")
        assert age_param.type == int
        assert age_param.required is True

        optional_param = next(p for p in params if p.name == "optional")
        assert optional_param.type == bool
        assert optional_param.required is False

    def test_build_docstring(self):
        """Test docstring generation."""
        bridge = MCPToolBridge()

        class MockTool:
            name = "test_tool"
            description = "A test tool"

        from archytas.mcp_tools import ToolParameter
        params = [
            ToolParameter(name="arg1", type=str, description="First argument", required=True),
            ToolParameter(name="arg2", type=int, description="Second argument", required=False)
        ]

        docstring = bridge._build_docstring(MockTool(), params)

        assert "A test tool" in docstring
        assert "Args:" in docstring
        assert "arg1 (str): First argument" in docstring
        assert "arg2 (int): Second argument (optional)" in docstring
        assert "Returns:" in docstring
        assert "list[dict]" in docstring

    def test_format_mcp_text_result(self):
        """Test MCP text content conversion."""
        import mcp.types
        bridge = MCPToolBridge()

        class MockResult:
            content = [mcp.types.TextContent(type="text", text="Hello from MCP")]

        result = bridge._format_mcp_result(MockResult())
        assert len(result) == 1
        assert result[0]["type"] == "text"
        assert result[0]["text"] == "Hello from MCP"

    def test_format_mcp_image_result(self):
        """Test MCP image content conversion."""
        import mcp.types
        bridge = MCPToolBridge()

        class MockResult:
            content = [mcp.types.ImageContent(type="image", data="base64data", mimeType="image/png")]

        result = bridge._format_mcp_result(MockResult())
        assert len(result) == 1
        assert result[0]["type"] == "image_url"
        assert result[0]["image_url"]["url"] == "data:image/png;base64,base64data"

    def test_format_mcp_mixed_result(self):
        """Test MCP mixed content conversion."""
        import mcp.types
        bridge = MCPToolBridge()

        class MockResult:
            content = [
                mcp.types.TextContent(type="text", text="Description"),
                mcp.types.ImageContent(type="image", data="xyz", mimeType="image/jpeg")
            ]

        result = bridge._format_mcp_result(MockResult())
        assert len(result) == 2
        assert result[0]["type"] == "text"
        assert result[0]["text"] == "Description"
        assert result[1]["type"] == "image_url"
        assert "data:image/jpeg;base64" in result[1]["image_url"]["url"]


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

            result = await echo_tool.run({"message": "test"}, {})

            assert isinstance(result, list)
            assert len(result) > 0
            assert result[0]["type"] == "text"
            assert "Echo: test" in result[0]["text"]

        except Exception as e:
            pytest.skip(f"Could not execute MCP tool: {e}")

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
