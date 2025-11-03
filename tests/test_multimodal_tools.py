# ruff: noqa
# type: ignore
"""Tests for multimodal tool returns."""

import pytest
from archytas.tool_utils import tool
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
