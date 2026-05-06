# ruff: noqa
# type: ignore
"""Tests for multimodal tool returns via ``MultiModalResponse``."""

import base64
import os

import pytest

from archytas.multimodal import MultiModalResponse, content_block_from_file
from archytas.tool_utils import tool


class TestMultimodalToolReturns:
    """Test that ``@tool``-decorated functions can return multimodal content."""

    @pytest.mark.asyncio
    async def test_string_return_backward_compatible(self):
        """String returns still work unchanged."""
        @tool()
        def my_tool() -> str:
            """Simple string tool."""
            return "Hello"

        result = await my_tool.run({}, {})
        assert result == "Hello"
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_multimodal_response_return(self):
        """A ``MultiModalResponse`` is unwrapped to its ``blocks`` list."""
        @tool()
        def my_tool() -> MultiModalResponse:
            """Multimodal tool."""
            return MultiModalResponse(blocks=[
                {"type": "image", "mime_type": "image/png", "base64": "abc"},
            ])

        result = await my_tool.run({}, {})
        assert isinstance(result, list)
        assert len(result) == 1
        block = result[0]
        assert block["type"] == "image"
        assert block["mime_type"] == "image/png"
        assert block["base64"] == "abc"

    @pytest.mark.asyncio
    async def test_multimodal_response_from_bytes(self):
        """``MultiModalResponse.from_bytes`` produces a single content block."""
        raw = b"\x89PNG\r\n\x1a\nfakepngbytes"

        @tool()
        def my_tool() -> MultiModalResponse:
            """Tool returning bytes-built multimodal content."""
            return MultiModalResponse.from_bytes(raw, "image/png")

        result = await my_tool.run({}, {})
        assert isinstance(result, list)
        assert len(result) == 1
        block = result[0]
        assert block["type"] == "image"
        assert block["mime_type"] == "image/png"
        # ``base64`` should round-trip back to the original bytes regardless of
        # whether the implementation stores ``str`` or ``bytes``.
        b64_value = block["base64"]
        if isinstance(b64_value, bytes):
            b64_value = b64_value.decode()
        assert base64.b64decode(b64_value) == raw

    @pytest.mark.asyncio
    async def test_multimodal_response_from_file(self, tmp_path):
        """``MultiModalResponse.from_file`` reads + encodes a file."""
        raw = b"fake-image-bytes"
        path = tmp_path / "img.png"
        path.write_bytes(raw)

        @tool()
        def my_tool() -> MultiModalResponse:
            """Tool returning file-built multimodal content."""
            return MultiModalResponse.from_file(str(path))

        result = await my_tool.run({}, {})
        assert isinstance(result, list)
        assert len(result) == 1
        block = result[0]
        assert block["type"] == "image"
        assert block["mime_type"] == "image/png"
        b64_value = block["base64"]
        if isinstance(b64_value, bytes):
            b64_value = b64_value.decode()
        assert base64.b64decode(b64_value) == raw

    @pytest.mark.asyncio
    async def test_multimodal_response_multiple_blocks(self):
        """Multiple blocks in a ``MultiModalResponse`` are preserved in order."""
        @tool()
        def my_tool() -> MultiModalResponse:
            """Tool returning multiple blocks."""
            return MultiModalResponse(blocks=[
                {"type": "image", "mime_type": "image/png", "base64": "aaa"},
                {"type": "image", "mime_type": "image/jpeg", "base64": "bbb"},
            ])

        result = await my_tool.run({}, {})
        assert len(result) == 2
        assert result[0]["mime_type"] == "image/png"
        assert result[1]["mime_type"] == "image/jpeg"

    @pytest.mark.asyncio
    async def test_none_return_converts_to_empty_string(self):
        """``None`` returns convert to empty string."""
        @tool()
        def none_tool():
            """Tool that returns None."""
            return None

        result = await none_tool.run({}, {})
        assert result == ""

    @pytest.mark.asyncio
    async def test_numeric_return_converts_to_string(self):
        """Numeric returns convert to string."""
        @tool()
        def number_tool() -> int:
            """Tool that returns a number."""
            return 42

        result = await number_tool.run({}, {})
        assert result == "42"


class TestContentBlockFromFile:
    """Tests for the ``content_block_from_file`` helper."""

    def test_reads_file_and_guesses_mimetype(self, tmp_path):
        raw = b"hello-bytes"
        path = tmp_path / "data.png"
        path.write_bytes(raw)

        block = content_block_from_file(str(path))
        assert block["type"] == "image"
        assert block["mime_type"] == "image/png"
        b64_value = block["base64"]
        if isinstance(b64_value, bytes):
            b64_value = b64_value.decode()
        assert base64.b64decode(b64_value) == raw

    def test_explicit_mimetype_override(self, tmp_path):
        path = tmp_path / "data.bin"
        path.write_bytes(b"x")
        block = content_block_from_file(str(path), mimetype="application/pdf")
        assert block["type"] == "application"
        assert block["mime_type"] == "application/pdf"

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            content_block_from_file(str(tmp_path / "does_not_exist.png"))

    def test_undeterminable_mimetype_raises(self, tmp_path):
        path = tmp_path / "no_extension"
        path.write_bytes(b"x")
        with pytest.raises(ValueError, match="mime-type"):
            content_block_from_file(str(path))
