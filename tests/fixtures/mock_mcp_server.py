"""Mock MCP server for testing."""

import base64
from pathlib import Path

from fastmcp import FastMCP
from mcp.types import TextContent, ImageContent

mcp = FastMCP("test-server")


@mcp.tool()
def echo(message: str) -> list[TextContent]:
    """
    Echo back a message.

    Args:
        message: Message to echo
    """
    return [TextContent(type="text", text=f"Echo: {message}")]


@mcp.tool()
def add(a: int, b: int) -> list[TextContent]:
    """
    Add two numbers.

    Args:
        a: First number
        b: Second number
    """
    return [TextContent(type="text", text=str(a + b))]


@mcp.tool()
def generate_image() -> list[ImageContent]:
    """Generate a cute pony image."""
    # Read the cute pony image from the fixtures directory
    image_path = Path(__file__).parent / "cute_pony.jpg"
    image_data = base64.b64encode(image_path.read_bytes()).decode('utf-8')

    return [ImageContent(
        type="image",
        data=image_data,
        mimeType="image/jpeg"
    )]


if __name__ == "__main__":
    mcp.run()
