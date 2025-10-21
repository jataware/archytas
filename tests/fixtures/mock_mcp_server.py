"""Mock MCP server for testing."""

from fastmcp import FastMCP

mcp = FastMCP("test-server")


@mcp.tool()
def echo(message: str) -> dict:
    """
    Echo back a message.

    Args:
        message: Message to echo
    """
    return {
        "content": [{
            "type": "text",
            "text": f"Echo: {message}"
        }]
    }


@mcp.tool()
def add(a: int, b: int) -> dict:
    """
    Add two numbers.

    Args:
        a: First number
        b: Second number
    """
    return {
        "content": [{
            "type": "text",
            "text": str(a + b)
        }]
    }


@mcp.tool()
def generate_image() -> dict:
    """Generate a simple test image."""
    # 1x1 red pixel PNG
    return {
        "content": [{
            "type": "image",
            "data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg==",
            "mimeType": "image/png"
        }]
    }


if __name__ == "__main__":
    mcp.run()
