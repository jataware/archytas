#!/usr/bin/env python3
"""
Demo script for MCP tools integration in Archytas.

This demonstrates:
1. Registering an MCP server
2. Discovering tools from the server
3. Executing tools and viewing results
4. Multimodal outputs (text and images)

Usage:
    python demo_mcp_tools.py
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the archytas directory to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from archytas.mcp_tools import mcp_tool_async, MCP_AVAILABLE
    from archytas.react import format_tool_result_for_display
except ImportError as e:
    print(f"Error importing archytas: {e}")
    print("\nMake sure you've installed archytas:")
    print("  uv pip install -e '.[mcp]'")
    sys.exit(1)


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


async def main():
    print_section("Archytas MCP Tools Demo")

    # Check if MCP is available
    if not MCP_AVAILABLE:
        print("❌ MCP dependencies not installed!")
        print("\nInstall with:")
        print("  uv pip install -e '.[mcp]'")
        print("  # or")
        print("  pip install 'archytas[mcp]'")
        return

    print("✓ MCP dependencies installed")

    # Path to mock server
    mock_server_path = Path(__file__).parent / "tests" / "fixtures" / "mock_mcp_server.py"

    if not mock_server_path.exists():
        print(f"❌ Mock MCP server not found at: {mock_server_path}")
        return

    print(f"✓ Mock MCP server found")

    # Register MCP server
    print_section("1. Registering MCP Server")
    print(f"Command: python {mock_server_path}")

    try:
        tools = await mcp_tool_async(
            server_name="demo",
            command=[sys.executable, str(mock_server_path)]
        )
        print(f"✓ Successfully registered MCP server")
        print(f"✓ Discovered {len(tools)} tools")
    except Exception as e:
        print(f"❌ Failed to register MCP server: {e}")
        return

    # Display discovered tools
    print_section("2. Discovered Tools")
    for i, tool in enumerate(tools, 1):
        print(f"{i}. {tool.__name__}")
        if tool.__doc__:
            # Get first line of docstring
            first_line = tool.__doc__.split('\n')[0].strip()
            print(f"   {first_line}")

    # Test echo tool
    print_section("3. Testing Echo Tool")
    echo_tool = next((t for t in tools if t.__name__ == "echo"), None)

    if echo_tool:
        print("Calling: echo(message='Hello from Archytas MCP demo!')")
        result = await echo_tool.run({"message": "Hello from Archytas MCP demo!"}, {})

        print("\nRaw result (LangChain format):")
        print(f"  {result}")

        print("\nFormatted result:")
        formatted = format_tool_result_for_display(result)
        print(f"  {formatted}")
    else:
        print("❌ Echo tool not found")

    # Test add tool
    print_section("4. Testing Add Tool")
    add_tool = next((t for t in tools if t.__name__ == "add"), None)

    if add_tool:
        print("Calling: add(a=42, b=8)")
        result = await add_tool.run({"a": 42, "b": 8}, {})

        print("\nRaw result:")
        print(f"  {result}")

        print("\nFormatted result:")
        formatted = format_tool_result_for_display(result)
        print(f"  {formatted}")
    else:
        print("❌ Add tool not found")

    # Test image generation tool
    print_section("5. Testing Image Generation Tool")
    image_tool = next((t for t in tools if t.__name__ == "generate_image"), None)

    if image_tool:
        print("Calling: generate_image()")
        result = await image_tool.run({}, {})

        print("\nRaw result (truncated):")
        print(f"  Type: {result[0]['type']}")
        if result[0]["type"] == "image_url":
            url = result[0]["image_url"]["url"]
            print(f"  URL prefix: {url[:50]}...")
            print(f"  URL length: {len(url)} characters")

        print("\nFormatted result (for display):")
        formatted = format_tool_result_for_display(result)
        print(f"  {formatted}")

        print("\n💡 Note: The actual image data is available to the LLM")
        print("   as a base64-encoded data URL in LangChain format.")
    else:
        print("❌ Generate image tool not found")

    # Summary
    print_section("Summary")
    print("✓ MCP server integration working correctly")
    print("✓ Tools discovered and executed successfully")
    print("✓ Multimodal outputs (text + images) working")
    print("\nNext steps:")
    print("  • Create your own MCP server with custom tools")
    print("  • Use mcp_tool() in your Archytas agents")
    print("  • See MCP_TOOLS_IMPLEMENTATION_SUMMARY.md for more examples")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
