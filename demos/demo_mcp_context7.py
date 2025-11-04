#!/usr/bin/env python3
"""
Demo using HTTP-based MCP server: Context7 Documentation API.

This demonstrates using an HTTP MCP server instead of stdio-based.
Context7 provides documentation lookup for various programming libraries.

Installation:
    1. Get a Context7 API key from https://context7.com
    2. Set the CONTEXT7_API_KEY environment variable

Usage:
    export CONTEXT7_API_KEY="your-api-key-here"
    python demos/demo_mcp_context7.py
"""

import asyncio
import os

from archytas.mcp_tools import mcp_tool_async, MCP_AVAILABLE
from archytas.react import ReActAgent
from archytas.models.openai import OpenAIModel


async def main():
    if not MCP_AVAILABLE:
        print("❌ MCP dependencies not installed!")
        print("Install with: uv pip install -e '.[mcp]'")
        return

    print("\n" + "=" * 70)
    print("  HTTP MCP Server Demo - Context7 Documentation")
    print("=" * 70 + "\n")

    # Check for API key
    api_key = os.environ.get("CONTEXT7_API_KEY")
    if not api_key:
        print("❌ CONTEXT7_API_KEY environment variable not set!")
        print("\nTo get started:")
        print("  1. Sign up at https://context7.com")
        print("  2. Get your API key")
        print("  3. Set environment variable:")
        print("     export CONTEXT7_API_KEY='your-api-key-here'")
        print(
            "\nFor this demo, you can skip Context7 and try the DuckDuckGo demo instead:"
        )
        print("  python demos/demo_mcp_duckduckgo.py")
        return

    print("Setting up Context7 HTTP MCP server...")

    # Connect to HTTP-based MCP server (using streamable_http transport)
    try:
        context7_tools = await mcp_tool_async(
            server_name="context7",
            url="https://mcp.context7.com/mcp",
            transport="streamable_http",  # Use streamable HTTP (not SSE)
            headers={"CONTEXT7_API_KEY": api_key},
        )
        print(f"✓ Registered {len(context7_tools)} Context7 tools\n")

        # Show available tools
        print("Available tools:")
        for tool in context7_tools:
            print(f"  - {tool.__name__}")
        print()

    except Exception as e:
        print(f"❌ Failed to setup Context7 HTTP MCP server: {e}")
        print("\nPlease check:")
        print("  - Your API key is valid")
        print("  - You have internet connectivity")
        print("  - The Context7 service is available")
        return

    # Create agent with Context7 tools
    print("Creating agent with vision-capable model...")
    agent = ReActAgent(
        model=OpenAIModel({"model_name": "gpt-4o-mini"}),
        tools=context7_tools,
        allow_ask_user=False,
        verbose=True,
    )
    print("✓ Agent ready!\n")

    # Test query
    query = "Look up the documentation for the Python requests library, specifically how to make POST requests with JSON data"

    print(f"Query: {query}\n")
    print("Agent searching...")
    print("-" * 70)

    response = await agent.react_async(query)

    print("-" * 70)
    print(f"\n✓ Agent Response:\n{response}\n")

    print("=" * 70)
    print("HTTP MCP server working perfectly!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted")
