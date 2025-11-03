#!/usr/bin/env python3
"""
Demo using external MCP server: DuckDuckGo Search.

This demonstrates using a published npm MCP package.
The server is automatically started and managed by Archytas.

Installation:
    Requires Node.js/npm installed on your system.
    The MCP server is installed automatically via npx.

Usage:
    python demos/demo_mcp_duckduckgo.py
"""

import asyncio


from archytas.mcp_tools import mcp_tool_async, MCP_AVAILABLE
from archytas.react import ReActAgent
from archytas.models.openai import OpenAIModel


async def main():
    if not MCP_AVAILABLE:
        print("❌ MCP dependencies not installed!")
        print("Install with: uv pip install -e '.[mcp]'")
        return

    print("\n" + "=" * 70)
    print("  External MCP Server Demo - DuckDuckGo Search")
    print("=" * 70 + "\n")

    print("Setting up DuckDuckGo MCP server...")
    print("(Note: This will download the npm package via npx on first run)\n")

    # Archytas automatically runs and manages the MCP server
    try:
        duckduckgo_tools = await mcp_tool_async(
            server_name="duckduckgo", command=["npx", "-y", "duckduckgo-mcp-server"]
        )
        print(f"✓ Registered {len(duckduckgo_tools)} DuckDuckGo tools\n")

        # Show available tools
        print("Available tools:")
        for tool in duckduckgo_tools:
            print(f"  - {tool.name}")
        print()

    except Exception as e:
        print(f"❌ Failed to setup DuckDuckGo MCP server: {e}")
        print("\nMake sure Node.js and npm are installed:")
        print("  - Check with: node --version && npm --version")
        return

    # Create agent with DuckDuckGo tools
    print("Creating agent with vision-capable model...")
    agent = ReActAgent(
        model=OpenAIModel({"model_name": "gpt-4o-mini"}),
        tools=duckduckgo_tools,
        allow_ask_user=False,
        verbose=True,
    )
    print("✓ Agent ready!\n")

    # Test query
    query = "What are the latest developments in AI agents?"

    print(f"Query: {query}\n")
    print("Agent searching...")
    print("-" * 70)

    response = await agent.react_async(query)

    print("-" * 70)
    print(f"\n✓ Agent Response:\n{response}\n")

    print("=" * 70)
    print("External MCP server working perfectly!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted")
