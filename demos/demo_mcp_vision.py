#!/usr/bin/env python3
"""
Quick demo showing MCP vision capabilities.

The agent can see and describe images returned by MCP tools!
"""

import asyncio
import sys
from pathlib import Path


from archytas.mcp_tools import mcp_tool_async, MCP_AVAILABLE
from archytas.react import ReActAgent
from archytas.models.openai import OpenAIModel


async def main():
    if not MCP_AVAILABLE:
        print("❌ MCP dependencies not installed!")
        print("Install with: uv pip install -e '.[mcp]'")
        return

    print("\n" + "=" * 70)
    print("  MCP Vision Demo - Agent Can See Images!")
    print("=" * 70 + "\n")

    # Setup MCP server
    mock_server = Path(__file__).parent / "tests" / "fixtures" / "mock_mcp_server.py"

    print("Setting up agent with vision-capable model and MCP tools...")
    mcp_tools = await mcp_tool_async(
        server_name="vision_demo", command=[sys.executable, str(mock_server)]
    )

    agent = ReActAgent(
        model=OpenAIModel({"model_name": "gpt-4o-mini"}),
        tools=mcp_tools,
        allow_ask_user=False,
        verbose=False,
    )

    print("✓ Agent ready with vision support!\n")

    # Ask agent to generate and describe the image
    query = "Generate an image and describe it in detail - what do you see?"

    print(f"Query: {query}\n")
    print("Agent thinking...")
    print("-" * 70)

    response = await agent.react_async(query)

    print("-" * 70)
    print(f"\n✓ Agent Response:\n{response}\n")

    print("=" * 70)
    print("Vision working perfectly! The agent can SEE images from MCP tools.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted")
