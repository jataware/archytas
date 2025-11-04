# MCP Tools Quick Start Guide

## Overview

Archytas supports the Model Context Protocol (MCP) via langchain-mcp-adapters. You can:
1. Create MCP servers with tools
2. Connect to them (local or remote)
3. Use the tools in Archytas agents

## 1. Creating an MCP Server

Use **FastMCP** to create MCP servers. Tools must return MCP types (`TextContent`, `ImageContent`).

```python
# my_server.py
from fastmcp import FastMCP
from mcp.types import TextContent, ImageContent
import base64

mcp = FastMCP("my-server")

@mcp.tool()
def my_text_tool(param: str) -> list[TextContent]:
    """Tool description here."""
    result = f"Processed: {param}"
    return [TextContent(type="text", text=result)]

@mcp.tool()
def my_image_tool() -> list[ImageContent]:
    """Generate an image."""
    # Your image generation logic
    image_base64 = generate_image()
    return [ImageContent(
        type="image",
        data=image_base64,
        mimeType="image/png"
    )]

if __name__ == "__main__":
    mcp.run()
```

**Key Points:**
- Use `@mcp.tool()` decorator
- Return `list[TextContent]` for text
- Return `list[ImageContent]` for images
- Can mix text and images in return list
- Install FastMCP: `pip install fastmcp`

## 2. Using MCP Tools in Archytas

### Option A: Single Server (Simple)

```python
import asyncio
import sys
from archytas.mcp_tools import mcp_tool_async
from archytas.react import ReActAgent

async def main():
    # Load tools from MCP server
    tools = await mcp_tool_async(
        server_name="myserver",
        command=[sys.executable, "my_server.py"]
    )

    # Create agent with tools
    agent = ReActAgent(model="gpt-4o-mini", tools=tools)

    # Use the agent
    response = await agent.react_async("Your task here")
    print(response)

asyncio.run(main())
```

### Option B: Multiple Servers

```python
from archytas.mcp_tools import MCPClient
from archytas.react import ReActAgent

async def main():
    # Create client with multiple servers
    client = MCPClient({
        "local": {
            "transport": "stdio",
            "command": ["python", "my_server.py"]
        },
        "api": {
            "transport": "streamable_http",
            "url": "https://api.example.com/mcp",
            "headers": {"API_KEY": "secret"}
        },
        "search": {
            "transport": "stdio",
            "command": ["npx", "-y", "duckduckgo-mcp-server"]
        }
    })

    # Get all tools
    tools = await client.get_tools()

    # Or get from specific server
    local_tools = await client.get_tools(server_name="local")

    # Create agent
    agent = ReActAgent(model="gpt-4o-mini", tools=tools)
    response = await agent.react_async("Your task")
```

### Option C: Mix with Native Tools

```python
from archytas.tool_utils import tool
from archytas.mcp_tools import mcp_tool_async

# Native Archytas tool
@tool()
def my_native_tool(x: int) -> str:
    """A native tool."""
    return f"Result: {x * 2}"

# Get MCP tools
mcp_tools = await mcp_tool_async(
    server_name="myserver",
    command=["python", "my_server.py"]
)

# Combine them
all_tools = [my_native_tool, *mcp_tools]

# Create agent with both
agent = ReActAgent(model="gpt-4o-mini", tools=all_tools)
```

## 3. Transport Types

MCP supports 4 transport types:

### Stdio (Local Process)
```python
tools = await mcp_tool_async(
    server_name="local",
    command=["python", "server.py"]
)
```

### Streamable HTTP
```python
tools = await mcp_tool_async(
    server_name="api",
    url="https://api.example.com/mcp",
    headers={"API_KEY": "secret"}
)
```

### WebSocket
```python
tools = await mcp_tool_async(
    server_name="realtime",
    url="ws://localhost:9000"
)
```

### SSE (Server-Sent Events)
```python
tools = await mcp_tool_async(
    server_name="sse",
    url="https://api.example.com/events",
    transport="sse",
    headers={"Authorization": "Bearer token"}
)
```

## 4. Tool Filtering

Only load specific tools:

```python
tools = await mcp_tool_async(
    server_name="myserver",
    command=["python", "server.py"],
    tools=["tool1", "tool2"]  # Only load these
)
```

## 5. Installation

```bash
# Install Archytas with MCP support (includes FastMCP for creating servers)
uv pip install -e ".[mcp]"

# Or with pip
pip install "archytas[mcp]"

# The [mcp] extra includes:
# - langchain-mcp-adapters (for connecting to MCP servers)
# - fastmcp (for creating your own MCP servers)
```

## 6. Vision Support

MCP tools can return images that agents can **actually see**:

```python
# In your MCP server
@mcp.tool()
def screenshot() -> list[ImageContent]:
    """Take a screenshot."""
    img_base64 = take_screenshot()
    return [ImageContent(
        type="image",
        data=img_base64,
        mimeType="image/png"
    )]

# Agent can see and describe the image
agent = ReActAgent(model="gpt-4o", tools=mcp_tools)
response = await agent.react_async("Take a screenshot and describe what you see")
# Agent will actually describe the visual content!
```

## 7. Common Patterns

### Error Handling
```python
try:
    tools = await mcp_tool_async(
        server_name="myserver",
        command=["python", "server.py"]
    )
except Exception as e:
    print(f"Failed to connect to MCP server: {e}")
```

### Dynamic Server Paths
```python
import sys
import os

server_path = os.path.join(os.path.dirname(__file__), "my_server.py")
tools = await mcp_tool_async(
    server_name="myserver",
    command=[sys.executable, server_path]
)
```

### Environment Variables
```python
# Pass env vars to stdio MCP server
tools = await mcp_tool_async(
    server_name="myserver",
    command=["python", "server.py"],
    env={"API_KEY": "secret", "DEBUG": "true"}
)
```

## 8. Examples in Repo

- `example_mcp_server.py` - Sample MCP server
- `example_using_mcp_tools.py` - Usage examples
- `demos/demo_mcp_vision.py` - Vision demo with image generation
- `tests/fixtures/mock_mcp_server.py` - Mock server for testing

## 9. Troubleshooting

**MCP dependencies not installed:**
```
ImportError: MCP dependencies not installed.
Install with: pip install 'archytas[mcp]'
```

**Tool not found:**
- Check tool name matches exactly
- Verify server is running
- Use `tools=None` to get all tools first

**Connection failed:**
- Verify command/path is correct
- Check server script runs standalone
- Look for errors in server output

## Quick Reference

| Task | Code |
|------|------|
| Single server | `tools = await mcp_tool_async(server_name="x", command=[...])` |
| Multiple servers | `client = MCPClient({...}); tools = await client.get_tools()` |
| HTTP server | `tools = await mcp_tool_async(server_name="x", url="https://...")` |
| Filter tools | `tools = await mcp_tool_async(..., tools=["tool1", "tool2"])` |
| Mix with native | `all_tools = [native_tool, *mcp_tools]` |
| Create agent | `agent = ReActAgent(model="gpt-4o-mini", tools=tools)` |

---

**Architecture**: Archytas → langchain-mcp-adapters → MCP Server

**Dependencies**: `langchain-mcp-adapters>=0.1.0`

**Documentation**: See `MCP_TOOLS_IMPLEMENTATION_SUMMARY.md` for details
