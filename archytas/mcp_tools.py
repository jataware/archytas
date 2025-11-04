"""
MCP tool integration for Archytas agents.

Thin wrapper around langchain-mcp-adapters that provides Archytas-friendly API.
Uses the official LangChain MCP adapters library for all MCP protocol handling.

Basic Usage:
    from archytas.mcp_tools import MCPClient
    from archytas.react import ReActAgent

    # Create client with multiple servers
    client = MCPClient({
        "weather": {
            "transport": "stdio",
            "command": ["python", "weather_server.py"]
        },
        "docs": {
            "transport": "streamable_http",
            "url": "https://mcp.context7.com/mcp",
            "headers": {"CONTEXT7_API_KEY": "your-key"}
        }
    })

    # Get all tools from all servers
    tools = await client.get_tools()

    agent = ReActAgent(model="gpt-4o", tools=tools)
    response = await agent.react("What's the weather?")

Simple Usage (single server):
    from archytas.mcp_tools import mcp_tool_async

    tools = await mcp_tool_async(
        server_name="weather",
        command=["python", "weather_server.py"]
    )
"""

import asyncio
import logging
from typing import Callable, Literal

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import (
    MultiServerMCPClient,
    StdioConnection,
    SSEConnection,
    StreamableHttpConnection,
    WebsocketConnection,
)

from .tool_utils import tool

logger = logging.getLogger(__name__)

# Re-export connection types for convenience
__all__ = [
    "MCPClient",
    "mcp_tool",
    "mcp_tool_async",
    "StdioConnection",
    "SSEConnection",
    "StreamableHttpConnection",
    "WebsocketConnection",
]


def _wrap_langchain_tool(langchain_tool: "BaseTool") -> Callable:
    """
    Wrap a LangChain BaseTool to make it compatible with Archytas agents.

    LangChain StructuredTool objects don't support sync invocation, but Archytas
    expects tools to have a run() method. This wrapper creates an async function
    that calls the tool's ainvoke() method and decorates it with @tool.

    Args:
        langchain_tool: A LangChain BaseTool object (from langchain-mcp-adapters)

    Returns:
        An Archytas-compatible tool function with run() method
    """
    # Sanitize tool name to be a valid Python identifier
    # Replace hyphens and other invalid chars with underscores
    sanitized_name = langchain_tool.name.replace("-", "_").replace(" ", "_")
    # Ensure it starts with a letter or underscore
    if sanitized_name and not (sanitized_name[0].isalpha() or sanitized_name[0] == "_"):
        sanitized_name = "_" + sanitized_name

    # Get the tool's input schema (JSON Schema dict)
    args_schema = langchain_tool.args_schema

    # Create parameter specifications from the schema
    if args_schema and "properties" in args_schema:
        # Extract field names and types from JSON Schema
        params_code = []
        properties = args_schema.get("properties", {})

        for field_name, field_spec in properties.items():
            # Map JSON Schema types to Python type strings
            json_type = field_spec.get("type", "string")
            if json_type == "string":
                params_code.append(f"{field_name}: str")
            elif json_type == "integer":
                params_code.append(f"{field_name}: int")
            elif json_type == "number":
                params_code.append(f"{field_name}: float")
            elif json_type == "boolean":
                params_code.append(f"{field_name}: bool")
            else:
                # Default to str for unknown types
                params_code.append(f"{field_name}: str")

        params_str = ", ".join(params_code)
    else:
        # No schema - just use generic kwargs
        params_str = "**kwargs"

    # Build the wrapper function dynamically with proper signatures
    # Build kwargs dict from actual parameter names
    if args_schema and "properties" in args_schema:
        param_names = list(args_schema.get("properties", {}).keys())
        kwargs_builder = "{" + ", ".join([f"'{name}': {name}" for name in param_names]) + "}"
    else:
        kwargs_builder = "kwargs"

    func_code = f"""
async def {sanitized_name}({params_str}):
    # Build kwargs from actual parameters
    tool_kwargs = {kwargs_builder}

    # Call the LangChain tool
    result = await langchain_tool.ainvoke(tool_kwargs)
    # Return result as-is (can be str, list, dict for multimodal)
    return result
"""

    # Execute the code to create the function
    namespace = {"langchain_tool": langchain_tool}
    exec(func_code, namespace)
    wrapper = namespace[sanitized_name]

    # Build a proper docstring with Args section for Archytas
    docstring_parts = [langchain_tool.description or "MCP tool"]

    if args_schema and "properties" in args_schema:
        docstring_parts.append("\n\nArgs:")
        properties = args_schema.get("properties", {})
        for field_name, field_spec in properties.items():
            field_desc = field_spec.get("description", "No description")
            field_type = field_spec.get("type", "string")
            docstring_parts.append(f"    {field_name} ({field_type}): {field_desc}")

    wrapper.__doc__ = "\n".join(docstring_parts)

    # Apply Archytas @tool decorator
    archytas_tool = tool(wrapper)

    return archytas_tool


class MCPClient:
    """
    Client for connecting to multiple MCP servers.

    Thin wrapper around langchain-mcp-adapters' MultiServerMCPClient that provides
    an Archytas-friendly API. All MCP protocol handling is delegated to the official
    LangChain MCP adapters library.

    Example:
        client = MCPClient({
            "weather": {
                "transport": "stdio",
                "command": "python",
                "args": ["weather_server.py"]
            },
            "docs": {
                "transport": "streamable_http",
                "url": "https://api.example.com/mcp",
                "headers": {"API_KEY": "secret"}
            }
        })
        tools = await client.get_tools()
    """

    def __init__(self, connections: dict | None = None):
        """
        Initialize MCPClient with server connections.

        Args:
            connections: Dict mapping server names to connection configs.
                Connection configs are TypedDicts from langchain-mcp-adapters:
                - StdioConnection
                - SSEConnection
                - StreamableHttpConnection
                - WebsocketConnection
        """

        # Delegate to langchain-mcp-adapters
        self._client = MultiServerMCPClient(connections)

    async def get_tools(self, server_name: str | None = None) -> list[Callable]:
        """
        Get tools from one or all servers.

        Tools are wrapped to be compatible with Archytas agents' execution model.

        Args:
            server_name: Optional server name. If None, returns all tools.

        Returns:
            List of Archytas-compatible tool functions
        """
        logger.debug("Getting tools from server_name=%s", server_name)
        langchain_tools = await self._client.get_tools(server_name=server_name)
        logger.info("Retrieved %d tools from MCP servers", len(langchain_tools))

        # Wrap LangChain tools to make them Archytas-compatible
        archytas_tools = [_wrap_langchain_tool(t) for t in langchain_tools]
        return archytas_tools


# Convenience helper functions
async def mcp_tool_async(
    server_name: str,
    command: str | list[str] | None = None,
    args: list[str] | None = None,
    url: str | None = None,
    headers: dict[str, str] | None = None,
    transport: Literal["stdio", "sse", "streamable_http", "websocket"] | None = None,
    tools: list[str] | None = None,
    env: dict[str, str] | None = None,
    cwd: str | None = None,
) -> list[Callable]:
    """
    Load tools from a single MCP server (async).

    Simple helper for single-server usage. For multiple servers, use MCPClient directly.

    Args:
        server_name: Unique identifier for this server
        command: Command executable (for stdio transport)
        args: Command arguments (for stdio transport)
        url: URL for HTTP-based transports (sse, streamable_http, websocket)
        headers: HTTP headers (for sse, streamable_http)
        transport: Transport type. Auto-detected if not specified:
            - If command/args provided: stdio
            - If url starts with ws:// or wss://: websocket
            - If url provided: streamable_http (default for HTTP)
        tools: Tool filtering not supported in langchain-mcp-adapters (use get_tools then filter)
        env: Optional environment variables (stdio only)
        cwd: Optional working directory (stdio only)

    Returns:
        List of Archytas-compatible tool functions (wrapped LangChain tools)

    Examples:
        # Stdio (local server)
        tools = await mcp_tool_async(
            server_name="weather",
            command="python",
            args=["weather_server.py"]
        )

        # Streamable HTTP (default for HTTP URLs)
        tools = await mcp_tool_async(
            server_name="api",
            url="http://localhost:8000/mcp",
            headers={"API_KEY": "secret"}
        )

        # WebSocket
        tools = await mcp_tool_async(
            server_name="realtime",
            url="ws://localhost:9000"
        )
    """

    # Auto-detect transport if not specified
    if transport is None:
        if command or args:
            transport = "stdio"
        elif url:
            if url.startswith(("ws://", "wss://")):
                transport = "websocket"
            else:
                transport = "streamable_http"  # Default for HTTP
        else:
            raise ValueError("Must specify either 'command'/'args' or 'url'")

    # Build connection config using langchain-mcp-adapters types
    connection: dict

    if transport == "stdio":
        # Handle command argument flexibility
        if isinstance(command, list):
            # If command is a list, use first element as command, rest as args
            cmd = command[0]
            cmd_args = command[1:]
        elif command and args:
            cmd = command
            cmd_args = args
        elif command and not args:
            # Single command string
            if isinstance(command, str) and " " in command:
                parts = command.split()
                cmd = parts[0]
                cmd_args = parts[1:]
            else:
                cmd = command
                cmd_args = []
        else:
            raise ValueError("'command' required for stdio transport")

        connection = {
            "transport": "stdio",
            "command": cmd,
            "args": cmd_args or []
        }
        if env:
            connection["env"] = env
        if cwd:
            connection["cwd"] = cwd

    elif transport == "sse":
        if not url:
            raise ValueError("'url' required for SSE transport")
        connection = {
            "transport": "sse",
            "url": url
        }
        if headers:
            connection["headers"] = headers

    elif transport == "streamable_http":
        if not url:
            raise ValueError("'url' required for streamable_http transport")
        connection = {
            "transport": "streamable_http",
            "url": url
        }
        if headers:
            connection["headers"] = headers

    elif transport == "websocket":
        if not url:
            raise ValueError("'url' required for websocket transport")
        connection = {
            "transport": "websocket",
            "url": url
        }

    else:
        raise ValueError(
            f"Invalid transport: {transport}. "
            f"Must be one of: 'stdio', 'sse', 'streamable_http', 'websocket'"
        )

    # Create client and get tools
    client = MCPClient({server_name: connection})
    all_tools = await client.get_tools(server_name=server_name)

    # Apply tool filtering if requested
    if tools:
        filtered = [t for t in all_tools if t.name in tools]
        logger.info(
            "Filtered %d/%d tools for server '%s'",
            len(filtered), len(all_tools), server_name
        )
        return filtered

    return all_tools


def mcp_tool(
    server_name: str,
    command: str | list[str] | None = None,
    args: list[str] | None = None,
    url: str | None = None,
    headers: dict[str, str] | None = None,
    transport: Literal["stdio", "sse", "streamable_http", "websocket"] | None = None,
    tools: list[str] | None = None,
    env: dict[str, str] | None = None,
    cwd: str | None = None,
) -> list[BaseTool]:
    """
    Load tools from a single MCP server (sync wrapper).

    This is a synchronous wrapper around mcp_tool_async(). For async contexts,
    use mcp_tool_async() directly.

    See mcp_tool_async() for full documentation.
    """

    return asyncio.run(
        mcp_tool_async(
            server_name=server_name,
            command=command,
            args=args,
            url=url,
            headers=headers,
            transport=transport,
            tools=tools,
            env=env,
            cwd=cwd
        )
    )
