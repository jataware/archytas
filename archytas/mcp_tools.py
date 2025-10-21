"""
MCP tool bridge for Archytas agents.

Enables MCP (Model Context Protocol) servers to be used as Archytas @tool functions.
Returns results in LangChain's native multimodal message format.

Basic Usage:
    from archytas.mcp_tools import mcp_tool
    from archytas.react import ReActAgent

    tools = [
        *mcp_tool(
            server_name="weather",
            command="uv run weather_server.py",
            tools=["get_weather"]  # optional filter
        )
    ]

    agent = ReActAgent(model="gpt-4o", tools=tools)
    response = await agent.react("What's the weather in SF?")
"""

import asyncio
import inspect
import logging
from dataclasses import dataclass
from typing import Any, Callable

try:
    import mcp
    from fastmcp import Client
    from fastmcp.client.transports import StdioTransport
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    _import_error_msg = (
        "MCP dependencies not installed. "
        "Install with: pip install 'archytas[mcp]' or uv pip install 'archytas[mcp]'"
    )

from archytas.tool_utils import sanitize_toolname, tool

logger = logging.getLogger(__name__)


@dataclass
class ToolParameter:
    """Parameter information for tool function."""
    name: str
    type: type
    description: str
    required: bool


@dataclass
class MCPServerHandle:
    """Handle for an MCP server connection."""
    name: str
    client: Client[Any]
    tools: dict[str, Any]


class MCPToolBridge:
    """
    Bridge MCP servers to Archytas @tool functions.

    Internal class. Use mcp_tool() helper function instead.
    """

    def __init__(self):
        if not MCP_AVAILABLE:
            raise ImportError(_import_error_msg)
        self._servers: dict[str, MCPServerHandle] = {}

    async def register_server(
        self,
        server_name: str,
        command: list[str],
        tool_filter: list[str] | None = None,
        **transport_kwargs
    ) -> list[Callable]:
        """
        Register MCP server and return Archytas-compatible tool functions.

        Args:
            server_name: Unique server identifier
            command: Command to start server (e.g., ["uv", "run", "server.py"])
            tool_filter: Optional list of tool names to register (None = all)
            **transport_kwargs: Additional StdioTransport args (env, cwd, etc.)

        Returns:
            List of Archytas @tool decorated functions
        """
        logger.info("Registering MCP server '%s' with command: %s", server_name, command)

        # Create transport and client
        transport = StdioTransport(command=command[0], args=command[1:], **transport_kwargs)
        client = Client(transport)

        try:
            # Connect and discover tools
            await client.__aenter__()
            mcp_tools = await client.list_tools()
            logger.info("Discovered %d tools from server '%s'", len(mcp_tools), server_name)

            # Build tool dict
            tool_dict = {mcp_tool.name: mcp_tool for mcp_tool in mcp_tools}

            # Store server handle
            self._servers[server_name] = MCPServerHandle(
                name=server_name,
                client=client,
                tools=tool_dict
            )

            # Filter tools if requested
            if tool_filter:
                filtered_tools = {name: tool_dict[name] for name in tool_filter if name in tool_dict}
                missing = set(tool_filter) - set(filtered_tools.keys())
                if missing:
                    logger.warning("Tools not found in server '%s': %s", server_name, missing)
            else:
                filtered_tools = tool_dict

            # Create Archytas tool functions
            archytas_tools = [
                self._create_tool_function(mcp_tool, server_name)
                for mcp_tool in filtered_tools.values()
            ]

            logger.info("Created %d Archytas tools from server '%s'", len(archytas_tools), server_name)
            return archytas_tools

        except Exception as e:
            logger.exception("Failed to register MCP server '%s'", server_name)
            # Clean up on failure
            try:
                await client.__aexit__(None, None, None)
            except Exception:
                pass
            raise RuntimeError(f"Failed to register MCP server '{server_name}': {e}") from e

    def _create_tool_function(
        self,
        mcp_tool: Any,
        server_name: str
    ) -> Callable:
        """
        Create Archytas @tool function from MCP tool.

        The generated function:
        1. Has proper type annotations from JSON schema
        2. Includes comprehensive docstring
        3. Returns list[dict] in LangChain multimodal format
        4. Is decorated with @tool
        """
        tool_name = sanitize_toolname(mcp_tool.name)
        params = self._extract_parameters(mcp_tool.inputSchema)
        docstring = self._build_docstring(mcp_tool, params)

        # Create async wrapper function
        async def mcp_tool_wrapper(**kwargs) -> list[dict]:
            return await self._execute_tool(server_name, mcp_tool.name, **kwargs)

        # Build function signature
        sig_params = [
            inspect.Parameter(
                name=param.name,
                kind=inspect.Parameter.KEYWORD_ONLY,
                annotation=param.type,
                default=inspect.Parameter.empty if param.required else None
            )
            for param in params
        ]

        # Apply metadata
        mcp_tool_wrapper.__name__ = tool_name
        mcp_tool_wrapper.__doc__ = docstring
        mcp_tool_wrapper.__signature__ = inspect.Signature(
            parameters=sig_params,
            return_annotation=list[dict]
        )
        mcp_tool_wrapper.__annotations__ = {
            p.name: p.type for p in params
        } | {"return": list[dict]}

        # Apply @tool decorator
        return tool(mcp_tool_wrapper)

    async def _execute_tool(
        self,
        server_name: str,
        tool_name: str,
        **kwargs
    ) -> list[dict]:
        """
        Execute MCP tool and return LangChain multimodal format.

        Returns:
            List of content dicts in LangChain format:
            [
                {"type": "text", "text": "..."},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
            ]
        """
        if server_name not in self._servers:
            raise ValueError(f"MCP server '{server_name}' not registered")

        handle = self._servers[server_name]
        logger.debug("Executing MCP tool '%s' on server '%s' with args: %s", tool_name, server_name, kwargs)

        try:
            result = await handle.client.call_tool(tool_name, kwargs)
            return self._format_mcp_result(result)
        except Exception as e:
            logger.exception("Error executing MCP tool '%s' on server '%s'", tool_name, server_name)
            # Return error as text content
            return [{"type": "text", "text": f"Error executing tool: {e}"}]

    def _extract_parameters(
        self,
        json_schema: dict[str, Any]
    ) -> list[ToolParameter]:
        """
        Extract parameters from JSON schema.

        Args:
            json_schema: MCP tool input schema

        Returns:
            List of parameter definitions
        """
        parameters = []
        properties = json_schema.get("properties", {})
        required = set(json_schema.get("required", []))

        for param_name, param_schema in properties.items():
            param_type = self._json_type_to_python(param_schema)
            param_desc = param_schema.get("description", "")

            parameters.append(ToolParameter(
                name=param_name,
                type=param_type,
                description=param_desc,
                required=param_name in required
            ))

        return parameters

    def _json_type_to_python(self, param_schema: dict[str, Any]) -> type:
        """
        Map JSON schema type to Python type.

        Mapping:
        - string -> str
        - integer -> int
        - number -> float
        - boolean -> bool
        - array -> list
        - object -> dict
        - null -> None
        """
        json_type = param_schema.get("type", "string")

        match json_type:
            case "string":
                return str
            case "integer":
                return int
            case "number":
                return float
            case "boolean":
                return bool
            case "array":
                return list
            case "object":
                return dict
            case "null":
                return type(None)
            case _:
                logger.warning("Unknown JSON type '%s', defaulting to str", json_type)
                return str

    def _build_docstring(
        self,
        mcp_tool: Any,
        params: list[ToolParameter]
    ) -> str:
        """
        Build Archytas-compatible docstring.

        Format matches Archytas conventions:
        - Short description
        - Args section with type annotations
        - Returns section
        """
        lines = [mcp_tool.description or "MCP tool"]

        if params:
            lines.append("")
            lines.append("Args:")
            for param in params:
                optional = "" if param.required else " (optional)"
                type_name = param.type.__name__
                lines.append(f"    {param.name} ({type_name}): {param.description}{optional}")

        lines.append("")
        lines.append("Returns:")
        lines.append("    list[dict]: Tool result in LangChain multimodal format")

        return "\n".join(lines)

    def _format_mcp_result(self, result: Any) -> list[dict]:
        """
        Convert MCP CallToolResult to LangChain multimodal format.

        Transforms MCP content blocks to LangChain's standard format.

        Returns:
            List of content dicts:
            [
                {"type": "text", "text": "..."},
                {"type": "image_url", "image_url": {"url": "data:..."}}
            ]
        """
        content_blocks = []

        for block in result.content:
            match block:
                case mcp.types.TextContent(text=text):
                    content_blocks.append({
                        "type": "text",
                        "text": text
                    })

                case mcp.types.ImageContent(data=data, mimeType=mime_type):
                    # Convert to base64 data URL format
                    image_url = f"data:{mime_type};base64,{data}"
                    content_blocks.append({
                        "type": "image_url",
                        "image_url": {"url": image_url}
                    })

                case _:
                    # Unknown content type - convert to text
                    content_blocks.append({
                        "type": "text",
                        "text": f"[{block.type}]: {str(block)}"
                    })

        return content_blocks

    async def close(self):
        """Close all MCP connections."""
        logger.info("Closing %d MCP server connections", len(self._servers))
        for handle in self._servers.values():
            try:
                await handle.client.__aexit__(None, None, None)
            except Exception:
                logger.exception("Error closing MCP server '%s'", handle.name)


# Singleton bridge
_global_bridge: MCPToolBridge | None = None


def mcp_tool(
    server_name: str,
    command: str | list[str],
    tools: list[str] | None = None,
    env: dict[str, str] | None = None,
    cwd: str | None = None,
) -> list[Callable]:
    """
    Convert MCP server tools to Archytas @tool functions.

    Usage:
        from archytas.mcp_tools import mcp_tool
        from archytas.react import ReActAgent

        tools = [
            my_native_tool,
            *mcp_tool(
                server_name="weather",
                command="uv run weather_server.py"
            )
        ]

        agent = ReActAgent(model="gpt-4o", tools=tools)

    Args:
        server_name: Unique identifier for this MCP server
        command: Command to start server (string or list)
        tools: Optional tool name filter (None = all tools)
        env: Optional environment variables for server
        cwd: Optional working directory for server

    Returns:
        List of @tool decorated functions

    Raises:
        ImportError: If MCP dependencies not installed
        ValueError: If command format invalid or server registration fails
        RuntimeError: If MCP server connection fails
    """
    if not MCP_AVAILABLE:
        raise ImportError(_import_error_msg)

    global _global_bridge

    if _global_bridge is None:
        _global_bridge = MCPToolBridge()

    # Convert string command to list
    if isinstance(command, str):
        command = command.split()

    # Build transport kwargs
    transport_kwargs = {}
    if env:
        transport_kwargs["env"] = env
    if cwd:
        transport_kwargs["cwd"] = cwd

    # Register server synchronously (wraps async call)
    return asyncio.run(
        _global_bridge.register_server(
            server_name=server_name,
            command=command,
            tool_filter=tools,
            **transport_kwargs
        )
    )


async def mcp_tool_async(
    server_name: str,
    command: str | list[str],
    tools: list[str] | None = None,
    env: dict[str, str] | None = None,
    cwd: str | None = None,
) -> list[Callable]:
    """
    Async version of mcp_tool() for use in async contexts.

    Usage:
        tools = [
            my_native_tool,
            *await mcp_tool_async(
                server_name="weather",
                command="uv run weather_server.py"
            )
        ]

    Args:
        server_name: Unique identifier for this MCP server
        command: Command to start server (string or list)
        tools: Optional tool name filter (None = all tools)
        env: Optional environment variables for server
        cwd: Optional working directory for server

    Returns:
        List of @tool decorated functions

    Raises:
        ImportError: If MCP dependencies not installed
        ValueError: If command format invalid or server registration fails
        RuntimeError: If MCP server connection fails
    """
    if not MCP_AVAILABLE:
        raise ImportError(_import_error_msg)

    global _global_bridge

    if _global_bridge is None:
        _global_bridge = MCPToolBridge()

    if isinstance(command, str):
        command = command.split()

    transport_kwargs = {}
    if env:
        transport_kwargs["env"] = env
    if cwd:
        transport_kwargs["cwd"] = cwd

    return await _global_bridge.register_server(
        server_name=server_name,
        command=command,
        tool_filter=tools,
        **transport_kwargs
    )
