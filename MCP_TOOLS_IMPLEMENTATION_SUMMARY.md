# MCP Tools Integration - Implementation Summary

## Overview

Successfully implemented MCP (Model Context Protocol) server integration for Archytas agents. This feature allows Archytas agents to use tools from external MCP servers while maintaining full support for multimodal outputs (text and images).

## Features Implemented

### 1. Multimodal Tool Returns
- **Modified**: `archytas/tool_utils.py`
- Tools can now return LangChain's native multimodal format: `str | list[dict] | dict`
- Backward compatible - existing string-only tools continue to work
- Validates multimodal content structure with helpful error messages
- Supported content types: `text` and `image_url`

### 2. Display Formatting
- **Modified**: `archytas/react.py`
- Added `format_tool_result_for_display()` helper function
- Converts multimodal content to readable terminal output
- Shows image placeholders: `[Image: image/png, base64 data]`
- Maintains original multimodal content for LLM consumption

### 3. MCP Tool Bridge
- **Created**: `archytas/mcp_tools.py` (new file)
- `MCPToolBridge` class manages MCP server connections
- Automatic tool discovery and registration
- Converts MCP tools to Archytas `@tool` decorated functions
- Translates MCP results to LangChain multimodal format
- Optional dependencies - graceful error messages if MCP not installed

### 4. Simple User API
- `mcp_tool(server_name, command, tools=None)` - synchronous helper
- `mcp_tool_async(server_name, command, tools=None)` - async helper
- Tool filtering support via `tools` parameter
- Environment and working directory configuration

### 5. Optional Dependencies
- **Modified**: `pyproject.toml`
- Added `[mcp]` optional dependency group
- Install with: `pip install 'archytas[mcp]'` or `uv pip install 'archytas[mcp]'`
- Includes `fastmcp>=0.1.0` and `mcp>=1.0.0`

### 6. Bug Fixes
- Fixed pre-existing langchain import issues (`StructuredTool` moved to `langchain_core.tools`)
- Fixed typo bug in `react.py` (`summarized_record_uuid` → `summary_record_uuid`)
- All linting errors resolved

## Files Changed

### Core Implementation
- `archytas/tool_utils.py` - Multimodal return support
- `archytas/react.py` - Display formatting helper
- `archytas/mcp_tools.py` - **NEW** - Complete MCP bridge implementation
- `archytas/models/base.py` - Fixed import
- `archytas/models/openai.py` - Fixed import

### Configuration
- `pyproject.toml` - Added MCP optional dependencies

### Testing
- `tests/test_multimodal_tools.py` - **NEW** - Comprehensive test suite (23 tests)
- `tests/fixtures/mock_mcp_server.py` - **NEW** - Mock MCP server for testing

## Testing Results

**All 23 tests passing ✓**

### Test Categories
- **Multimodal Tool Returns** (7 tests)
  - String backward compatibility
  - List/dict multimodal formats
  - Validation and error handling
  - Type conversions

- **Display Formatting** (7 tests)
  - String, text content, images
  - Mixed content, unknown types
  - Base64 and remote URLs

- **MCP Bridge Core** (6 tests)
  - JSON schema to Python type mapping
  - Parameter extraction
  - Docstring generation
  - MCP result formatting

- **MCP Integration** (3 tests)
  - Server registration
  - Tool execution
  - Tool filtering

## Usage Examples

### Basic Usage

```python
from archytas.mcp_tools import mcp_tool
from archytas.react import ReActAgent

# Register MCP server tools
tools = [
    *mcp_tool(
        server_name="weather",
        command="uv run weather_server.py"
    )
]

# Create agent with MCP tools
agent = ReActAgent(model="gpt-4o", tools=tools)
response = await agent.react("What's the weather in San Francisco?")
```

### Multiple MCP Servers

```python
tools = [
    my_native_tool,  # Native Archytas tool
    *mcp_tool(
        server_name="weather",
        command="uv run weather_server.py",
        tools=["get_weather", "get_forecast"]  # Filter specific tools
    ),
    *mcp_tool(
        server_name="database",
        command=["python", "db_server.py"],
        env={"DB_HOST": "localhost"},
        cwd="/opt/servers"
    )
]
```

### Multimodal Native Tools

```python
from archytas.tool_utils import tool

@tool()
def get_chart(data: list[int]) -> list[dict]:
    """Generate a chart and return as image."""
    # Generate chart...
    image_base64 = generate_chart_base64(data)

    return [
        {"type": "text", "text": "Generated chart:"},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{image_base64}"}
        }
    ]
```

## Quick Demo & Testing

### 1. Install Dependencies

```bash
# Basic install
uv pip install -e .

# With MCP support
uv pip install -e ".[mcp]"

# With dev dependencies
uv pip install -e ".[dev,mcp]"
```

### 2. Run Tests

```bash
source .venv/bin/activate
pytest tests/test_multimodal_tools.py -v
```

Expected output: `23 passed in ~2s`

### 3. Test with Mock MCP Server

```python
# tests/fixtures/mock_mcp_server.py provides:
# - echo(message: str) -> text
# - add(a: int, b: int) -> text
# - generate_image() -> image

import asyncio
import sys
from archytas.mcp_tools import mcp_tool_async

async def demo():
    tools = await mcp_tool_async(
        server_name="test",
        command=[sys.executable, "tests/fixtures/mock_mcp_server.py"]
    )

    # Test echo tool
    echo_tool = next(t for t in tools if t.__name__ == "echo")
    result = await echo_tool.run({"message": "Hello!"}, {})
    print(result)
    # Output: [{'type': 'text', 'text': 'Echo: Hello!'}]

asyncio.run(demo())
```

### 4. Create Custom MCP Server

```python
# my_server.py
from fastmcp import FastMCP

mcp = FastMCP("my-tools")

@mcp.tool()
def greet(name: str) -> dict:
    """Greet someone by name."""
    return {
        "content": [{
            "type": "text",
            "text": f"Hello, {name}!"
        }]
    }

if __name__ == "__main__":
    mcp.run()
```

Then use it:

```python
tools = mcp_tool(
    server_name="myserver",
    command="python my_server.py"
)
```

## Architecture Decisions

### 1. LangChain Native Format
- Uses LangChain's existing `{"type": "text"|"image_url", ...}` format
- No custom content block classes needed
- Maximum compatibility with LangChain ecosystem
- Future-proof as LangChain evolves

### 2. Optional Dependencies
- MCP dependencies not required for core Archytas functionality
- Clear error messages guide users to install extras
- Keeps base installation lightweight

### 3. Singleton Bridge Pattern
- Reuses MCP client connections for performance
- Automatic connection management
- Single global bridge per process

### 4. Backward Compatibility
- All existing string-returning tools continue to work
- Runtime validation with helpful error messages
- No breaking changes to existing code

## Technical Details

### JSON Schema to Python Mapping
- `string` → `str`
- `integer` → `int`
- `number` → `float`
- `boolean` → `bool`
- `array` → `list`
- `object` → `dict`
- `null` → `type(None)`

### MCP to LangChain Translation
- `mcp.types.TextContent` → `{"type": "text", "text": "..."}`
- `mcp.types.ImageContent` → `{"type": "image_url", "image_url": {"url": "data:mime;base64,..."}}`

### Tool Function Generation
- Dynamically creates functions with proper signatures
- Preserves parameter types from JSON schemas
- Generates comprehensive docstrings
- Applies `@tool` decorator automatically

## Performance Considerations

- **Connection Reuse**: MCP clients kept alive (StdioTransport `keep_alive=True`)
- **Parallel Execution**: Multiple tools can execute concurrently
- **Lazy Loading**: MCP bridge only created when first tool registered
- **Minimal Overhead**: Direct conversion without intermediate formats

## Future Enhancements (Not Implemented)

- Audio/video content support
- Tool result caching
- Connection pooling
- Health checks and auto-recovery
- Configuration file support
- Hot reload capabilities

## Known Limitations

- Only `text` and `image_url` content types currently supported
- No automatic image compression/resizing
- Synchronous `mcp_tool()` blocks during server registration
- One global bridge per process (not configurable)

## Documentation

- Comprehensive docstrings in all functions
- Type annotations throughout
- Usage examples in docstrings
- This implementation summary

## Branch

- **Feature Branch**: `feature/mcp-tools-integration`
- **Base Branch**: `main`

## Next Steps

1. Merge feature branch to main after review
2. Update main README with MCP tools section
3. Add example MCP servers to repository
4. Consider adding to official Archytas documentation
5. Monitor for issues with real-world MCP servers

---

**Implementation Date**: 2025-10-21
**Tests**: 23/23 passing ✓
**Linting**: All checks passed ✓
**Python Version**: 3.13.1
**Key Dependencies**: fastmcp>=0.1.0, mcp>=1.0.0
