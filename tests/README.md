# Archytas Test Suite

This directory contains the test suite for Archytas, focusing on testing ReAct agents with deterministic outputs (temperature=0).

## Structure

```
tests/
├── conftest.py              # Shared pytest fixtures
├── pytest.ini               # Pytest configuration
├── test_react_basic.py      # Basic ReAct agent tests
├── test_tools.py            # Tool creation and execution tests
├── test_python_tool.py      # PythonTool specific tests
├── test_context_management.py # Context and auto-context tests
├── test_chat_history.py     # Chat history management tests
└── legacy/                  # Old tests (deprecated)
```

## Running Tests

### Prerequisites

The test suite supports multiple LLM providers. Set the API keys for the providers you want to test:

```bash
# OpenAI (GPT models)
export OPENAI_API_KEY="sk-..."

# Anthropic (Claude models)
export ANTHROPIC_API_KEY="sk-ant-..."

# Google Gemini
export GEMINI_API_KEY="..."
```

By default, tests will run against all providers with available API keys. You can limit testing to specific providers using the `--model-provider` flag.

### Run All Tests (All Providers)
```bash
pytest
# or for faster parallel execution
pytest -n auto
```

### Run Tests for Specific Provider
```bash
pytest --model-provider=openai
pytest --model-provider=anthropic
pytest --model-provider=gemini
```

### Run Specific Test File
```bash
pytest tests/test_react_basic.py
```

### Run Specific Test Class
```bash
pytest tests/test_react_basic.py::TestBasicReActLoop
```

### Run Specific Test
```bash
pytest tests/test_react_basic.py::TestBasicReActLoop::test_react_simple_query_no_tools
```

### Run with Verbose Output
```bash
pytest -v
```

### Run Tests Matching Pattern
```bash
pytest -k "tool"  # Runs all tests with "tool" in the name
```

### Run Only Async Tests
```bash
pytest -m asyncio
```

### Skip Slow Tests
```bash
pytest -m "not slow"
```

## Test Categories

### Unit Tests
Tests that don't require external API calls:
- Tool decorator validation
- Tool signature parsing
- Type normalization

### Integration Tests
Tests that require API keys and make actual LLM calls:
- ReAct agent execution
- Tool execution with agents
- Context management
- Chat history

Most tests in this suite are integration tests since they validate end-to-end agent behavior.

## Writing New Tests

### Basic Test Structure
```python
def test_my_feature(react_agent_with_tools):
    """Test description."""
    @tool()
    def my_tool(x: int) -> str:
        """
        Tool description.

        Args:
            x (int): Description

        Returns:
            str: Description
        """
        return str(x * 2)

    agent = react_agent_with_tools([my_tool])
    result = agent.react("Use my_tool with 21")

    assert "42" in result
```

### Async Test Structure
```python
@pytest.mark.asyncio
async def test_async_feature(react_agent):
    """Test async functionality."""
    result = await react_agent.react_async("What is 2+2?")
    assert "4" in result
```

### Testing Deterministic Outputs

All tests use `temperature=0.0` to get deterministic outputs from the LLM. This means:
- Same queries should produce consistent tool calls
- Numeric calculations should be exact
- String manipulations should be predictable

However, the exact phrasing of the final answer may vary slightly, so tests should:
- Assert on key facts/numbers being present
- Check for specific tool outputs
- Verify expected tool calls were made
- Not require exact string matches for natural language responses

## Fixtures

### `api_key`
Retrieves OpenAI API key from environment, skips test if not available.

### `openai_model`
Creates an OpenAI model instance (gpt-4o-mini) with the API key.

### `react_agent`
Creates a basic ReActAgent with temperature=0, no custom tools.

### `react_agent_with_tools`
Factory fixture that creates a ReActAgent with custom tools.

Usage:
```python
def test_example(react_agent_with_tools):
    agent = react_agent_with_tools([tool1, tool2], max_errors=5)
    # ... test code
```

## Common Patterns

### Testing Tool Execution
```python
@tool()
def my_tool(x: int) -> str:
    """Tool docstring..."""
    return str(x)

agent = react_agent_with_tools([my_tool])
result = agent.react("Use my_tool with 42")
assert "42" in result
```

### Testing Error Handling
```python
from archytas.react import FailedTaskError

with pytest.raises(FailedTaskError):
    agent.react("Impossible task")
```

### Testing Async Methods
```python
@pytest.mark.asyncio
async def test_async(react_agent):
    result = await react_agent.react_async("query")
    assert isinstance(result, str)
```

### Testing Context
```python
agent.add_context("Important info")
result = agent.react("What did I tell you?")
assert "Important info" in result
```

## Troubleshooting

### Tests Failing Due to API Rate Limits
If you hit rate limits, add delays between tests or reduce parallelism:
```bash
pytest -n 1  # Run tests sequentially
```

### Tests Skipping Due to Missing API Key
Export your OpenAI API key:
```bash
export OPENAI_API_KEY="sk-..."
```

### Flaky Tests Due to LLM Non-Determinism
Even with temperature=0, LLMs may occasionally vary. If a test is flaky:
- Make assertions more flexible
- Check for key facts rather than exact strings
- Verify tool calls instead of natural language responses

### Async Test Issues
Make sure tests are marked with `@pytest.mark.asyncio` and pytest-asyncio is installed:
```bash
pip install pytest-asyncio
```
