# Archytas: A Tools Interface for AI Agents
<img src="https://raw.githubusercontent.com/jataware/archytas/main/assets/logo.png" width="150" height="150" align="left" style="padding-right:0.5em;"/>

Implementation of the [ReAct (Reason & Action)](https://arxiv.org/abs/2210.03629) framework for Large Language Model (LLM) agents. Mainly targeting OpenAI's GPT-4.

Easily create tools from simple python functions or classes with the `@tool` decorator. A tools list can then be passed to the `ReActAgent` which will automagically generate a prompt for the LLM containing usage instructions for each tool, as well as manage the ReAct decision loop while the LLM performs its task.

Tools can be anything from internet searches to custom interpreters for your domain. Archytas provides a few built-in demo tools e.g. datetime, fibonacci numbers, and a simple calculator.

<div style="clear:left;"></div>

# Demos

Short demo of using the `PythonTool` to download a COVID-19 dataset, and perform some basic processing/visualization/analysis/etc.
<div align="center">
  <a href="https://youtu.be/52e4xN8SIi8">
    <img src="https://raw.githubusercontent.com/jataware/archytas/main/assets/covid_repl_demo.gif" alt="Watch the video">
  </a>
  <br/>
  click to watch original video on youtube
</div>

## MCP Demos

```bash
# Install with MCP support (includes langchain-mcp-adapters + fastmcp)
uv pip install -e ".[mcp]"

# Vision demo with local MCP server
python demos/demo_mcp_vision.py

# External MCP server (DuckDuckGo search via npx)
python demos/demo_mcp_duckduckgo.py

# HTTP MCP server (Context7 - requires API key)
export CONTEXT7_API_KEY="your-api-key"
python demos/demo_mcp_context7.py
```
To use mcp tools with archytas, you will need to install the optional deps
`uv pip install -e ".[mcp]"`

Read the [mcp quick start doc](docs/MCP_QUICK_START.md)  for more info

# Quickstart
```bash
# make sure poetry is installed
pip install poetry

# clone and install
git clone git@github.com:jataware/archytas.git
cd archytas
poetry install

# make sure OPENAI_API_KEY var is set
# or pass it in as an argument to the agent
export OPENAI_API_KEY="sk-..."

# run demo
poetry run chat-repl
```

# Simple Usage
Import pre-made tools from the tools module
```python
from archytas.react import ReActAgent, FailedTaskError
from archytas.tools import PythonTool

from easyrepl import REPL

# create the agent with the tools list or a `custom_prelude` (if desired)
some_tools = [PythonTool, ..., etc.]
agent = ReActAgent(tools=some_tools, verbose=True)

# REPL to interact with agent
for query in REPL():
    try:
        answer = agent.react(query)
        print(answer)
    except FailedTaskError as e:
        print(f"Error: {e}")
```

# Documentation
See the [wiki docs](https://github.com/jataware/archytas/wiki) for details.

