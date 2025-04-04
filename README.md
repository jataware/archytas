# Archytas: A Tools Interface for AI Agents
<img src="assets/logo.png" width="150" height="150" align="left" style="padding-right:0.5em;"/>

Implementation of the [ReAct (Reason & Action)](https://arxiv.org/abs/2210.03629) framework for Large Language Model (LLM) agents. Mainly targeting OpenAI's GPT-4.

Easily create tools from simple python functions or classes with the `@tool` decorator. A tools list can then be passed to the `ReActAgent` which will automagically generate a prompt for the LLM containing usage instructions for each tool, as well as manage the ReAct decision loop while the LLM performs its task.

Tools can be anything from internet searches to custom interpreters for your domain. Archytas provides a few built-in demo tools e.g. datetime, fibonacci numbers, and a simple calculator.

<div style="clear:left;"></div>

# Demo
Short demo of using the `PythonTool` to download a COVID-19 dataset, and perform some basic processing/visualization/analysis/etc.
<div align="center">
  <a href="https://youtu.be/52e4xN8SIi8">
    <img src="assets/covid_repl_demo.gif" alt="Watch the video">
  </a>
  <br/>
  click to watch original video on youtube
</div>

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
