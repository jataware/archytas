# Archytas: A Tools Interface for AI Agents
<img src="assets/logo.png" width="150" height="150" align="left" style="padding-right:0.5em;"/>

Implementation of the [ReAct (Reason & Action)](https://arxiv.org/abs/2210.03629) framework for Large Language Model (LLM) agents. Mainly targeting OpenAI's GPT-4.

Easily create tools from simple python functions or classes with the `@tool` decorator. A tools list can then be passed to the `ReActAgent` which will automagially generate a prompt for the LLM containing usage instructions for each tool, as well as manage the ReAct decision loop while the LLM performs its task.

Tools can be anything from internet searches to custom interpreters for your domain. Archytas provides a few built-in demo tools e.g. datetime, fibonacci numbers, and a simple calculator.

<div style="clear:left;"></div>
 
# Quicksart
```bash
# make sure poetry is installed
pip install poetry

# clone and install
git clone git@github.com:jataware/archytas.git
cd archytas
poetry install

# make sure OPENAI_API_KEY var is set
# or set openai_key in .openai.toml
export OPENAI_API_KEY="sk-..."

# run demo
poetry run chat-repl
```

# Simple Usage
Import pre-made tools from the tools module
```python
from archytas.react import ReActAgent, FailedTaskError
from archytas.tools import datetime_tool, fib_n, calculator

from easyrepl import REPL

# create the agent with the tools list
some_tools = [datetime_tool, fib_n, calculator]
agent = ReActAgent(tools=some_tools+[mytool], verbose=True)

# REPL to interact with agent
for query in REPL()
    try:
        answer = agent.react(query)
        print(answer)
    except FailedTaskError as e:
        print(f"Error: {e}")
```

## Built-in Tools
(TODO)
- ask_user
- datetime
- timestamp
- fib_n
- calculator
- ...

# Custom Tools
(TODO)
```python
from archytas.tools import tool

@tool()
def example_tool(arg1:int, arg2:str='', arg3:dict=None) -> int:
    """
    Simple 1 sentence description of the tool

    More detailed description of the tool. This can be multiple lines.
    Explain more what the tool does, and what it is used for.

    Args:
        arg1 (int): Description of the first argument.
        arg2 (str): Description of the second argument. Defaults to ''.
        arg3 (dict): Description of the third argument. Defaults to {}.

    Returns:
        int: Description of the return value

    Examples:
        >>> example_tool(1, 'hello', {'a': 1, 'b': 2})
        3
        >>> example_tool(2, 'world', {'a': 1, 'b': 2})
        4
    """
    return 42

# TODO: class tool example
```