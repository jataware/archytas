# from agent import Agent
# from prompt import prompt
import json
from rich import traceback, print; traceback.install()
from archytas.react import ReAct, FailedTaskError
from archytas.tools import calculator, ask_user, datetime_tool, timestamp

from easyrepl import REPL, readl
history_file = 'chat_history.txt'


import pdb


"""
[notes]
- GPT-3.5 doesn't follow directions. It will add extra comment text to the beginning of its response.
    
    >>> can you tell me what my age squared is?
    Sure! To calculate your age squared, I will use the calculator tool. Here's what I'm thinking:

    {
    "thought": "I need to use the calculator to find your age squared.",
    "tool": "calculator",
    "tool_input": "age^2"
    }

    Please replace "age" with your actual age.

"""


def main():

    # agent = Agent(prompt=prompt, model='gpt-4')
    tools = {
        'calculator': calculator,
        'ask_user': ask_user,
        'datetime': datetime_tool,
        'timestamp': timestamp,
    }

    agent = ReAct(tools=tools, verbose=True)

    for query in REPL(history_file=history_file):
        if not query: continue
        try:
            answer = agent.react(query)
            print(f'[green]{answer}[/green]')
        except FailedTaskError as e:
            print(f"[red]{e}[/red]")

    # pdb.set_trace()
    # 1


if __name__ == '__main__':
    main()