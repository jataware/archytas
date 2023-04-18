from archytas.react import ReActAgent, FailedTaskError
from archytas.tools import datetime_tool, timestamp, fib_n, example_tool, calculator

from rich import traceback, print; traceback.install()
from easyrepl import REPL

import pdb


def start_repl():

    # make a list of the tools to use
    # tools = [calculator, datetime_tool, timestamp]
    tools = [datetime_tool, timestamp, fib_n, example_tool, calculator]

    # create the agent
    agent = ReActAgent(tools=tools, verbose=True)

    # print the agent's prompt
    # print(agent.prompt)

    # run the REPL
    for query in REPL(history_file='chat_history.txt'):
        try:
            answer = agent.react(query)
            print(f'[green]{answer}[/green]')
        except FailedTaskError as e:
            print(f"[red]{e}[/red]")




if __name__ == '__main__':
    start_repl()