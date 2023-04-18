from archytas.react import ReActAgent, FailedTaskError
from archytas.tools import datetime_tool, timestamp
from archytas.demo_tools import fib_n, example_tool, calculator, Jackpot

from rich import traceback, print; traceback.install()
from easyrepl import REPL

import pdb


def start_repl():
    # make an instance of a class tool
    jackpot = Jackpot(chips=1000)

    # make a list of the tools to use
    tools = [datetime_tool, timestamp, fib_n, example_tool, calculator, jackpot]

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
        except KeyboardInterrupt:
            print('[yellow]KeyboardInterrupt[/yellow]')




if __name__ == '__main__':
    start_repl()