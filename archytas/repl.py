from archytas.react import ReActAgent, FailedTaskError
from archytas.tools import datetime_tool, timestamp, PythonTool
from archytas.demo_tools import fib_n, example_tool, calculator, Jackpot, ModelSimulation, pirate_subquery

from rich import traceback, print; traceback.install(show_locals=True)
from easyrepl import REPL

import pdb


def start_repl():
    # make an instance of a class tool
    jackpot = Jackpot(chips=1000)

    # make a list of the tools to use
    tools = [datetime_tool, timestamp, fib_n, example_tool, calculator, jackpot, ModelSimulation, PythonTool, pirate_subquery]

    # # example of making a python tool with a prelude and some pre-initialized local variables
    # import numpy as np
    # python = PythonTool(
    #     prelude='import numpy as np\nfrom matplotlib import pyplot as plt',
    #     locals={'fib_n': fib_n, 'jackpot': jackpot, 'ModelSimulation': ModelSimulation},
    # )
    # tools = [python]

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