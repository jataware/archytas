from archytas.react import ReActAgent, FailedTaskError
from archytas.tools import ask_user, datetime, timestamp, fib_n, example_tool, calculator

from rich import traceback, print; traceback.install()
from easyrepl import REPL

import pdb


def main():

    # make a list of the tools to use
    # tools = [calculator, ask_user, datetime, timestamp]
    tools = [ask_user, datetime, timestamp, fib_n, example_tool, calculator]

    # create the agent
    agent = ReActAgent(tools=tools, verbose=True)

    # run the REPL
    for query in REPL(history_file='chat_history.txt'):
        try:
            answer = agent.react(query)
            print(f'[green]{answer}[/green]')
        except FailedTaskError as e:
            print(f"[red]{e}[/red]")




if __name__ == '__main__':
    main()